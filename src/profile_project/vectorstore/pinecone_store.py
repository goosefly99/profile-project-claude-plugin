# src/profile_project/vectorstore/pinecone_store.py
from __future__ import annotations

from pinecone import Pinecone

from profile_project.vectorstore.protocols import QueryResult


class IndexDimensionMismatch(Exception):
    """Raised when an existing Pinecone index's dimension != the embedder's.

    This is the last-line geometry guard (§10.1, non-goal): the plugin connects
    to existing indexes only and refuses to write into a mismatched geometry.
    """

    def __init__(self, index: str, index_dim: int, embedding_dim: int) -> None:
        super().__init__(
            f"Pinecone index {index!r} has dimension {index_dim} but the "
            f"configured embedder emits {embedding_dim}; refusing to write into a "
            f"mismatched geometry (index is never auto-created)."
        )
        self.index = index
        self.index_dim = index_dim
        self.embedding_dim = embedding_dim


class PineconeStore:
    """Remote, existing-index-only store (§10.1).

    NEVER creates indexes. Flow: has_index -> describe_index -> validate the
    effective embedding dimension against the index dimension (raise
    IndexDimensionMismatch on mismatch) -> confirm ready -> connect by host.
    Default path is BYO-vectors: index.upsert(vectors=...) / index.query(...).
    """

    def __init__(
        self,
        api_key: str,
        index: str,
        embedding_dim: int,
        collection: str,
        namespace: str | None = None,
    ) -> None:
        self._index_name = index
        # Pinecone uses "" for the default namespace; never None at call sites.
        self._namespace: str = namespace if namespace is not None else ""
        self._collection = collection
        pc = Pinecone(api_key=api_key)
        if not pc.has_index(index):
            raise ValueError(
                f"Pinecone index {index!r} does not exist; profile-project never "
                f"creates indexes. Create it first or point at an existing index."
            )
        desc = pc.describe_index(index)
        index_dim: int = int(desc.dimension) if desc.dimension is not None else -1
        if index_dim != embedding_dim:
            raise IndexDimensionMismatch(index, index_dim, embedding_dim)
        if not desc.status["ready"]:
            raise ValueError(f"Pinecone index {index!r} is not ready.")
        # Connect by host (production-safe), resolved once.
        host: str = str(desc.host) if desc.host is not None else ""
        self._index = pc.Index(host=host)

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, object]] | None = None,
    ) -> None:
        metas: list[dict[str, object]] = (
            metadatas if metadatas is not None else [{} for _ in ids]
        )
        vectors: list[dict[str, object]] = []
        for chunk_id, values, document, meta in zip(
            ids, embeddings, documents, metas, strict=True
        ):
            # Pinecone has no document column; fold it into metadata (scalars only).
            payload: dict[str, object] = dict(meta)
            payload["document"] = document
            vectors.append({"id": chunk_id, "values": values, "metadata": payload})
        self._index.upsert(vectors=vectors, namespace=self._namespace)

    def query(
        self,
        embedding: list[float],
        top_k: int = 10,
        where: dict[str, object] | None = None,
        where_document: dict[str, object] | None = None,
    ) -> list[QueryResult]:
        res = self._index.query(
            vector=embedding,
            top_k=top_k,
            namespace=self._namespace,
            filter=where,
            include_metadata=True,
        )
        out: list[QueryResult] = []
        for match in res.get("matches", []):
            meta: dict[str, object] = dict(match.get("metadata") or {})
            document = str(meta.pop("document", ""))
            out.append(
                QueryResult(
                    id=str(match["id"]),
                    document=document,
                    score=float(match["score"]),
                    metadata=meta,
                )
            )
        return out

    def delete(
        self,
        ids: list[str] | None = None,
        where: dict[str, object] | None = None,
    ) -> None:
        if ids is not None:
            self._index.delete(ids=ids, namespace=self._namespace)
        elif where is not None:
            self._index.delete(filter=where, namespace=self._namespace)
        else:
            self._index.delete(delete_all=True, namespace=self._namespace)

    def count(self) -> int:
        stats = self._index.describe_index_stats()
        return int(stats.get("total_vector_count", 0))

    def reset(self) -> None:
        # Never delete the (existing, user-owned) index; clear our namespace only.
        self._index.delete(delete_all=True, namespace=self._namespace)
