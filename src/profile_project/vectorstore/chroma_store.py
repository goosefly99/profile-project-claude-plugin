# src/profile_project/vectorstore/chroma_store.py
from __future__ import annotations

import chromadb

from profile_project.vectorstore.protocols import QueryResult


class ChromaDBStore:
    """Local default store (§10.1).

    Embeddings are precomputed by the caller (same embedder on write and read)
    and passed straight to ``upsert``. The collection is created with cosine
    space; ``hnsw:space`` is immutable after creation, so ``reset()`` (delete +
    recreate) is the only way to change geometry.
    """

    def __init__(self, persist_path: str, collection: str) -> None:
        self._persist_path = persist_path
        self._collection_name = collection
        # The Path -> str cast is load-bearing: PersistentClient wants a str.
        self._client = chromadb.PersistentClient(path=str(persist_path))
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, object]] | None = None,
    ) -> None:
        # upsert (not add) so re-indexing the same id overwrites idempotently.
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,  # type: ignore[arg-type]
            documents=documents,
            metadatas=metadatas,  # type: ignore[arg-type]
        )

    def query(
        self,
        embedding: list[float],
        top_k: int = 10,
        where: dict[str, object] | None = None,
        where_document: dict[str, object] | None = None,
    ) -> list[QueryResult]:
        res = self._collection.query(
            query_embeddings=[embedding],  # type: ignore[arg-type]
            n_results=top_k,
            where=where,  # type: ignore[arg-type]
            where_document=where_document,  # type: ignore[arg-type]
        )
        ids = res.get("ids") or [[]]
        if not ids[0]:  # guard the empty case
            return []
        documents = res.get("documents") or [[]]
        metadatas = res.get("metadatas") or [[]]
        distances = res.get("distances") or [[]]
        out: list[QueryResult] = []
        for i, chunk_id in enumerate(ids[0]):
            distance = float(distances[0][i])
            # cosine distance -> similarity, clamped into [0, 1]
            score = max(0.0, min(1.0, 1.0 - distance))
            meta = metadatas[0][i]
            out.append(
                QueryResult(
                    id=str(chunk_id),
                    document=str(documents[0][i]),
                    score=score,
                    metadata=dict(meta) if meta else {},
                )
            )
        return out

    def delete(
        self,
        ids: list[str] | None = None,
        where: dict[str, object] | None = None,
    ) -> None:
        self._collection.delete(ids=ids, where=where)  # type: ignore[arg-type]

    def count(self) -> int:
        # count() doubles as an HNSW warm-load.
        return int(self._collection.count())

    def reset(self) -> None:
        # hnsw:space is immutable, so delete + recreate to reset geometry.
        self._client.delete_collection(name=self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
