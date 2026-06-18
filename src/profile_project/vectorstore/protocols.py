from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class QueryResult:
    """A single ranked hit returned by ``VectorStore.query`` (sec.10.1).

    ``score`` is a cosine similarity in ``[0.0, 1.0]`` (the store converts the
    backend distance via ``max(0.0, min(1.0, 1.0 - distance))``); ``metadata``
    carries the non-secret routing fields stamped at index time (sec.10.4).
    """

    id: str
    document: str
    score: float
    metadata: dict[str, object]


@runtime_checkable
class Embedder(Protocol):
    """Embedding provider contract (sec.10.3).

    The same embedder MUST be used on write and read so query vectors share
    geometry with the stored vectors; ``embedder_version`` is the byte-identical
    geometry guard (sec.10.3).
    """

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    @property
    def dimension(self) -> int:  # may probe for remote providers
        ...

    @property
    def model_name(self) -> str: ...

    @property
    def embedder_version(self) -> str:  # canonical "<provider>/<model>@<tag>"
        ...

    @property
    def embedding_provider(self) -> str:  # 'sentence-transformers'|'openai'|'ollama'
        ...

    def probe_dimension(self) -> int:  # one bounded dry-run embed; cached
        ...


class VectorStore(Protocol):
    """Vector store contract over precomputed embeddings (sec.10.1/sec.10.3).

    Embeddings are computed by the caller (same embedder on write and read) and
    passed in directly; the store never embeds. ``add`` is upsert semantics so
    re-indexing the same id overwrites idempotently.
    """

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, object]] | None = None,
    ) -> None: ...

    def query(
        self,
        embedding: list[float],
        top_k: int = 10,
        where: dict[str, object] | None = None,
        where_document: dict[str, object] | None = None,
    ) -> list[QueryResult]: ...

    def delete(
        self, ids: list[str] | None = None, where: dict[str, object] | None = None
    ) -> None: ...

    def count(self) -> int: ...

    def reset(self) -> None: ...
