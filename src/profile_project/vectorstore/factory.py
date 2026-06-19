# src/profile_project/vectorstore/factory.py
from __future__ import annotations

from typing import TYPE_CHECKING

from profile_project.config.conflicts import run_conflict_detection
from profile_project.vectorstore.chroma_store import ChromaDBStore
from profile_project.vectorstore.embedders import (
    EmbedderExtraMissing,
    build_embedder,
)
from profile_project.vectorstore.pinecone_store import (
    IndexDimensionMismatch,
    PineconeStore,
)

if TYPE_CHECKING:
    from profile_project.config.settings import Settings
    from profile_project.vectorstore.protocols import Embedder, VectorStore


def build_store(settings: Settings, embedder: Embedder) -> VectorStore:
    """Construct the configured store, using the embedder's effective dimension.

    Dispatches on ``settings.vectorstore.backend``. The Pinecone path passes the
    *effective* embedding dim (``embedder.dimension``) so the store's last-line
    geometry guard (§10.1) validates against what the embedder really emits.
    """
    backend = settings.vectorstore.backend
    if backend == "chromadb":
        return ChromaDBStore(
            persist_path=settings.vectorstore.chromadb.path,
            collection=settings.vectorstore.collection,
        )
    if backend == "pinecone":
        api_key = settings.pinecone_api_key
        index = settings.vectorstore.pinecone.index
        if api_key is None or index is None:
            # Should be unreachable: C1/C1b warn+disable upstream. Defense-in-depth.
            raise ValueError(
                "pinecone backend requires PROFILE_PROJECT_PINECONE_API_KEY and a "
                "configured existing index."
            )
        return PineconeStore(
            api_key=api_key.get_secret_value(),
            index=index,
            embedding_dim=embedder.dimension,
            collection=settings.vectorstore.collection,
            namespace=settings.vectorstore.pinecone.namespace,
        )
    raise ValueError(f"unsupported vectorstore backend: {backend!r}")


def build_backend(settings: Settings) -> tuple[Embedder, VectorStore] | None:
    """Build (embedder, store) or None when the vectorstore is disabled.

    Returns None when: the master switch is off, the conflict matrix (§6.5)
    warned + disabled the vectorstore, the backend is "disabled", the selected
    embedder's optional python extra is missing (C5), or the existing Pinecone
    index dimension does not match the effective embedding dimension (§10.1
    invariant #4: warn+disable on mismatch, never abort). Never raises for those
    disable paths — a misconfigured vectorstore is disabled, never aborts.
    """
    if not settings.vectorstore.enabled:
        return None
    if settings.vectorstore.backend == "disabled":
        return None
    _warnings, enabled_post = run_conflict_detection(settings)
    if not enabled_post:
        return None
    try:
        embedder = build_embedder(settings)
    except EmbedderExtraMissing:
        # C5: missing extra -> warn+disable (warning already recorded upstream)
        return None
    try:
        store = build_store(settings, embedder)
    except IndexDimensionMismatch:
        # §10.1 #4: existing Pinecone index geometry != embedder dim -> warn+disable
        # at the tool layer (callers return graceful index_disabled, never crash).
        return None
    return embedder, store
