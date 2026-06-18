# src/profile_project/tools/vectorstore_tools.py
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from profile_project.artifacts.schemas import VectorstoreIndex
from profile_project.artifacts.store import (
    load_artifact,
    store_artifact,
)
from profile_project.config.conflicts import run_conflict_detection
from profile_project.config.init_gate import (
    is_initialized,
    resolve_project_root,
)
from profile_project.config.settings import Settings
from profile_project.config.sources import load_settings
from profile_project.tools._envelope import ToolError, require_init, tool_envelope
from profile_project.vectorstore.chunking import (
    ChunkConfig,
    RawContent,
    presplit,
    token_chunk,
)
from profile_project.vectorstore.embedders import (
    EmbedderExtraMissing,
    build_embedder,
)
from profile_project.vectorstore.factory import build_backend
from profile_project.vectorstore.ids import chunk_id

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

CHUNK_CONFIG = ChunkConfig(
    chunk_size=512, chunk_overlap=64, token_encoding="cl100k_base"
)


def _resolved_settings() -> tuple[Settings, Path]:
    root = resolve_project_root()
    return load_settings(root), root


def _content_hash(text: str) -> str:
    """sha256 hex of the chunk text (the content_hash arg of chunk_id, §10.4)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_agent_page_contents(settings: Settings, root: Path) -> list[RawContent]:
    """Read the agent-pages manifest, pre-split each page on its headings."""
    manifest = load_artifact(root, "agent-pages")
    if manifest is None:
        return []
    pages = manifest.get("pages", [])
    if not isinstance(pages, list):
        return []
    contents: list[RawContent] = []
    for page in pages:
        rel = str(page["path"])
        abs_path = root / rel
        if not abs_path.exists():
            continue
        text = abs_path.read_text(encoding="utf-8")
        base_meta: dict[str, object] = {
            "source_id": str(page.get("id", rel)),
            "source_type": "agent-page",
            "path": rel,
            "page_type": page.get("page_type"),
            "title": page.get("title"),
        }
        # presplit returns raw string segments; wrap each into RawContent here.
        for segment in presplit(text, "agent-page"):
            contents.append(RawContent(text=segment, metadata=dict(base_meta)))
    return contents


def _stored_embedder_version(root: Path) -> str | None:
    """The embedder_version of the prior vectorstore-index artifact, if any."""
    prior = load_artifact(root, "vectorstore-index")
    if prior is None:
        return None
    version = prior.get("embedder_version")
    return str(version) if version is not None else None


def _build_index(run_id: str | None, *, reset_geometry: bool) -> dict[str, object]:
    settings, root = _resolved_settings()
    bundle = build_backend(settings)
    if bundle is None:
        raise ToolError(
            "index_disabled",
            "vectorstore is disabled for this project; cannot build index.",
            retriable=False,
        )
    embedder, store = bundle
    embedder_version = embedder.embedder_version

    # embedder_version mismatch guard: never interleave geometries. A build that
    # discovers the store already holds a different embedder_version routes to the
    # geometry-safe reset (rebuild) path rather than mixing vector spaces (§10.4).
    prior_version = _stored_embedder_version(root)
    if prior_version is not None and prior_version != embedder_version:
        reset_geometry = True

    raw = _load_agent_page_contents(settings, root)
    chunks = token_chunk(raw, CHUNK_CONFIG)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, object]] = []
    for chunk in chunks:
        span = chunk.metadata.get("span", {})
        start_token = int(span.get("start_token", 0)) if isinstance(span, dict) else 0
        end_token = int(span.get("end_token", 0)) if isinstance(span, dict) else 0
        c_hash = _content_hash(chunk.text)
        cid = chunk_id(
            source_id=str(chunk.metadata.get("source_id", "")),
            path=str(chunk.metadata.get("path", "")),
            start_token=start_token,
            end_token=end_token,
            embedder_version=embedder_version,
            content_hash=c_hash,
        )
        meta = dict(chunk.metadata)
        meta["embedder_version"] = embedder_version
        meta["embedding_model"] = embedder.model_name
        meta["embedding_provider"] = embedder.embedding_provider
        if run_id is not None:
            meta["run_id"] = run_id
        ids.append(cid)
        documents.append(chunk.text)
        metadatas.append(meta)

    if reset_geometry:
        # Geometry-safe full re-index: drop the prior collection wholesale.
        store.reset()
    else:
        # Agent pages are regenerated wholesale: replace the prior chunk set
        # (delete existing agent-page chunks before upserting the new set, §9.3).
        store.delete(where={"source_type": "agent-page"})

    if ids:
        embeddings = embedder.embed(documents)
        store.add(ids, embeddings, documents, metadatas)

    artifact = VectorstoreIndex(
        artifact_type="vectorstore-index",
        schema_version=1,
        run_id=run_id or "",
        backend=settings.vectorstore.backend,  # type: ignore[arg-type]
        collection=settings.vectorstore.collection,
        namespace=settings.vectorstore.pinecone.namespace,
        embedder_version=embedder_version,
        dimension=embedder.dimension,
        chunk_count=len(ids),
        source_types_indexed=["agent-page"] if ids else [],
        chunk_config={  # type: ignore[arg-type]
            "chunk_size": CHUNK_CONFIG.chunk_size,
            "chunk_overlap": CHUNK_CONFIG.chunk_overlap,
            "token_encoding": CHUNK_CONFIG.token_encoding,
        },
    )
    content = artifact.model_dump()
    store_artifact(
        root, run_id or "", "build_vectorstore", "vectorstore-index", content
    )
    return {"vectorstore_index": content}


@tool_envelope
@require_init
def pp_index_build(run_id: str | None = None) -> dict[str, object]:
    """Chunk -> embed -> upsert agent pages into the store (§9.3/§11)."""
    return _build_index(run_id, reset_geometry=False)


@tool_envelope
@require_init
def pp_index_rebuild(run_id: str | None = None) -> dict[str, object]:
    """Full geometry-safe re-index (reset + rebuild, §10.4/§11)."""
    return _build_index(run_id, reset_geometry=True)


@tool_envelope
@require_init
def pp_query(
    query: str,
    top_k: int = 10,
    where: dict[str, object] | None = None,
) -> dict[str, object]:
    """Embed query -> vector search -> ranked, attributed hits (§11)."""
    settings, _root = _resolved_settings()
    bundle = build_backend(settings)
    if bundle is None:
        raise ToolError(
            "index_disabled",
            "vectorstore is disabled for this project; cannot query.",
            retriable=False,
        )
    embedder, store = bundle
    if store.count() == 0:
        raise ToolError(
            "index_empty",
            "no vectors indexed yet; run pp_index_build first.",
            retriable=True,
        )
    vector = embedder.embed_query(query)
    results = store.query(vector, top_k=top_k, where=where)
    hits: list[dict[str, object]] = [
        {
            "id": r.id,
            "document": r.document,
            "score": r.score,
            "metadata": r.metadata,
            "path": r.metadata.get("path"),
            "title": r.metadata.get("title"),
        }
        for r in results
    ]
    hits.sort(key=lambda h: float(h["score"]), reverse=True)  # type: ignore[arg-type]
    return {"hits": hits}


@tool_envelope
def pp_index_status() -> dict[str, object]:
    """Store stats / warm-load. In-handler read-only init guard (§11/§6b.3)."""
    settings, root = _resolved_settings()
    backend = settings.vectorstore.backend
    if not is_initialized(root):
        # NEVER construct a store client pre-init: PersistentClient + count()
        # would materialize .profile_project/chroma on disk.
        return {
            "backend": backend,
            "count": 0,
            "dimension": None,
            "embedder_version": None,
            "status": "uninitialized",
        }
    bundle = build_backend(settings)
    if bundle is None:
        return {
            "backend": backend,
            "count": 0,
            "dimension": None,
            "embedder_version": None,
            "status": "disabled",
        }
    embedder, store = bundle
    count = store.count()
    return {
        "backend": backend,
        "count": count,
        "dimension": embedder.dimension,
        "embedder_version": embedder.embedder_version,
        "status": "ready" if count > 0 else "empty",
    }


@tool_envelope
def pp_vectorstore_check(dry_run: bool = True) -> dict[str, object]:
    """Diagnose reachability + dimension; dry-run never writes (§6.5/§10.1)."""
    settings, _root = _resolved_settings()
    backend = settings.vectorstore.backend
    # Bounded fail-closed conflict probes (C3/C4); (warnings, vectorstore_enabled).
    warnings, vectorstore_enabled = run_conflict_detection(settings)
    dimension: int | None = None
    reachable = False
    try:
        # Building the embedder may probe (ollama) but never writes to the store;
        # the store client is intentionally NOT constructed on a dry run.
        embedder = build_embedder(settings)
        dimension = embedder.dimension
        # chromadb local is always reachable; remote backends fold their
        # bounded fail-closed probe result into the conflict warnings (§6.5 C3/C4).
        reachable = backend == "chromadb" or vectorstore_enabled
    except EmbedderExtraMissing as exc:
        warnings.append(f"embedder extra missing (treated as unreachable): {exc}")
        reachable = False
    except Exception as exc:  # noqa: BLE001 - fail closed: unreachable, never raise
        warnings.append(f"vectorstore check failed (treated as unreachable): {exc}")
        reachable = False
    return {
        "ok": True,
        "backend": backend,
        "reachable": reachable,
        "dimension": dimension,
        "warnings": warnings,
    }


def register_vectorstore_tools(mcp: FastMCP) -> None:
    mcp.tool()(pp_index_build)
    mcp.tool()(pp_index_rebuild)
    mcp.tool()(pp_query)
    mcp.tool()(pp_index_status)
    mcp.tool()(pp_vectorstore_check)
