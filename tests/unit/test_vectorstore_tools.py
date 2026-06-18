# tests/unit/test_vectorstore_tools.py
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from mcp.server.fastmcp import FastMCP

from profile_project.tools import vectorstore_tools as vt


def _write_config(root: Path, **overrides: object) -> None:
    cfg: dict[str, object] = {
        "vectorstore": {"enabled": True, "backend": "chromadb"},
        "embeddings": {"method": "sentence-transformers"},
    }
    cfg.update(overrides)
    (root / ".profile_project_config.json").write_text(
        json.dumps(cfg), encoding="utf-8"
    )


def test_index_status_preinit_returns_uninitialized_stub_no_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pre-init: config present but no .profile_project/ tree, no stamp.
    _write_config(tmp_path)
    monkeypatch.setenv("PROFILE_PROJECT_PROJECT_DIR", str(tmp_path))

    # Fail loudly if any backend (store/embedder) is constructed pre-init.
    def _boom(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("build_backend must NOT run pre-init")

    monkeypatch.setattr(vt, "build_backend", _boom)

    result = vt.pp_index_status()
    assert result["ok"] is True
    assert result["status"] == "uninitialized"
    assert result["count"] == 0
    assert result["dimension"] is None
    assert result["embedder_version"] is None
    assert result["backend"] == "chromadb"
    # Zero residue: the read-only predicate never created the store dir.
    assert not (tmp_path / ".profile_project").exists()
    assert not (tmp_path / ".profile_project" / "chroma").exists()


def test_query_refuses_pre_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(tmp_path)
    monkeypatch.setenv("PROFILE_PROJECT_PROJECT_DIR", str(tmp_path))

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("backend must NOT be built pre-init")

    monkeypatch.setattr(vt, "build_backend", _boom)

    result = vt.pp_query("how is config resolved?")
    assert result["ok"] is False
    assert result["error"]["code"] == "not_initialized"
    assert result["error"]["remedy"] == "/profile-project:init"
    assert not (tmp_path / ".profile_project").exists()


@dataclass
class _FakeResult:
    id: str
    document: str
    score: float
    metadata: dict[str, object]


class _FakeStore:
    def __init__(self, count: int) -> None:
        self._count = count
        self.queried: list[list[float]] = []

    def count(self) -> int:
        return self._count

    def query(
        self,
        embedding: list[float],
        top_k: int = 10,
        where: dict[str, object] | None = None,
        where_document: dict[str, object] | None = None,
    ) -> list[_FakeResult]:
        self.queried.append(embedding)
        return [
            _FakeResult(
                "c1",
                "config is resolved by a layered resolver",
                0.42,
                {"path": "profile/context/architecture.md", "title": "Architecture"},
            ),
            _FakeResult(
                "c2",
                "the gate is enforced in the server",
                0.91,
                {"path": "profile/context/overview.md", "title": "Overview"},
            ),
        ]


class _FakeEmbedder:
    dimension = 384
    model_name = "all-MiniLM-L6-v2"
    embedder_version = "sentence-transformers/all-MiniLM-L6-v2@hf-fp32"
    embedding_provider = "sentence-transformers"

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


def _force_initialized(monkeypatch: pytest.MonkeyPatch) -> None:
    # Past the gate: both the envelope gate and the in-handler predicate see init.
    monkeypatch.setattr(vt, "is_initialized", lambda _root: True)
    from profile_project.tools import _envelope

    monkeypatch.setattr(_envelope, "is_initialized", lambda _root: True)


def test_query_index_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(tmp_path, vectorstore={"enabled": False, "backend": "disabled"})
    monkeypatch.setenv("PROFILE_PROJECT_PROJECT_DIR", str(tmp_path))
    _force_initialized(monkeypatch)
    # Disabled vectorstore: build_backend returns None.
    monkeypatch.setattr(vt, "build_backend", lambda _s: None)
    result = vt.pp_query("q")
    assert result["ok"] is False
    assert result["error"]["code"] == "index_disabled"


def test_query_index_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(tmp_path)
    monkeypatch.setenv("PROFILE_PROJECT_PROJECT_DIR", str(tmp_path))
    _force_initialized(monkeypatch)

    class _NoEmbed(_FakeEmbedder):
        def embed_query(self, text: str) -> list[float]:
            raise AssertionError("embed before empty check")

    monkeypatch.setattr(vt, "build_backend", lambda _s: (_NoEmbed(), _FakeStore(0)))
    result = vt.pp_query("q")
    assert result["ok"] is False
    assert result["error"]["code"] == "index_empty"
    assert result["error"]["retriable"] is True


def test_query_returns_ranked_attributed_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(tmp_path)
    monkeypatch.setenv("PROFILE_PROJECT_PROJECT_DIR", str(tmp_path))
    _force_initialized(monkeypatch)
    store = _FakeStore(2)
    monkeypatch.setattr(vt, "build_backend", lambda _s: (_FakeEmbedder(), store))
    result = vt.pp_query("how is config resolved?", top_k=5)
    assert result["ok"] is True
    hits = result["hits"]
    assert [h["id"] for h in hits] == ["c2", "c1"]  # sorted by descending score
    assert hits[0]["path"] == "profile/context/overview.md"
    assert hits[0]["title"] == "Overview"
    assert store.queried == [[0.1, 0.2, 0.3]]


class _RecordingStore(_FakeStore):
    def __init__(self, count: int = 0) -> None:
        super().__init__(count)
        self.added_ids: list[str] = []
        self.deleted_ids: list[str] = []
        self.reset_called = False

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, object]] | None = None,
    ) -> None:
        self.added_ids.extend(ids)
        self._count = len(self.added_ids)

    def delete(
        self,
        ids: list[str] | None = None,
        where: dict[str, object] | None = None,
    ) -> None:
        if ids:
            self.deleted_ids.extend(ids)

    def reset(self) -> None:
        self.reset_called = True
        self.added_ids.clear()
        self._count = 0


def _stub_build_subsystem(
    monkeypatch: pytest.MonkeyPatch, store: _RecordingStore
) -> dict[str, object]:
    monkeypatch.setattr(
        vt, "build_backend", lambda _s: (_FakeEmbedder(), store)
    )
    # One agent page on disk, loaded via the agent-pages artifact manifest.
    # run_id must be None (flat path); a non-None run_id means the code is reading
    # the wrong run-scoped path — return None so assertions catch the regression.
    monkeypatch.setattr(
        vt,
        "load_artifact",
        lambda root, t, run_id=None: {
            "artifact_type": "agent-pages",
            "output_dir": "profile/context",
            "pages": [
                {"id": "overview", "path": "profile/context/overview.md",
                 "page_type": "overview", "title": "Overview"},
            ],
            "page_count": 1,
        } if t == "agent-pages" and run_id is None else None,
    )
    stored: dict[str, object] = {}

    def _store_artifact(
        root: Path,
        run_id: str,
        phase: str,
        artifact_type: str,
        content: dict[str, object],
    ) -> dict[str, object]:
        stored["content"] = content
        stored["type"] = artifact_type
        return {
            "type": artifact_type,
            "path": ".profile_project/artifacts/vectorstore-index.json",
        }

    monkeypatch.setattr(vt, "store_artifact", _store_artifact)
    return stored


def test_index_build_chunks_embeds_upserts_and_stores_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(tmp_path)
    monkeypatch.setenv("PROFILE_PROJECT_PROJECT_DIR", str(tmp_path))
    _force_initialized(monkeypatch)
    ctx = tmp_path / "profile" / "context"
    ctx.mkdir(parents=True)
    (ctx / "overview.md").write_text(
        "# Overview\n\nprofile-project profiles a project into context pages.\n",
        encoding="utf-8",
    )
    store = _RecordingStore()
    stored = _stub_build_subsystem(monkeypatch, store)

    result = vt.pp_index_build(run_id="r1")
    assert result["ok"] is True
    vsi = result["vectorstore_index"]
    assert vsi["artifact_type"] == "vectorstore-index"
    assert vsi["backend"] == "chromadb"
    assert vsi["embedder_version"] == "sentence-transformers/all-MiniLM-L6-v2@hf-fp32"
    assert vsi["dimension"] == 384
    assert vsi["chunk_count"] == len(store.added_ids) > 0
    assert "agent-page" in vsi["source_types_indexed"]
    assert vsi["chunk_config"] == {
        "chunk_size": 512,
        "chunk_overlap": 64,
        "token_encoding": "cl100k_base",
    }
    # The artifact was registered server-side (deterministic-phase contract §7.7).
    assert stored["type"] == "vectorstore-index"
    # Build does NOT reset geometry.
    assert store.reset_called is False


def test_index_build_replaces_prior_chunk_set_without_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(tmp_path)
    monkeypatch.setenv("PROFILE_PROJECT_PROJECT_DIR", str(tmp_path))
    _force_initialized(monkeypatch)
    ctx = tmp_path / "profile" / "context"
    ctx.mkdir(parents=True)
    (ctx / "overview.md").write_text("# Overview\n\nbody\n", encoding="utf-8")
    store = _RecordingStore(count=5)
    captured: dict[str, object] = {}

    real_delete = store.delete

    def _tracking_delete(
        ids: list[str] | None = None, where: dict[str, object] | None = None
    ) -> None:
        captured["delete_where"] = where
        real_delete(ids=ids, where=where)

    monkeypatch.setattr(store, "delete", _tracking_delete)
    _stub_build_subsystem(monkeypatch, store)

    result = vt.pp_index_build(run_id="r1")
    assert result["ok"] is True
    # Build replaces the agent-page chunk set by a scoped delete, NOT a reset.
    assert store.reset_called is False
    assert captured["delete_where"] == {"source_type": "agent-page"}


def test_index_rebuild_resets_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(tmp_path)
    monkeypatch.setenv("PROFILE_PROJECT_PROJECT_DIR", str(tmp_path))
    _force_initialized(monkeypatch)
    ctx = tmp_path / "profile" / "context"
    ctx.mkdir(parents=True)
    (ctx / "overview.md").write_text("# Overview\n\nbody\n", encoding="utf-8")
    store = _RecordingStore(count=9)
    _stub_build_subsystem(monkeypatch, store)

    result = vt.pp_index_rebuild(run_id="r1")
    assert result["ok"] is True
    # Rebuild is geometry-safe: reset the whole collection before re-adding.
    assert store.reset_called is True
    assert len(store.added_ids) > 0


def test_index_build_embedder_version_mismatch_routes_to_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(tmp_path)
    monkeypatch.setenv("PROFILE_PROJECT_PROJECT_DIR", str(tmp_path))
    _force_initialized(monkeypatch)
    ctx = tmp_path / "profile" / "context"
    ctx.mkdir(parents=True)
    (ctx / "overview.md").write_text("# Overview\n\nbody\n", encoding="utf-8")
    store = _RecordingStore(count=4)
    _stub_build_subsystem(monkeypatch, store)
    # A prior vectorstore-index artifact stamped with a DIFFERENT geometry.
    # Both branches gate on run_id is None (flat path); non-None run_id returns None
    # so we verify the code loads from the correct flat path.
    monkeypatch.setattr(
        vt,
        "load_artifact",
        lambda root, t, run_id=None: (
            {
                "artifact_type": "agent-pages",
                "pages": [
                    {"id": "overview", "path": "profile/context/overview.md",
                     "page_type": "overview", "title": "Overview"},
                ],
            }
            if t == "agent-pages" and run_id is None
            else (
                {"embedder_version": "openai/text-embedding-3-small@v1"}
                if t == "vectorstore-index" and run_id is None
                else None
            )
        ),
    )

    result = vt.pp_index_build(run_id="r1")
    assert result["ok"] is True
    # Mismatched embedder_version forces the geometry-safe reset, never a
    # scoped delete that would interleave vector spaces.
    assert store.reset_called is True


def test_vectorstore_check_dry_run_never_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(tmp_path)
    monkeypatch.setenv("PROFILE_PROJECT_PROJECT_DIR", str(tmp_path))

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("dry-run must NOT construct a store")

    monkeypatch.setattr(vt, "build_backend", _boom)
    monkeypatch.setattr(vt, "build_embedder", lambda _s: _FakeEmbedder())

    result = vt.pp_vectorstore_check(dry_run=True)
    assert result["ok"] is True
    assert result["backend"] == "chromadb"
    assert result["reachable"] is True  # chromadb local is always reachable
    assert result["dimension"] == 384
    assert isinstance(result["warnings"], list)
    # Zero residue: dry-run never created the store tree.
    assert not (tmp_path / ".profile_project").exists()


def test_register_vectorstore_tools_registers_all_five() -> None:
    mcp = FastMCP("profile-project-test")
    vt.register_vectorstore_tools(mcp)
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert {
        "pp_index_build",
        "pp_index_rebuild",
        "pp_query",
        "pp_index_status",
        "pp_vectorstore_check",
    } <= names
