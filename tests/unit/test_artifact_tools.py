from __future__ import annotations

import json
from pathlib import Path

import pytest

from profile_project.tools import artifact_tools, config_tools, dag_tools


@pytest.fixture()
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / ".profile_project_config.json").write_text(
        json.dumps(
            {
                "vectorstore": {"enabled": True, "backend": "chromadb"},
                "embeddings": {"method": "sentence-transformers"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PROFILE_PROJECT_PROJECT_DIR", str(tmp_path))
    monkeypatch.delenv("PROFILE_PROJECT_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PROFILE_PROJECT_PINECONE_API_KEY", raising=False)
    return tmp_path


_SOURCE_INDEX = {
    "artifact_type": "source-index",
    "schema_version": 1,
    "run_id": "r",
    "project_root": "/abs",
    "sources": [],
    "counts": {"code": 0, "doc": 0, "transcript": 0, "note": 0, "external": 0},
    "excluded_dirs": [],
    "gitignore_applied": True,
}


def test_pp_validate_artifact_pre_init(project: Path) -> None:
    ok = artifact_tools.pp_validate_artifact("source-index", _SOURCE_INDEX)
    assert ok["ok"] is True
    assert ok["errors"] == []

    bad = artifact_tools.pp_validate_artifact("source-index", {"artifact_type": "x"})
    assert bad["ok"] is False
    assert bad["errors"]
    assert not (project / ".profile_project").exists()


def test_pp_list_and_load_artifact_pre_init_returns_empty_none(project: Path) -> None:
    listed = artifact_tools.pp_list_artifacts()
    assert listed == {"ok": True, "artifacts": []}
    loaded = artifact_tools.pp_load_artifact("source-index")
    assert loaded == {"ok": True, "artifact": None}
    assert not (project / ".profile_project").exists()


def test_pp_store_artifact_gated_pre_init(project: Path) -> None:
    result = artifact_tools.pp_store_artifact("r1", "discover_context", "source-index", _SOURCE_INDEX)
    assert result["ok"] is False
    assert result["error"]["code"] == "not_initialized"
    assert not (project / ".profile_project").exists()


def test_pp_store_artifact_writes_and_registers_post_init(project: Path) -> None:
    config_tools.pp_init_project(
        {
            "vectorstore": {"enabled": True, "backend": "chromadb"},
            "embeddings": {"method": "sentence-transformers"},
        }
    )
    run_id = dag_tools.pp_init_run({})["run_id"]
    dag_tools.pp_start_phase(run_id, "discover_context")
    stored = artifact_tools.pp_store_artifact(
        run_id, "discover_context", "source-index", _SOURCE_INDEX
    )
    assert stored["ok"] is True
    assert stored["artifact_ref"]["type"] == "source-index"
    assert (
        project / ".profile_project" / "artifacts" / "source-index.json"
    ).exists()

    loaded = artifact_tools.pp_load_artifact("source-index", run_id=run_id)
    assert loaded["artifact"]["artifact_type"] == "source-index"
    listed = artifact_tools.pp_list_artifacts(run_id=run_id)
    assert any(a["type"] == "source-index" for a in listed["artifacts"])
