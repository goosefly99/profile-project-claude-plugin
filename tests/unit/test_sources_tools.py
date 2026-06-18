from __future__ import annotations

import json
from pathlib import Path

import pytest

from profile_project.tools import config_tools, sources_tools


@pytest.fixture()
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Tiny\n", encoding="utf-8")
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


def test_discover_persist_false_pre_init_no_write(project: Path) -> None:
    result = sources_tools.pp_discover_sources(persist=False)
    assert result["ok"] is True
    assert result["source_index"]["artifact_type"] == "source-index"
    assert result["source_index"]["counts"]["code"] >= 1
    # in-memory only — no artifact tree materialized pre-init.
    assert not (project / ".profile_project").exists()


def test_discover_persist_true_refused_pre_init(project: Path) -> None:
    result = sources_tools.pp_discover_sources(persist=True)
    assert result["ok"] is False
    assert result["error"]["code"] == "not_initialized"
    # in-handler gate fired BEFORE any write — zero residue.
    assert not (project / ".profile_project").exists()


def test_discover_persist_true_writes_post_init(project: Path) -> None:
    config_tools.pp_init_project(
        {
            "vectorstore": {"enabled": True, "backend": "chromadb"},
            "embeddings": {"method": "sentence-transformers"},
        }
    )
    result = sources_tools.pp_discover_sources(persist=True)
    assert result["ok"] is True
    assert (
        project / ".profile_project" / "artifacts" / "source-index.json"
    ).exists()


def test_pp_list_sources_and_get_source_pre_init(project: Path) -> None:
    listed = sources_tools.pp_list_sources()
    assert listed["ok"] is True
    assert listed["counts"]["code"] >= 1
    assert not (project / ".profile_project").exists()

    one_id = listed["sources"][0]["source_id"]
    got = sources_tools.pp_get_source(one_id)
    assert got["ok"] is True
    assert got["source"]["source_id"] == one_id

    missing = sources_tools.pp_get_source("nope")
    assert missing["source"] is None


def test_pp_add_source_gated_pre_init(project: Path) -> None:
    result = sources_tools.pp_add_source("https://example.com/adr", kind="external")
    assert result["ok"] is False
    assert result["error"]["code"] == "not_initialized"
