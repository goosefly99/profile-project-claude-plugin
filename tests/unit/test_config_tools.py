from __future__ import annotations

import json
from pathlib import Path

import pytest

from profile_project.config.settings import CONFIG_FILENAME
from profile_project.tools import config_tools


def _write_config(root: Path) -> Path:
    cfg = {
        "vectorstore": {"enabled": True, "backend": "chromadb"},
        "embeddings": {"method": "sentence-transformers"},
    }
    path = root / CONFIG_FILENAME
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


@pytest.fixture()
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _write_config(tmp_path)
    monkeypatch.setenv("PROFILE_PROJECT_PROJECT_DIR", str(tmp_path))
    monkeypatch.delenv("PROFILE_PROJECT_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PROFILE_PROJECT_PINECONE_API_KEY", raising=False)
    # Simulate extras installed so C5 conflict detection doesn't fire.
    import profile_project.config.conflicts as _conflicts
    monkeypatch.setattr(_conflicts, "_extra_installed", lambda _m: True)
    return tmp_path


def test_pp_config_path_reports_path_and_not_initialized(project: Path) -> None:
    result = config_tools.pp_config_path()
    assert result["config_path"] == str(project / CONFIG_FILENAME)
    assert result["initialized"] is False


def test_pp_config_show_masks_and_reports_vectorstore(project: Path) -> None:
    result = config_tools.pp_config_show()
    assert result["vectorstore_enabled"] is True
    assert "openai_api_key" not in json.dumps(result["settings"])


def test_pp_config_get_returns_value_and_source(project: Path) -> None:
    result = config_tools.pp_config_get("embeddings.method")
    assert result["key"] == "embeddings.method"
    assert result["value"] == "sentence-transformers"
    assert result["source"] == "project_json"


def test_pp_config_validate_returns_envelope(project: Path) -> None:
    result = config_tools.pp_config_validate()
    assert result["ok"] is True
    assert result["initialized"] is False
    assert result["vectorstore_enabled"] is True
    assert result["errors"] == []


def test_pp_init_project_bootstraps_all_or_nothing(project: Path) -> None:
    config = {
        "vectorstore": {"enabled": True, "backend": "chromadb"},
        "embeddings": {"method": "sentence-transformers"},
    }
    result = config_tools.pp_init_project(config)
    assert result["ok"] is True
    assert result["initialized"] is True
    assert (project / CONFIG_FILENAME).exists()
    assert (project / ".profile_project" / ".initialized").exists()
    assert (project / ".profile_project" / "runs").is_dir()
    assert (project / ".profile_project" / "artifacts").is_dir()
    gitignore = (project / ".gitignore").read_text(encoding="utf-8")
    assert ".profile_project/" in gitignore
    assert ".profile_project/" in result["created"]


def test_pp_init_project_idempotent_preserves_runs(project: Path) -> None:
    config = {"vectorstore": {"enabled": True, "backend": "chromadb"},
              "embeddings": {"method": "sentence-transformers"}}
    config_tools.pp_init_project(config)
    sentinel = project / ".profile_project" / "runs" / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    result = config_tools.pp_init_project(config)
    assert result["ok"] is True
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_pp_init_project_rejects_secret_in_json(project: Path) -> None:
    config = {"openai_api_key": "sk-leaked", "embeddings": {"method": "openai"}}
    result = config_tools.pp_init_project(config)
    assert result["ok"] is False
    assert result["error"]["code"] == "forbidden_secret"
    assert not (project / ".profile_project").exists()


def test_pp_init_project_requires_env_secret(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PROFILE_PROJECT_OPENAI_API_KEY", raising=False)
    config = {"vectorstore": {"enabled": True, "backend": "chromadb"},
              "embeddings": {"method": "openai"}}
    result = config_tools.pp_init_project(config)
    assert result["ok"] is False
    assert result["error"]["code"] == "missing_secret"
    assert not (project / ".profile_project").exists()


def test_pp_config_set_refused_pre_init(project: Path) -> None:
    result = config_tools.pp_config_set("vectorstore.collection", "renamed")
    assert result["ok"] is False
    assert result["error"]["code"] == "not_initialized"
    # no write happened: collection still default in resolved config
    assert "renamed" not in (project / CONFIG_FILENAME).read_text(encoding="utf-8")


def test_pp_config_set_writes_post_init(project: Path) -> None:
    config = {"vectorstore": {"enabled": True, "backend": "chromadb"},
              "embeddings": {"method": "sentence-transformers"}}
    config_tools.pp_init_project(config)
    result = config_tools.pp_config_set("vectorstore.collection", "renamed")
    assert result["ok"] is True
    assert result["written"] is True
    assert "renamed" in (project / CONFIG_FILENAME).read_text(encoding="utf-8")


def test_pp_init_project_rejects_invalid_candidate_and_persists_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PROFILE_PROJECT_PROJECT_DIR", str(tmp_path))
    bad = {
        "totally_unknown_key": "x",
        "embeddings": {"method": "sentence-transformers"},
    }
    result = config_tools.pp_init_project(bad)
    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_config"
    assert not (tmp_path / CONFIG_FILENAME).exists()
    assert not (tmp_path / ".profile_project").exists()


def test_pp_init_project_rolls_back_tree_on_late_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PROFILE_PROJECT_PROJECT_DIR", str(tmp_path))
    (tmp_path / CONFIG_FILENAME).write_text(
        json.dumps({"vectorstore": {"enabled": True, "backend": "chromadb"},
                    "embeddings": {"method": "sentence-transformers"}}),
        encoding="utf-8",
    )

    def _boom(_root: Path, _entry: str) -> bool:
        raise OSError("boom after tree creation")

    monkeypatch.setattr(config_tools, "ensure_gitignore_entry", _boom)
    cfg = {"vectorstore": {"enabled": True, "backend": "chromadb"},
           "embeddings": {"method": "sentence-transformers"}}
    result = config_tools.pp_init_project(cfg)
    assert result["ok"] is False
    # genuine all-or-nothing: the tree the bootstrap created is rolled back
    assert not (tmp_path / ".profile_project").exists()


def test_pp_config_set_invalid_value_restores_prior_config(project: Path) -> None:
    cfg = {"vectorstore": {"enabled": True, "backend": "chromadb"},
           "embeddings": {"method": "sentence-transformers"}}
    config_tools.pp_init_project(cfg)
    before = (project / CONFIG_FILENAME).read_text(encoding="utf-8")
    result = config_tools.pp_config_set("totally_unknown_key", "x")
    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_config"
    # the live config is byte-for-byte restored
    assert (project / CONFIG_FILENAME).read_text(encoding="utf-8") == before
