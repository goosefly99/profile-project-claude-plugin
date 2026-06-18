from __future__ import annotations

import json
from pathlib import Path

import pytest

from profile_project.config.provenance import compute_provenance, validate_config


def _write_json_config(root: Path, payload: dict[str, object]) -> Path:
    cfg = root / ".profile_project_config.json"
    cfg.write_text(json.dumps(payload), encoding="utf-8")
    return cfg


# ---------------------------------------------------------------------------
# compute_provenance
# ---------------------------------------------------------------------------


def test_compute_provenance_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # project JSON sets embeddings.method (overrides env); env sets vectorstore.backend;
    # vectorstore.collection is left to the field default.
    _write_json_config(tmp_path, {"embeddings": {"method": "openai"}})
    monkeypatch.setenv("PROFILE_PROJECT_VECTORSTORE__BACKEND", "pinecone")
    monkeypatch.setenv("PROFILE_PROJECT_DEFAULT_EMBEDDINGS_METHOD", "ollama")
    monkeypatch.delenv("PROFILE_PROJECT_VECTORSTORE__COLLECTION", raising=False)

    prov = compute_provenance(tmp_path)

    assert prov["embeddings.method"] == "project_json"  # JSON overrides env
    assert prov["vectorstore.backend"] == "env"
    assert prov["vectorstore.collection"] == "default"


def test_compute_provenance_alias_env_var_attributed_as_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PROFILE_PROJECT_DEFAULT_EMBEDDINGS_METHOD via _TopLevelAliasSource -> 'env'."""
    monkeypatch.setenv("PROFILE_PROJECT_DEFAULT_EMBEDDINGS_METHOD", "ollama")
    monkeypatch.delenv("PROFILE_PROJECT_EMBEDDINGS__METHOD", raising=False)
    # no JSON config
    prov = compute_provenance(tmp_path)
    assert prov["embeddings.method"] == "env"


def test_compute_provenance_no_json_no_env_all_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PROFILE_PROJECT_EMBEDDINGS__METHOD", raising=False)
    monkeypatch.delenv("PROFILE_PROJECT_DEFAULT_EMBEDDINGS_METHOD", raising=False)
    monkeypatch.delenv("PROFILE_PROJECT_VECTORSTORE__BACKEND", raising=False)
    monkeypatch.delenv("PROFILE_PROJECT_VECTORSTORE__ENABLED", raising=False)
    monkeypatch.delenv("PROFILE_PROJECT_VECTORSTORE__COLLECTION", raising=False)
    prov = compute_provenance(tmp_path)
    for val in prov.values():
        assert val == "default"


# ---------------------------------------------------------------------------
# resolve_field
# ---------------------------------------------------------------------------


def test_resolve_field_value_and_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from profile_project.config.provenance import resolve_field
    from profile_project.config.sources import load_settings

    _write_json_config(tmp_path, {"embeddings": {"method": "openai"}})
    monkeypatch.setenv("PROFILE_PROJECT_VECTORSTORE__BACKEND", "pinecone")
    monkeypatch.delenv("PROFILE_PROJECT_DEFAULT_EMBEDDINGS_METHOD", raising=False)
    monkeypatch.delenv("PROFILE_PROJECT_OPENAI_API_KEY", raising=False)
    # Set project_dir so resolve_field can locate the JSON config.
    monkeypatch.setenv("PROFILE_PROJECT_PROJECT_DIR", str(tmp_path))

    settings = load_settings(tmp_path)

    value, source = resolve_field(settings, "embeddings.method")
    assert value == "openai"
    assert source == "project_json"

    backend_value, backend_source = resolve_field(settings, "vectorstore.backend")
    assert backend_value == "pinecone"
    assert backend_source == "env"


# ---------------------------------------------------------------------------
# validate_config – happy path
# ---------------------------------------------------------------------------


def test_validate_config_ok_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Zero-conflict config: sentence-transformers + chromadb, vectorstore enabled.
    _write_json_config(
        tmp_path,
        {
            "embeddings": {"method": "sentence-transformers"},
            "vectorstore": {"enabled": True, "backend": "chromadb"},
        },
    )
    monkeypatch.delenv("PROFILE_PROJECT_VECTORSTORE__BACKEND", raising=False)
    monkeypatch.delenv("PROFILE_PROJECT_DEFAULT_EMBEDDINGS_METHOD", raising=False)

    # Simulate extras installed so C5 doesn't fire.
    import profile_project.config.conflicts as _conflicts
    monkeypatch.setattr(_conflicts, "_extra_installed", lambda _m: True)

    result = validate_config(tmp_path)

    assert result["ok"] is True
    assert result["errors"] == []
    assert result["config_path"] == str(tmp_path / ".profile_project_config.json")
    assert result["initialized"] is False  # no .initialized stamp written
    assert result["vectorstore_enabled"] is True
    assert result["provenance"]["embeddings.method"] == "project_json"
    assert result["provenance"]["vectorstore.collection"] == "default"
    assert result["settings"]["embeddings"]["method"] == "sentence-transformers"


def test_validate_config_initialized_true_when_stamp_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_json_config(tmp_path, {})
    (tmp_path / ".profile_project_initialized").write_text("", encoding="utf-8")

    result = validate_config(tmp_path)

    assert result["initialized"] is True


# ---------------------------------------------------------------------------
# validate_config – conflict-driven warn+disable (ok stays True)
# ---------------------------------------------------------------------------


def test_validate_config_warn_disables_vectorstore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # method=openai with no OPENAI_API_KEY => C2: warn + disable, but ok stays True.
    _write_json_config(
        tmp_path,
        {
            "embeddings": {"method": "openai"},
            "vectorstore": {"enabled": True, "backend": "chromadb"},
        },
    )
    monkeypatch.delenv("PROFILE_PROJECT_OPENAI_API_KEY", raising=False)

    result = validate_config(tmp_path)

    assert result["ok"] is True
    assert result["errors"] == []
    assert result["vectorstore_enabled"] is False
    assert any("OPENAI_API_KEY" in w for w in result["warnings"])


def test_validate_config_c11_nonexistent_root_warns_and_disables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C11: if resolved project_dir does not exist, warn + disable vectorstore."""
    nonexistent = tmp_path / "does_not_exist"
    _write_json_config(tmp_path, {})
    monkeypatch.setenv("PROFILE_PROJECT_PROJECT_DIR", str(nonexistent))

    result = validate_config(tmp_path)

    assert result["ok"] is True
    assert result["vectorstore_enabled"] is False
    assert any("does not exist" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# validate_config – hard failures (ok=False)
# ---------------------------------------------------------------------------


def test_validate_config_hard_fail_forbidden_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A forbidden secret in JSON is a hard failure: ok=false, errors populated,
    # warnings/provenance empty, settings empty, vectorstore_enabled false.
    _write_json_config(tmp_path, {"openai_api_key": "sk-should-never-be-here"})

    result = validate_config(tmp_path)

    assert result["ok"] is False
    assert result["warnings"] == []
    assert result["provenance"] == {}
    assert result["settings"] == {}
    assert result["vectorstore_enabled"] is False
    assert len(result["errors"]) == 1
    assert "openai_api_key" in result["errors"][0]
    assert "sk-should-never-be-here" not in result["errors"][0]
    assert "sk-should-never-be-here" not in json.dumps(result)


def test_validate_config_hard_fail_unknown_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Unknown key => extra="forbid" => ValidationError => hard failure.
    _write_json_config(tmp_path, {"bogus_unknown_field": 123})

    result = validate_config(tmp_path)

    assert result["ok"] is False
    assert result["provenance"] == {}
    assert result["errors"]  # non-empty loud message
