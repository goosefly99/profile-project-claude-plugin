from __future__ import annotations

import json
from pathlib import Path

import pytest

from profile_project.config.init_gate import (
    CONFIG_FILENAME,
    STAMP_DIRNAME,
    STAMP_FILENAME,
    STAMP_SCHEMA_VERSION,
    SUPPORTED_STAMP_SCHEMA_VERSIONS,
    InitStamp,
)


def test_stamp_constants_match_spec() -> None:
    assert STAMP_DIRNAME == ".profile_project"
    assert STAMP_FILENAME == ".initialized"
    assert CONFIG_FILENAME == ".profile_project_config.json"
    assert STAMP_SCHEMA_VERSION == 1
    assert 1 in SUPPORTED_STAMP_SCHEMA_VERSIONS


def test_init_stamp_round_trips_and_forbids_extra() -> None:
    stamp = InitStamp(
        schema_version=1,
        initialized_at="2026-06-18T14:03:11Z",
        project_root="/abs/path/to/project",
        config_path="/abs/path/to/project/.profile_project_config.json",
    )
    assert stamp.schema_version == 1
    assert stamp.project_root == "/abs/path/to/project"
    with pytest.raises(Exception):
        InitStamp.model_validate(
            {
                "schema_version": 1,
                "initialized_at": "2026-06-18T14:03:11Z",
                "project_root": "/p",
                "config_path": "/p/.profile_project_config.json",
                "unexpected": "boom",
            }
        )


from profile_project.config.init_gate import resolve_project_root


def test_resolve_project_root_prefers_settings_then_env_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from profile_project.config.settings import Settings

    explicit = tmp_path / "explicit"
    explicit.mkdir()
    project_dir_env = tmp_path / "project_dir_env"
    project_dir_env.mkdir()
    claude_dir = tmp_path / "claude_dir"
    claude_dir.mkdir()

    monkeypatch.setenv("PROFILE_PROJECT_PROJECT_DIR", str(project_dir_env))
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(claude_dir))

    # settings.project_dir beats every env var.
    settings = Settings(project_dir=explicit)
    assert resolve_project_root(settings) == explicit.resolve()

    # No settings -> PROFILE_PROJECT_PROJECT_DIR wins over CLAUDE_PROJECT_DIR.
    assert resolve_project_root(None) == project_dir_env.resolve()

    # Drop the primary env override -> falls to CLAUDE_PROJECT_DIR.
    monkeypatch.delenv("PROFILE_PROJECT_PROJECT_DIR")
    assert resolve_project_root(None) == claude_dir.resolve()


def test_resolve_project_root_falls_back_to_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PROFILE_PROJECT_PROJECT_DIR", raising=False)
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    monkeypatch.delenv("PWD", raising=False)
    monkeypatch.chdir(tmp_path)
    assert resolve_project_root(None) == tmp_path.resolve()
