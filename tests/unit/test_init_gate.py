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


from profile_project.config.init_gate import read_stamp


def _write_stamp(root: Path, **overrides: object) -> Path:
    tree = root / STAMP_DIRNAME
    tree.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema_version": 1,
        "initialized_at": "2026-06-18T14:03:11Z",
        "project_root": str(root),
        "config_path": str(root / CONFIG_FILENAME),
    }
    payload.update(overrides)
    stamp_file = tree / STAMP_FILENAME
    stamp_file.write_text(json.dumps(payload), encoding="utf-8")
    return stamp_file


def test_read_stamp_returns_none_when_absent(tmp_path: Path) -> None:
    assert read_stamp(tmp_path) is None


def test_read_stamp_returns_none_on_malformed_json(tmp_path: Path) -> None:
    tree = tmp_path / STAMP_DIRNAME
    tree.mkdir()
    (tree / STAMP_FILENAME).write_text("{not json", encoding="utf-8")
    assert read_stamp(tmp_path) is None


def test_read_stamp_parses_valid_stamp(tmp_path: Path) -> None:
    _write_stamp(tmp_path)
    stamp = read_stamp(tmp_path)
    assert stamp is not None
    assert stamp.schema_version == 1
    assert stamp.project_root == str(tmp_path)


from profile_project.config.init_gate import write_init_stamp


def test_write_init_stamp_creates_tree_and_round_trips(tmp_path: Path) -> None:
    config_path = tmp_path / CONFIG_FILENAME
    stamp_file = write_init_stamp(tmp_path, config_path)

    # Returns the absolute stamp-file path inside the gitignored tree.
    assert stamp_file == tmp_path / STAMP_DIRNAME / STAMP_FILENAME
    assert stamp_file.is_file()

    # The written stamp is readable back via the read-only reader.
    stamp = read_stamp(tmp_path)
    assert stamp is not None
    assert stamp.schema_version == STAMP_SCHEMA_VERSION
    assert stamp.project_root == str(tmp_path.resolve())
    assert stamp.config_path == str(config_path.resolve())
    # initialized_at is ISO-8601 UTC (Z suffix).
    assert stamp.initialized_at.endswith("Z")

    # Persisted JSON has exactly the four stamp fields (extra="forbid").
    on_disk = json.loads(stamp_file.read_text(encoding="utf-8"))
    assert set(on_disk) == {
        "schema_version",
        "initialized_at",
        "project_root",
        "config_path",
    }


def test_write_init_stamp_honors_schema_version_kwarg(tmp_path: Path) -> None:
    write_init_stamp(tmp_path, tmp_path / CONFIG_FILENAME, schema_version=1)
    stamp = read_stamp(tmp_path)
    assert stamp is not None
    assert stamp.schema_version == 1


from profile_project.config.init_gate import is_initialized


def test_is_initialized_false_when_uninitialized_and_leaves_no_residue(
    tmp_path: Path,
) -> None:
    assert is_initialized(tmp_path) is False
    # Strictly read-only: the predicate must NOT create the tree or config.
    assert not (tmp_path / STAMP_DIRNAME).exists()
    assert not (tmp_path / CONFIG_FILENAME).exists()


def test_is_initialized_true_when_stamp_and_config_present(tmp_path: Path) -> None:
    _write_stamp(tmp_path)
    (tmp_path / CONFIG_FILENAME).write_text("{}", encoding="utf-8")
    assert is_initialized(tmp_path) is True


def test_is_initialized_false_when_config_missing(tmp_path: Path) -> None:
    _write_stamp(tmp_path)
    assert is_initialized(tmp_path) is False


def test_is_initialized_false_on_unsupported_schema_version(tmp_path: Path) -> None:
    _write_stamp(tmp_path, schema_version=999)
    (tmp_path / CONFIG_FILENAME).write_text("{}", encoding="utf-8")
    assert is_initialized(tmp_path) is False


def test_is_initialized_false_on_root_mismatch(tmp_path: Path) -> None:
    _write_stamp(tmp_path, project_root="/some/other/root")
    (tmp_path / CONFIG_FILENAME).write_text("{}", encoding="utf-8")
    assert is_initialized(tmp_path) is False


from profile_project.config.init_gate import detect_root_move


def test_detect_root_move_no_stamp(tmp_path: Path) -> None:
    moved, stamped = detect_root_move(tmp_path)
    assert moved is False
    assert stamped is None


def test_detect_root_move_same_root(tmp_path: Path) -> None:
    _write_stamp(tmp_path)
    moved, stamped = detect_root_move(tmp_path)
    assert moved is False
    assert stamped == str(tmp_path)


def test_detect_root_move_detects_mismatch(tmp_path: Path) -> None:
    _write_stamp(tmp_path, project_root="/old/abs/path")
    moved, stamped = detect_root_move(tmp_path)
    assert moved is True
    assert stamped == "/old/abs/path"
