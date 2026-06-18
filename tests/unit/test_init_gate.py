from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

import profile_project.config.conflicts as _conflicts
from profile_project.config.init_gate import (
    CONFIG_FILENAME,
    STAMP_DIRNAME,
    STAMP_FILENAME,
    STAMP_SCHEMA_VERSION,
    SUPPORTED_STAMP_SCHEMA_VERSIONS,
    InitStamp,
    detect_root_move,
    is_initialized,
    not_initialized_error,
    project_root_moved_error,
    read_stamp,
    resolve_project_root,
    write_init_stamp,
)
from profile_project.config.settings import CONFIG_FILENAME as _CONFIG_FILENAME
from profile_project.tools import config_tools as _ct


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
    with pytest.raises(ValidationError):
        InitStamp.model_validate(
            {
                "schema_version": 1,
                "initialized_at": "2026-06-18T14:03:11Z",
                "project_root": "/p",
                "config_path": "/p/.profile_project_config.json",
                "unexpected": "boom",
            }
        )


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


def test_build_init_stamp_returns_payload_without_writing(tmp_path: Path) -> None:
    from profile_project.config.init_gate import build_init_stamp

    payload = build_init_stamp(tmp_path, tmp_path / CONFIG_FILENAME)
    assert set(payload) == {
        "schema_version",
        "initialized_at",
        "project_root",
        "config_path",
    }
    assert payload["project_root"] == str(tmp_path.resolve())
    assert payload["schema_version"] == 1
    # No filesystem side effects.
    assert not (tmp_path / ".profile_project").exists()


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


def test_not_initialized_error_shape(tmp_path: Path) -> None:
    env = not_initialized_error(tmp_path)
    assert env["ok"] is False
    err = env["error"]
    assert isinstance(err, dict)
    assert err["code"] == "not_initialized"
    assert err["retriable"] is False
    assert err["resolved_root"] == str(tmp_path)
    assert err["remedy"] == "/profile-project:init"
    assert "Run /profile-project:init first." in err["message"]


def test_project_root_moved_error_shape(tmp_path: Path) -> None:
    env = project_root_moved_error("/old/abs/path", tmp_path)
    assert env["ok"] is False
    err = env["error"]
    assert isinstance(err, dict)
    assert err["code"] == "project_root_moved"
    assert err["stamped_root"] == "/old/abs/path"
    assert err["resolved_root"] == str(tmp_path)
    assert err["retriable"] is False
    assert err["remedy"] == "/profile-project:init --reinit"


# ---------------------------------------------------------------------------
# Task 9: gate-residue + move-reinit regression tests
# ---------------------------------------------------------------------------


def test_gated_tool_leaves_zero_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / _CONFIG_FILENAME).write_text(
        json.dumps(
            {
                "vectorstore": {"enabled": True, "backend": "chromadb"},
                "embeddings": {"method": "sentence-transformers"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PROFILE_PROJECT_PROJECT_DIR", str(tmp_path))
    result = _ct.pp_config_set("vectorstore.collection", "x")
    assert result["error"]["code"] == "not_initialized"
    assert not (tmp_path / ".profile_project").exists()
    assert not (tmp_path / ".profile_project" / "chroma").exists()


def test_move_then_force_reinit_rewrites_old_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from profile_project.config.files import atomic_write_json

    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    cfg = {
        "vectorstore": {"enabled": True, "backend": "chromadb"},
        "embeddings": {"method": "sentence-transformers"},
    }
    (old / _CONFIG_FILENAME).write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("PROFILE_PROJECT_PROJECT_DIR", str(old))
    monkeypatch.setattr(_conflicts, "_extra_installed", lambda _m: True)
    _ct.pp_init_project(cfg)
    run_dir = old / ".profile_project" / "runs" / "r1"
    run_dir.mkdir(parents=True)
    old_root_str = str(old.resolve())
    # Write run-state exactly as the pipeline does (atomic_write_json -> json.dumps),
    # so the test exercises the real on-disk encoding rather than a hand-crafted one.
    atomic_write_json(
        run_dir / "run-state.json",
        {
            "run_id": "r1",
            "project_root": old_root_str,
            "config_path": str((old / _CONFIG_FILENAME).resolve()),
            "run_data_dir": str(run_dir.resolve()),
        },
    )
    shutil.move(str(old), str(new))
    monkeypatch.setenv("PROFILE_PROJECT_PROJECT_DIR", str(new))
    result = _ct.pp_init_project(cfg, force=True)
    assert result["ok"] is True
    data = json.loads(
        (new / ".profile_project" / "runs" / "r1" / "run-state.json").read_text(
            encoding="utf-8"
        )
    )
    assert data["project_root"] == str(new.resolve())
    assert old_root_str not in json.dumps(data)
