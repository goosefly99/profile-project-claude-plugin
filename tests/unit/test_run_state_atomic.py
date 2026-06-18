from __future__ import annotations

import errno
import json
from pathlib import Path

import pytest

from profile_project.config.files import (
    append_jsonl,
    atomic_rename,
    atomic_write_json,
    ensure_gitignore_entry,
    rewrite_root_prefix,
    transaction,
)

# ---------------------------------------------------------------------------
# atomic_write_json
# ---------------------------------------------------------------------------


def test_atomic_write_json_round_trips(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "run-state.json"
    atomic_write_json(target, {"run_id": "r1", "status": "initialized"})
    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "run_id": "r1",
        "status": "initialized",
    }


def test_atomic_write_json_leaves_no_temp_file(tmp_path: Path) -> None:
    target = tmp_path / "run-state.json"
    atomic_write_json(target, {"a": 1})
    siblings = [p.name for p in tmp_path.iterdir()]
    assert siblings == ["run-state.json"]


# ---------------------------------------------------------------------------
# atomic_rename — EPERM/EACCES fallback and re-raise
# ---------------------------------------------------------------------------


def test_atomic_rename_falls_back_on_eperm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "src.tmp"
    dst = tmp_path / "dst.json"
    src.write_text("payload", encoding="utf-8")

    def boom(_a: object, _b: object) -> None:
        raise PermissionError(errno.EPERM, "locked")

    monkeypatch.setattr("profile_project.config.files.os.replace", boom)
    atomic_rename(src, dst)
    assert dst.read_text(encoding="utf-8") == "payload"
    assert not src.exists()


def test_atomic_rename_reraises_other_permission_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "src.tmp"
    dst = tmp_path / "dst.json"
    src.write_text("payload", encoding="utf-8")

    def boom(_a: object, _b: object) -> None:
        raise PermissionError(errno.ENOSPC, "no space")

    monkeypatch.setattr("profile_project.config.files.os.replace", boom)
    with pytest.raises(PermissionError):
        atomic_rename(src, dst)


# ---------------------------------------------------------------------------
# append_jsonl
# ---------------------------------------------------------------------------


def test_append_jsonl_writes_one_object_per_line(tmp_path: Path) -> None:
    log = tmp_path / "runs" / "r1" / "events.jsonl"
    append_jsonl(log, {"event": "phase_started", "phase": "discover_context"})
    append_jsonl(log, {"event": "phase_completed", "phase": "discover_context"})
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {
        "event": "phase_started",
        "phase": "discover_context",
    }
    assert json.loads(lines[1]) == {
        "event": "phase_completed",
        "phase": "discover_context",
    }


# ---------------------------------------------------------------------------
# ensure_gitignore_entry
# ---------------------------------------------------------------------------


def test_ensure_gitignore_entry_creates_and_appends(tmp_path: Path) -> None:
    # fresh project: no .gitignore yet -> file is created with the entry
    added = ensure_gitignore_entry(tmp_path, ".profile_project/")
    assert added is True
    gitignore = tmp_path / ".gitignore"
    assert gitignore.read_text(encoding="utf-8").splitlines() == [".profile_project/"]


def test_ensure_gitignore_entry_is_idempotent(tmp_path: Path) -> None:
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("node_modules/\n.profile_project/\n", encoding="utf-8")
    added = ensure_gitignore_entry(tmp_path, ".profile_project/")
    assert added is False
    # untouched: no duplicate line appended, existing content preserved
    assert gitignore.read_text(encoding="utf-8").splitlines() == [
        "node_modules/",
        ".profile_project/",
    ]


def test_ensure_gitignore_entry_appends_to_existing(tmp_path: Path) -> None:
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("node_modules/\n", encoding="utf-8")
    added = ensure_gitignore_entry(tmp_path, ".profile_project/")
    assert added is True
    assert gitignore.read_text(encoding="utf-8").splitlines() == [
        "node_modules/",
        ".profile_project/",
    ]


# ---------------------------------------------------------------------------
# rewrite_root_prefix
# ---------------------------------------------------------------------------


def test_rewrite_root_prefix_rewrites_absolute_paths(tmp_path: Path) -> None:
    old_root = "/old/abs/proj"
    new_root = "/new/abs/proj"
    run_data_root = tmp_path / ".profile_project" / "runs"
    state = run_data_root / "r1" / "run-state.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(
        json.dumps(
            {
                "run_id": "r1",
                "config_path": "/old/abs/proj/.profile_project_config.json",
                "run_data_dir": "/old/abs/proj/.profile_project/runs/r1",
                "project_root": "/old/abs/proj",
            }
        ),
        encoding="utf-8",
    )
    events = run_data_root / "r1" / "events.jsonl"
    events.write_text(
        json.dumps({"event": "init", "run_data_dir": "/old/abs/proj/.profile_project"})
        + "\n",
        encoding="utf-8",
    )

    rewritten = rewrite_root_prefix(run_data_root, old_root, new_root)

    assert rewritten == 2
    reloaded = json.loads(state.read_text(encoding="utf-8"))
    assert reloaded["config_path"] == "/new/abs/proj/.profile_project_config.json"
    assert reloaded["run_data_dir"] == "/new/abs/proj/.profile_project/runs/r1"
    assert reloaded["project_root"] == "/new/abs/proj"
    # no persisted absolute path still points at the old root
    assert old_root not in state.read_text(encoding="utf-8")
    assert old_root not in events.read_text(encoding="utf-8")


def test_rewrite_root_prefix_skips_unaffected_files(tmp_path: Path) -> None:
    run_data_root = tmp_path / "runs"
    untouched = run_data_root / "r2" / "run-state.json"
    untouched.parent.mkdir(parents=True, exist_ok=True)
    untouched.write_text(
        json.dumps({"project_root": "/somewhere/else"}), encoding="utf-8"
    )

    rewritten = rewrite_root_prefix(run_data_root, "/old/abs/proj", "/new/abs/proj")

    assert rewritten == 0
    assert json.loads(untouched.read_text(encoding="utf-8")) == {
        "project_root": "/somewhere/else"
    }


# ---------------------------------------------------------------------------
# transaction
# ---------------------------------------------------------------------------


def test_transaction_commits_all_writes_on_clean_exit(tmp_path: Path) -> None:
    with transaction(tmp_path) as txn:
        txn.write_json(tmp_path / ".profile_project_config.json", {"v": 1})
        txn.write_json(tmp_path / ".profile_project" / ".initialized", {"sv": 1})
        txn.append_jsonl(
            tmp_path / ".profile_project" / "runs" / "r1" / "events.jsonl",
            {"e": "init"},
        )
    assert (tmp_path / ".profile_project_config.json").exists()
    assert (tmp_path / ".profile_project" / ".initialized").exists()
    assert (tmp_path / ".profile_project" / "runs" / "r1" / "events.jsonl").exists()


def test_transaction_rolls_back_created_files_and_dirs_on_exception(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError):
        with transaction(tmp_path) as txn:
            txn.write_json(tmp_path / ".profile_project_config.json", {"v": 1})
            txn.write_json(tmp_path / ".profile_project" / ".initialized", {"sv": 1})
            raise RuntimeError("bootstrap blew up after partial write")
    # all-or-nothing: nothing the transaction created survives
    assert not (tmp_path / ".profile_project_config.json").exists()
    assert not (tmp_path / ".profile_project").exists()
    # tmp_path itself (pre-existing, not created by the txn) is untouched
    assert tmp_path.exists()


def test_transaction_preserves_preexisting_dirs_on_rollback(tmp_path: Path) -> None:
    keep = tmp_path / ".profile_project"
    keep.mkdir()
    (keep / "sentinel.txt").write_text("keep me", encoding="utf-8")
    with pytest.raises(RuntimeError):
        with transaction(tmp_path) as txn:
            txn.write_json(keep / "runs" / "r1" / "run-state.json", {"v": 1})
            raise RuntimeError("fail")
    # the txn-created runs/ subtree is gone...
    assert not (keep / "runs").exists()
    # ...but the pre-existing dir + file it did not create survive
    assert (keep / "sentinel.txt").read_text(encoding="utf-8") == "keep me"
