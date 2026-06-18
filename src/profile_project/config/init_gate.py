from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from profile_project.config.files import atomic_write_json
from profile_project.config.settings import CONFIG_FILENAME, Settings

STAMP_SCHEMA_VERSION: int = 1
SUPPORTED_STAMP_SCHEMA_VERSIONS: frozenset[int] = frozenset({1})
STAMP_DIRNAME: str = ".profile_project"
STAMP_FILENAME: str = ".initialized"

# Note: `CONFIG_FILENAME` is NOT redefined here. It is the single source of
# truth in `config.settings` and is imported above; `is_initialized` and
# `write_init_stamp` use it directly, and `config/__init__.py` re-exports it.

__all__ = [
    "CONFIG_FILENAME",
    "STAMP_DIRNAME",
    "STAMP_FILENAME",
    "STAMP_SCHEMA_VERSION",
    "SUPPORTED_STAMP_SCHEMA_VERSIONS",
    "InitStamp",
    "build_init_stamp",
    "detect_root_move",
    "is_initialized",
    "not_initialized_error",
    "project_root_moved_error",
    "read_stamp",
    "resolve_project_root",
    "write_init_stamp",
]


class InitStamp(BaseModel):
    """The `.profile_project/.initialized` stamp (spec §6b.2)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    initialized_at: str
    project_root: str
    config_path: str


def resolve_project_root(settings: Settings | None = None) -> Path:
    """Resolve the absolute project root (spec §6b.4 step 1).

    Precedence: settings.project_dir -> $PROFILE_PROJECT_PROJECT_DIR ->
    $CLAUDE_PROJECT_DIR -> $PWD -> Path.cwd(). Read-only: never creates dirs.
    """
    if settings is not None and settings.project_dir is not None:
        return Path(settings.project_dir).resolve()
    for env_name in ("PROFILE_PROJECT_PROJECT_DIR", "CLAUDE_PROJECT_DIR", "PWD"):
        value = os.environ.get(env_name)
        if value:
            return Path(value).resolve()
    return Path.cwd().resolve()


def read_stamp(root: Path) -> InitStamp | None:
    """Read + validate the init stamp for `root`; None if absent/invalid.

    Read-only: opens the file for reading only and never creates the tree.
    """
    stamp_file = root / STAMP_DIRNAME / STAMP_FILENAME
    try:
        raw = stamp_file.read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError, OSError):
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    try:
        return InitStamp.model_validate(data)
    except ValueError:
        return None


def build_init_stamp(
    root: Path, config_path: Path, *, schema_version: int = 1
) -> dict[str, object]:
    """Build the init-stamp payload dict (spec §6b.2). No filesystem side effects."""
    return InitStamp(
        schema_version=schema_version,
        initialized_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        project_root=str(root.resolve()),
        config_path=str(config_path.resolve()),
    ).model_dump(mode="json")


def write_init_stamp(
    root: Path, config_path: Path, *, schema_version: int = 1
) -> Path:
    """Write the `.profile_project/.initialized` stamp atomically (spec §6b.2).

    This is the **only** mutating function in this module. It creates the
    gitignored `.profile_project/` tree if absent, then writes the stamp via
    `atomic_write_json` (temp file + fsync + os.replace, §7.6). Records the
    resolved absolute `project_root` and `config_path`, a fresh UTC ISO-8601
    `initialized_at`, and `schema_version`. Returns the stamp-file path.

    Called solely by `pp_init_project` (§6b.8) as one step of its
    all-or-nothing bootstrap transaction. The gate predicates never call it.
    """
    tree = root / STAMP_DIRNAME
    tree.mkdir(parents=True, exist_ok=True)
    stamp_file = tree / STAMP_FILENAME
    atomic_write_json(
        stamp_file, build_init_stamp(root, config_path, schema_version=schema_version)
    )
    return stamp_file


def is_initialized(root: Path) -> bool:
    """READ-ONLY init predicate (spec §6b.2 invariant).

    True iff a valid, supported, root-matching stamp exists AND the root config
    file is present. Never creates `.profile_project/`, never touches the stamp,
    never writes config. Any absence -> not initialized.
    """
    stamp = read_stamp(root)
    if stamp is None:
        return False
    if stamp.schema_version not in SUPPORTED_STAMP_SCHEMA_VERSIONS:
        return False
    if stamp.project_root != str(root):
        return False
    return (root / CONFIG_FILENAME).is_file()


def detect_root_move(root: Path) -> tuple[bool, str | None]:
    """Detect a moved/renamed project root (spec §6b.7). Read-only.

    Returns (moved, stamped_root): moved is True iff a stamp exists whose
    recorded project_root differs from str(root). stamped_root is the stamp's
    project_root, or None when no stamp is present.
    """
    stamp = read_stamp(root)
    if stamp is None:
        return (False, None)
    return (stamp.project_root != str(root), stamp.project_root)


def not_initialized_error(resolved_root: Path) -> dict[str, object]:
    """Structured `not_initialized` envelope (spec §6b.3). Non-destructive."""
    return {
        "ok": False,
        "error": {
            "code": "not_initialized",
            "message": (
                "profile-project is not initialized for this project. "
                "Run /profile-project:init first."
            ),
            "retriable": False,
            "resolved_root": str(resolved_root),
            "remedy": "/profile-project:init",
        },
    }


def project_root_moved_error(
    stamped_root: str, resolved_root: Path
) -> dict[str, object]:
    """Structured `project_root_moved` envelope (spec §6b.7). Non-destructive."""
    return {
        "ok": False,
        "error": {
            "code": "project_root_moved",
            "message": (
                "This project was initialized for a different root. "
                "Re-run /profile-project:init."
            ),
            "stamped_root": stamped_root,
            "resolved_root": str(resolved_root),
            "retriable": False,
            "remedy": "/profile-project:init --reinit",
        },
    }
