from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from profile_project.config.settings import CONFIG_FILENAME, Settings

STAMP_SCHEMA_VERSION: int = 1
SUPPORTED_STAMP_SCHEMA_VERSIONS: frozenset[int] = frozenset({1})
STAMP_DIRNAME: str = ".profile_project"
STAMP_FILENAME: str = ".initialized"

# Note: `CONFIG_FILENAME` is NOT redefined here. It is the single source of
# truth in `config.settings` and is imported above; `is_initialized` and
# `write_init_stamp` use it directly, and `config/__init__.py` re-exports it.


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
