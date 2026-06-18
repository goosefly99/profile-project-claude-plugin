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
