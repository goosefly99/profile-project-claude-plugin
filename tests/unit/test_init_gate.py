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
