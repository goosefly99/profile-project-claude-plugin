from __future__ import annotations

import json
from pathlib import Path

import pytest
from mcp.server.fastmcp import FastMCP

from profile_project.tools import register_tools


@pytest.fixture()
def uninit_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / ".profile_project_config.json").write_text(
        json.dumps(
            {
                "vectorstore": {"enabled": True, "backend": "chromadb"},
                "embeddings": {"method": "sentence-transformers"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PROFILE_PROJECT_PROJECT_DIR", str(tmp_path))
    return tmp_path


def test_register_tools_exposes_full_surface() -> None:
    import asyncio

    mcp = FastMCP("profile-project-test")
    register_tools(mcp)
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    expected = {
        "pp_config_path", "pp_config_show", "pp_config_get", "pp_config_set",
        "pp_config_validate", "pp_init_project",
        "pp_discover_sources", "pp_list_sources", "pp_get_source", "pp_add_source",
        "pp_init_run", "pp_next_phases", "pp_start_phase", "pp_complete_phase",
        "pp_fail_phase", "pp_retry_phase", "pp_run_status", "pp_list_runs",
        "pp_store_artifact", "pp_load_artifact", "pp_list_artifacts",
        "pp_validate_artifact",
        "pp_index_build", "pp_index_rebuild", "pp_query", "pp_index_status",
        "pp_vectorstore_check",
    }
    assert expected <= names


def test_register_is_idempotent() -> None:
    # FastMCP silently overwrites on duplicate tool names (verified: emits a
    # WARNING log but does not raise). Calling register_tools twice on the same
    # instance must not crash.
    mcp = FastMCP("profile-project-test")
    register_tools(mcp)
    register_tools(mcp)  # second call must not raise on duplicate registration
