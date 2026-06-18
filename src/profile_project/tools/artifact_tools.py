from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from profile_project.artifacts.store import (
    list_artifact_refs,
    load_artifact,
    store_artifact,
    validate_artifact,
)
from profile_project.config.init_gate import is_initialized, resolve_project_root
from profile_project.tools._envelope import ToolError, require_init, tool_envelope

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


def _runs_root(root: Path) -> Path:
    return root / ".profile_project" / "runs"


@tool_envelope
def pp_validate_artifact(type: str, content: dict[str, object]) -> dict[str, object]:
    errors = validate_artifact(type, content)
    return {"ok": len(errors) == 0, "errors": errors}


@tool_envelope
def pp_list_artifacts(run_id: str | None = None) -> dict[str, object]:
    root = resolve_project_root()
    if not is_initialized(root):
        return {"artifacts": []}
    return {"artifacts": [ref.to_dict() for ref in list_artifact_refs(root, run_id)]}


@tool_envelope
def pp_load_artifact(type: str, run_id: str | None = None) -> dict[str, object]:
    root = resolve_project_root()
    if not is_initialized(root):
        return {"artifact": None}
    return {"artifact": load_artifact(root, type, run_id)}


@tool_envelope
@require_init
def pp_store_artifact(
    run_id: str, phase: str, type: str, content: dict[str, object]
) -> dict[str, object]:
    root = resolve_project_root()
    run_dir = _runs_root(root) / run_id
    if not run_dir.is_dir():
        raise ToolError(
            "run_not_found",
            f"No run with id {run_id!r} under {_runs_root(root)}.",
        )
    ref = store_artifact(root, run_id, phase, type, content)
    return {"artifact_ref": ref.to_dict()}


def register_artifact_tools(mcp: FastMCP) -> None:
    mcp.tool()(pp_store_artifact)
    mcp.tool()(pp_load_artifact)
    mcp.tool()(pp_list_artifacts)
    mcp.tool()(pp_validate_artifact)
