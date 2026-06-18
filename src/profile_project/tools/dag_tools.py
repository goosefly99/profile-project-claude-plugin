from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from profile_project.config.init_gate import is_initialized, resolve_project_root
from profile_project.dag.brief import build_phase_brief
from profile_project.dag.graph import EDGES, PHASES
from profile_project.dag.lifecycle import (
    complete_phase,
    fail_phase,
    retry_phase,
    start_phase,
)
from profile_project.dag.resolver import resolve_next_phases
from profile_project.dag.run_state import RunState, init_run, list_runs, load_run, persist
from profile_project.tools._envelope import ToolError, require_init, tool_envelope

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


def _runs_root(root: Path) -> Path:
    return root / ".profile_project" / "runs"


def _load_or_raise(root: Path, run_id: str) -> RunState:
    run_dir = _runs_root(root) / run_id
    if not run_dir.is_dir():
        raise ToolError(
            "run_not_found",
            f"No run with id {run_id!r} under {_runs_root(root)}.",
        )
    return load_run(run_dir)


@tool_envelope
def pp_run_status(run_id: str) -> dict[str, object]:
    root = resolve_project_root()
    run_dir = _runs_root(root) / run_id
    if not is_initialized(root) or not run_dir.is_dir():
        return {"run_state": None}
    return {"run_state": load_run(run_dir).to_dict()}


@tool_envelope
def pp_list_runs() -> dict[str, object]:
    root = resolve_project_root()
    runs_root = _runs_root(root)
    if not is_initialized(root) or not runs_root.is_dir():
        return {"runs": []}
    return {
        "runs": [
            {"run_id": s.run_id, "status": s.status}
            for s in list_runs(runs_root)
        ]
    }


@tool_envelope
@require_init
def pp_init_run(run_parameters: dict[str, object] | None = None) -> dict[str, object]:
    root = resolve_project_root()
    run_id = uuid.uuid4().hex
    run_dir = _runs_root(root) / run_id
    state = init_run(run_parameters or {}, run_dir)
    persist(state)
    return {"run_id": state.run_id, "run_state": state.to_dict()}


@tool_envelope
@require_init
def pp_next_phases(run_id: str) -> dict[str, object]:
    root = resolve_project_root()
    state = _load_or_raise(root, run_id)
    next_phases = resolve_next_phases(
        PHASES,
        EDGES,
        set(state.completed_phase_names()),
        set(state.available_artifact_types()),
        set(state.skipped_phase_names()),
    )
    recommended = next_phases[0] if next_phases else None
    return {"next_phases": next_phases, "recommended": recommended}


@tool_envelope
@require_init
def pp_start_phase(run_id: str, phase: str) -> dict[str, object]:
    root = resolve_project_root()
    state = _load_or_raise(root, run_id)
    # start_phase mutates the RunState in place (pending->in_progress) and
    # persists; it returns a PhaseState, so we MUST NOT rebind `state` to its
    # result — build_phase_brief takes the (unchanged) RunState.
    start_phase(state, phase)
    return build_phase_brief(state, phase)


@tool_envelope
@require_init
def pp_complete_phase(run_id: str, phase: str) -> dict[str, object]:
    root = resolve_project_root()
    state = _load_or_raise(root, run_id)
    state = complete_phase(state, phase)
    return {"run_state": state.to_dict()}


@tool_envelope
@require_init
def pp_fail_phase(run_id: str, phase: str, error: str) -> dict[str, object]:
    root = resolve_project_root()
    state = _load_or_raise(root, run_id)
    state = fail_phase(state, phase, error)
    return {"run_state": state.to_dict()}


@tool_envelope
@require_init
def pp_retry_phase(run_id: str, phase: str) -> dict[str, object]:
    root = resolve_project_root()
    state = _load_or_raise(root, run_id)
    state = retry_phase(state, phase)
    return {"run_state": state.to_dict()}


def register_dag_tools(mcp: FastMCP) -> None:
    mcp.tool()(pp_init_run)
    mcp.tool()(pp_next_phases)
    mcp.tool()(pp_start_phase)
    mcp.tool()(pp_complete_phase)
    mcp.tool()(pp_fail_phase)
    mcp.tool()(pp_retry_phase)
    mcp.tool()(pp_run_status)
    mcp.tool()(pp_list_runs)
