from __future__ import annotations

from pathlib import Path

from profile_project.artifacts.paths import artifact_path
from profile_project.dag.graph import EDGES, PHASES
from profile_project.dag.resolver import resolve_next_phases
from profile_project.dag.run_state import (
    ArtifactRef,
    PhaseState,
    PipelineError,
    RunState,
    append_event,
    persist,
    utc_now_iso,
)

_PHASES_BY_NAME = {p.name: p for p in PHASES}


def _register_phase_outputs(state: RunState, phase: str, now: str) -> None:
    """Register a completing phase's declared outputs into ``available_artifacts``.

    Idempotent: an output type already present on the run (e.g. a hand-seeded ref
    or a retry) is skipped, never duplicated. The path is the canonical flat
    repo-relative POSIX artifact location. ``ps.output_artifacts`` is set to the
    phase's declared output types. This is the single uniform point both agent and
    deterministic phases pass through, so it is where ``available_artifacts``
    becomes correctly populated as a side effect of the completion contract.
    """
    phase_def = _PHASES_BY_NAME.get(phase)
    if phase_def is None:
        return
    root = Path(state.config_path).parent
    present = set(state.available_artifact_types())
    ps = state.phases[phase]
    ps.output_artifacts = list(phase_def.outputs)
    for out_type in phase_def.outputs:
        if out_type in present:
            continue
        rel = artifact_path(root, out_type).relative_to(root).as_posix()
        state.available_artifacts.append(
            ArtifactRef(
                type=out_type,
                path=rel,
                phase=phase,
                created_at=now,
            )
        )
        present.add(out_type)


def state_transition_error(phase: str, current: str, attempted: str) -> PipelineError:
    """Build the structured `state_transition_error` (spec §7.6)."""
    return PipelineError(
        (
            f"Cannot {attempted} phase '{phase}' from status "
            f"'{current}'. See the §7.6 transition table."
        ),
        code="state_transition_error",
        retriable=False,
        phase=phase,
        current_status=current,
        attempted_transition=attempted,
    )


def _get_phase(state: RunState, phase: str, attempted: str) -> PhaseState:
    ps = state.phases.get(phase)
    if ps is None:
        raise state_transition_error(phase, current="unknown", attempted=attempted)
    return ps


def start_phase(state: RunState, phase: str) -> PhaseState:
    """Guard `pending -> in_progress` (spec §7.6/§7.7/§7.8 step 3)."""
    ps = _get_phase(state, phase, attempted="start")
    if ps.status != "pending":
        raise state_transition_error(phase, current=ps.status, attempted="start")
    now = utc_now_iso()
    ps.status = "in_progress"
    ps.started_at = now
    if state.status == "initialized":
        state.status = "running"
    state.updated_at = now
    append_event(state, "phase_started", phase=phase, at=now)
    persist(state)
    return ps


def _is_run_terminal(state: RunState) -> bool:
    """Run is terminal iff no phase is failed/in_progress AND the resolver is
    empty (spec §7.8 step 7)."""
    statuses = {ps.status for ps in state.phases.values()}
    if "failed" in statuses or "in_progress" in statuses:
        return False
    completed = {n for n, ps in state.phases.items() if ps.status == "completed"}
    skipped = {n for n, ps in state.phases.items() if ps.status == "skipped"}
    available = {a.type for a in state.available_artifacts}
    return resolve_next_phases(PHASES, EDGES, completed, available, skipped) == []


def complete_phase(state: RunState, phase: str) -> RunState:
    """Guard `in_progress -> completed`; reconcile run status (spec §7.6/§7.8)."""
    ps = _get_phase(state, phase, attempted="complete")
    if ps.status != "in_progress":
        raise state_transition_error(phase, current=ps.status, attempted="complete")
    now = utc_now_iso()
    ps.status = "completed"
    ps.completed_at = now
    _register_phase_outputs(state, phase, now)
    state.updated_at = now
    append_event(state, "phase_completed", phase=phase, at=now)
    if _is_run_terminal(state):
        state.status = "completed"
        state.completed_at = now
    persist(state)
    return state


def fail_phase(state: RunState, phase: str, error: str) -> RunState:
    """Guard `in_progress -> failed`; set run status failed (spec §7.6)."""
    ps = _get_phase(state, phase, attempted="fail")
    if ps.status != "in_progress":
        raise state_transition_error(phase, current=ps.status, attempted="fail")
    now = utc_now_iso()
    ps.status = "failed"
    ps.error = error
    state.status = "failed"
    state.updated_at = now
    append_event(state, "phase_failed", phase=phase, error=error, at=now)
    persist(state)
    return state


def skip_phase(state: RunState, phase: str) -> PhaseState:
    """Guard `pending -> skipped` (spec §7.3 toggle-skip / skip-if-empty)."""
    ps = _get_phase(state, phase, attempted="skip")
    if ps.status != "pending":
        raise state_transition_error(phase, current=ps.status, attempted="skip")
    now = utc_now_iso()
    ps.status = "skipped"
    state.updated_at = now
    append_event(state, "phase_skipped", phase=phase, at=now)
    persist(state)
    return ps


def retry_phase(state: RunState, phase: str) -> RunState:
    """Guard `failed -> pending`; clear error, bump retry_count (spec §7.6)."""
    ps = _get_phase(state, phase, attempted="retry")
    if ps.status != "failed":
        raise state_transition_error(phase, current=ps.status, attempted="retry")
    now = utc_now_iso()
    ps.status = "pending"
    ps.error = None
    ps.retry_count += 1
    ps.started_at = None
    ps.completed_at = None
    if state.status == "failed" and not any(
        p.status == "failed" for p in state.phases.values()
    ):
        state.status = "running"
    state.updated_at = now
    append_event(state, "phase_retried", phase=phase, at=now)
    persist(state)
    return state
