from __future__ import annotations

from pathlib import Path

import pytest

from profile_project.dag.lifecycle import (
    complete_phase,
    fail_phase,
    retry_phase,
    skip_phase,
    start_phase,
    state_transition_error,
)
from profile_project.dag.run_state import ArtifactRef, PipelineError, RunState


def test_state_transition_error_envelope_shape() -> None:
    err = state_transition_error(
        "analyze_codebase", current="completed", attempted="start"
    )
    assert isinstance(err, PipelineError)
    envelope = err.envelope
    assert envelope["ok"] is False
    inner = envelope["error"]
    assert isinstance(inner, dict)
    assert inner["code"] == "state_transition_error"
    assert inner["phase"] == "analyze_codebase"
    assert inner["current_status"] == "completed"
    assert inner["attempted_transition"] == "start"
    assert inner["retriable"] is False
    assert "analyze_codebase" in inner["message"]


def _new_state(tmp_path: Path) -> RunState:
    run_dir = tmp_path / ".profile_project" / "runs" / "r1"
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunState.model_validate(
        {
            "run_id": "r1",
            "pipeline_version": "profile-project/1",
            "created_at": "2026-06-18T00:00:00Z",
            "updated_at": "2026-06-18T00:00:00Z",
            "completed_at": None,
            "status": "initialized",
            "config_path": str(tmp_path / ".profile_project_config.json"),
            "run_data_dir": str(run_dir),
            "run_parameters": {
                "include_docs": True,
                "include_transcripts": True,
                "build_vectorstore": True,
                "phase_models": {"default": None},
            },
            "phases": {
                "discover_context": {
                    "phase_name": "discover_context",
                    "status": "pending",
                    "input_artifacts": [],
                    "output_artifacts": [],
                    "retry_count": 0,
                    "started_at": None,
                    "completed_at": None,
                    "error": None,
                },
                "analyze_codebase": {
                    "phase_name": "analyze_codebase",
                    "status": "pending",
                    "input_artifacts": [],
                    "output_artifacts": [],
                    "retry_count": 0,
                    "started_at": None,
                    "completed_at": None,
                    "error": None,
                },
            },
            "available_artifacts": [],
        }
    )


def test_start_phase_transitions_pending_to_in_progress(tmp_path: Path) -> None:
    state = _new_state(tmp_path)
    ps = start_phase(state, "discover_context")
    assert ps.status == "in_progress"
    assert ps.started_at is not None
    assert state.status == "running"  # initialized -> running on first start
    events_path = Path(str(state.run_data_dir)) / "events.jsonl"
    events = events_path.read_text(encoding="utf-8")
    assert '"phase_started"' in events


def test_start_phase_rejects_non_pending(tmp_path: Path) -> None:
    state = _new_state(tmp_path)
    start_phase(state, "discover_context")
    with pytest.raises(PipelineError) as exc:
        start_phase(state, "discover_context")
    assert exc.value.envelope["error"]["current_status"] == "in_progress"


def test_complete_phase_transitions_in_progress_to_completed(tmp_path: Path) -> None:
    state = _new_state(tmp_path)
    start_phase(state, "discover_context")
    # CORRECTION A: register the source-index artifact before complete_phase,
    # matching the real store-before-complete contract (pp_store_artifact is called
    # before pp_complete_phase in actual orchestration). Without it,
    # analyze_codebase is not input-satisfied and the resolver returns [],
    # making _is_run_terminal True, which would incorrectly flip state.status
    # to "completed".
    state.available_artifacts.append(
        ArtifactRef(
            type="source-index",
            path="/abs/.profile_project/artifacts/source-index.json",
            phase="discover_context",
            created_at="2026-06-18T00:00:00Z",
        )
    )
    returned = complete_phase(state, "discover_context")
    assert returned is state
    assert state.phases["discover_context"].status == "completed"
    assert state.phases["discover_context"].completed_at is not None
    # analyze_codebase still pending -> run not terminal yet.
    assert state.status == "running"
    assert state.completed_at is None


def test_complete_phase_marks_run_completed_when_terminal(tmp_path: Path) -> None:
    state = _new_state(tmp_path)
    # Drop analyze_codebase so discover_context is the only schedulable phase,
    # and pre-skip it so the resolver yields [] after completion.
    state.phases["analyze_codebase"].status = "skipped"
    start_phase(state, "discover_context")
    complete_phase(state, "discover_context")
    assert state.status == "completed"
    assert state.completed_at is not None


def test_complete_phase_rejects_non_in_progress(tmp_path: Path) -> None:
    state = _new_state(tmp_path)
    with pytest.raises(PipelineError) as exc:
        complete_phase(state, "discover_context")
    assert exc.value.envelope["error"]["current_status"] == "pending"


def test_fail_phase_sets_error_and_run_failed(tmp_path: Path) -> None:
    state = _new_state(tmp_path)
    start_phase(state, "discover_context")
    fail_phase(state, "discover_context", "boom")
    assert state.phases["discover_context"].status == "failed"
    assert state.phases["discover_context"].error == "boom"
    assert state.status == "failed"
    events_path = Path(str(state.run_data_dir)) / "events.jsonl"
    events = events_path.read_text(encoding="utf-8")
    assert '"phase_failed"' in events


def test_fail_phase_rejects_non_in_progress(tmp_path: Path) -> None:
    state = _new_state(tmp_path)
    with pytest.raises(PipelineError) as exc:
        fail_phase(state, "discover_context", "boom")
    assert exc.value.envelope["error"]["current_status"] == "pending"


def test_skip_phase_transitions_pending_to_skipped(tmp_path: Path) -> None:
    state = _new_state(tmp_path)
    ps = skip_phase(state, "analyze_codebase")
    assert ps.status == "skipped"


def test_skip_phase_rejects_non_pending(tmp_path: Path) -> None:
    state = _new_state(tmp_path)
    start_phase(state, "discover_context")
    with pytest.raises(PipelineError) as exc:
        skip_phase(state, "discover_context")
    assert exc.value.envelope["error"]["current_status"] == "in_progress"


def test_retry_phase_resets_failed_to_pending(tmp_path: Path) -> None:
    state = _new_state(tmp_path)
    start_phase(state, "discover_context")
    fail_phase(state, "discover_context", "boom")
    retry_phase(state, "discover_context")
    ps = state.phases["discover_context"]
    assert ps.status == "pending"
    assert ps.error is None
    assert ps.retry_count == 1
    assert ps.started_at is None
    assert ps.completed_at is None
    # No phase is failed any more -> run status reconciled off "failed".
    assert state.status == "running"


def test_retry_phase_rejects_non_failed(tmp_path: Path) -> None:
    state = _new_state(tmp_path)
    with pytest.raises(PipelineError) as exc:
        retry_phase(state, "discover_context")
    assert exc.value.envelope["error"]["current_status"] == "pending"
