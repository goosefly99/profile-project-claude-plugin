from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from profile_project.dag.run_state import (
    EVENTS_FILENAME,
    PIPELINE_VERSION,
    RUN_STATE_FILENAME,
    ArtifactRef,
    PhaseState,
    PipelineError,
    RunState,
    append_event,
    init_run,
    list_runs,
    load_run,
    persist,
    recover_run,
    run_dir_for,
    runs_root_for,
    utc_now_iso,
)
from profile_project.dag.run_state import (
    EVENTS_FILENAME as _EVENTS_FILENAME,
)
from profile_project.dag.run_state import (
    RUN_STATE_FILENAME as _RUN_STATE_FILENAME,
)


def test_module_constants_match_spec() -> None:
    assert PIPELINE_VERSION == "profile-project/1"
    assert RUN_STATE_FILENAME == "run-state.json"
    assert EVENTS_FILENAME == "events.jsonl"


def test_phase_state_defaults_to_pending_and_forbids_extra() -> None:
    ps = PhaseState(phase_name="discover_context")
    assert ps.status == "pending"
    assert ps.input_artifacts == []
    assert ps.output_artifacts == []
    assert ps.retry_count == 0
    assert ps.started_at is None
    assert ps.completed_at is None
    assert ps.error is None
    with pytest.raises(ValidationError):
        PhaseState.model_validate({"phase_name": "x", "bogus": 1})


def test_artifact_ref_shape_and_forbids_extra() -> None:
    ref = ArtifactRef(
        type="source-index",
        path=".profile_project/artifacts/source-index.json",
        phase="discover_context",
        created_at="2026-06-18T14:03:11Z",
    )
    assert ref.version == 1
    assert ref.parent_artifact is None
    # to_dict() mirrors model_dump(mode="json")
    assert ref.to_dict() == ref.model_dump(mode="json")
    assert ref.to_dict()["type"] == "source-index"
    with pytest.raises(ValidationError):
        ArtifactRef.model_validate(
            {
                "type": "t",
                "path": "p",
                "phase": "f",
                "created_at": "2026-06-18T14:03:11Z",
                "bogus": 1,
            }
        )


def test_run_state_serializes_to_the_section_7_5_keys() -> None:
    state = RunState(
        run_id="run-001",
        created_at="2026-06-18T14:03:11Z",
        updated_at="2026-06-18T14:03:11Z",
        config_path="/abs/.profile_project_config.json",
        phases={"discover_context": PhaseState(phase_name="discover_context")},
    )
    dumped = state.model_dump(mode="json")
    assert set(dumped) == {
        "run_id",
        "pipeline_version",
        "created_at",
        "updated_at",
        "completed_at",
        "status",
        "config_path",
        "run_data_dir",
        "run_parameters",
        "phases",
        "available_artifacts",
    }
    assert dumped["pipeline_version"] == "profile-project/1"
    assert dumped["status"] == "initialized"
    assert dumped["completed_at"] is None
    # to_dict() is the canonical JSON-mode dump
    assert state.to_dict() == dumped
    # round-trips through JSON + validation unchanged
    assert RunState.model_validate(json.loads(json.dumps(dumped))) == state


def test_run_state_read_only_accessors() -> None:
    state = RunState(
        run_id="run-acc",
        created_at="2026-06-18T14:03:11Z",
        updated_at="2026-06-18T14:03:11Z",
        config_path="/abs/.profile_project_config.json",
        phases={
            "discover_context": PhaseState(
                phase_name="discover_context", status="completed"
            ),
            "analyze_docs": PhaseState(phase_name="analyze_docs", status="skipped"),
            "analyze_codebase": PhaseState(
                phase_name="analyze_codebase", status="pending"
            ),
        },
        available_artifacts=[
            ArtifactRef(
                type="source-index",
                path="a.json",
                phase="discover_context",
                created_at="2026-06-18T14:03:11Z",
            ),
            ArtifactRef(
                type="codebase-analysis",
                path="b.json",
                phase="analyze_codebase",
                created_at="2026-06-18T14:03:11Z",
            ),
        ],
    )
    assert state.completed_phase_names() == ["discover_context"]
    assert state.skipped_phase_names() == ["analyze_docs"]
    assert state.available_artifact_types() == ["source-index", "codebase-analysis"]


def test_pipeline_error_carries_a_structured_envelope() -> None:
    err = PipelineError("boom", code="run_state_corrupt", remedy="delete + re-init")
    assert err.envelope["ok"] is False
    inner = err.envelope["error"]
    assert isinstance(inner, dict)
    assert inner["code"] == "run_state_corrupt"
    assert inner["message"] == "boom"
    assert inner["remedy"] == "delete + re-init"
    assert inner["retriable"] is False


def test_pipeline_error_defaults_and_extra_fields() -> None:
    err = PipelineError(
        "nope", code="run_state_unanchored", retriable=True, run_id="r1"
    )
    inner = err.envelope["error"]
    assert isinstance(inner, dict)
    assert inner["remedy"] == ""
    assert inner["retriable"] is True
    assert inner["run_id"] == "r1"


def test_runs_root_for_and_run_dir_for(tmp_path: Path) -> None:
    assert runs_root_for(tmp_path) == tmp_path / ".profile_project" / "runs"
    assert run_dir_for(tmp_path, "run-abc") == (
        tmp_path / ".profile_project" / "runs" / "run-abc"
    )


def test_init_run_creates_all_phases_pending_when_no_toggles_off(
    tmp_path: Path,
) -> None:
    run_dir = run_dir_for(tmp_path, "run-abc")
    params: dict[str, object] = {
        "include_docs": True,
        "include_transcripts": True,
        "build_vectorstore": True,
        "phase_models": {"default": None},
    }
    state = init_run(params, run_dir)
    assert state.run_id == "run-abc"
    assert state.pipeline_version == "profile-project/1"
    assert state.status == "initialized"
    assert state.run_parameters == params
    assert list(state.phases) == [
        "discover_context",
        "analyze_codebase",
        "analyze_docs",
        "analyze_transcripts_notes",
        "synthesize_knowledge",
        "build_agent_pages",
        "build_human_spec",
        "build_vectorstore",
        "verify_profile",
    ]
    assert all(ph.status == "pending" for ph in state.phases.values())
    # run_data_dir is anchored to run dir; config_path points at the project root config
    assert state.run_data_dir == str(run_dir)
    assert state.config_path == str(tmp_path / ".profile_project_config.json")


def test_init_run_toggle_skips_each_disabled_branch_at_creation(
    tmp_path: Path,
) -> None:
    run_dir = run_dir_for(tmp_path, "run-skip")
    params: dict[str, object] = {
        "include_docs": False,
        "include_transcripts": False,
        "build_vectorstore": False,
    }
    state = init_run(params, run_dir)
    assert state.phases["analyze_docs"].status == "skipped"
    assert state.phases["analyze_transcripts_notes"].status == "skipped"
    assert state.phases["build_vectorstore"].status == "skipped"
    # always-on phases are untouched by toggle-skip
    assert state.phases["analyze_codebase"].status == "pending"
    assert state.phases["synthesize_knowledge"].status == "pending"
    assert state.phases["build_agent_pages"].status == "pending"
    assert state.phases["verify_profile"].status == "pending"
    # the accessor reflects the toggle-skip
    assert state.skipped_phase_names() == [
        "analyze_docs",
        "analyze_transcripts_notes",
        "build_vectorstore",
    ]


def test_init_run_missing_toggle_key_defaults_to_not_skipped(tmp_path: Path) -> None:
    # An absent toggle key is NOT False, so the phase stays pending (skip-if-empty
    # is decided later at discover time, not here).
    state = init_run({}, run_dir_for(tmp_path, "r"))
    assert state.phases["analyze_docs"].status == "pending"
    assert state.phases["build_vectorstore"].status == "pending"


def test_init_run_does_not_write_to_disk(tmp_path: Path) -> None:
    run_dir = run_dir_for(tmp_path, "run-nodisk")
    init_run({}, run_dir)
    assert not run_dir.exists()
    assert not (tmp_path / ".profile_project").exists()


def test_utc_now_iso_is_zulu_suffixed() -> None:
    stamp = utc_now_iso()
    assert stamp.endswith("Z")
    assert "T" in stamp


def test_persist_atomically_writes_run_state_and_bumps_updated_at(
    tmp_path: Path,
) -> None:
    run_dir = run_dir_for(tmp_path, "run-persist")
    state = init_run({}, run_dir)
    original_updated = state.updated_at
    persist(state)
    written = run_dir / _RUN_STATE_FILENAME
    assert written.exists()
    data = json.loads(written.read_text(encoding="utf-8"))
    assert data["run_id"] == "run-persist"
    assert data["pipeline_version"] == "profile-project/1"
    # persist() refreshes updated_at on the in-memory object too
    assert state.updated_at >= original_updated
    assert data["updated_at"] == state.updated_at
    # no stray temp file from the atomic write
    siblings = sorted(p.name for p in run_dir.iterdir())
    assert siblings == [_RUN_STATE_FILENAME]


def test_persist_raises_when_run_data_dir_is_none() -> None:
    state = RunState(
        run_id="x",
        created_at="2026-06-18T14:03:11Z",
        updated_at="2026-06-18T14:03:11Z",
        config_path="/abs/.profile_project_config.json",
        run_data_dir=None,
    )
    with pytest.raises(PipelineError) as exc:
        persist(state)
    inner = exc.value.envelope["error"]
    assert isinstance(inner, dict)
    assert inner["code"] == "run_state_unanchored"


def test_append_event_writes_one_jsonl_line_per_call(tmp_path: Path) -> None:
    run_dir = run_dir_for(tmp_path, "run-events")
    state = init_run({}, run_dir)
    append_event(state, "phase_started", phase="discover_context")
    append_event(
        state, "artifact_stored", phase="discover_context", type="source-index"
    )
    lines = (run_dir / _EVENTS_FILENAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["event"] == "phase_started"
    assert first["phase"] == "discover_context"
    assert first["run_id"] == "run-events"
    assert first["ts"].endswith("Z")
    second = json.loads(lines[1])
    assert second["event"] == "artifact_stored"
    assert second["type"] == "source-index"


def test_load_run_round_trips_a_persisted_state(tmp_path: Path) -> None:
    run_dir = run_dir_for(tmp_path, "run-load")
    state = init_run({"include_docs": False}, run_dir)
    persist(state)
    loaded = load_run(run_dir)
    assert loaded.run_id == "run-load"
    assert loaded.phases["analyze_docs"].status == "skipped"
    assert loaded.pipeline_version == "profile-project/1"


def test_load_run_raises_pipeline_error_when_file_absent(tmp_path: Path) -> None:
    run_dir = run_dir_for(tmp_path, "missing")
    with pytest.raises(PipelineError) as exc:
        load_run(run_dir)
    inner = exc.value.envelope["error"]
    assert isinstance(inner, dict)
    assert inner["code"] == "run_state_corrupt"
    assert inner["remedy"] != ""


def test_load_run_raises_pipeline_error_on_malformed_json(tmp_path: Path) -> None:
    run_dir = run_dir_for(tmp_path, "broken")
    run_dir.mkdir(parents=True)
    (run_dir / RUN_STATE_FILENAME).write_text("{not valid json", encoding="utf-8")
    with pytest.raises(PipelineError) as exc:
        load_run(run_dir)
    inner = exc.value.envelope["error"]
    assert isinstance(inner, dict)
    assert inner["code"] == "run_state_corrupt"


def test_load_run_raises_pipeline_error_on_schema_violation(tmp_path: Path) -> None:
    run_dir = run_dir_for(tmp_path, "badschema")
    run_dir.mkdir(parents=True)
    (run_dir / RUN_STATE_FILENAME).write_text(
        json.dumps({"run_id": "x", "unexpected_key": 1}), encoding="utf-8"
    )
    with pytest.raises(PipelineError) as exc:
        load_run(run_dir)
    inner = exc.value.envelope["error"]
    assert isinstance(inner, dict)
    assert inner["code"] == "run_state_corrupt"


def test_list_runs_returns_every_persisted_run_sorted_by_id(tmp_path: Path) -> None:
    persist(init_run({}, run_dir_for(tmp_path, "run-b")))
    persist(init_run({}, run_dir_for(tmp_path, "run-a")))
    # a directory with no run-state.json is silently skipped
    (runs_root_for(tmp_path) / "not-a-run").mkdir(parents=True)
    runs = list_runs(runs_root_for(tmp_path))
    assert [r.run_id for r in runs] == ["run-a", "run-b"]


def test_list_runs_missing_root_returns_empty(tmp_path: Path) -> None:
    assert list_runs(runs_root_for(tmp_path)) == []


def test_recover_run_resets_in_progress_to_pending_with_retry_bump(
    tmp_path: Path,
) -> None:
    run_dir = run_dir_for(tmp_path, "run-crashed")
    state = init_run({}, run_dir)
    state.status = "running"
    state.phases["analyze_codebase"].status = "in_progress"
    state.phases["analyze_codebase"].started_at = "2026-06-18T14:03:11Z"
    state.phases["analyze_codebase"].error = "stale"
    persist(state)

    recovered, recovered_phases, warnings = recover_run(run_dir)
    ph = recovered.phases["analyze_codebase"]
    assert ph.status == "pending"
    assert ph.retry_count == 1
    assert ph.started_at is None
    assert ph.error is None
    assert recovered_phases == ["analyze_codebase"]
    # the reset is persisted, not just in-memory
    reloaded = load_run(run_dir)
    assert reloaded.phases["analyze_codebase"].status == "pending"
    assert reloaded.phases["analyze_codebase"].retry_count == 1


def test_recover_run_reconciles_failed_status_with_no_failed_phase(
    tmp_path: Path,
) -> None:
    run_dir = run_dir_for(tmp_path, "run-mislabeled")
    state = init_run({}, run_dir)
    state.status = "failed"  # stale: no phase is actually failed
    state.phases["discover_context"].status = "completed"
    persist(state)

    recovered, recovered_phases, warnings = recover_run(run_dir)
    assert recovered.status == "running"
    assert recovered_phases == []
    assert any("failed" in w for w in warnings)


def test_recover_run_leaves_clean_state_unchanged(tmp_path: Path) -> None:
    run_dir = run_dir_for(tmp_path, "run-clean")
    state = init_run({}, run_dir)
    state.status = "running"
    state.phases["discover_context"].status = "completed"
    persist(state)

    recovered, recovered_phases, warnings = recover_run(run_dir)
    assert recovered.status == "running"
    assert recovered.phases["discover_context"].status == "completed"
    assert recovered_phases == []
    assert warnings == []


def test_recover_run_raises_pipeline_error_on_corrupt_state(tmp_path: Path) -> None:
    run_dir = run_dir_for(tmp_path, "run-corrupt")
    run_dir.mkdir(parents=True)
    (run_dir / RUN_STATE_FILENAME).write_text("{broken", encoding="utf-8")
    with pytest.raises(PipelineError) as exc:
        recover_run(run_dir)
    inner = exc.value.envelope["error"]
    assert isinstance(inner, dict)
    assert inner["code"] == "run_state_corrupt"
