from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from profile_project.dag.brief import (
    AgentDirective,
    PhaseBrief,
    build_phase_brief,
    resolve_input_artifacts,
    resolve_model,
)
from profile_project.dag.graph import EDGES
from profile_project.dag.run_state import RunState

# The flat repo-relative POSIX artifacts dir the real lifecycle emits paths under.
ARTIFACT_BASE = ".profile_project/artifacts"


def test_agent_directive_forbids_extra() -> None:
    d = AgentDirective(
        subagent_type="codebase-analyzer",
        model="opus-tier",
        description="Analyze the codebase for profile-project",
        prompt="do the work",
        isolation=None,
    )
    assert d.subagent_type == "codebase-analyzer"
    assert d.isolation is None
    with pytest.raises(ValidationError):
        AgentDirective.model_validate(
            {
                "subagent_type": "x",
                "model": None,
                "description": "d",
                "prompt": "p",
                "isolation": None,
                "boom": 1,
            }
        )


def test_phase_brief_forbids_extra_and_allows_null_directive() -> None:
    b = PhaseBrief(
        run_id="r1",
        phase="discover_context",
        description="Deterministic source discovery.",
        input_artifacts=[],
        expected_outputs=["source-index"],
        input_mode="all",
        optional=False,
        agent_directive=None,
        completion_contract="...",
        next_step="...",
        warnings=[],
    )
    assert b.agent_directive is None
    with pytest.raises(ValidationError):
        PhaseBrief.model_validate({"run_id": "r1", "extra": 1})


def _state_with_outputs(tmp_path: Path) -> RunState:
    run_dir = tmp_path / ".profile_project" / "runs" / "r1"
    run_dir.mkdir(parents=True, exist_ok=True)
    base: dict[str, object] = {
        "run_id": "r1",
        "pipeline_version": "profile-project/1",
        "created_at": "2026-06-18T00:00:00Z",
        "updated_at": "2026-06-18T00:00:00Z",
        "completed_at": None,
        "status": "running",
        "config_path": str(tmp_path / ".profile_project_config.json"),
        "run_data_dir": str(run_dir),
        "run_parameters": {"phase_models": {"default": None}},
        "available_artifacts": [],
    }

    def _phase(name: str, outs: list[str]) -> dict[str, object]:
        return {
            "phase_name": name,
            "status": "completed",
            "input_artifacts": [],
            "output_artifacts": outs,
            "retry_count": 0,
            "started_at": None,
            "completed_at": None,
            "error": None,
        }

    # output_artifacts are the flat repo-relative POSIX *paths* the real
    # lifecycle produces (artifact_path(root, t).relative_to(root).as_posix()),
    # i.e. ``.profile_project/artifacts/<type>.json`` — NOT artifact type strings
    # and NOT abs paths.
    base["phases"] = {
        "analyze_codebase": _phase(
            "analyze_codebase",
            [f"{ARTIFACT_BASE}/codebase-analysis.json"],
        ),
        "analyze_docs": _phase(
            "analyze_docs",
            [f"{ARTIFACT_BASE}/docs-analysis.json"],
        ),
        "analyze_transcripts_notes": _phase(
            "analyze_transcripts_notes",
            [f"{ARTIFACT_BASE}/context-analysis.json"],
        ),
        "synthesize_knowledge": _phase("synthesize_knowledge", []),
    }
    return RunState.model_validate(base)


def test_resolve_input_artifacts_collects_upstream_paths_first_seen(
    tmp_path: Path,
) -> None:
    state = _state_with_outputs(tmp_path)
    paths = resolve_input_artifacts(state, "synthesize_knowledge", EDGES)
    assert paths == [
        f"{ARTIFACT_BASE}/codebase-analysis.json",
        f"{ARTIFACT_BASE}/docs-analysis.json",
        f"{ARTIFACT_BASE}/context-analysis.json",
    ]


def test_resolve_input_artifacts_dedups(tmp_path: Path) -> None:
    state = _state_with_outputs(tmp_path)
    dup = f"{ARTIFACT_BASE}/codebase-analysis.json"
    state.phases["analyze_docs"].output_artifacts = [dup]
    paths = resolve_input_artifacts(state, "synthesize_knowledge", EDGES)
    assert paths.count(dup) == 1


def test_real_lifecycle_sets_output_artifacts_to_paths(tmp_path: Path) -> None:
    """The REAL lifecycle (C-1): complete_phase sets ps.output_artifacts to flat
    repo-relative POSIX *paths*, never artifact TYPE strings — so downstream
    briefs hand sub-agents openable inputs. Guards against regressing to types."""
    from profile_project.dag.lifecycle import complete_phase, start_phase
    from profile_project.dag.run_state import init_run

    run_dir = tmp_path / ".profile_project" / "runs" / "r1"
    run_dir.mkdir(parents=True, exist_ok=True)
    state = init_run({}, run_dir)

    start_phase(state, "discover_context")
    complete_phase(state, "discover_context")
    start_phase(state, "analyze_codebase")
    complete_phase(state, "analyze_codebase")

    ac = state.phases["analyze_codebase"]
    # PATH, not the "codebase-analysis" TYPE string.
    assert ac.output_artifacts == [f"{ARTIFACT_BASE}/codebase-analysis.json"]
    assert "codebase-analysis" not in ac.output_artifacts

    # And the downstream brief therefore resolves an openable PATH.
    paths = resolve_input_artifacts(state, "synthesize_knowledge", EDGES)
    assert f"{ARTIFACT_BASE}/codebase-analysis.json" in paths


def test_resolve_model_prefers_phase_then_default_then_none() -> None:
    pm: dict[str, object] = {
        "default": "lite-tier",
        "analyze_codebase": "opus-tier",
    }
    assert resolve_model("analyze_codebase", pm) == "opus-tier"
    assert resolve_model("build_human_spec", pm) == "lite-tier"


def test_resolve_model_null_phase_falls_through_to_default() -> None:
    pm: dict[str, object] = {"default": "lite-tier", "analyze_codebase": None}
    assert resolve_model("analyze_codebase", pm) == "lite-tier"


def test_resolve_model_null_default_inherits_session() -> None:
    pm: dict[str, object] = {"default": None}
    assert resolve_model("analyze_codebase", pm) is None
    assert resolve_model("anything", {}) is None


def test_build_phase_brief_agent_phase_has_directive_with_resolved_model(
    tmp_path: Path,
) -> None:
    state = _state_with_outputs(tmp_path)
    state.run_parameters["phase_models"] = {
        "default": None,
        "analyze_codebase": "opus-tier",
    }
    state.phases["synthesize_knowledge"].status = "in_progress"
    brief = build_phase_brief(state, "synthesize_knowledge")
    assert brief["phase"] == "synthesize_knowledge"
    assert brief["expected_outputs"] == ["knowledge-graph"]
    assert brief["input_mode"] == "required_optional"
    assert brief["optional"] is False
    assert brief["agent_directive"] is not None
    directive = brief["agent_directive"]
    assert isinstance(directive, dict)
    assert directive["subagent_type"] == "synthesize_knowledge"
    assert directive["model"] is None  # not in phase_models -> inherit
    assert brief["input_artifacts"] == [
        f"{ARTIFACT_BASE}/codebase-analysis.json",
        f"{ARTIFACT_BASE}/docs-analysis.json",
        f"{ARTIFACT_BASE}/context-analysis.json",
    ]
    assert "pp_store_artifact" in str(brief["completion_contract"])


def test_build_phase_brief_deterministic_phase_has_null_directive_and_server_tool(
    tmp_path: Path,
) -> None:
    state = _state_with_outputs(tmp_path)
    brief = build_phase_brief(state, "discover_context")
    assert brief["agent_directive"] is None
    assert brief["expected_outputs"] == ["source-index"]
    assert "pp_discover_sources(persist=true)" in str(brief["next_step"])
    assert "pp_store_artifact" not in str(brief["completion_contract"])

    brief2 = build_phase_brief(state, "build_vectorstore")
    assert brief2["agent_directive"] is None
    assert "pp_index_build" in str(brief2["next_step"])
