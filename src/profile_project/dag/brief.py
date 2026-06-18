from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from profile_project.dag.graph import EDGES, PHASES, Edge, Phase
from profile_project.dag.run_state import RunState


class AgentDirective(BaseModel):
    """Sub-agent dispatch directive (spec §7.7)."""

    model_config = ConfigDict(extra="forbid")

    subagent_type: str
    model: str | None
    description: str
    prompt: str
    isolation: str | None


class PhaseBrief(BaseModel):
    """The `pp_start_phase` brief (spec §7.7)."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    phase: str
    description: str
    input_artifacts: list[str]
    expected_outputs: list[str]
    input_mode: str
    optional: bool
    agent_directive: AgentDirective | None
    completion_contract: str
    next_step: str
    warnings: list[str]


def resolve_input_artifacts(
    state: RunState, phase: str, edges: list[Edge]
) -> list[str]:
    """Resolve upstream input-artifact paths along incoming edges (spec §7.7).

    Pulls each predecessor phase's `output_artifacts` paths in edge order,
    deduplicating while preserving first-seen order.
    """
    seen: set[str] = set()
    resolved: list[str] = []
    for edge in edges:
        if edge.dst != phase:
            continue
        upstream = state.phases.get(edge.src)
        if upstream is None:
            continue
        for path in upstream.output_artifacts:
            if path not in seen:
                seen.add(path)
                resolved.append(path)
    return resolved


def resolve_model(phase: str, phase_models: dict[str, object]) -> str | None:
    """Resolve the executor model (spec §6/§7.7).

    Order: phase_models[phase] > phase_models["default"] > None (inherit the
    orchestrator/session model). A null/missing entry falls through to the next
    level; no concrete model id is ever silently pinned.
    """
    specific = phase_models.get(phase)
    if isinstance(specific, str) and specific and specific != "default":
        return specific
    fallback = phase_models.get("default")
    if isinstance(fallback, str) and fallback and fallback != "default":
        return fallback
    return None


_PHASES_BY_NAME: dict[str, Phase] = {p.name: p for p in PHASES}

# Deterministic (server) phases -> the server tool that does compute+store (§7.7).
_DETERMINISTIC_TOOL: dict[str, str] = {
    "discover_context": "pp_discover_sources(persist=true)",
    "build_vectorstore": "pp_index_build",
}


def build_phase_brief(state: RunState, phase: str) -> dict[str, object]:
    """Build the §7.7 PhaseBrief for `phase` (returns model_dump(mode="json")).

    Deterministic phases (discover_context, build_vectorstore) carry
    agent_directive=None and a next_step that names the server tool which does
    the compute AND stores the artifact; the agent then calls pp_complete_phase.
    Agent phases carry a populated AgentDirective whose model is resolved from
    run_parameters["phase_models"] (phase > default > inherit).
    """
    spec = _PHASES_BY_NAME[phase]
    input_artifacts = resolve_input_artifacts(state, phase, EDGES)
    phase_models_raw = state.run_parameters.get("phase_models", {})
    phase_models: dict[str, object] = (
        phase_models_raw if isinstance(phase_models_raw, dict) else {}
    )

    if spec.executor == "deterministic":
        tool = _DETERMINISTIC_TOOL[phase]
        outputs_label = ", ".join(spec.outputs)
        brief = PhaseBrief(
            run_id=state.run_id,
            phase=phase,
            description=f"Deterministic (server) phase: {phase}.",
            input_artifacts=input_artifacts,
            expected_outputs=list(spec.outputs),
            input_mode=spec.input_mode,
            optional=spec.optional,
            agent_directive=None,
            completion_contract=(
                f"Call {tool}; the server computes and stores the "
                f"{outputs_label} artifact, then call "
                f"pp_complete_phase(run_id, phase) (or pp_fail_phase on error)."
            ),
            next_step=f"Call the server tool {tool}.",
            warnings=[],
        )
        return brief.model_dump(mode="json")

    model = resolve_model(phase, phase_models)
    outputs_label = ", ".join(spec.outputs)
    directive = AgentDirective(
        subagent_type=phase,
        model=model,
        description=f"Run the {phase} phase for profile-project.",
        prompt=(
            f"Phase '{phase}': produce the {outputs_label} artifact from "
            f"the resolved input artifacts. Honor the completion contract."
        ),
        isolation="worktree" if spec.parallel else None,
    )
    brief = PhaseBrief(
        run_id=state.run_id,
        phase=phase,
        description=f"Agent phase: {phase}.",
        input_artifacts=input_artifacts,
        expected_outputs=list(spec.outputs),
        input_mode=spec.input_mode,
        optional=spec.optional,
        agent_directive=directive,
        completion_contract=(
            "On success call pp_store_artifact(...) then "
            "pp_complete_phase(run_id, phase); on failure call "
            "pp_fail_phase(run_id, phase, error)."
        ),
        next_step="Spawn the sub-agent described by agent_directive.",
        warnings=[],
    )
    return brief.model_dump(mode="json")
