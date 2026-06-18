from __future__ import annotations

from profile_project.dag.graph import Edge, Phase


def input_satisfied(phase: Phase, available_artifact_types: set[str]) -> bool:
    """Gate 1: input-artifact satisfaction keyed on the closed input_mode enum."""
    inputs = phase.inputs
    if phase.input_mode == "all":
        return all(t in available_artifact_types for t in inputs)
    if phase.input_mode == "any":
        return any(t in available_artifact_types for t in inputs)
    if phase.input_mode == "required_optional":
        return len(inputs) > 0 and inputs[0] in available_artifact_types
    raise ValueError(f"unknown input_mode {phase.input_mode!r} for phase {phase.name!r}")


def required_predecessors_satisfied(
    phase: Phase,
    edges: list[Edge],
    completed: set[str],
    skipped: set[str],
) -> bool:
    """Gate 2: OR over required incoming edges; a skipped source counts as satisfied."""
    required_sources = [e.src for e in edges if e.dst == phase.name and e.required]
    if not required_sources:
        return True
    satisfied = completed | skipped
    return any(src in satisfied for src in required_sources)


def resolve_next_phases(
    phases: list[Phase],
    edges: list[Edge],
    completed: set[str],
    available_artifact_types: set[str],
    skipped: set[str],
) -> list[str]:
    """Single non-recursive pass: return runnable phase names in insertion order."""
    raise NotImplementedError
