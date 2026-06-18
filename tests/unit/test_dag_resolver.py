from __future__ import annotations

from profile_project.dag.graph import Edge, Phase
from profile_project.dag.resolver import (
    input_satisfied,
    required_predecessors_satisfied,
    resolve_next_phases,
)


def _phase(
    name: str,
    *,
    entry: bool = False,
    toggle_key: str | None = None,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    executor: str = "agent",
    parallel: bool = False,
    input_mode: str = "all",
    optional: bool = False,
) -> Phase:
    return Phase(
        name=name,
        entry=entry,
        toggle_key=toggle_key,
        inputs=inputs if inputs is not None else [],
        outputs=outputs if outputs is not None else [],
        executor=executor,
        parallel=parallel,
        input_mode=input_mode,
        optional=optional,
    )


def test_input_satisfied_all_mode() -> None:
    # all: empty inputs are trivially satisfied; otherwise every input type required.
    entry = _phase("discover_context", entry=True, inputs=[], input_mode="all")
    assert input_satisfied(entry, set()) is True

    pages = _phase("build_agent_pages", inputs=["knowledge-graph"], input_mode="all")
    assert input_satisfied(pages, {"knowledge-graph"}) is True
    assert input_satisfied(pages, set()) is False

    synth_all = _phase(
        "x",
        inputs=["codebase-analysis", "docs-analysis"],
        input_mode="all",
    )
    assert input_satisfied(synth_all, {"codebase-analysis"}) is False
    assert input_satisfied(synth_all, {"codebase-analysis", "docs-analysis"}) is True


def test_input_satisfied_any_mode() -> None:
    verify = _phase(
        "verify_profile",
        inputs=["agent-pages", "human-spec", "vectorstore-index"],
        input_mode="any",
    )
    assert input_satisfied(verify, set()) is False
    assert input_satisfied(verify, {"human-spec"}) is True
    assert input_satisfied(verify, {"agent-pages", "vectorstore-index"}) is True


def test_input_satisfied_required_optional_mode() -> None:
    # required_optional: inputs non-empty AND inputs[0] available; the rest optional.
    synth = _phase(
        "synthesize_knowledge",
        inputs=["codebase-analysis", "docs-analysis", "context-analysis"],
        input_mode="required_optional",
    )
    assert input_satisfied(synth, set()) is False
    assert input_satisfied(synth, {"docs-analysis", "context-analysis"}) is False
    assert input_satisfied(synth, {"codebase-analysis"}) is True
    assert input_satisfied(synth, {"codebase-analysis", "docs-analysis"}) is True
