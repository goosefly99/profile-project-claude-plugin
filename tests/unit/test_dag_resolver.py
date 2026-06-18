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


def test_required_predecessors_or_over_required_edges() -> None:
    # synthesize_knowledge has one required edge (from analyze_codebase) plus two
    # optional edges; the OR-over-required gate ignores the optional ones.
    edges = [
        Edge(src="analyze_codebase", dst="synthesize_knowledge", required=True),
        Edge(src="analyze_docs", dst="synthesize_knowledge", required=False),
        Edge(src="analyze_transcripts_notes", dst="synthesize_knowledge", required=False),
    ]
    synth = _phase("synthesize_knowledge", input_mode="required_optional")

    # No required predecessor done yet -> not satisfied.
    assert required_predecessors_satisfied(synth, edges, set(), set()) is False
    # Required predecessor completed -> satisfied.
    assert (
        required_predecessors_satisfied(synth, edges, {"analyze_codebase"}, set())
        is True
    )
    # An optional predecessor done is NOT enough.
    assert (
        required_predecessors_satisfied(synth, edges, {"analyze_docs"}, set())
        is False
    )


def test_required_predecessor_skipped_counts_as_satisfied() -> None:
    # verify_profile's required edge is build_agent_pages -> verify_profile; a skipped
    # source counts the same as completed for the predecessor gate.
    edges = [
        Edge(src="build_agent_pages", dst="verify_profile", required=True),
        Edge(src="build_human_spec", dst="verify_profile", required=False),
        Edge(src="build_vectorstore", dst="verify_profile", required=False),
    ]
    verify = _phase(
        "verify_profile",
        inputs=["agent-pages", "human-spec", "vectorstore-index"],
        input_mode="any",
    )
    assert required_predecessors_satisfied(verify, edges, set(), set()) is False
    assert (
        required_predecessors_satisfied(verify, edges, set(), {"build_agent_pages"})
        is True
    )


def test_required_predecessors_satisfied_entry_phase() -> None:
    # An entry phase has no required incoming edge -> gate 2 is vacuously satisfied.
    discover = _phase("discover_context", entry=True, input_mode="all")
    assert required_predecessors_satisfied(discover, [], set(), set()) is True
