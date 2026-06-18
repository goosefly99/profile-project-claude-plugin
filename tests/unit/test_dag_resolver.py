from __future__ import annotations

from profile_project.dag.graph import EDGES, PHASES, Edge, Phase
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
        Edge(
            src="analyze_transcripts_notes",
            dst="synthesize_knowledge",
            required=False,
        ),
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


def test_resolve_offers_entry_phase_first() -> None:
    phases = [
        _phase("discover_context", entry=True, inputs=[], input_mode="all"),
        _phase("analyze_codebase", inputs=["source-index"], input_mode="all"),
    ]
    edges = [Edge(src="discover_context", dst="analyze_codebase", required=True)]
    # Nothing completed, no artifacts yet: only the entry phase is runnable.
    runnable = resolve_next_phases(
        phases,
        edges,
        completed=set(),
        available_artifact_types=set(),
        skipped=set(),
    )
    assert runnable == ["discover_context"]


def test_resolve_advances_after_entry_completes() -> None:
    phases = [
        _phase("discover_context", entry=True, inputs=[], input_mode="all"),
        _phase("analyze_codebase", inputs=["source-index"], input_mode="all"),
    ]
    edges = [Edge(src="discover_context", dst="analyze_codebase", required=True)]
    runnable = resolve_next_phases(
        phases,
        edges,
        completed={"discover_context"},
        available_artifact_types={"source-index"},
        skipped=set(),
    )
    assert runnable == ["analyze_codebase"]


def test_resolve_returns_insertion_order_for_parallel_phases() -> None:
    # After discover completes, the three analyze_* phases are all runnable; the
    # resolver must return them in PHASES insertion order, deterministically.
    phases = [
        _phase("discover_context", entry=True, inputs=[], input_mode="all"),
        _phase("analyze_codebase", inputs=["source-index"], input_mode="all"),
        _phase(
            "analyze_docs", inputs=["source-index"], input_mode="all", optional=True
        ),
        _phase(
            "analyze_transcripts_notes",
            inputs=["source-index"],
            input_mode="all",
            optional=True,
        ),
    ]
    edges = [
        Edge(src="discover_context", dst="analyze_codebase", required=True),
        Edge(src="discover_context", dst="analyze_docs", required=False),
        Edge(src="discover_context", dst="analyze_transcripts_notes", required=False),
    ]
    runnable = resolve_next_phases(
        phases,
        edges,
        completed={"discover_context"},
        available_artifact_types={"source-index"},
        skipped=set(),
    )
    assert runnable == [
        "analyze_codebase",
        "analyze_docs",
        "analyze_transcripts_notes",
    ]


def _build_branch() -> tuple[list[Phase], list[Edge]]:
    phases = [
        _phase("synthesize_knowledge", input_mode="required_optional"),
        _phase("build_agent_pages", inputs=["knowledge-graph"], input_mode="all"),
        _phase("build_human_spec", inputs=["knowledge-graph"], input_mode="all"),
        _phase(
            "build_vectorstore",
            inputs=["knowledge-graph"],
            input_mode="all",
            optional=True,
        ),
        _phase(
            "verify_profile",
            inputs=["agent-pages", "human-spec", "vectorstore-index"],
            input_mode="any",
        ),
    ]
    edges = [
        Edge(src="synthesize_knowledge", dst="build_agent_pages", required=True),
        Edge(src="synthesize_knowledge", dst="build_human_spec", required=True),
        Edge(src="synthesize_knowledge", dst="build_vectorstore", required=True),
        Edge(src="build_agent_pages", dst="verify_profile", required=True),
        Edge(src="build_human_spec", dst="verify_profile", required=False),
        Edge(src="build_vectorstore", dst="verify_profile", required=False),
    ]
    return phases, edges


def test_skipped_branch_satisfies_successor_and_is_never_offered() -> None:
    # build_vectorstore toggled off (skipped). Its required edge into verify_profile
    # is satisfied by it being skipped; build_vectorstore is never offered.
    phases, edges = _build_branch()
    runnable = resolve_next_phases(
        phases,
        edges,
        completed={"synthesize_knowledge"},
        available_artifact_types={"knowledge-graph"},
        skipped={"build_vectorstore"},
    )
    assert "build_vectorstore" not in runnable
    assert runnable == ["build_agent_pages", "build_human_spec"]


def test_verify_runs_over_what_was_built_with_skipped_vectorstore() -> None:
    # After the two build phases complete (vectorstore skipped), verify_profile is
    # runnable: input_mode=any is met by agent-pages/human-spec, and its required
    # predecessor build_agent_pages is completed.
    phases, edges = _build_branch()
    runnable = resolve_next_phases(
        phases,
        edges,
        completed={"synthesize_knowledge", "build_agent_pages", "build_human_spec"},
        available_artifact_types={"knowledge-graph", "agent-pages", "human-spec"},
        skipped={"build_vectorstore"},
    )
    assert runnable == ["verify_profile"]


def test_terminal_detection_returns_empty_list() -> None:
    # Everything done or skipped -> resolver returns [] (terminal signal).
    phases, edges = _build_branch()
    runnable = resolve_next_phases(
        phases,
        edges,
        completed={
            "synthesize_knowledge",
            "build_agent_pages",
            "build_human_spec",
            "verify_profile",
        },
        available_artifact_types={"knowledge-graph", "agent-pages", "human-spec"},
        skipped={"build_vectorstore"},
    )
    assert runnable == []


def test_resolve_drives_full_fixed_graph_to_terminal() -> None:
    # Walk the real fixed graph (all toggles on) to completion, asserting the
    # resolver advances deterministically and terminates with [].
    completed: set[str] = set()
    available: set[str] = set()
    produces = {
        "discover_context": "source-index",
        "analyze_codebase": "codebase-analysis",
        "analyze_docs": "docs-analysis",
        "analyze_transcripts_notes": "context-analysis",
        "synthesize_knowledge": "knowledge-graph",
        "build_agent_pages": "agent-pages",
        "build_human_spec": "human-spec",
        "build_vectorstore": "vectorstore-index",
        "verify_profile": "verification-report",
    }
    guard = 0
    while True:
        guard += 1
        assert guard <= len(PHASES) + 1  # single-pass resolver must terminate fast
        runnable = resolve_next_phases(
            PHASES, EDGES, completed, available, skipped=set()
        )
        if not runnable:
            break
        for name in runnable:
            completed.add(name)
            available.add(produces[name])
    assert completed == {p.name for p in PHASES}
    # Once everything is completed, the resolver reports terminal.
    assert resolve_next_phases(PHASES, EDGES, completed, available, set()) == []
