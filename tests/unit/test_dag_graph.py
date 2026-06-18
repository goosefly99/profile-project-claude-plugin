from __future__ import annotations

import dataclasses

import pytest

from profile_project.dag.graph import (
    EDGES,
    PHASES,
    Edge,
    Phase,
    assert_dag,
)


def _by_name() -> dict[str, Phase]:
    return {p.name: p for p in PHASES}


def test_phase_and_edge_are_frozen_dataclasses() -> None:
    p = Phase(
        name="x",
        entry=True,
        toggle_key=None,
        inputs=[],
        outputs=["source-index"],
        executor="deterministic",
        parallel=False,
        input_mode="all",
        optional=False,
    )
    e = Edge(src="a", dst="b", required=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.name = "y"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.required = False  # type: ignore[misc]


def test_phases_are_the_nine_fixed_nodes_in_order() -> None:
    assert [p.name for p in PHASES] == [
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


def test_only_discover_context_is_an_entry_point() -> None:
    entries = [p.name for p in PHASES if p.entry]
    assert entries == ["discover_context"]


def test_phase_toggle_keys_match_spec() -> None:
    phases = _by_name()
    assert phases["discover_context"].toggle_key is None
    assert phases["analyze_codebase"].toggle_key is None
    assert phases["analyze_docs"].toggle_key == "include_docs"
    assert phases["analyze_transcripts_notes"].toggle_key == "include_transcripts"
    assert phases["synthesize_knowledge"].toggle_key is None
    assert phases["build_agent_pages"].toggle_key is None
    assert phases["build_human_spec"].toggle_key is None
    assert phases["build_vectorstore"].toggle_key == "build_vectorstore"
    assert phases["verify_profile"].toggle_key is None


def test_phase_inputs_and_outputs_match_spec() -> None:
    phases = _by_name()
    assert phases["discover_context"].inputs == []
    assert phases["discover_context"].outputs == ["source-index"]
    assert phases["analyze_codebase"].inputs == ["source-index"]
    assert phases["analyze_codebase"].outputs == ["codebase-analysis"]
    assert phases["analyze_docs"].inputs == ["source-index"]
    assert phases["analyze_docs"].outputs == ["docs-analysis"]
    assert phases["analyze_transcripts_notes"].inputs == ["source-index"]
    assert phases["analyze_transcripts_notes"].outputs == ["context-analysis"]
    assert phases["synthesize_knowledge"].inputs == [
        "codebase-analysis",
        "docs-analysis",
        "context-analysis",
    ]
    assert phases["synthesize_knowledge"].outputs == ["knowledge-graph"]
    assert phases["build_agent_pages"].inputs == ["knowledge-graph"]
    assert phases["build_agent_pages"].outputs == ["agent-pages"]
    assert phases["build_human_spec"].inputs == ["knowledge-graph"]
    assert phases["build_human_spec"].outputs == ["human-spec"]
    assert phases["build_vectorstore"].inputs == ["knowledge-graph", "agent-pages"]
    assert phases["build_vectorstore"].outputs == ["vectorstore-index"]
    assert phases["verify_profile"].inputs == [
        "agent-pages",
        "human-spec",
        "vectorstore-index",
    ]
    assert phases["verify_profile"].outputs == ["verification-report"]


def test_phase_executor_input_mode_and_optional_match_spec() -> None:
    phases = _by_name()
    assert phases["discover_context"].executor == "deterministic"
    assert phases["build_vectorstore"].executor == "deterministic"
    assert phases["verify_profile"].executor == "mixed"
    for name in (
        "analyze_codebase",
        "analyze_docs",
        "analyze_transcripts_notes",
        "synthesize_knowledge",
        "build_agent_pages",
        "build_human_spec",
    ):
        assert phases[name].executor == "agent"
    assert phases["synthesize_knowledge"].input_mode == "required_optional"
    assert phases["verify_profile"].input_mode == "any"
    for name in (
        "discover_context",
        "analyze_codebase",
        "analyze_docs",
        "analyze_transcripts_notes",
        "build_agent_pages",
        "build_human_spec",
        "build_vectorstore",
    ):
        assert phases[name].input_mode == "all"
    optional = {p.name for p in PHASES if p.optional}
    assert optional == {
        "analyze_docs",
        "analyze_transcripts_notes",
        "build_vectorstore",
    }


def test_analyze_phases_are_parallel() -> None:
    phases = _by_name()
    assert phases["analyze_codebase"].parallel is True
    assert phases["analyze_docs"].parallel is True
    assert phases["analyze_transcripts_notes"].parallel is True
    assert phases["discover_context"].parallel is False
    assert phases["synthesize_knowledge"].parallel is False
    assert phases["verify_profile"].parallel is False


def test_edges_are_the_twelve_fixed_edges_in_order() -> None:
    assert [(e.src, e.dst, e.required) for e in EDGES] == [
        ("discover_context", "analyze_codebase", True),
        ("discover_context", "analyze_docs", False),
        ("discover_context", "analyze_transcripts_notes", False),
        ("analyze_codebase", "synthesize_knowledge", True),
        ("analyze_docs", "synthesize_knowledge", False),
        ("analyze_transcripts_notes", "synthesize_knowledge", False),
        ("synthesize_knowledge", "build_agent_pages", True),
        ("synthesize_knowledge", "build_human_spec", True),
        ("synthesize_knowledge", "build_vectorstore", True),
        ("build_agent_pages", "verify_profile", True),
        ("build_human_spec", "verify_profile", False),
        ("build_vectorstore", "verify_profile", False),
    ]


def test_assert_dag_accepts_the_fixed_graph() -> None:
    assert assert_dag(PHASES, EDGES) is None


def test_assert_dag_rejects_a_cycle() -> None:
    phases = [
        Phase(
            name="a",
            entry=True,
            toggle_key=None,
            inputs=[],
            outputs=["t-a"],
            executor="agent",
            parallel=False,
            input_mode="all",
            optional=False,
        ),
        Phase(
            name="b",
            entry=False,
            toggle_key=None,
            inputs=["t-a"],
            outputs=["t-b"],
            executor="agent",
            parallel=False,
            input_mode="all",
            optional=False,
        ),
    ]
    edges = [
        Edge(src="a", dst="b", required=True),
        Edge(src="b", dst="a", required=True),
    ]
    with pytest.raises(ValueError, match="cycle"):
        assert_dag(phases, edges)


def test_assert_dag_rejects_dangling_edge_source() -> None:
    phases = [
        Phase(
            name="a",
            entry=True,
            toggle_key=None,
            inputs=[],
            outputs=["t-a"],
            executor="agent",
            parallel=False,
            input_mode="all",
            optional=False,
        ),
    ]
    edges = [Edge(src="ghost", dst="a", required=True)]
    with pytest.raises(ValueError, match="unknown phase"):
        assert_dag(phases, edges)


def test_assert_dag_rejects_dangling_edge_target() -> None:
    phases = [
        Phase(
            name="a",
            entry=True,
            toggle_key=None,
            inputs=[],
            outputs=["t-a"],
            executor="agent",
            parallel=False,
            input_mode="all",
            optional=False,
        ),
    ]
    edges = [Edge(src="a", dst="ghost", required=True)]
    with pytest.raises(ValueError, match="unknown phase"):
        assert_dag(phases, edges)


def test_assert_dag_rejects_duplicate_phase_name() -> None:
    phases = [
        Phase(
            name="a",
            entry=True,
            toggle_key=None,
            inputs=[],
            outputs=["t-a"],
            executor="agent",
            parallel=False,
            input_mode="all",
            optional=False,
        ),
        Phase(
            name="a",
            entry=False,
            toggle_key=None,
            inputs=["t-a"],
            outputs=["t-a2"],
            executor="agent",
            parallel=False,
            input_mode="all",
            optional=False,
        ),
    ]
    with pytest.raises(ValueError, match="duplicate phase"):
        assert_dag(phases, [])
