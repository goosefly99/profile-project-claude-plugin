# src/profile_project/dag/graph.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Phase:
    """A single node in the fixed profiling DAG (§7.1).

    inputs/outputs are artifact *types* (e.g. "source-index"), not paths.
    ``input_mode`` is closed to ``all|any|required_optional`` (§7.3) and
    ``executor`` to ``deterministic|agent|mixed``; both are typed ``str``.
    """

    name: str
    entry: bool
    toggle_key: str | None
    inputs: list[str]
    outputs: list[str]
    executor: str
    parallel: bool
    input_mode: str
    optional: bool


@dataclass(frozen=True)
class Edge:
    """A directed dependency between two phases (§7.2)."""

    src: str
    dst: str
    required: bool


PHASES: list[Phase] = [
    Phase(
        name="discover_context",
        entry=True,
        toggle_key=None,
        inputs=[],
        outputs=["source-index"],
        executor="deterministic",
        parallel=False,
        input_mode="all",
        optional=False,
    ),
    Phase(
        name="analyze_codebase",
        entry=False,
        toggle_key=None,
        inputs=["source-index"],
        outputs=["codebase-analysis"],
        executor="agent",
        parallel=True,
        input_mode="all",
        optional=False,
    ),
    Phase(
        name="analyze_docs",
        entry=False,
        toggle_key="include_docs",
        inputs=["source-index"],
        outputs=["docs-analysis"],
        executor="agent",
        parallel=True,
        input_mode="all",
        optional=True,
    ),
    Phase(
        name="analyze_transcripts_notes",
        entry=False,
        toggle_key="include_transcripts",
        inputs=["source-index"],
        outputs=["context-analysis"],
        executor="agent",
        parallel=True,
        input_mode="all",
        optional=True,
    ),
    Phase(
        name="synthesize_knowledge",
        entry=False,
        toggle_key=None,
        inputs=["codebase-analysis", "docs-analysis", "context-analysis"],
        outputs=["knowledge-graph"],
        executor="agent",
        parallel=False,
        input_mode="required_optional",
        optional=False,
    ),
    Phase(
        name="build_agent_pages",
        entry=False,
        toggle_key=None,
        inputs=["knowledge-graph"],
        outputs=["agent-pages"],
        executor="agent",
        parallel=False,
        input_mode="all",
        optional=False,
    ),
    Phase(
        name="build_human_spec",
        entry=False,
        toggle_key=None,
        inputs=["knowledge-graph"],
        outputs=["human-spec"],
        executor="agent",
        parallel=False,
        input_mode="all",
        optional=False,
    ),
    Phase(
        name="build_vectorstore",
        entry=False,
        toggle_key="build_vectorstore",
        inputs=["knowledge-graph", "agent-pages"],
        outputs=["vectorstore-index"],
        executor="deterministic",
        parallel=False,
        input_mode="all",
        optional=True,
    ),
    Phase(
        name="verify_profile",
        entry=False,
        toggle_key=None,
        inputs=["agent-pages", "human-spec", "vectorstore-index"],
        outputs=["verification-report"],
        executor="mixed",
        parallel=False,
        input_mode="any",
        optional=False,
    ),
]


EDGES: list[Edge] = [
    Edge(src="discover_context", dst="analyze_codebase", required=True),
    Edge(src="discover_context", dst="analyze_docs", required=False),
    Edge(src="discover_context", dst="analyze_transcripts_notes", required=False),
    Edge(src="analyze_codebase", dst="synthesize_knowledge", required=True),
    Edge(src="analyze_docs", dst="synthesize_knowledge", required=False),
    Edge(src="analyze_transcripts_notes", dst="synthesize_knowledge", required=False),
    Edge(src="synthesize_knowledge", dst="build_agent_pages", required=True),
    Edge(src="synthesize_knowledge", dst="build_human_spec", required=True),
    Edge(src="synthesize_knowledge", dst="build_vectorstore", required=True),
    Edge(src="build_agent_pages", dst="verify_profile", required=True),
    Edge(src="build_human_spec", dst="verify_profile", required=False),
    Edge(src="build_vectorstore", dst="verify_profile", required=False),
]


def assert_dag(phases: list[Phase], edges: list[Edge]) -> None:
    """Validate the fixed graph at startup (§3, §7.3).

    Raises ``ValueError`` on a duplicate phase name, an edge whose source or
    target is not a declared phase (dangling endpoint), or a cycle (detected
    via Kahn's algorithm). Returns ``None`` when the graph is a valid DAG.
    """
    names: set[str] = set()
    for phase in phases:
        if phase.name in names:
            raise ValueError(f"duplicate phase name: {phase.name!r}")
        names.add(phase.name)

    indegree: dict[str, int] = {name: 0 for name in names}
    adjacency: dict[str, list[str]] = {name: [] for name in names}
    for edge in edges:
        if edge.src not in names:
            raise ValueError(
                f"edge {edge.src!r}->{edge.dst!r} references"
                f" unknown phase: {edge.src!r}"
            )
        if edge.dst not in names:
            raise ValueError(
                f"edge {edge.src!r}->{edge.dst!r} references"
                f" unknown phase: {edge.dst!r}"
            )
        adjacency[edge.src].append(edge.dst)
        indegree[edge.dst] += 1

    # Kahn's algorithm: repeatedly remove zero-indegree nodes.
    queue: list[str] = [name for name in names if indegree[name] == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for successor in adjacency[node]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                queue.append(successor)

    if visited != len(names):
        remaining = sorted(name for name in names if indegree[name] > 0)
        raise ValueError(f"graph contains a cycle among phases: {remaining}")
