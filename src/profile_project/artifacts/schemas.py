from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SourceKind = Literal["code", "doc", "transcript", "note", "external"]


class _Artifact(BaseModel):
    """Base for every artifact: forbids unknown keys; stamps schema_version."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    run_id: str


# --- §8.1 source-index ---
class SourceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    kind: SourceKind
    path_or_url: str
    bytes: int
    # Correction A: nullable for external/unknown-extension sources
    language: str | None
    discovered_by: Literal["auto", "manifest"]
    excluded: bool = False


class SourceIndex(_Artifact):
    artifact_type: Literal["source-index"] = "source-index"
    project_root: str
    sources: list[SourceEntry]
    counts: dict[str, int]
    excluded_dirs: list[str]
    gitignore_applied: bool


# --- §8.2 codebase-analysis ---
class ModuleEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    path: str
    responsibility: str
    public_symbols: list[str]
    depends_on: list[str]


class ComponentEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: str
    files: list[str]
    summary: str


class Dependencies(BaseModel):
    model_config = ConfigDict(extra="forbid")

    internal: list[list[str]]
    external: list[str]


class Hotspot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    reason: str


class CodebaseAnalysis(_Artifact):
    artifact_type: Literal["codebase-analysis"] = "codebase-analysis"
    modules: list[ModuleEntry]
    components: list[ComponentEntry]
    dependencies: Dependencies
    entry_points: list[str]
    hotspots: list[Hotspot]
    notes: str = ""


# --- §8.3 docs-analysis ---
class DocumentEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    title: str
    topics: list[str]
    summary: str
    claims: list[str]
    links_out: list[str]


class DocsAnalysis(_Artifact):
    artifact_type: Literal["docs-analysis"] = "docs-analysis"
    documents: list[DocumentEntry]
    coverage_gaps: list[str]
    notes: str = ""


# --- §8.4 context-analysis ---
class ContextItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    kind: Literal["transcript", "note"]
    speaker: str | None = None
    timecode: str | None = None
    decisions: list[str]
    requirements: list[str]
    open_questions: list[str]
    summary: str


class ContextAnalysis(_Artifact):
    artifact_type: Literal["context-analysis"] = "context-analysis"
    items: list[ContextItem]
    notes: str = ""


# --- §8.5 knowledge-graph ---
class Entity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    type: str
    summary: str
    evidence: list[str]


class Concept(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    definition: str
    related: list[str]


class CrossLink(BaseModel):
    # "from" is a reserved word in the JSON; expose it via the from_ alias.
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_: str = Field(alias="from")
    to: str
    relation: str
    evidence: list[str]


class GlossaryTerm(BaseModel):
    model_config = ConfigDict(extra="forbid")

    term: str
    definition: str
    see_also: list[str]


class KeyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: str
    rationale: str
    source: str


class KnowledgeGraph(_Artifact):
    artifact_type: Literal["knowledge-graph"] = "knowledge-graph"
    entities: list[Entity]
    concepts: list[Concept]
    cross_links: list[CrossLink]
    glossary: list[GlossaryTerm]
    key_decisions: list[KeyDecision]
    coverage: dict[str, float]


# --- §8.6 agent-pages ---
class AgentPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    path: str
    page_type: Literal[
        "overview",
        "architecture",
        "module-map",
        "data-flows",
        "glossary",
        "key-decisions",
        "onboarding",
    ]
    entity_refs: list[str]
    indexed: bool
    token_estimate: int


class AgentPages(_Artifact):
    artifact_type: Literal["agent-pages"] = "agent-pages"
    output_dir: str
    pages: list[AgentPage]
    page_count: int


# --- §8.7 human-spec ---
class GuideSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    path: str
    has_mermaid: bool


class HumanSpec(_Artifact):
    artifact_type: Literal["human-spec"] = "human-spec"
    output_dir: str
    sections: list[GuideSection]
    section_count: int


# --- §8.8 vectorstore-index ---
class ChunkConfigMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_size: int
    chunk_overlap: int
    token_encoding: str


class VectorstoreIndex(_Artifact):
    artifact_type: Literal["vectorstore-index"] = "vectorstore-index"
    backend: Literal["chromadb", "pinecone"]
    collection: str
    namespace: str | None = None
    embedder_version: str
    dimension: int
    chunk_count: int
    source_types_indexed: list[str]
    chunk_config: ChunkConfigMeta


# --- §8.9 verification-report ---
class CoverageReport(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    modules: float
    docs: float
    decisions_traced: float
    pass_: bool = Field(alias="pass")


class QuerySmokeTest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    ran: bool
    query: str
    top_hit: str | None = None
    score: float | None = None
    pass_: bool = Field(alias="pass")


class VerificationReport(_Artifact):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    artifact_type: Literal["verification-report"] = "verification-report"
    coverage: CoverageReport
    broken_links: list[str]
    query_smoke_test: QuerySmokeTest
    warnings: list[str]
    overall_pass: bool


ARTIFACT_MODELS: dict[str, type[BaseModel]] = {
    "source-index": SourceIndex,
    "codebase-analysis": CodebaseAnalysis,
    "docs-analysis": DocsAnalysis,
    "context-analysis": ContextAnalysis,
    "knowledge-graph": KnowledgeGraph,
    "agent-pages": AgentPages,
    "human-spec": HumanSpec,
    "vectorstore-index": VectorstoreIndex,
    "verification-report": VerificationReport,
}

ARTIFACT_TYPES: frozenset[str] = frozenset(ARTIFACT_MODELS)
