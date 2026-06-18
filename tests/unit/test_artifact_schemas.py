from __future__ import annotations

import pytest
from pydantic import ValidationError

from profile_project.artifacts.schemas import (
    ARTIFACT_MODELS,
    ARTIFACT_TYPES,
    AgentPages,
    CodebaseAnalysis,
    ContextAnalysis,
    DocsAnalysis,
    HumanSpec,
    KnowledgeGraph,
    SourceIndex,
    VectorstoreIndex,
    VerificationReport,
)


def test_source_index_accepts_the_spec_shape() -> None:
    si = SourceIndex.model_validate(
        {
            "artifact_type": "source-index",
            "schema_version": 1,
            "run_id": "r1",
            "project_root": "/abs/path",
            "sources": [
                {
                    "source_id": "sha1abc",
                    "kind": "code",
                    "path_or_url": "src/app/main.py",
                    "bytes": 12345,
                    "language": "python",
                    "discovered_by": "auto",
                    "excluded": False,
                }
            ],
            "counts": {
                "code": 312,
                "doc": 18,
                "transcript": 4,
                "note": 2,
                "external": 1,
            },
            "excluded_dirs": ["node_modules", ".venv"],
            "gitignore_applied": True,
        }
    )
    assert si.artifact_type == "source-index"
    assert si.sources[0].kind == "code"
    assert si.counts["code"] == 312


def test_source_index_rejects_unknown_top_level_key() -> None:
    with pytest.raises(ValidationError):
        SourceIndex.model_validate(
            {
                "artifact_type": "source-index",
                "schema_version": 1,
                "run_id": "r1",
                "project_root": "/abs/path",
                "sources": [],
                "counts": {},
                "excluded_dirs": [],
                "gitignore_applied": True,
                "surprise": "boom",
            }
        )


def test_source_index_rejects_bad_kind_literal() -> None:
    with pytest.raises(ValidationError):
        SourceIndex.model_validate(
            {
                "artifact_type": "source-index",
                "schema_version": 1,
                "run_id": "r1",
                "project_root": "/abs/path",
                "sources": [
                    {
                        "source_id": "x",
                        "kind": "binary",
                        "path_or_url": "x",
                        "bytes": 1,
                        "language": "?",
                        "discovered_by": "auto",
                        "excluded": False,
                    }
                ],
                "counts": {},
                "excluded_dirs": [],
                "gitignore_applied": True,
            }
        )


def test_codebase_analysis_accepts_the_spec_shape() -> None:
    ca = CodebaseAnalysis.model_validate(
        {
            "artifact_type": "codebase-analysis",
            "schema_version": 1,
            "run_id": "r1",
            "modules": [
                {
                    "name": "config",
                    "path": "src/profile_project/config",
                    "responsibility": "layered settings",
                    "public_symbols": ["Settings"],
                    "depends_on": ["pydantic"],
                }
            ],
            "components": [
                {
                    "name": "server",
                    "kind": "service",
                    "files": ["server.py"],
                    "summary": "fastmcp",
                }
            ],
            "dependencies": {
                "internal": [["dag", "artifacts"]],
                "external": ["pydantic", "chromadb"],
            },
            "entry_points": ["src/profile_project/__main__.py"],
            "hotspots": [{"path": "config/settings.py", "reason": "high fan-in"}],
            "notes": "free-form",
        }
    )
    assert ca.modules[0].public_symbols == ["Settings"]
    assert ca.dependencies.internal == [["dag", "artifacts"]]


def test_source_index_accepts_null_language() -> None:
    si = SourceIndex.model_validate(
        {
            "artifact_type": "source-index",
            "schema_version": 1,
            "run_id": "r1",
            "project_root": "/abs",
            "sources": [
                {
                    "source_id": "x",
                    "kind": "external",
                    "path_or_url": "https://e.com",
                    "bytes": 0,
                    "language": None,
                    "discovered_by": "manifest",
                    "excluded": False,
                }
            ],
            "counts": {},
            "excluded_dirs": [],
            "gitignore_applied": True,
        }
    )
    assert si.sources[0].language is None


def test_docs_analysis_shape() -> None:
    da = DocsAnalysis.model_validate(
        {
            "artifact_type": "docs-analysis",
            "schema_version": 1,
            "run_id": "r1",
            "documents": [
                {
                    "path": "README.md",
                    "title": "Readme",
                    "topics": ["install"],
                    "summary": "...",
                    "claims": ["claim"],
                    "links_out": ["http://x"],
                }
            ],
            "coverage_gaps": ["no deployment doc"],
            "notes": "...",
        }
    )
    assert da.documents[0].path == "README.md"


def test_context_analysis_shape() -> None:
    ca = ContextAnalysis.model_validate(
        {
            "artifact_type": "context-analysis",
            "schema_version": 1,
            "run_id": "r1",
            "items": [
                {
                    "source_id": "s1",
                    "kind": "transcript",
                    "speaker": "alice",
                    "timecode": "00:01",
                    "decisions": ["d"],
                    "requirements": ["req"],
                    "open_questions": ["q"],
                    "summary": "...",
                }
            ],
            "notes": "...",
        }
    )
    assert ca.items[0].kind == "transcript"


def test_knowledge_graph_shape() -> None:
    kg = KnowledgeGraph.model_validate(
        {
            "artifact_type": "knowledge-graph",
            "schema_version": 1,
            "run_id": "r1",
            "entities": [
                {
                    "id": "ent:config",
                    "name": "Config",
                    "type": "subsystem",
                    "summary": "...",
                    "evidence": ["src"],
                }
            ],
            "concepts": [
                {
                    "id": "con:lc",
                    "name": "Layered config",
                    "definition": "...",
                    "related": ["ent:config"],
                }
            ],
            "cross_links": [
                {
                    "from": "ent:dag",
                    "to": "ent:artifacts",
                    "relation": "produces",
                    "evidence": [],
                }
            ],
            "glossary": [
                {
                    "term": "input_mode",
                    "definition": "...",
                    "see_also": ["resolve_next_phases"],
                }
            ],
            "key_decisions": [
                {"decision": "d", "rationale": "r", "source": "context-analysis#3"}
            ],
            "coverage": {"code_modules_covered": 0.92, "docs_covered": 1.0},
        }
    )
    assert kg.cross_links[0].from_ == "ent:dag"
    assert kg.coverage["docs_covered"] == 1.0


def test_agent_pages_shape() -> None:
    ap = AgentPages.model_validate(
        {
            "artifact_type": "agent-pages",
            "schema_version": 1,
            "run_id": "r1",
            "output_dir": "profile/context",
            "pages": [
                {
                    "id": "overview",
                    "title": "Overview",
                    "path": "profile/context/overview.md",
                    "page_type": "overview",
                    "entity_refs": ["ent:config"],
                    "indexed": True,
                    "token_estimate": 1800,
                }
            ],
            "page_count": 7,
        }
    )
    assert ap.pages[0].page_type == "overview"


def test_human_spec_shape() -> None:
    hs = HumanSpec.model_validate(
        {
            "artifact_type": "human-spec",
            "schema_version": 1,
            "run_id": "r1",
            "output_dir": "profile/guide",
            "sections": [
                {
                    "id": "system-overview",
                    "title": "System Overview",
                    "path": "profile/guide/01-system-overview.md",
                    "has_mermaid": True,
                }
            ],
            "section_count": 6,
        }
    )
    assert hs.sections[0].has_mermaid is True


def test_vectorstore_index_shape_uses_canonical_embedder_version() -> None:
    vi = VectorstoreIndex.model_validate(
        {
            "artifact_type": "vectorstore-index",
            "schema_version": 1,
            "run_id": "r1",
            "backend": "chromadb",
            "collection": "profile-project",
            "namespace": None,
            "embedder_version": "sentence-transformers/all-MiniLM-L6-v2@hf-fp32",
            "dimension": 384,
            "chunk_count": 1240,
            "source_types_indexed": ["agent-page", "code", "doc"],
            "chunk_config": {
                "chunk_size": 512,
                "chunk_overlap": 64,
                "token_encoding": "cl100k_base",
            },
        }
    )
    assert vi.embedder_version == "sentence-transformers/all-MiniLM-L6-v2@hf-fp32"
    assert vi.chunk_config.chunk_size == 512


def test_verification_report_shape() -> None:
    vr = VerificationReport.model_validate(
        {
            "artifact_type": "verification-report",
            "schema_version": 1,
            "run_id": "r1",
            "coverage": {
                "modules": 0.92,
                "docs": 1.0,
                "decisions_traced": 0.88,
                "pass": True,
            },
            "broken_links": [],
            "query_smoke_test": {
                "ran": True,
                "query": "how is config resolved?",
                "top_hit": "profile/context/architecture.md",
                "score": 0.81,
                "pass": True,
            },
            "warnings": [],
            "overall_pass": True,
        }
    )
    assert vr.overall_pass is True
    assert vr.query_smoke_test.ran is True


def test_artifact_models_registry_covers_all_nine_types() -> None:
    assert set(ARTIFACT_MODELS) == {
        "source-index",
        "codebase-analysis",
        "docs-analysis",
        "context-analysis",
        "knowledge-graph",
        "agent-pages",
        "human-spec",
        "vectorstore-index",
        "verification-report",
    }
    assert ARTIFACT_TYPES == frozenset(ARTIFACT_MODELS)


def test_artifact_models_keys_match_each_models_artifact_type_default() -> None:
    for type_str, model in ARTIFACT_MODELS.items():
        default = model.model_fields["artifact_type"].default
        assert default == type_str
