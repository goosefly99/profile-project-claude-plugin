from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from profile_project.artifacts.paths import profile_dirs
from profile_project.config.sources import load_settings
from profile_project.tools.artifact_tools import pp_load_artifact, pp_store_artifact
from profile_project.tools.config_tools import pp_config_validate, pp_init_project
from profile_project.tools.dag_tools import (
    pp_complete_phase,
    pp_init_run,
    pp_next_phases,
    pp_run_status,
    pp_start_phase,
)
from profile_project.tools.sources_tools import pp_discover_sources
from profile_project.tools.vectorstore_tools import (
    pp_index_build,
    pp_index_status,
    pp_query,
)
from profile_project.vectorstore.ids import DEFAULT_ST_EMBEDDER_VERSION

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_project"

EXPECTED_PAGES: frozenset[str] = frozenset(
    {
        "overview.md",
        "architecture.md",
        "module-map.md",
        "data-flows.md",
        "glossary.md",
        "key-decisions.md",
        "onboarding.md",
    }
)
EXPECTED_SECTIONS: frozenset[str] = frozenset(
    {
        "01-system-overview.md",
        "02-components.md",
        "03-setup-onboarding.md",
        "04-flows.md",
        "05-decisions.md",
        "06-open-questions.md",
    }
)

KNOWN_FACT = "Widget retries failed writes up to three times using exponential backoff."

# Maps each §9.1 page filename to its schema-valid AgentPage.page_type literal.
_PAGE_TYPES: dict[str, str] = {
    "overview.md": "overview",
    "architecture.md": "architecture",
    "module-map.md": "module-map",
    "data-flows.md": "data-flows",
    "glossary.md": "glossary",
    "key-decisions.md": "key-decisions",
    "onboarding.md": "onboarding",
}


def _copy_fixture(tmp_path: Path) -> Path:
    """Copy the read-only tiny_project fixture into a writable temp root."""
    project_root = tmp_path / "tiny_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    return project_root


def _write_deliverables(project_root: Path) -> None:
    """Write the §9.1 agent pages and §9.2 guide sections to profile/."""
    settings = load_settings(project_root)
    context_dir, guide_dir = profile_dirs(settings, project_root)
    context_dir.mkdir(parents=True, exist_ok=True)
    guide_dir.mkdir(parents=True, exist_ok=True)
    for page in EXPECTED_PAGES:
        body = (
            f"# {page}\n\n{KNOWN_FACT}\n"
            if page == "architecture.md"
            else f"# {page}\n\nDense agent-facing content for tiny_project.\n"
        )
        (context_dir / page).write_text(body, encoding="utf-8")
    for section in EXPECTED_SECTIONS:
        (guide_dir / section).write_text(
            f"# {section}\n\nNarrative human-facing content for tiny_project.\n",
            encoding="utf-8",
        )


def _artifact_content(phase: str, project_root: Path) -> dict[str, object]:
    """Schema-valid §8 artifact body for each agent-produced phase."""
    bodies: dict[str, dict[str, object]] = {
        "analyze_codebase": {
            "artifact_type": "codebase-analysis",
            "schema_version": 1,
            "run_id": "tiny",
            "modules": [
                {
                    "name": "widget",
                    "path": "src/widget",
                    "responsibility": "core widget logic",
                    "public_symbols": ["Widget", "build_widget"],
                    "depends_on": [],
                }
            ],
            "components": [
                {
                    "name": "Widget",
                    "kind": "model",
                    "files": ["src/widget/core.py"],
                    "summary": "stateful widget with retrying writes",
                }
            ],
            "dependencies": {"internal": [], "external": []},
            "entry_points": ["src/widget/cli.py"],
            "hotspots": [],
            "notes": "tiny fixture",
        },
        "analyze_docs": {
            "artifact_type": "docs-analysis",
            "schema_version": 1,
            "run_id": "tiny",
            "documents": [
                {
                    "path": "docs/architecture.md",
                    "title": "Widget Architecture",
                    "topics": ["write reliability"],
                    "summary": KNOWN_FACT,
                    "claims": [KNOWN_FACT],
                    "links_out": [],
                }
            ],
            "coverage_gaps": [],
            "notes": "",
        },
        "analyze_transcripts_notes": {
            "artifact_type": "context-analysis",
            "schema_version": 1,
            "run_id": "tiny",
            "items": [
                {
                    "source_id": "notes/design-notes.md",
                    "kind": "note",
                    "speaker": None,
                    "timecode": None,
                    "decisions": ["keep public surface to Widget and build_widget"],
                    "requirements": [],
                    "open_questions": ["should backoff be jittered?"],
                    "summary": "design notes",
                }
            ],
            "notes": "",
        },
        "synthesize_knowledge": {
            "artifact_type": "knowledge-graph",
            "schema_version": 1,
            "run_id": "tiny",
            "entities": [
                {
                    "id": "ent:widget",
                    "name": "Widget",
                    "type": "subsystem",
                    "summary": "retrying widget",
                    "evidence": ["src/widget/core.py"],
                }
            ],
            "concepts": [
                {
                    "id": "con:backoff",
                    "name": "Exponential backoff retry",
                    "definition": KNOWN_FACT,
                    "related": ["ent:widget"],
                }
            ],
            "cross_links": [],
            "glossary": [
                {"term": "backoff", "definition": "retry delay growth", "see_also": []}
            ],
            "key_decisions": [
                {
                    "decision": "limit public surface",
                    "rationale": "simplicity",
                    "source": "context-analysis#1",
                }
            ],
            "coverage": {"code_modules_covered": 1.0, "docs_covered": 1.0},
        },
        "build_agent_pages": {
            "artifact_type": "agent-pages",
            "schema_version": 1,
            "run_id": "tiny",
            "output_dir": "profile/context",
            "pages": [
                {
                    "id": Path(name).stem,
                    "title": Path(name).stem.replace("-", " ").title(),
                    "path": f"profile/context/{name}",
                    "page_type": _PAGE_TYPES[name],
                    "entity_refs": ["ent:widget"],
                    "indexed": True,
                    "token_estimate": 64,
                }
                for name in sorted(EXPECTED_PAGES)
            ],
            "page_count": len(EXPECTED_PAGES),
        },
        "build_human_spec": {
            "artifact_type": "human-spec",
            "schema_version": 1,
            "run_id": "tiny",
            "output_dir": "profile/guide",
            "sections": [
                {
                    "id": Path(name).stem,
                    "title": Path(name).stem,
                    "path": f"profile/guide/{name}",
                    "has_mermaid": False,
                }
                for name in sorted(EXPECTED_SECTIONS)
            ],
            "section_count": len(EXPECTED_SECTIONS),
        },
        "verify_profile": {
            "artifact_type": "verification-report",
            "schema_version": 1,
            "run_id": "tiny",
            "coverage": {
                "modules": 1.0,
                "docs": 1.0,
                "decisions_traced": 1.0,
                "pass": True,
            },
            "broken_links": [],
            "query_smoke_test": {
                "ran": True,
                "query": "how does the widget handle failed writes?",
                "top_hit": "profile/context/architecture.md",
                "score": 0.5,
                "pass": True,
            },
            "warnings": [],
            "overall_pass": True,
        },
    }
    return bodies[phase]


def _cost_estimate_line(source_index: dict[str, object]) -> str:
    """The §16 pre-flight cost/latency line derived from the source-index counts."""
    sources = source_index["sources"]
    assert isinstance(sources, list)
    counts = source_index["counts"]
    assert isinstance(counts, dict)
    n_files = sum(int(v) for v in counts.values())
    total_bytes = sum(int(s.get("bytes", 0)) for s in sources if not s.get("excluded"))
    # §16 authoritative formula (matches the skill orchestration-loop pre-flight):
    #   tokens ≈ total_bytes / 4
    #   M (chunks) = ceil(tokens / (CHUNK_SIZE - CHUNK_OVERLAP))  with default 512/64
    #   est_input_tokens = M * CHUNK_SIZE
    chunk_size, chunk_overlap = 512, 64
    n_chunks = -(-(total_bytes // 4) // (chunk_size - chunk_overlap))  # ceil division
    est_tokens = n_chunks * chunk_size
    # OpenAI text-embedding-3-small: $0.02 / 1M input tokens; batch cap 2048/call.
    est_openai_usd = est_tokens / 1_000_000 * 0.02
    return (
        f"this will read ~{n_files} files / embed ~{n_chunks} chunks "
        f"(est. ~{est_tokens} input tokens; for openai est. ~${est_openai_usd:.4f}; "
        f"batch cap 2048 inputs/call)"
    )


DETERMINISTIC = {"discover_context", "build_vectorstore"}


def _drive_to_terminal(run_id: str, project_root: Path) -> dict[str, object]:
    """Run the §7.8 loop to terminal, acting as the agent for reasoning phases."""
    for _ in range(50):  # bounded; the fixed graph has 9 phases
        nxt = pp_next_phases(run_id=run_id)
        assert nxt["ok"] is True, nxt
        runnable = nxt["next_phases"]
        assert isinstance(runnable, list)
        if not runnable:
            break
        for phase in runnable:
            brief = pp_start_phase(run_id=run_id, phase=phase)
            assert brief["ok"] is True, brief
            # C-1 regression guard: a brief's input_artifacts must be repo-relative
            # PATHS sub-agents can open (".profile_project/artifacts/<t>.json"),
            # never artifact TYPE strings ("codebase-analysis").
            inputs = brief["input_artifacts"]
            assert isinstance(inputs, list)
            for art in inputs:
                assert isinstance(art, str)
                assert ".profile_project/artifacts/" in art, art
                assert art.endswith(".json"), art
            if phase == "synthesize_knowledge":
                # An agent phase with multiple upstream inputs: must be non-empty
                # and every entry a path, proving the type->path fix end-to-end.
                assert len(inputs) > 0
            if phase == "discover_context":
                disc = pp_discover_sources(persist=True)
                assert disc["ok"] is True, disc
                # §16 pre-flight cost/latency estimate: produced AFTER discover_context
                # (the source-index counts exist only now) and BEFORE the heavy phases.
                loaded = pp_load_artifact(type="source-index", run_id=run_id)
                assert loaded["ok"] is True, loaded
                source_index = loaded["artifact"]
                assert isinstance(source_index, dict), loaded
                cost_line = _cost_estimate_line(source_index)
                assert cost_line.startswith("this will read ~")
                assert "embed ~" in cost_line and "input tokens" in cost_line
                assert "for openai est. ~$" in cost_line
            elif phase == "build_vectorstore":
                built = pp_index_build(run_id=run_id)
                assert built["ok"] is True, built
            else:
                if phase == "build_agent_pages":
                    _write_deliverables(project_root)  # pages + sections written once
                expected_outputs = brief["expected_outputs"]
                assert isinstance(expected_outputs, list)
                expected_type = str(expected_outputs[0])
                stored = pp_store_artifact(
                    run_id=run_id,
                    phase=phase,
                    type=expected_type,
                    content=_artifact_content(phase, project_root),
                )
                assert stored["ok"] is True, stored
            done = pp_complete_phase(run_id=run_id, phase=phase)
            assert done["ok"] is True, done
    final = pp_run_status(run_id=run_id)
    assert final["ok"] is True, final
    run_state = final["run_state"]
    assert isinstance(run_state, dict), final
    return run_state


@pytest.mark.integration
def test_end_to_end_tiny_init_dag_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = _copy_fixture(tmp_path)
    # Route every tool to the temp fixture root (resolve_project_root precedence).
    monkeypatch.setenv("PROFILE_PROJECT_PROJECT_DIR", str(project_root))
    # Secrets-are-env-only: the zero-setup pair needs no keys; ensure none leak in.
    monkeypatch.delenv("PROFILE_PROJECT_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PROFILE_PROJECT_PINECONE_API_KEY", raising=False)

    # --- diagnose (read-only, pre-init) ---
    diag = pp_config_validate(project_path=str(project_root))
    assert diag["ok"] is True, diag
    assert diag["vectorstore_enabled"] is True
    provenance = diag["provenance"]
    assert isinstance(provenance, dict)
    assert provenance["embeddings.method"] == "project_json"

    # --- init (the only pre-init mutator) ---
    config = (project_root / ".profile_project_config.json").read_text("utf-8")
    init = pp_init_project(config=json.loads(config), force=False)
    assert init["ok"] is True, init
    assert init["initialized"] is True
    assert (project_root / ".profile_project" / ".initialized").exists()
    gitignore = (project_root / ".gitignore").read_text("utf-8")
    assert ".profile_project/" in gitignore

    # --- run the DAG to terminal ---
    started = pp_init_run(
        run_parameters={
            "include_docs": True,
            "include_transcripts": True,
            "build_vectorstore": True,
        }
    )
    assert started["ok"] is True, started
    run_id = started["run_id"]
    assert isinstance(run_id, str)
    run_state = _drive_to_terminal(run_id, project_root)
    assert run_state["status"] == "completed"
    assert pp_next_phases(run_id=run_id)["next_phases"] == []

    # --- assert the full agent-page set (S4) ---
    settings = load_settings(project_root)
    context_dir, guide_dir = profile_dirs(settings, project_root)
    assert {p.name for p in context_dir.glob("*.md")} == EXPECTED_PAGES

    # --- assert the full guide section set (S5) ---
    assert {p.name for p in guide_dir.glob("*.md")} == EXPECTED_SECTIONS

    # --- assert verification passed (S7) ---
    report = pp_load_artifact(run_id=run_id, type="verification-report")
    assert report["ok"] is True, report
    artifact = report["artifact"]
    assert isinstance(artifact, dict), report
    assert artifact["overall_pass"] is True
    coverage = artifact["coverage"]
    assert isinstance(coverage, dict)
    assert coverage["pass"] is True

    # --- assert the vectorstore is populated with the default geometry (S6) ---
    status = pp_index_status()
    assert status["ok"] is True, status
    assert status["status"] != "uninitialized"
    count = status["count"]
    assert isinstance(count, int) and count > 0
    assert status["dimension"] == 384
    assert status["embedder_version"] == DEFAULT_ST_EMBEDDER_VERSION

    # --- assert a working, ranked, attributable pp_query hit (S6) ---
    hits = pp_query(query="how does the widget handle failed writes?", top_k=5)
    assert hits["ok"] is True, hits
    hit_list = hits["hits"]
    assert isinstance(hit_list, list)
    assert len(hit_list) > 0
    top = hit_list[0]
    assert isinstance(top, dict)
    score = top["score"]
    assert isinstance(score, (int, float))
    assert 0.0 <= float(score) <= 1.0
    metadata = top["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["embedder_version"] == DEFAULT_ST_EMBEDDER_VERSION
    assert "architecture" in str(metadata["path"])
