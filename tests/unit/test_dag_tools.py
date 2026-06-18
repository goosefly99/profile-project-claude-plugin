from __future__ import annotations

import json
from pathlib import Path

import pytest

from profile_project.tools import dag_tools


@pytest.fixture()
def uninit_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / ".profile_project_config.json").write_text(
        json.dumps(
            {
                "vectorstore": {"enabled": True, "backend": "chromadb"},
                "embeddings": {"method": "sentence-transformers"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PROFILE_PROJECT_PROJECT_DIR", str(tmp_path))
    monkeypatch.delenv("PROFILE_PROJECT_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PROFILE_PROJECT_PINECONE_API_KEY", raising=False)
    return tmp_path


def test_pp_run_status_pre_init_returns_none(uninit_project: Path) -> None:
    result = dag_tools.pp_run_status("nope")
    assert result == {"ok": True, "run_state": None}
    assert not (uninit_project / ".profile_project").exists()


def test_pp_list_runs_pre_init_returns_empty(uninit_project: Path) -> None:
    result = dag_tools.pp_list_runs()
    assert result == {"ok": True, "runs": []}
    assert not (uninit_project / ".profile_project").exists()


from profile_project.tools import config_tools


def _init(project: Path) -> None:
    config_tools.pp_init_project(
        {
            "vectorstore": {"enabled": True, "backend": "chromadb"},
            "embeddings": {"method": "sentence-transformers"},
        }
    )


def test_pp_init_run_gated_pre_init(uninit_project: Path) -> None:
    result = dag_tools.pp_init_run({})
    assert result["ok"] is False
    assert result["error"]["code"] == "not_initialized"
    assert not (uninit_project / ".profile_project" / "runs").exists()


def test_pp_init_run_creates_run_with_pending_phases(uninit_project: Path) -> None:
    _init(uninit_project)
    result = dag_tools.pp_init_run(
        {"include_docs": True, "include_transcripts": True, "build_vectorstore": True}
    )
    assert result["ok"] is True
    run_id = result["run_id"]
    assert isinstance(run_id, str) and run_id
    state = result["run_state"]
    assert state["status"] == "initialized"
    assert state["phases"]["discover_context"]["status"] == "pending"
    assert (
        uninit_project / ".profile_project" / "runs" / run_id / "run-state.json"
    ).exists()


def test_pp_init_run_toggle_skips_vectorstore(uninit_project: Path) -> None:
    _init(uninit_project)
    result = dag_tools.pp_init_run({"build_vectorstore": False})
    state = result["run_state"]
    assert state["phases"]["build_vectorstore"]["status"] == "skipped"


def test_pp_next_phases_offers_entry_then_run_not_found(uninit_project: Path) -> None:
    _init(uninit_project)
    run_id = dag_tools.pp_init_run({})["run_id"]
    result = dag_tools.pp_next_phases(run_id)
    assert result["ok"] is True
    assert result["next_phases"] == ["discover_context"]
    assert result["recommended"] == "discover_context"

    missing = dag_tools.pp_next_phases("does-not-exist")
    assert missing["ok"] is False
    assert missing["error"]["code"] == "run_not_found"


def test_pp_start_phase_returns_brief_and_marks_in_progress(uninit_project: Path) -> None:
    _init(uninit_project)
    run_id = dag_tools.pp_init_run({})["run_id"]
    brief = dag_tools.pp_start_phase(run_id, "discover_context")
    assert brief["ok"] is True
    assert brief["phase"] == "discover_context"
    # deterministic phase carries no sub-agent directive (§7.7).
    assert brief["agent_directive"] is None
    status = dag_tools.pp_run_status(run_id)["run_state"]
    assert status["phases"]["discover_context"]["status"] == "in_progress"
    assert status["status"] == "running"


def test_pp_complete_then_fail_then_retry(uninit_project: Path) -> None:
    _init(uninit_project)
    run_id = dag_tools.pp_init_run({})["run_id"]
    dag_tools.pp_start_phase(run_id, "discover_context")
    done = dag_tools.pp_complete_phase(run_id, "discover_context")
    assert done["ok"] is True
    assert (
        done["run_state"]["phases"]["discover_context"]["status"] == "completed"
    )

    dag_tools.pp_start_phase(run_id, "analyze_codebase")
    failed = dag_tools.pp_fail_phase(run_id, "analyze_codebase", "boom")
    assert failed["run_state"]["phases"]["analyze_codebase"]["status"] == "failed"
    assert failed["run_state"]["status"] == "failed"

    retried = dag_tools.pp_retry_phase(run_id, "analyze_codebase")
    assert retried["ok"] is True
    assert (
        retried["run_state"]["phases"]["analyze_codebase"]["status"] == "pending"
    )


def test_pp_retry_phase_on_non_failed_errors(uninit_project: Path) -> None:
    _init(uninit_project)
    run_id = dag_tools.pp_init_run({})["run_id"]
    result = dag_tools.pp_retry_phase(run_id, "discover_context")
    assert result["ok"] is False
    assert result["error"]["code"] == "state_transition_error"
