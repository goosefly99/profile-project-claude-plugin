from __future__ import annotations

import json
from pathlib import Path

import pytest

from profile_project.artifacts.paths import artifact_path
from profile_project.artifacts.store import (
    list_artifact_refs,
    load_artifact,
    store_artifact,
    validate_artifact,
)


def _source_index_content() -> dict[str, object]:
    return {
        "artifact_type": "source-index",
        "schema_version": 1,
        "run_id": "r1",
        "project_root": "/abs/path",
        "sources": [],
        "counts": {"code": 0},
        "excluded_dirs": [],
        "gitignore_applied": True,
    }


def test_validate_artifact_returns_empty_list_when_valid() -> None:
    errors = validate_artifact("source-index", _source_index_content())
    assert errors == []


def test_validate_artifact_reports_unknown_type() -> None:
    errors = validate_artifact("nope", {})
    assert len(errors) == 1
    assert "unknown artifact type" in errors[0]


def test_validate_artifact_reports_schema_errors() -> None:
    bad = _source_index_content()
    del bad["project_root"]
    errors = validate_artifact("source-index", bad)
    assert errors
    assert any("project_root" in e for e in errors)


def test_store_artifact_validates_writes_and_returns_ref(tmp_path: Path) -> None:
    ref = store_artifact(
        tmp_path,
        run_id="r1",
        phase="discover_context",
        artifact_type="source-index",
        content=_source_index_content(),
    )
    assert ref.type == "source-index"
    assert ref.phase == "discover_context"
    assert ref.version == 1
    assert ref.parent_artifact is None
    # path is project-root-relative POSIX (§7.5)
    assert ref.path == ".profile_project/artifacts/source-index.json"
    # the file actually landed on disk with the normalized JSON
    written = artifact_path(tmp_path, "source-index")
    assert written.exists()
    on_disk = json.loads(written.read_text(encoding="utf-8"))
    assert on_disk["artifact_type"] == "source-index"
    assert on_disk["run_id"] == "r1"


def test_store_artifact_writes_nothing_when_validation_fails(
    tmp_path: Path,
) -> None:
    bad = _source_index_content()
    del bad["project_root"]
    with pytest.raises(ValueError, match="project_root"):
        store_artifact(
            tmp_path,
            run_id="r1",
            phase="discover_context",
            artifact_type="source-index",
            content=bad,
        )
    # validate-before-write: no artifact file or tree residue from the failed call
    artifact_file = (
        tmp_path / ".profile_project" / "artifacts" / "source-index.json"
    )
    assert not artifact_file.exists()


def test_load_artifact_round_trips_the_dict(tmp_path: Path) -> None:
    store_artifact(
        tmp_path,
        run_id="r1",
        phase="discover_context",
        artifact_type="source-index",
        content=_source_index_content(),
    )
    loaded = load_artifact(tmp_path, "source-index")
    assert loaded is not None
    assert loaded["artifact_type"] == "source-index"
    assert loaded["run_id"] == "r1"


def test_load_artifact_returns_none_when_absent(tmp_path: Path) -> None:
    assert load_artifact(tmp_path, "source-index") is None


def test_list_artifact_refs_empty_when_dir_absent(tmp_path: Path) -> None:
    assert list_artifact_refs(tmp_path) == []


def test_load_artifact_with_run_id_reads_flat_store(tmp_path: Path) -> None:
    # store_artifact writes flat regardless of run_id; load_artifact with a
    # run_id must still find it (artifacts are flat; run_id is provenance only).
    store_artifact(
        tmp_path, run_id="r1", phase="discover_context",
        artifact_type="source-index", content=_source_index_content(),
    )
    loaded = load_artifact(tmp_path, "source-index", run_id="r1")
    assert loaded is not None
    assert loaded["artifact_type"] == "source-index"
    refs = list_artifact_refs(tmp_path, run_id="r1")
    assert any(r.type == "source-index" for r in refs)


def test_list_artifact_refs_returns_one_ref_per_present_artifact_sorted(
    tmp_path: Path,
) -> None:
    store_artifact(
        tmp_path,
        run_id="r1",
        phase="discover_context",
        artifact_type="source-index",
        content=_source_index_content(),
    )
    store_artifact(
        tmp_path,
        run_id="r1",
        phase="build_human_spec",
        artifact_type="human-spec",
        content={
            "artifact_type": "human-spec",
            "schema_version": 1,
            "run_id": "r1",
            "output_dir": "profile/guide",
            "sections": [],
            "section_count": 0,
        },
    )
    refs = list_artifact_refs(tmp_path)
    assert [r.type for r in refs] == ["human-spec", "source-index"]
    assert all(r.created_at for r in refs)
