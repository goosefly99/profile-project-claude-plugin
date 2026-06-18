from __future__ import annotations

import json
from pathlib import Path

from profile_project.config.settings import CONFIG_FILENAME, Settings, SourcesSettings
from profile_project.sources.manifest import add_manifest_source, read_manifest


def _settings(**source_kw: object) -> Settings:
    return Settings(sources=SourcesSettings(**source_kw))  # type: ignore[arg-type]


def test_manifest_resolves_globs_with_kind_hints(tmp_path: Path) -> None:
    (tmp_path / "meetings").mkdir()
    (tmp_path / "meetings" / "q2.txt").write_text("a: hi\n", encoding="utf-8")
    (tmp_path / "rfcs").mkdir()
    (tmp_path / "rfcs" / "001.md").write_text("# rfc\n", encoding="utf-8")
    notes_content = b"note\n"
    (tmp_path / "notes.md").write_bytes(notes_content)
    settings = _settings(
        extra_doc_globs=["rfcs/**/*.md"],
        transcripts=["meetings/*.txt"],
        notes=["notes.md"],
        external=["https://example.com/adr"],
    )
    sources = read_manifest(tmp_path, settings)
    by_path = {s.path_or_url: s for s in sources}
    assert by_path["meetings/q2.txt"].kind == "transcript"
    assert by_path["rfcs/001.md"].kind == "doc"
    assert by_path["notes.md"].kind == "note"
    assert by_path["https://example.com/adr"].kind == "external"
    for s in sources:
        assert s.discovered_by == "manifest"
    # external has no on-disk bytes
    assert by_path["https://example.com/adr"].bytes == 0
    # local manifest entries carry real sizes
    assert by_path["notes.md"].bytes == len(notes_content)


def test_manifest_empty_blocks_yield_nothing(tmp_path: Path) -> None:
    assert read_manifest(tmp_path, _settings()) == []


def test_manifest_unmatched_glob_yields_nothing(tmp_path: Path) -> None:
    settings = _settings(transcripts=["meetings/*.txt"])
    assert read_manifest(tmp_path, settings) == []


def test_add_manifest_source_persists_to_config(tmp_path: Path) -> None:
    config_path = tmp_path / CONFIG_FILENAME
    config_path.write_text(
        json.dumps({"sources": {"transcripts": []}}), encoding="utf-8"
    )
    new = add_manifest_source(tmp_path, "meetings/q3.txt", kind="transcript")
    assert new["path_or_url"] == "meetings/q3.txt"
    assert new["kind"] == "transcript"
    assert new["discovered_by"] == "manifest"
    written = json.loads(config_path.read_text(encoding="utf-8"))
    assert written["sources"]["transcripts"] == ["meetings/q3.txt"]


def test_add_manifest_source_external_routes_to_external_list(tmp_path: Path) -> None:
    config_path = tmp_path / CONFIG_FILENAME
    config_path.write_text(json.dumps({"sources": {}}), encoding="utf-8")
    new = add_manifest_source(tmp_path, "https://example.com/adr")
    assert new["kind"] == "external"
    assert new["bytes"] == 0
    written = json.loads(config_path.read_text(encoding="utf-8"))
    assert written["sources"]["external"] == ["https://example.com/adr"]
