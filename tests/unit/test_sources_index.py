from __future__ import annotations

from pathlib import Path

import pytest

from profile_project.config.settings import Settings, SourcesSettings
from profile_project.sources.discover import RawSource
from profile_project.sources.index import build_source_index


def _settings(**source_kw: list[str]) -> Settings:
    return Settings(sources=SourcesSettings(**source_kw))


def _make_repo(root: Path) -> None:
    (root / "src" / "app").mkdir(parents=True)
    (root / "src" / "app" / "main.py").write_text("print('hi')\n", encoding="utf-8")
    (root / "src" / "app" / "util.py").write_text("x = 1\n", encoding="utf-8")
    (root / "README.md").write_text("# Title\n", encoding="utf-8")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "dep.js").write_text("var a=1;\n", encoding="utf-8")


def test_rawsource_is_frozen() -> None:
    s = RawSource(
        source_id="abc",
        kind="code",
        path_or_url="src/app/main.py",
        bytes=12,
        language="python",
        discovered_by="auto",
        excluded=False,
    )
    with pytest.raises((AttributeError, TypeError, ValueError)):
        s.bytes = 99


def test_index_shape_and_counts(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    index = build_source_index(tmp_path, _settings())
    assert index["artifact_type"] == "source-index"
    assert index["schema_version"] == 1
    assert index["run_id"] is None
    assert index["project_root"] == str(tmp_path)
    assert index["gitignore_applied"] is True
    excluded = index["excluded_dirs"]
    assert isinstance(excluded, list)
    assert "node_modules" in excluded
    sources = index["sources"]
    assert isinstance(sources, list)
    paths = {s["path_or_url"] for s in sources if isinstance(s, dict)}
    assert "src/app/main.py" in paths
    assert "src/app/util.py" in paths
    assert "README.md" in paths
    # node_modules is an excluded dir -> not in the index
    assert "node_modules/dep.js" not in paths
    counts = index["counts"]
    assert isinstance(counts, dict)
    assert counts["code"] == 2
    assert counts["doc"] == 1
    assert counts == {
        "code": 2,
        "doc": 1,
        "transcript": 0,
        "note": 0,
        "external": 0,
    }


def test_index_every_source_has_required_fields(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    index = build_source_index(tmp_path, _settings())
    sources = index["sources"]
    assert isinstance(sources, list)
    for src in sources:
        assert isinstance(src, dict)
        assert set(src) == {
            "source_id",
            "kind",
            "path_or_url",
            "bytes",
            "language",
            "discovered_by",
            "excluded",
        }
        assert src["kind"] in {"code", "doc", "transcript", "note", "external"}
        assert src["discovered_by"] in {"auto", "manifest"}
        assert src["excluded"] is False
        assert isinstance(src["bytes"], int)


def test_index_source_id_is_stable_sha1_of_path(tmp_path: Path) -> None:
    import hashlib

    _make_repo(tmp_path)
    index = build_source_index(tmp_path, _settings())
    sources = index["sources"]
    assert isinstance(sources, list)
    by_path = {s["path_or_url"]: s for s in sources if isinstance(s, dict)}
    expected = hashlib.sha1(b"src/app/main.py").hexdigest()
    entry = by_path["src/app/main.py"]
    assert isinstance(entry, dict)
    assert entry["source_id"] == expected


def test_index_manifest_merge_and_dedup(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    (tmp_path / "meetings").mkdir()
    (tmp_path / "meetings" / "kickoff.txt").write_text(
        "alice: hello\n", encoding="utf-8"
    )
    settings = _settings(
        transcripts=["meetings/*.txt"],
        external=["https://example.com/adr"],
        notes=[],
        extra_doc_globs=[],
    )
    index = build_source_index(tmp_path, settings)
    sources = index["sources"]
    assert isinstance(sources, list)
    by_path = {s["path_or_url"]: s for s in sources if isinstance(s, dict)}
    kickoff = by_path["meetings/kickoff.txt"]
    assert isinstance(kickoff, dict)
    assert kickoff["kind"] == "transcript"
    assert kickoff["discovered_by"] == "manifest"
    adr = by_path["https://example.com/adr"]
    assert isinstance(adr, dict)
    assert adr["kind"] == "external"
    assert adr["bytes"] == 0
    counts = index["counts"]
    assert isinstance(counts, dict)
    assert counts["transcript"] == 1
    assert counts["external"] == 1


def test_index_manifest_hint_wins_over_auto_on_dedup(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    # README.md is auto-discovered as a doc; also list it under notes in the manifest.
    settings = _settings(notes=["README.md"])
    index = build_source_index(tmp_path, settings)
    sources = index["sources"]
    assert isinstance(sources, list)
    matches = [
        s
        for s in sources
        if isinstance(s, dict) and s["path_or_url"] == "README.md"
    ]
    assert len(matches) == 1  # deduped
    assert matches[0]["kind"] == "note"  # manifest hint wins
    assert matches[0]["discovered_by"] == "manifest"
