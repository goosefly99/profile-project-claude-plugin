from __future__ import annotations

import pytest

from profile_project.sources.classify import classify


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("src/app/main.py", "code"),
        ("pkg/mod.ts", "code"),
        ("lib/Widget.java", "code"),
        ("cmd/run.go", "code"),
        ("README.md", "doc"),
        ("docs/guide.rst", "doc"),
        ("notes/scratch.txt", "doc"),
    ],
)
def test_classify_by_extension(path: str, expected: str) -> None:
    assert classify(path) == expected


def test_classify_external_url() -> None:
    assert classify("https://example.com/adr") == "external"
    assert classify("http://example.com/x") == "external"


def test_manifest_hint_overrides_extension() -> None:
    # A .txt transcript would heuristically be a doc, but the manifest hint pins it.
    assert classify("meetings/kickoff.txt", hint="transcript") == "transcript"


def test_manifest_note_hint_overrides_markdown() -> None:
    assert classify("notes/design.md", hint="note") == "note"


def test_external_hint_for_local_looking_path() -> None:
    assert classify("https://example.com/adr", hint="external") == "external"
