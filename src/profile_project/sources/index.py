from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from profile_project.sources.discover import (
    DEFAULT_EXCLUDED_DIRS,
    RawSource,
    discover,
)
from profile_project.sources.manifest import read_manifest

if TYPE_CHECKING:
    from profile_project.config.settings import Settings


def _excluded_dirs(settings: Settings) -> list[str]:
    seen: dict[str, None] = {}
    for d in (*DEFAULT_EXCLUDED_DIRS, *settings.sources.excluded_dirs):
        seen[d] = None
    return list(seen)


def build_source_index(root: Path, settings: Settings) -> dict[str, object]:
    merged: dict[str, RawSource] = {}
    # Auto-discovered first; manifest entries win on a path collision.
    for src in discover(root, settings):
        if src.excluded:
            continue
        merged[src.path_or_url] = src
    for src in read_manifest(root, settings):
        merged[src.path_or_url] = src

    counts: dict[str, int] = {
        "code": 0,
        "doc": 0,
        "transcript": 0,
        "note": 0,
        "external": 0,
    }
    sources: list[dict[str, object]] = []
    for src in merged.values():
        counts[src.kind] += 1
        sources.append(
            {
                "source_id": src.source_id,
                "kind": src.kind,
                "path_or_url": src.path_or_url,
                "bytes": src.bytes,
                "language": src.language,
                "discovered_by": src.discovered_by,
                "excluded": False,
            }
        )

    return {
        "artifact_type": "source-index",
        "schema_version": 1,
        "run_id": None,
        "project_root": str(root),
        "sources": sources,
        "counts": counts,
        "excluded_dirs": _excluded_dirs(settings),
        "gitignore_applied": True,
    }
