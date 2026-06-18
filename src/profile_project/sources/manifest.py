from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from profile_project.config.files import atomic_write_json
from profile_project.config.settings import CONFIG_FILENAME
from profile_project.sources.classify import classify
from profile_project.sources.discover import RawSource, _language_for, source_id_for

if TYPE_CHECKING:
    from profile_project.config.settings import Settings

# Maps a source kind to the project-config sources.* list it persists into.
_KIND_TO_LIST: dict[str, str] = {
    "doc": "extra_doc_globs",
    "transcript": "transcripts",
    "note": "notes",
    "external": "external",
}


def _expand(root: Path, globs: list[str], hint: str) -> list[RawSource]:
    out: list[RawSource] = []
    seen: set[str] = set()
    for pattern in globs:
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            rel_posix = path.relative_to(root).as_posix()
            if rel_posix in seen:
                continue
            seen.add(rel_posix)
            try:
                size = path.stat().st_size
            except OSError:
                continue
            out.append(
                RawSource(
                    source_id=source_id_for(rel_posix),
                    kind=classify(rel_posix, hint=hint),
                    path_or_url=rel_posix,
                    bytes=size,
                    language=_language_for(rel_posix),
                    discovered_by="manifest",
                    excluded=False,
                )
            )
    return out


def read_manifest(root: Path, settings: Settings) -> list[RawSource]:
    sources: list[RawSource] = []
    sources.extend(_expand(root, settings.sources.extra_doc_globs, "doc"))
    sources.extend(_expand(root, settings.sources.transcripts, "transcript"))
    sources.extend(_expand(root, settings.sources.notes, "note"))
    for url in settings.sources.external:
        sources.append(
            RawSource(
                source_id=source_id_for(url),
                kind=classify(url, hint="external"),
                path_or_url=url,
                bytes=0,
                language=None,
                discovered_by="manifest",
                excluded=False,
            )
        )
    return sources


def add_manifest_source(
    root: Path, path_or_url: str, kind: str | None = None
) -> dict[str, object]:
    resolved_kind = classify(path_or_url, hint=kind)
    config_path = root / CONFIG_FILENAME
    config: dict[str, object] = json.loads(config_path.read_text(encoding="utf-8"))
    sources_block = config.setdefault("sources", {})
    assert isinstance(sources_block, dict)
    list_key = _KIND_TO_LIST.get(resolved_kind, "external")
    entries = sources_block.setdefault(list_key, [])
    assert isinstance(entries, list)
    if path_or_url not in entries:
        entries.append(path_or_url)
    atomic_write_json(config_path, config)

    is_external = resolved_kind == "external"
    size = 0
    language: str | None = None
    if not is_external:
        target = root / path_or_url
        try:
            size = target.stat().st_size
        except OSError:
            size = 0
        language = _language_for(path_or_url)
    return {
        "source_id": source_id_for(path_or_url),
        "kind": resolved_kind,
        "path_or_url": path_or_url,
        "bytes": size,
        "language": language,
        "discovered_by": "manifest",
        "excluded": False,
    }
