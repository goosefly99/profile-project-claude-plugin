from __future__ import annotations

import fnmatch
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from profile_project.sources.classify import classify

if TYPE_CHECKING:
    from profile_project.config.settings import Settings

DEFAULT_EXCLUDED_DIRS: tuple[str, ...] = (
    "node_modules",
    ".venv",
    "build",
    "dist",
    ".git",
    "__pycache__",
)
MAX_FILE_BYTES = 2 * 1024 * 1024  # 2 MiB — oversized files are skipped (excluded=True)

_LANGUAGE_BY_EXT: dict[str, str] = {
    ".py": "python", ".pyi": "python", ".ts": "typescript", ".tsx": "typescript",
    ".js": "javascript", ".jsx": "javascript", ".java": "java", ".go": "go",
    ".rs": "rust", ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp",
    ".hpp": "cpp", ".cs": "csharp", ".rb": "ruby", ".php": "php",
    ".md": "markdown", ".rst": "restructuredtext", ".txt": "text",
    ".toml": "toml", ".yaml": "yaml", ".yml": "yaml", ".sql": "sql",
    ".sh": "shell",
}


class RawSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str
    kind: str
    path_or_url: str
    bytes: int
    language: str | None
    discovered_by: str
    excluded: bool


def source_id_for(path_or_url: str) -> str:
    return hashlib.sha1(path_or_url.encode("utf-8")).hexdigest()


def _language_for(rel_posix: str) -> str | None:
    name = rel_posix.rsplit("/", 1)[-1]
    dot = name.rfind(".")
    if dot == -1:
        return None
    return _LANGUAGE_BY_EXT.get(name[dot:].lower())


def _read_gitignore_patterns(root: Path) -> list[str]:
    gi = root / ".gitignore"
    if not gi.is_file():
        return []
    patterns: list[str] = []
    for line in gi.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            patterns.append(stripped.rstrip("/"))
    return patterns


def _gitignored(rel_posix: str, patterns: list[str]) -> bool:
    name = rel_posix.rsplit("/", 1)[-1]
    for pat in patterns:
        if fnmatch.fnmatch(rel_posix, pat) or fnmatch.fnmatch(name, pat):
            return True
        if rel_posix == pat or rel_posix.startswith(pat + "/"):
            return True
    return False


def _is_binary(path: Path) -> bool:
    try:
        chunk = path.read_bytes()[:1024]
    except OSError:
        return True
    return b"\x00" in chunk


def discover(root: Path, settings: Settings) -> list[RawSource]:
    excluded_dirs = set(DEFAULT_EXCLUDED_DIRS) | set(settings.sources.excluded_dirs)
    gitignore = _read_gitignore_patterns(root)
    results: list[RawSource] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        parts = rel.parts
        if any(part in excluded_dirs for part in parts[:-1]):
            continue
        rel_posix = rel.as_posix()
        if rel_posix == ".gitignore":
            continue
        if _gitignored(rel_posix, gitignore):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        excluded = size > MAX_FILE_BYTES or _is_binary(path)
        results.append(
            RawSource(
                source_id=source_id_for(rel_posix),
                kind=classify(rel_posix),
                path_or_url=rel_posix,
                bytes=size,
                language=_language_for(rel_posix),
                discovered_by="auto",
                excluded=excluded,
            )
        )
    return results
