from __future__ import annotations

from pathlib import Path

from profile_project.config.settings import Settings, SourcesSettings
from profile_project.sources.discover import discover


def _settings(**source_kw: object) -> Settings:
    return Settings(sources=SourcesSettings(**source_kw))  # type: ignore[arg-type]


def test_discover_skips_default_excluded_dirs(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    for d in ("node_modules", ".venv", "build", "dist", ".git", "__pycache__"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "junk.py").write_text("y = 2\n", encoding="utf-8")
    kept = {s.path_or_url for s in discover(tmp_path, _settings()) if not s.excluded}
    assert kept == {"src/a.py"}


def test_discover_skips_config_excluded_dirs(tmp_path: Path) -> None:
    (tmp_path / "keep.py").write_text("a = 1\n", encoding="utf-8")
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "x.py").write_text("b = 2\n", encoding="utf-8")
    settings = _settings(excluded_dirs=["vendor"])
    kept = {s.path_or_url for s in discover(tmp_path, settings) if not s.excluded}
    assert kept == {"keep.py"}


def test_discover_respects_gitignore(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("secret.py\nlogs/\n", encoding="utf-8")
    (tmp_path / "keep.py").write_text("a = 1\n", encoding="utf-8")
    (tmp_path / "secret.py").write_text("KEY = 'x'\n", encoding="utf-8")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "run.log").write_text("noise\n", encoding="utf-8")
    kept = {s.path_or_url for s in discover(tmp_path, _settings()) if not s.excluded}
    assert kept == {"keep.py"}


def test_discover_marks_binary_excluded(tmp_path: Path) -> None:
    (tmp_path / "img.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x01\x02\x03")
    (tmp_path / "ok.py").write_text("a = 1\n", encoding="utf-8")
    sources = discover(tmp_path, _settings())
    by_path = {s.path_or_url: s for s in sources}
    assert by_path["img.png"].excluded is True
    assert by_path["ok.py"].excluded is False


def test_discover_marks_oversized_excluded(tmp_path: Path) -> None:
    big = tmp_path / "huge.py"
    big.write_text("# pad\n" * 400_000, encoding="utf-8")  # > 2 MiB
    small = tmp_path / "small.py"
    small.write_text("a = 1\n", encoding="utf-8")
    by_path = {s.path_or_url: s for s in discover(tmp_path, _settings())}
    assert by_path["huge.py"].excluded is True
    assert by_path["small.py"].excluded is False


def test_discover_sets_auto_provenance_and_relative_posix_paths(tmp_path: Path) -> None:
    content = b"a = 1\n"
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_bytes(content)
    sources = [s for s in discover(tmp_path, _settings()) if not s.excluded]
    assert len(sources) == 1
    src = sources[0]
    assert src.path_or_url == "pkg/mod.py"  # POSIX separators, repo-relative
    assert src.discovered_by == "auto"
    assert src.kind == "code"
    assert src.bytes == len(content)
    assert src.language == "python"
