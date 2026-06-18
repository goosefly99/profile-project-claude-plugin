from __future__ import annotations

_CODE_EXTENSIONS = frozenset(
    {
        ".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".java", ".go", ".rs",
        ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".rb", ".php", ".kt",
        ".swift", ".scala", ".sh", ".sql", ".toml", ".yaml", ".yml",
    }
)
_DOC_EXTENSIONS = frozenset({".md", ".rst", ".txt", ".adoc"})


def _is_url(path_or_url: str) -> bool:
    return path_or_url.startswith("http://") or path_or_url.startswith("https://")


def classify(path_or_url: str, *, hint: str | None = None) -> str:
    if hint is not None:
        return hint
    if _is_url(path_or_url):
        return "external"
    suffix = path_or_url.rsplit("/", 1)[-1]
    dot = suffix.rfind(".")
    ext = suffix[dot:].lower() if dot != -1 else ""
    if ext in _CODE_EXTENSIONS:
        return "code"
    if ext in _DOC_EXTENSIONS:
        return "doc"
    return "doc"
