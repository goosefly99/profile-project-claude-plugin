from __future__ import annotations

import json
import pathlib

PLUGIN_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parents[2]
README_PATH: pathlib.Path = PLUGIN_ROOT / "README.md"
ENV_EXAMPLE_PATH: pathlib.Path = PLUGIN_ROOT / ".env.example"
MARKETPLACE_DOC_PATH: pathlib.Path = PLUGIN_ROOT / "docs" / "marketplace-entry.md"


def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def test_pyproject_is_the_plugin_root_anchor() -> None:
    # parents[2] from tests/unit/<file> must be the plugin root holding pyproject.toml.
    assert (PLUGIN_ROOT / "pyproject.toml").is_file()
    assert (PLUGIN_ROOT / ".claude-plugin" / "plugin.json").is_file()


def test_readme_exists_and_covers_required_sections() -> None:
    text = _read(README_PATH)
    for heading in (
        "# profile-project",
        "## Installation",
        "## Initialization",
        "## Usage",
        "## Configuration",
        "## Troubleshooting",
        "## Security & hygiene",
    ):
        assert heading in text, f"README missing section: {heading!r}"


def test_readme_documents_zero_setup_default_and_extras() -> None:
    text = _read(README_PATH)
    # Zero-setup default backends (sentence-transformers + chromadb) + install set.
    assert "sentence-transformers" in text
    assert "chromadb" in text
    assert "[local-embeddings]" in text
    assert "[chroma]" in text
    # Every §14.1 extra name is documented.
    extras = (
        "[chroma]",
        "[pinecone]",
        "[openai]",
        "[local-embeddings]",
        "[ollama]",
        "[all]",
    )
    for extra in extras:
        assert extra in text, f"README missing extra: {extra}"


def test_readme_places_chromadb_under_the_chroma_extra() -> None:
    # §14.1: chromadb is the [chroma] EXTRA (NOT a base dependency). The extras matrix
    # row for [chroma] must name chromadb on the same line.
    text = _read(README_PATH)
    chroma_rows = [
        line for line in text.splitlines() if "[chroma]" in line and "chromadb" in line
    ]
    assert chroma_rows, "extras matrix must list chromadb under the [chroma] extra"


def test_readme_documents_stdio_launch_line() -> None:
    text = _read(README_PATH)
    assert "uv run --directory ${CLAUDE_PLUGIN_ROOT} python -m profile_project" in text


def test_readme_lists_every_slash_command() -> None:
    text = _read(README_PATH)
    for cmd in (
        "/profile-project:init",
        "/profile-project:profile",
        "/profile-project:status",
        "/profile-project:query",
        "/profile-project:navigate",
        "/profile-project:refresh",
    ):
        assert cmd in text, f"README missing command: {cmd}"


def test_readme_states_secrets_are_env_only_and_json_overrides_env() -> None:
    text = _read(README_PATH)
    assert "PROFILE_PROJECT_OPENAI_API_KEY" in text
    assert "PROFILE_PROJECT_PINECONE_API_KEY" in text
    # The two load-bearing config rules.
    assert "env-only" in text.lower() or "environment-only" in text.lower()
    assert "project json overrides env" in text.lower()


def test_readme_pins_default_embedder_version_literal() -> None:
    text = _read(README_PATH)
    assert "sentence-transformers/all-MiniLM-L6-v2@hf-fp32" in text


def test_readme_never_instructs_creating_a_pinecone_index() -> None:
    text = _read(README_PATH)
    lowered = text.lower()
    # Existing-index-only invariant: never tell the user to create/provision an index.
    assert "create_index" not in lowered
    assert "existing" in lowered  # README must say Pinecone uses an EXISTING index


def test_env_example_documents_all_non_secret_env_vars() -> None:
    text = _read(ENV_EXAMPLE_PATH)
    required = (
        "PROFILE_PROJECT_OPENAI_API_KEY",
        "PROFILE_PROJECT_PINECONE_API_KEY",
        "PROFILE_PROJECT_DEFAULT_EMBEDDINGS_METHOD",
        "PROFILE_PROJECT_EMBEDDINGS__METHOD",
        "PROFILE_PROJECT_VECTORSTORE__BACKEND",
        "PROFILE_PROJECT_VECTORSTORE__ENABLED",
        "PROFILE_PROJECT_VECTORSTORE__COLLECTION",
        "PROFILE_PROJECT_CHROMADB__PATH",
        "PROFILE_PROJECT_PINECONE__INDEX",
        "PROFILE_PROJECT_PINECONE__NAMESPACE",
        "PROFILE_PROJECT_PINECONE__EMBEDDINGS_MODEL",
        "PROFILE_PROJECT_EMBEDDINGS__SENTENCE_TRANSFORMERS__MODEL",
        "PROFILE_PROJECT_EMBEDDINGS__OPENAI__MODEL",
        "PROFILE_PROJECT_EMBEDDINGS__OLLAMA__BASE_URL",
        "PROFILE_PROJECT_EMBEDDINGS__OLLAMA__MODEL",
        "PROFILE_PROJECT_EMBED_TIMEOUT_SECONDS",
        "PROFILE_PROJECT_EMBED_MAX_RETRIES",
        "PROFILE_PROJECT_PHASES__INCLUDE_DOCS",
        "PROFILE_PROJECT_PHASES__INCLUDE_TRANSCRIPTS",
        "PROFILE_PROJECT_PHASES__BUILD_VECTORSTORE",
        "PROFILE_PROJECT_OUTPUT__CONTEXT_DIR",
        "PROFILE_PROJECT_OUTPUT__GUIDE_DIR",
        "PROFILE_PROJECT_PROJECT_DIR",
    )
    for var in required:
        assert var in text, f".env.example missing env var: {var}"


def test_env_example_secrets_have_no_real_values() -> None:
    # The two secret rows MUST be present but empty (no value after '=').
    for line in _read(ENV_EXAMPLE_PATH).splitlines():
        stripped = line.strip()
        if stripped.startswith(
            ("PROFILE_PROJECT_OPENAI_API_KEY=", "PROFILE_PROJECT_PINECONE_API_KEY=")
        ):
            key, _, value = stripped.partition("=")
            assert value == "", f"secret {key} must be empty, got {value!r}"


def test_marketplace_doc_has_verbatim_entry_and_publish_precondition() -> None:
    text = _read(MARKETPLACE_DOC_PATH)
    # The §14.4 entry fields, exact.
    assert '"name": "profile-project-auto-dev"' in text
    url_field = (
        '"url": "https://github.com/goosefly99/profile-project-claude-plugin.git"'
    )
    assert url_field in text
    assert '"ref": "auto_dev"' in text
    assert '"version": "0.1.0"' in text
    homepage_field = (
        '"homepage": "https://github.com/goosefly99/profile-project-claude-plugin"'
    )
    assert homepage_field in text
    # Where to add it.
    assert "claude-plugins-auto-dev/.claude-plugin/marketplace.json" in text
    # The publish precondition (§14.4): branch must exist + manifest name/version match.
    lowered = text.lower()
    assert "publish precondition" in lowered
    assert "auto_dev" in text and "branch must" in lowered
    assert "tag ref is more stable than a branch ref" in lowered


def test_marketplace_doc_entry_is_valid_json_object() -> None:
    # The fenced ```json block must parse and carry the four load-bearing keys.
    text = _read(MARKETPLACE_DOC_PATH)
    start = text.index("```json") + len("```json")
    end = text.index("```", start)
    entry = json.loads(text[start:end])
    assert entry["name"] == "profile-project-auto-dev"
    assert entry["version"] == "0.1.0"
    assert entry["source"]["ref"] == "auto_dev"
    assert entry["source"]["url"] == "https://github.com/goosefly99/profile-project-claude-plugin.git"


def test_plugin_manifest_version_matches_marketplace_entry() -> None:
    manifest = json.loads(_read(PLUGIN_ROOT / ".claude-plugin" / "plugin.json"))
    entry_text = _read(MARKETPLACE_DOC_PATH)
    start = entry_text.index("```json") + len("```json")
    end = entry_text.index("```", start)
    entry = json.loads(entry_text[start:end])
    # The publish precondition requires manifest name+version to match the entry.
    assert manifest["name"] == entry["name"] == "profile-project-auto-dev"
    assert manifest["version"] == entry["version"] == "0.1.0"
