from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from profile_project.config.settings import (
    CONFIG_FILENAME,
    EmbeddingsSettings,
    ProfileSettings,
    Settings,
    VectorStoreSettings,
)


def test_defaults_are_zero_setup_local(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in list(__import__("os").environ):
        if var.startswith("PROFILE_PROJECT_"):
            monkeypatch.delenv(var, raising=False)
    s = Settings()
    assert s.embeddings.method == "sentence-transformers"
    assert s.embeddings.sentence_transformers.model == "all-MiniLM-L6-v2"
    assert s.vectorstore.backend == "chromadb"
    assert s.vectorstore.enabled is True
    assert s.vectorstore.collection == "profile-project"
    assert s.vectorstore.chromadb.path == ".profile_project/chroma"
    assert s.embeddings.ollama.base_url == "http://localhost:11434"
    assert s.embeddings.ollama.model == "qwen3-embedding:8b"
    assert s.phases.include_docs is True
    assert s.output.context_dir == "profile/context"
    assert s.sources.excluded_dirs == ["build", "dist", ".venv", "node_modules"]
    assert s.phase_models == {"default": None}
    assert s.embed_timeout_seconds == 30.0
    assert s.embed_max_retries == 0
    assert s.project_dir is None
    assert s.openai_api_key is None
    assert s.pinecone_api_key is None


def test_nested_delimiter_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROFILE_PROJECT_VECTORSTORE__COLLECTION", "from-env")
    monkeypatch.setenv("PROFILE_PROJECT_CHROMADB__PATH", "custom/chroma")
    monkeypatch.setenv(
        "PROFILE_PROJECT_EMBEDDINGS__SENTENCE_TRANSFORMERS__MODEL", "all-mpnet-base-v2"
    )
    s = Settings()
    assert s.vectorstore.collection == "from-env"
    assert s.vectorstore.chromadb.path == "custom/chroma"
    assert s.embeddings.sentence_transformers.model == "all-mpnet-base-v2"


def test_embeddings_method_primary_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROFILE_PROJECT_EMBEDDINGS__METHOD", raising=False)
    monkeypatch.setenv("PROFILE_PROJECT_DEFAULT_EMBEDDINGS_METHOD", "openai")
    assert Settings().embeddings.method == "openai"


def test_embeddings_method_nested_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROFILE_PROJECT_DEFAULT_EMBEDDINGS_METHOD", raising=False)
    monkeypatch.setenv("PROFILE_PROJECT_EMBEDDINGS__METHOD", "ollama")
    assert Settings().embeddings.method == "ollama"


def test_vectorstore_backend_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROFILE_PROJECT_VECTORSTORE__BACKEND", "pinecone")
    assert Settings().vectorstore.backend == "pinecone"


def test_secrets_are_secretstr_and_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROFILE_PROJECT_OPENAI_API_KEY", "sk-super-secret")
    monkeypatch.setenv("PROFILE_PROJECT_PINECONE_API_KEY", "pc-super-secret")
    s = Settings()
    assert isinstance(s.openai_api_key, SecretStr)
    assert s.openai_api_key.get_secret_value() == "sk-super-secret"
    assert s.pinecone_api_key is not None
    dumped = s.model_dump()
    assert "sk-super-secret" not in repr(dumped["openai_api_key"])
    assert "pc-super-secret" not in repr(dumped["pinecone_api_key"])
    assert str(dumped["openai_api_key"]) == "**********"


def test_embed_timeout_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROFILE_PROJECT_EMBED_TIMEOUT_SECONDS", "0")
    with pytest.raises(ValidationError) as exc:
        Settings()
    assert "embed_timeout_seconds" in str(exc.value)


def test_extra_forbid_rejects_unknown_top_level_key() -> None:
    with pytest.raises(ValidationError):
        Settings(bogus_field="x")  # type: ignore[call-arg]


def test_init_kwargs_override_construct_nested() -> None:
    s = Settings(vectorstore=VectorStoreSettings(backend="disabled", enabled=False))
    assert s.vectorstore.backend == "disabled"
    assert s.vectorstore.enabled is False
    assert EmbeddingsSettings().method == "sentence-transformers"


def test_config_filename_constant() -> None:
    assert CONFIG_FILENAME == ".profile_project_config.json"


def test_project_dir_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROFILE_PROJECT_PROJECT_DIR", "/abs/project")
    assert Settings().project_dir == Path("/abs/project")


def test_profile_settings_has_no_persisted_root_dir() -> None:
    assert "root_dir" not in ProfileSettings.model_fields
    assert set(ProfileSettings.model_fields) == {"name"}
