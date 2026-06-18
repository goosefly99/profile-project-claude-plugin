from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

CONFIG_FILENAME = ".profile_project_config.json"

# ---------------------------------------------------------------------------
# Top-level env-var aliases that don't fit the standard nested-delimiter
# scheme (e.g. PROFILE_PROJECT_DEFAULT_EMBEDDINGS_METHOD maps to
# embeddings.method, PROFILE_PROJECT_CHROMADB__PATH maps to
# vectorstore.chromadb.path).  Declared here so they're easy to audit.
# ---------------------------------------------------------------------------
_TOP_LEVEL_ALIASES: dict[str, list[str]] = {
    "PROFILE_PROJECT_DEFAULT_EMBEDDINGS_METHOD": ["embeddings", "method"],
    "PROFILE_PROJECT_CHROMADB__PATH": ["vectorstore", "chromadb", "path"],
}


class _TopLevelAliasSource(PydanticBaseSettingsSource):
    """Custom pydantic-settings source that resolves non-standard env aliases."""

    def get_field_value(
        self,
        field: Any,
        field_name: str,
    ) -> Any:  # pragma: no cover
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for env_key, path in _TOP_LEVEL_ALIASES.items():
            val = os.environ.get(env_key)
            if val is None:
                continue
            node: dict[str, Any] = result
            for segment in path[:-1]:
                node = node.setdefault(segment, {})
            node[path[-1]] = val
        return result


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class SentenceTransformersSettings(BaseModel):
    model: str = "all-MiniLM-L6-v2"


class OpenAISettings(BaseModel):
    model: str = "text-embedding-3-small"
    base_url: str | None = None


class OllamaSettings(BaseModel):
    base_url: str = "http://localhost:11434"
    model: str = "qwen3-embedding:8b"


class EmbeddingsSettings(BaseModel):
    # populate_by_name allows field name "method" to be accepted alongside
    # validation_alias values when data is provided by pydantic-settings'
    # env_nested_delimiter resolution.
    model_config = {"populate_by_name": True}

    method: Literal["sentence-transformers", "openai", "ollama", "disabled"] = Field(
        default="sentence-transformers",
        validation_alias=AliasChoices(
            "PROFILE_PROJECT_DEFAULT_EMBEDDINGS_METHOD",
            "PROFILE_PROJECT_EMBEDDINGS__METHOD",
        ),
    )
    sentence_transformers: SentenceTransformersSettings = Field(
        default_factory=SentenceTransformersSettings
    )
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)


class ChromadbSettings(BaseModel):
    path: str = ".profile_project/chroma"


class PineconeSettings(BaseModel):
    index: str | None = None
    namespace: str | None = None
    embeddings_model: str | None = None


class VectorStoreSettings(BaseModel):
    # populate_by_name allows field name "backend" alongside its alias when
    # data arrives via env_nested_delimiter or direct constructor kwargs.
    model_config = {"populate_by_name": True}

    backend: Literal["chromadb", "pinecone", "disabled"] = Field(
        default="chromadb",
        validation_alias=AliasChoices("PROFILE_PROJECT_VECTORSTORE__BACKEND"),
    )
    enabled: bool = True
    collection: str = "profile-project"
    chromadb: ChromadbSettings = Field(default_factory=ChromadbSettings)
    pinecone: PineconeSettings = Field(default_factory=PineconeSettings)


class PhasesSettings(BaseModel):
    include_docs: bool = True
    include_transcripts: bool = True
    build_vectorstore: bool = True


class OutputSettings(BaseModel):
    context_dir: str = "profile/context"
    guide_dir: str = "profile/guide"


class SourcesSettings(BaseModel):
    extra_doc_globs: list[str] = Field(default_factory=list)
    transcripts: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    external: list[str] = Field(default_factory=list)
    excluded_dirs: list[str] = Field(
        default_factory=lambda: ["build", "dist", ".venv", "node_modules"]
    )


class ProfileSettings(BaseModel):
    name: str | None = None


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PROFILE_PROJECT_",
        env_nested_delimiter="__",
        nested_model_default_partial_update=True,
        extra="forbid",
    )

    vectorstore: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    embeddings: EmbeddingsSettings = Field(default_factory=EmbeddingsSettings)
    phases: PhasesSettings = Field(default_factory=PhasesSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)
    sources: SourcesSettings = Field(default_factory=SourcesSettings)
    profile: ProfileSettings = Field(default_factory=ProfileSettings)
    phase_models: dict[str, str | None] = Field(
        default_factory=lambda: dict[str, str | None]({"default": None})
    )
    embed_timeout_seconds: float = Field(default=30.0, gt=0.0)
    embed_max_retries: int = Field(default=0, ge=0)
    project_dir: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("PROFILE_PROJECT_PROJECT_DIR"),
    )
    openai_api_key: SecretStr | None = None
    pinecone_api_key: SecretStr | None = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Top-level aliases sit between init_settings (highest priority) and
        # the standard env source so that direct constructor kwargs still win.
        return (
            init_settings,
            _TopLevelAliasSource(settings_cls),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )
