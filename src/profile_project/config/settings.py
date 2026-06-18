from __future__ import annotations

import os
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, SecretStr, field_serializer
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

CONFIG_FILENAME = ".profile_project_config.json"

# ---------------------------------------------------------------------------
# Context variable for threading the project root into settings_customise_sources
# without changing the classmethod signature (which pydantic-settings controls).
# ---------------------------------------------------------------------------
_PROJECT_ROOT: ContextVar[Path | None] = ContextVar(
    "_PROFILE_PROJECT_ROOT", default=None
)

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
    # the env_nested_delimiter resolution used by pydantic-settings.
    model_config = {"populate_by_name": True}

    method: Literal["sentence-transformers", "openai", "ollama", "disabled"] = Field(
        default="sentence-transformers",
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
    # populate_by_name allows field name "backend" to be accepted alongside
    # the env_nested_delimiter resolution used by pydantic-settings.
    model_config = {"populate_by_name": True}

    backend: Literal["chromadb", "pinecone", "disabled"] = Field(
        default="chromadb",
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

    @field_serializer("openai_api_key", "pinecone_api_key", when_used="always")
    def _mask_secret(self, value: SecretStr | None) -> str | None:
        """Return a fixed mask for any set secret, None when unset.

        Ensures neither model_dump() nor model_dump(mode='json') ever exposes
        the real secret value — pydantic's default only masks in json/repr mode.
        """
        if value is None:
            return None
        return "**********"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Lazy import to break the circular dependency:
        # settings.py is imported by sources.py (for CONFIG_FILENAME / _PROJECT_ROOT),
        # and sources.py is imported here. Deferring the import to call-time avoids
        # the circular import at module load time.
        from profile_project.config.sources import (  # noqa: PLC0415
            ProjectJsonConfigSettingsSource,
        )

        project_root = _PROJECT_ROOT.get() or Path.cwd()
        project_source = ProjectJsonConfigSettingsSource(settings_cls, project_root)
        # Precedence: init > project-JSON > alias-env > env > dotenv > secrets
        # _TopLevelAliasSource sits after project JSON so the JSON can override
        # alias-env vars (PROFILE_PROJECT_DEFAULT_EMBEDDINGS_METHOD etc.) while
        # direct constructor kwargs (init_settings) still win everything.
        return (
            init_settings,
            project_source,
            _TopLevelAliasSource(settings_cls),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )
