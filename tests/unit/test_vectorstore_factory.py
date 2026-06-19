from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from profile_project.config.settings import (
    EmbeddingsSettings,
    PineconeSettings,
    Settings,
    VectorStoreSettings,
)
from profile_project.vectorstore.factory import build_backend, build_store


class _FakeEmbedder:
    @property
    def dimension(self) -> int:
        return 384

    @property
    def model_name(self) -> str:
        return "all-MiniLM-L6-v2"

    @property
    def embedder_version(self) -> str:
        return "sentence-transformers/all-MiniLM-L6-v2@hf-fp32"

    @property
    def embedding_provider(self) -> str:
        return "sentence-transformers"

    def probe_dimension(self) -> int:
        return 384

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 384 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * 384


def test_build_store_dispatches_to_chromadb(monkeypatch: pytest.MonkeyPatch) -> None:
    made = MagicMock(name="ChromaDBStore")
    monkeypatch.setattr(
        "profile_project.vectorstore.factory.ChromaDBStore", made
    )
    s = Settings(vectorstore=VectorStoreSettings(backend="chromadb", enabled=True))
    store = build_store(s, _FakeEmbedder())
    assert store is made.return_value
    kwargs: Mapping[str, Any] = made.call_args.kwargs
    assert kwargs["collection"] == "profile-project"
    assert kwargs["persist_path"] == ".profile_project/chroma"


def test_build_store_dispatches_to_pinecone_with_effective_dim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made = MagicMock(name="PineconeStore")
    monkeypatch.setattr(
        "profile_project.vectorstore.factory.PineconeStore", made
    )
    s = Settings(
        vectorstore=VectorStoreSettings(
            backend="pinecone",
            enabled=True,
            pinecone=PineconeSettings(
                index="my-index",
                namespace="profile-v1",
                embeddings_model="text-embedding-3-small",
            ),
        ),
        pinecone_api_key=SecretStr("pk-123"),
    )
    store = build_store(s, _FakeEmbedder())
    assert store is made.return_value
    kwargs: Mapping[str, Any] = made.call_args.kwargs
    assert kwargs["index"] == "my-index"
    assert kwargs["namespace"] == "profile-v1"
    assert kwargs["api_key"] == "pk-123"
    # effective embedding dim comes from the embedder, never a static table
    assert kwargs["embedding_dim"] == 384


def test_build_backend_returns_none_when_disabled() -> None:
    s = Settings(vectorstore=VectorStoreSettings(backend="disabled", enabled=False))
    assert build_backend(s) is None


def test_build_backend_returns_none_when_backend_disabled_but_enabled() -> None:
    # The backend="disabled" guard must fire on its own, independent of the
    # master enabled switch (which is True here so it cannot short-circuit first).
    s = Settings(vectorstore=VectorStoreSettings(backend="disabled", enabled=True))
    assert build_backend(s) is None


def test_build_backend_returns_none_on_conflict_disable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # openai method + no key => C2 warn+disable => factory yields None.
    monkeypatch.delenv("PROFILE_PROJECT_OPENAI_API_KEY", raising=False)
    s = Settings(
        vectorstore=VectorStoreSettings(backend="chromadb", enabled=True),
        embeddings=EmbeddingsSettings(method="openai"),
    )
    assert build_backend(s) is None


def test_build_backend_returns_none_when_extra_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Patch the extra-installed seam so C5 doesn't short-circuit before
    # we reach the patched build_embedder raising EmbedderExtraMissing.
    monkeypatch.setattr(
        "profile_project.config.conflicts._extra_installed", lambda _m: True
    )

    from profile_project.vectorstore.embedders import EmbedderExtraMissing

    def _boom(_settings: Settings) -> Any:
        raise EmbedderExtraMissing("openai extra not installed")

    monkeypatch.setattr(
        "profile_project.vectorstore.factory.build_embedder", _boom
    )
    s = Settings(
        vectorstore=VectorStoreSettings(backend="chromadb", enabled=True),
    )
    assert build_backend(s) is None


def test_build_backend_returns_embedder_and_store_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Patch _extra_installed so C5 (sentence-transformers not installed) doesn't
    # disable before we reach the patched build_embedder.
    monkeypatch.setattr(
        "profile_project.config.conflicts._extra_installed", lambda _m: True
    )

    fake = _FakeEmbedder()
    monkeypatch.setattr(
        "profile_project.vectorstore.factory.build_embedder", lambda _s: fake
    )
    made = MagicMock(name="ChromaDBStore")
    monkeypatch.setattr(
        "profile_project.vectorstore.factory.ChromaDBStore", made
    )
    s = Settings(vectorstore=VectorStoreSettings(backend="chromadb", enabled=True))
    result = build_backend(s)
    assert result is not None
    embedder, store = result
    assert embedder is fake
    assert store is made.return_value
