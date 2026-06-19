from __future__ import annotations

import types
from typing import Any

import pytest
from pydantic import SecretStr

from profile_project.config.settings import EmbeddingsSettings, Settings
from profile_project.vectorstore.embedders import (
    OPENAI_BATCH_CAP,
    OPENAI_NO_DIMENSIONS_MODELS,
    OPENAI_STATIC_DIMS,
    EmbedderExtraMissing,
    OllamaEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    build_embedder,
)
from profile_project.vectorstore.protocols import Embedder


def test_openai_constants_match_spec() -> None:
    assert OPENAI_BATCH_CAP == 2048
    assert OPENAI_STATIC_DIMS == {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    assert OPENAI_NO_DIMENSIONS_MODELS == frozenset({"text-embedding-ada-002"})


def test_embedder_extra_missing_is_import_error() -> None:
    assert issubclass(EmbedderExtraMissing, ImportError)


# ---------------------------------------------------------------------------
# SentenceTransformerEmbedder tests
# ---------------------------------------------------------------------------


class _FakeSTModel:
    def __init__(self, name: str) -> None:
        self.name = name
        self.last_kwargs: dict[str, Any] = {}

    def encode(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        self.last_kwargs = kwargs
        return [[0.1, 0.2, 0.3] for _ in texts]

    def get_sentence_embedding_dimension(self) -> int:
        return 3


@pytest.fixture
def _patch_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.machinery
    import sys

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _FakeSTModel  # type: ignore[attr-defined]
    module.__spec__ = importlib.machinery.ModuleSpec(
        "sentence_transformers", None
    )
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_sentence_transformer_is_structural_embedder(
    _patch_sentence_transformers: None,
) -> None:
    emb = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    assert isinstance(emb, Embedder)


def test_sentence_transformer_normalizes_and_reports_dim(
    _patch_sentence_transformers: None,
) -> None:
    emb = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    vectors = emb.embed(["a", "b"])
    assert vectors == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    assert emb._model.last_kwargs["normalize_embeddings"] is True
    assert emb._model.last_kwargs["convert_to_numpy"] is True
    assert emb.embed_query("a") == [0.1, 0.2, 0.3]
    assert emb.dimension == 3
    assert emb.probe_dimension() == 3


def test_sentence_transformer_default_version_literal(
    _patch_sentence_transformers: None,
) -> None:
    emb = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    assert emb.model_name == "all-MiniLM-L6-v2"
    assert emb.embedder_version == "sentence-transformers/all-MiniLM-L6-v2@hf-fp32"
    assert emb.embedding_provider == "sentence-transformers"


# ---------------------------------------------------------------------------
# OpenAIEmbedder tests
# ---------------------------------------------------------------------------


class _FakeOpenAIEmbeddings:
    def __init__(self, calls: list[dict[str, Any]]) -> None:
        self._calls = calls

    def create(
        self,
        *,
        input: list[str],
        model: str,
        encoding_format: str,
        dimensions: int | None = None,
    ) -> Any:
        self._calls.append(
            {
                "input": list(input),
                "model": model,
                "encoding_format": encoding_format,
                "dimensions": dimensions,
            }
        )
        dim = dimensions if dimensions is not None else 1536
        data = [
            types.SimpleNamespace(embedding=[float(i)] * dim, index=i)
            for i, _ in enumerate(input)
        ]
        return types.SimpleNamespace(data=data)


class _FakeOpenAIClient:
    def __init__(
        self, *, api_key: str, base_url: str | None, timeout: float, max_retries: int
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.calls: list[dict[str, Any]] = []
        self.embeddings = _FakeOpenAIEmbeddings(self.calls)


@pytest.fixture
def _patch_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.machinery
    import sys

    module = types.ModuleType("openai")
    module.OpenAI = _FakeOpenAIClient  # type: ignore[attr-defined]
    module.__spec__ = importlib.machinery.ModuleSpec(
        "openai", None
    )
    monkeypatch.setitem(sys.modules, "openai", module)


def test_openai_is_structural_embedder(_patch_openai: None) -> None:
    emb = OpenAIEmbedder("text-embedding-3-small", api_key="sk-test")
    assert isinstance(emb, Embedder)
    assert emb.embedding_provider == "openai"


def test_openai_sub_batches_at_cap_and_preserves_order(_patch_openai: None) -> None:
    emb = OpenAIEmbedder("text-embedding-3-small", api_key="sk-test")
    texts = [f"t{i}" for i in range(OPENAI_BATCH_CAP + 5)]
    vectors = emb.embed(texts)
    assert len(vectors) == OPENAI_BATCH_CAP + 5
    # two calls: 2048 then 5, none exceeding the cap, concatenated in order
    sizes = [len(c["input"]) for c in emb._client.calls]
    assert sizes == [OPENAI_BATCH_CAP, 5]
    assert all(size <= OPENAI_BATCH_CAP for size in sizes)
    assert emb._client.calls[0]["encoding_format"] == "float"


def test_openai_effective_dim_prefers_dimensions_arg(_patch_openai: None) -> None:
    emb = OpenAIEmbedder("text-embedding-3-small", api_key="sk-test", dimensions=256)
    assert emb.dimension == 256
    assert emb.embedder_version == "openai/text-embedding-3-small@dim256"
    # the dimensions arg is forwarded to the API call
    emb.embed(["x"])
    assert emb._client.calls[-1]["dimensions"] == 256


def test_openai_effective_dim_falls_back_to_static_table(_patch_openai: None) -> None:
    emb = OpenAIEmbedder("text-embedding-3-large", api_key="sk-test")
    assert emb.dimension == 3072
    assert emb.embedder_version == "openai/text-embedding-3-large@dim3072"


def test_openai_ada_never_sends_dimensions(_patch_openai: None) -> None:
    emb = OpenAIEmbedder("text-embedding-ada-002", api_key="sk-test", dimensions=256)
    assert emb.dimension == 1536  # ada ignores dimensions; static table wins
    emb.embed(["x"])
    assert emb._client.calls[-1]["dimensions"] is None


# ---------------------------------------------------------------------------
# OllamaEmbedder tests
# ---------------------------------------------------------------------------


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeHTTPXClient:
    def __init__(self, *, timeout: float, boom: bool = False) -> None:
        self.timeout = timeout
        self.boom = boom
        self.posts: list[dict[str, Any]] = []

    def post(self, url: str, json: dict[str, Any]) -> _FakeHTTPResponse:
        if self.boom:
            raise RuntimeError("connection refused")
        self.posts.append({"url": url, "json": json})
        return _FakeHTTPResponse({"embeddings": [[0.5, 0.6, 0.7, 0.8]]})

    def close(self) -> None:
        return None

    def __enter__(self) -> _FakeHTTPXClient:
        return self

    def __exit__(self, *args: Any) -> None:
        return None


def _patch_httpx(
    monkeypatch: pytest.MonkeyPatch, *, boom: bool = False
) -> list[_FakeHTTPXClient]:
    import httpx

    created: list[_FakeHTTPXClient] = []

    def _factory(*, timeout: float, **_: Any) -> _FakeHTTPXClient:
        client = _FakeHTTPXClient(timeout=timeout, boom=boom)
        created.append(client)
        return client

    monkeypatch.setattr(httpx, "Client", _factory)
    return created


def test_ollama_is_structural_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_httpx(monkeypatch)
    emb = OllamaEmbedder("qwen3-embedding:8b")
    assert isinstance(emb, Embedder)
    assert emb.embedding_provider == "ollama"


def test_ollama_posts_one_at_a_time_to_api_embed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _patch_httpx(monkeypatch)
    emb = OllamaEmbedder("qwen3-embedding:8b")
    vectors = emb.embed(["a", "b", "c"])
    assert vectors == [[0.5, 0.6, 0.7, 0.8]] * 3
    posts = [p for client in created for p in client.posts]
    assert len(posts) == 3  # one request per text (multi-input 400 workaround)
    assert posts[0]["url"] == "http://localhost:11434/api/embed"
    assert posts[0]["json"] == {"model": "qwen3-embedding:8b", "input": "a"}


def test_ollama_dimension_is_probed_live(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_httpx(monkeypatch)
    emb = OllamaEmbedder("qwen3-embedding:8b")
    assert emb.dimension == 4
    assert emb.probe_dimension() == 4
    assert emb.model_name == "qwen3-embedding:8b"
    assert emb.embedder_version == "ollama/qwen3-embedding:8b@dim4"


def test_ollama_probe_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_httpx(monkeypatch, boom=True)
    emb = OllamaEmbedder("qwen3-embedding:8b", timeout=1.0)
    with pytest.raises(RuntimeError, match="connection refused"):
        emb.probe_dimension()


# ---------------------------------------------------------------------------
# build_embedder dispatch tests
# ---------------------------------------------------------------------------


def test_build_embedder_dispatches_sentence_transformers(
    _patch_sentence_transformers: None,
) -> None:
    settings = Settings(
        embeddings=EmbeddingsSettings(method="sentence-transformers")
    )
    emb = build_embedder(settings)
    assert isinstance(emb, SentenceTransformerEmbedder)
    assert emb.model_name == "all-MiniLM-L6-v2"


def test_build_embedder_dispatches_openai(_patch_openai: None) -> None:
    settings = Settings(
        embeddings=EmbeddingsSettings(method="openai"),
        openai_api_key=SecretStr("sk-live"),
    )
    emb = build_embedder(settings)
    assert isinstance(emb, OpenAIEmbedder)
    assert emb._client.api_key == "sk-live"
    assert emb._client.timeout == 30.0


def test_build_embedder_dispatches_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_httpx(monkeypatch)
    settings = Settings(embeddings=EmbeddingsSettings(method="ollama"))
    emb = build_embedder(settings)
    assert isinstance(emb, OllamaEmbedder)
    assert emb.model_name == "qwen3-embedding:8b"


def test_build_embedder_missing_extra_raises_embedder_extra_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    # Force the lazy `import sentence_transformers` inside the embedder to fail,
    # as it would when the [local-embeddings] extra is not installed.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    settings = Settings(
        embeddings=EmbeddingsSettings(method="sentence-transformers")
    )
    with pytest.raises(EmbedderExtraMissing, match="local-embeddings"):
        build_embedder(settings)


def test_build_embedder_rejects_disabled() -> None:
    settings = Settings(embeddings=EmbeddingsSettings(method="disabled"))
    with pytest.raises(ValueError, match="disabled"):
        build_embedder(settings)
