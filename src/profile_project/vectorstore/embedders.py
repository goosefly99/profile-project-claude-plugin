# src/profile_project/vectorstore/embedders.py
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from profile_project.config.settings import Settings
    from profile_project.vectorstore.protocols import Embedder

OPENAI_BATCH_CAP: int = 2048
OPENAI_STATIC_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
OPENAI_NO_DIMENSIONS_MODELS: frozenset[str] = frozenset({"text-embedding-ada-002"})


class EmbedderExtraMissing(ImportError):
    """Raised when the selected provider's optional extra is not installed.

    A subclass of ``ImportError`` so callers can also catch the broad case.
    The conflict-C5 caller (``pp_config_validate`` / ``pp_vectorstore_check``)
    catches this and warns + disables the vectorstore rather than crashing.
    """


class SentenceTransformerEmbedder:
    """Local, offline-after-first-pull embedder (§10.2 default).

    Embeddings are L2-normalized at the model (``normalize_embeddings=True``)
    so a cosine store and a dot-product store agree.
    """

    def __init__(self, model_name: str, *, embedder_version: str | None = None) -> None:
        from sentence_transformers import (
            SentenceTransformer,  # lazy: needs [local-embeddings]
        )

        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._embedder_version = (
            embedder_version
            if embedder_version is not None
            else f"sentence-transformers/{model_name}@hf-fp32"
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return [[float(x) for x in row] for row in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedder_version(self) -> str:
        return self._embedder_version

    @property
    def embedding_provider(self) -> str:
        return "sentence-transformers"

    def probe_dimension(self) -> int:
        return len(self.embed_query("probe"))


class OpenAIEmbedder:
    """Remote OpenAI embedder (§10.2). Batches at ``OPENAI_BATCH_CAP`` (2048).

    Effective dimension resolution order: explicit ``dimensions`` arg (if the
    model accepts it) > ``OPENAI_STATIC_DIMS`` > one bounded live probe.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        *,
        base_url: str | None = None,
        dimensions: int | None = None,
        timeout: float = 30.0,
        max_retries: int = 0,
    ) -> None:
        from openai import OpenAI  # lazy: needs [openai]

        self._model_name = model_name
        # ada-002 does not accept `dimensions`; never send it (§10.2).
        self._dimensions = (
            None if model_name in OPENAI_NO_DIMENSIONS_MODELS else dimensions
        )
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._cached_dim: int | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), OPENAI_BATCH_CAP):
            batch = texts[start : start + OPENAI_BATCH_CAP]
            response = self._client.embeddings.create(
                input=batch,
                model=self._model_name,
                encoding_format="float",
                dimensions=self._dimensions,
            )
            ordered = sorted(response.data, key=lambda d: d.index)
            out.extend([float(x) for x in d.embedding] for d in ordered)
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        if self._dimensions is not None:
            return self._dimensions
        static = OPENAI_STATIC_DIMS.get(self._model_name)
        if static is not None:
            return static
        return self.probe_dimension()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedder_version(self) -> str:
        return f"openai/{self._model_name}@dim{self.dimension}"

    @property
    def embedding_provider(self) -> str:
        return "openai"

    def probe_dimension(self) -> int:
        if self._cached_dim is None:
            self._cached_dim = len(self.embed_query("probe"))
        return self._cached_dim


class OllamaEmbedder:
    """Local Ollama daemon embedder (§10.2). No API key.

    POSTs one text at a time to ``{base_url}/api/embed`` (multi-input 400
    workaround). Dimension is probed live; probes are bounded by ``timeout``
    and fail closed (errors propagate — the caller warns + disables).
    """

    def __init__(
        self,
        model_name: str,
        *,
        base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
        max_retries: int = 0,
    ) -> None:
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._cached_dim: int | None = None

    def _embed_one(self, text: str) -> list[float]:
        import httpx  # base dependency

        url = f"{self._base_url}/api/embed"
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(url, json={"model": self._model_name, "input": text})
            response.raise_for_status()
            payload = response.json()
        return [float(x) for x in payload["embeddings"][0]]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    @property
    def dimension(self) -> int:
        return self.probe_dimension()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedder_version(self) -> str:
        return f"ollama/{self._model_name}@dim{self.dimension}"

    @property
    def embedding_provider(self) -> str:
        return "ollama"

    def probe_dimension(self) -> int:
        if self._cached_dim is None:
            self._cached_dim = len(self._embed_one("probe"))
        return self._cached_dim


def build_embedder(settings: Settings) -> Embedder:
    """Construct the configured concrete embedder (§10.2 / §10.3).

    Dispatches on ``settings.embeddings.method``. Raises
    ``EmbedderExtraMissing`` when the selected provider's optional extra is not
    installed (conflict C5: the caller warns + disables). Raises ``ValueError``
    for ``"disabled"`` or any unrecognized method. Secrets are unwrapped only
    at this boundary and never logged.

    The scoped ollama-unreachable -> sentence-transformers auto-fallback is
    omitted in v0.1.0 (deferred to Future Work, §18); an unreachable ollama
    endpoint is governed by the conflict matrix (C3: warn + disable), and this
    factory never silently swaps geometries.
    """
    method = settings.embeddings.method
    try:
        if method == "sentence-transformers":
            return SentenceTransformerEmbedder(
                settings.embeddings.sentence_transformers.model
            )
        if method == "openai":
            if settings.openai_api_key is None:
                raise ValueError(
                    "embeddings.method='openai' but "
                    "PROFILE_PROJECT_OPENAI_API_KEY is unset"
                )
            return OpenAIEmbedder(
                settings.embeddings.openai.model,
                api_key=settings.openai_api_key.get_secret_value(),
                base_url=settings.embeddings.openai.base_url,
                timeout=settings.embed_timeout_seconds,
                max_retries=settings.embed_max_retries,
            )
        if method == "ollama":
            return OllamaEmbedder(
                settings.embeddings.ollama.model,
                base_url=settings.embeddings.ollama.base_url,
                timeout=settings.embed_timeout_seconds,
                max_retries=settings.embed_max_retries,
            )
    except EmbedderExtraMissing:
        raise
    except ImportError as exc:
        extra = {
            "sentence-transformers": "local-embeddings",
            "openai": "openai",
        }.get(method, method)
        raise EmbedderExtraMissing(
            f"embeddings.method={method!r} requires the optional '[{extra}]' "
            f"extra, which is not installed: {exc}"
        ) from exc
    raise ValueError(
        f"embeddings.method={method!r} has no embedder (disabled/unknown)"
    )
