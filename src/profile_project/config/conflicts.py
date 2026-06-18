from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, NamedTuple

if TYPE_CHECKING:
    from profile_project.config.settings import Settings


class ConflictWarning(NamedTuple):
    code: str
    severity: Literal["warn"]
    message: str
    disables_vectorstore: bool


# (base_url, timeout_seconds) -> reachable
OllamaProbe = Callable[[str, float], bool]
# (settings, timeout_seconds) -> (effective_embedding_dim, index_dim) or None on failure
DimProbe = Callable[["Settings", float], "tuple[int, int] | None"]

_BACKEND_EXTRA_MODULE: dict[str, str] = {
    "chromadb": "chromadb",
    "pinecone": "pinecone",
}

_EMBEDDINGS_EXTRA_MODULE: dict[str, str] = {
    "sentence-transformers": "sentence_transformers",
    "openai": "openai",
}

_OPENAI_HOSTS = ("api.openai.com",)


def _extra_installed(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def run_conflict_detection(
    settings: Settings,
    *,
    ollama_probe: OllamaProbe | None = None,
    dim_probe: DimProbe | None = None,
) -> tuple[list[str], bool]:
    """Run the §6.5 conflict matrix.

    Returns (warning_messages, vectorstore_enabled_post). Never raises:
    secrets-in-JSON and structurally-invalid config are rejected earlier (the
    field/source layer), so this detector only ever warns + disables.
    """
    warnings: list[ConflictWarning] = []
    vs = settings.vectorstore
    emb = settings.embeddings
    enabled = vs.enabled

    # C1: pinecone backend but missing PINECONE_API_KEY -> warn+disable
    if enabled and vs.backend == "pinecone" and settings.pinecone_api_key is None:
        warnings.append(ConflictWarning(
            "C1", "warn",
            "vectorstore.backend='pinecone' but PROFILE_PROJECT_PINECONE_API_KEY "
            "is unset; disabling vectorstore (cannot connect).",
            True,
        ))

    # C1b: pinecone backend but missing index ref -> warn+disable
    if enabled and vs.backend == "pinecone" and not vs.pinecone.index:
        warnings.append(ConflictWarning(
            "C1b", "warn",
            "vectorstore.backend='pinecone' but no existing index ref configured; "
            "disabling vectorstore (index ref required; never auto-created).",
            True,
        ))

    # C2: openai embeddings but missing OPENAI_API_KEY -> warn+disable
    if enabled and emb.method == "openai" and settings.openai_api_key is None:
        warnings.append(ConflictWarning(
            "C2", "warn",
            "embeddings.method='openai' but PROFILE_PROJECT_OPENAI_API_KEY is unset; "
            "disabling vectorstore (cannot embed).",
            True,
        ))

    # C3: ollama unreachable (probe-gated, bounded by embed_timeout_seconds)
    if enabled and emb.method == "ollama" and ollama_probe is not None:
        reachable = ollama_probe(emb.ollama.base_url, settings.embed_timeout_seconds)
        if not reachable:
            warnings.append(ConflictWarning(
                "C3", "warn",
                f"embeddings.method='ollama' but the endpoint "
                f"'{emb.ollama.base_url}' is unreachable; disabling vectorstore "
                "until reachable.",
                True,
            ))

    # C4: dimension mismatch (probe-gated, fail-closed)
    if enabled and vs.backend == "pinecone" and dim_probe is not None:
        dims = dim_probe(settings, settings.embed_timeout_seconds)
        if dims is None:
            warnings.append(ConflictWarning(
                "C4", "warn",
                "the Pinecone index dimension could not be verified (probe failed "
                "or timed out); disabling vectorstore (fail closed).",
                True,
            ))
        else:
            embed_dim, index_dim = dims
            if embed_dim != index_dim:
                warnings.append(ConflictWarning(
                    "C4", "warn",
                    f"embedding dimension {embed_dim} != existing Pinecone index "
                    f"dimension {index_dim}; disabling vectorstore (geometry "
                    "mismatch).",
                    True,
                ))

    # C5: required python extra not installed -> warn+disable
    # C5a: check vectorstore backend extra
    extra_mod = _BACKEND_EXTRA_MODULE.get(vs.backend)
    if enabled and extra_mod is not None and not _extra_installed(extra_mod):
        warnings.append(ConflictWarning(
            "C5", "warn",
            f"vectorstore.backend='{vs.backend}' but the '{extra_mod}' python "
            "extra is not installed; disabling vectorstore (dependency missing).",
            True,
        ))
    # C5b: check embeddings method extra (only when vectorstore still enabled)
    emb_mod = _EMBEDDINGS_EXTRA_MODULE.get(emb.method)
    if enabled and emb_mod is not None and not _extra_installed(emb_mod):
        warnings.append(ConflictWarning(
            "C5", "warn",
            f"embeddings.method='{emb.method}' but the '{emb_mod}' python "
            "extra is not installed; disabling vectorstore (dependency missing).",
            True,
        ))

    # C6: openai method but non-OpenAI base_url -> advisory (keep enabled)
    if emb.method == "openai" and emb.openai.base_url is not None \
            and not any(h in emb.openai.base_url for h in _OPENAI_HOSTS):
        warnings.append(ConflictWarning(
            "C6", "warn",
            f"embeddings.method='openai' but openai.base_url "
            f"'{emb.openai.base_url}' is a non-OpenAI host; provider/base_url "
            "mismatch (kept enabled; provenance flagged).",
            False,
        ))

    # C7: OpenAI key set but ollama selected -> advisory (keep enabled)
    if emb.method == "ollama" and settings.openai_api_key is not None:
        warnings.append(ConflictWarning(
            "C7", "warn",
            "OpenAI key set but ollama selected as the embeddings method; the "
            "OpenAI key is ignored.",
            False,
        ))

    # C8: pinecone key set but chromadb backend -> advisory (keep enabled)
    if vs.backend == "chromadb" and settings.pinecone_api_key is not None:
        warnings.append(ConflictWarning(
            "C8", "warn",
            "pinecone key set but chromadb backend selected; the pinecone key is "
            "unused.",
            False,
        ))

    # C9: vectorstore disabled but non-default config set -> advisory
    if not vs.enabled and (vs.backend == "pinecone" or vs.pinecone.index):
        warnings.append(ConflictWarning(
            "C9", "warn",
            "vectorstore.enabled=False but non-default vectorstore.* config is "
            "set; dead config.",
            False,
        ))

    if any(w.disables_vectorstore for w in warnings):
        enabled = False
    return [w.message for w in warnings], enabled
