from __future__ import annotations

import pydantic
import pytest

from profile_project.config.conflicts import (
    ConflictWarning,
    run_conflict_detection,
)
from profile_project.config.settings import Settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_extras_all_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate all optional extras present so C5 never fires spuriously."""
    monkeypatch.setattr(
        "profile_project.config.conflicts._extra_installed",
        lambda _name: True,
    )


def _patch_extras_missing(
    monkeypatch: pytest.MonkeyPatch, *missing_modules: str
) -> None:
    """Return False for the given module names, True for everything else."""
    missing = set(missing_modules)
    monkeypatch.setattr(
        "profile_project.config.conflicts._extra_installed",
        lambda name: name not in missing,
    )


# ---------------------------------------------------------------------------
# Structural test — ConflictWarning NamedTuple shape
# ---------------------------------------------------------------------------

def test_conflict_warning_is_named_tuple_with_expected_fields() -> None:
    w = ConflictWarning(
        code="C2",
        severity="warn",
        message="cannot embed",
        disables_vectorstore=True,
    )
    assert w.code == "C2"
    assert w.severity == "warn"
    assert w.message == "cannot embed"
    assert w.disables_vectorstore is True
    # NamedTuple fields, in order
    assert ConflictWarning._fields == (
        "code",
        "severity",
        "message",
        "disables_vectorstore",
    )


# ---------------------------------------------------------------------------
# C1 / C1b / C2 — credential-missing conflicts
# ---------------------------------------------------------------------------

def test_c1_pinecone_missing_key_warns_and_disables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_extras_all_installed(monkeypatch)
    s = Settings(
        vectorstore={"enabled": True, "backend": "pinecone",  # type: ignore[arg-type]
                     "pinecone": {"index": "my-idx"}},
        embeddings={"method": "sentence-transformers"},  # type: ignore[arg-type]
    )
    warnings, enabled = run_conflict_detection(s)
    assert enabled is False
    assert any("PROFILE_PROJECT_PINECONE_API_KEY" in w for w in warnings)


def test_c1b_pinecone_missing_index_warns_and_disables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_extras_all_installed(monkeypatch)
    s = Settings(
        vectorstore={"enabled": True, "backend": "pinecone", "pinecone": {}},  # type: ignore[arg-type]
        embeddings={"method": "sentence-transformers"},  # type: ignore[arg-type]
        pinecone_api_key="pc-secret",  # type: ignore[arg-type]
    )
    warnings, enabled = run_conflict_detection(s)
    assert enabled is False
    assert any("existing index ref" in w for w in warnings)


def test_c2_openai_missing_key_warns_and_disables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_extras_all_installed(monkeypatch)
    s = Settings(
        vectorstore={"enabled": True, "backend": "chromadb"},  # type: ignore[arg-type]
        embeddings={"method": "openai"},  # type: ignore[arg-type]
    )
    warnings, enabled = run_conflict_detection(s)
    assert enabled is False
    assert any("PROFILE_PROJECT_OPENAI_API_KEY" in w for w in warnings)
    assert any("cannot embed" in w for w in warnings)


# ---------------------------------------------------------------------------
# C3 — Ollama reachability
# ---------------------------------------------------------------------------

def test_c3_ollama_unreachable_probe_warns_and_disables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_extras_all_installed(monkeypatch)
    s = Settings(
        vectorstore={"enabled": True, "backend": "chromadb"},  # type: ignore[arg-type]
        embeddings={"method": "ollama",  # type: ignore[arg-type]
                    "ollama": {"base_url": "http://localhost:11434"}},
    )
    calls: list[tuple[str, float]] = []

    def probe(base_url: str, timeout: float) -> bool:
        calls.append((base_url, timeout))
        return False  # unreachable

    warnings, enabled = run_conflict_detection(s, ollama_probe=probe)
    assert enabled is False
    assert calls == [("http://localhost:11434", 30.0)]
    assert any("ollama" in w and "unreachable" in w for w in warnings)


def test_c3_ollama_not_probed_when_no_probe_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_extras_all_installed(monkeypatch)
    s = Settings(
        vectorstore={"enabled": True, "backend": "chromadb"},  # type: ignore[arg-type]
        embeddings={"method": "ollama",  # type: ignore[arg-type]
                    "ollama": {"base_url": "http://localhost:11434"}},
    )
    warnings, enabled = run_conflict_detection(s)  # no probe
    assert enabled is True
    assert not any("unreachable" in w for w in warnings)


# ---------------------------------------------------------------------------
# C4 — Dimension mismatch
# ---------------------------------------------------------------------------

def test_c4_dim_mismatch_warns_and_disables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_extras_all_installed(monkeypatch)
    s = Settings(
        vectorstore={"enabled": True, "backend": "pinecone",  # type: ignore[arg-type]
                     "pinecone": {"index": "my-idx",
                                  "embeddings_model": "text-embedding-3-small"}},
        embeddings={"method": "openai"},  # type: ignore[arg-type]
        openai_api_key="oa-secret",  # type: ignore[arg-type]
        pinecone_api_key="pc-secret",  # type: ignore[arg-type]
    )

    def dim_probe(settings: Settings, timeout: float) -> tuple[int, int] | None:
        # (effective_embedding_dim, index_dim) — geometry mismatch
        return (384, 1536)

    warnings, enabled = run_conflict_detection(s, dim_probe=dim_probe)
    assert enabled is False
    assert any("dimension" in w and "384" in w and "1536" in w for w in warnings)


def test_c4_dim_probe_fail_closed_disables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_extras_all_installed(monkeypatch)
    s = Settings(
        vectorstore={"enabled": True, "backend": "pinecone",  # type: ignore[arg-type]
                     "pinecone": {"index": "my-idx",
                                  "embeddings_model": "text-embedding-3-small"}},
        embeddings={"method": "openai"},  # type: ignore[arg-type]
        openai_api_key="oa-secret",  # type: ignore[arg-type]
        pinecone_api_key="pc-secret",  # type: ignore[arg-type]
    )

    def dim_probe(settings: Settings, timeout: float) -> tuple[int, int] | None:
        return None  # probe failed -> fail closed

    warnings, enabled = run_conflict_detection(s, dim_probe=dim_probe)
    assert enabled is False
    assert any("could not be verified" in w for w in warnings)


# ---------------------------------------------------------------------------
# C5 — Missing python extras (vectorstore backend + embeddings method)
# ---------------------------------------------------------------------------

def test_c5_missing_store_extra_warns_and_disables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_extras_missing(monkeypatch, "chromadb")
    s = Settings(
        vectorstore={"enabled": True, "backend": "chromadb"},  # type: ignore[arg-type]
        embeddings={"method": "sentence-transformers"},  # type: ignore[arg-type]
    )
    warnings, enabled = run_conflict_detection(s)
    assert enabled is False
    assert any("chromadb" in w and "not installed" in w for w in warnings)


def test_c5_missing_embedder_extra_warns_and_disables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C5b: sentence-transformers extra absent → warn + disable."""
    _patch_extras_missing(monkeypatch, "sentence_transformers")
    s = Settings(
        vectorstore={"enabled": True, "backend": "chromadb"},  # type: ignore[arg-type]
        embeddings={"method": "sentence-transformers"},  # type: ignore[arg-type]
    )
    warnings, enabled = run_conflict_detection(s)
    assert enabled is False
    assert any(
        "sentence-transformers" in w and "not installed" in w for w in warnings
    )


def test_c5_missing_openai_embedder_extra_warns_and_disables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C5b: openai extra absent (but key is present) → warn + disable."""
    _patch_extras_missing(monkeypatch, "openai")
    s = Settings(
        vectorstore={"enabled": True, "backend": "chromadb"},  # type: ignore[arg-type]
        embeddings={"method": "openai"},  # type: ignore[arg-type]
        openai_api_key="oa-secret",  # type: ignore[arg-type]
    )
    warnings, enabled = run_conflict_detection(s)
    assert enabled is False
    assert any("openai" in w and "not installed" in w for w in warnings)


def test_c5_ollama_embedder_has_no_extra_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ollama uses httpx (base dep); no C5 should fire for the embedder."""
    _patch_extras_all_installed(monkeypatch)
    s = Settings(
        vectorstore={"enabled": True, "backend": "chromadb"},  # type: ignore[arg-type]
        embeddings={"method": "ollama"},  # type: ignore[arg-type]
    )
    warnings, enabled = run_conflict_detection(s)
    assert enabled is True
    assert not any("not installed" in w for w in warnings)


# ---------------------------------------------------------------------------
# C6 / C7 / C8 / C9 — Advisory (non-disabling) conflicts
# ---------------------------------------------------------------------------

def test_c6_openai_base_url_non_openai_host_warns_keeps_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_extras_all_installed(monkeypatch)
    s = Settings(
        vectorstore={"enabled": True, "backend": "chromadb"},  # type: ignore[arg-type]
        embeddings={"method": "openai",  # type: ignore[arg-type]
                    "openai": {"base_url": "http://localhost:11434"}},
        openai_api_key="oa-secret",  # type: ignore[arg-type]
    )
    warnings, enabled = run_conflict_detection(s)
    assert enabled is True
    assert any("base_url" in w and "non-OpenAI" in w for w in warnings)


def test_c7_ollama_with_openai_key_warns_keeps_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_extras_all_installed(monkeypatch)
    s = Settings(
        vectorstore={"enabled": True, "backend": "chromadb"},  # type: ignore[arg-type]
        embeddings={"method": "ollama"},  # type: ignore[arg-type]
        openai_api_key="oa-secret",  # type: ignore[arg-type]
    )
    warnings, enabled = run_conflict_detection(s)
    assert enabled is True
    assert any("OpenAI key set but ollama selected" in w for w in warnings)


def test_c8_chromadb_with_pinecone_key_warns_keeps_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_extras_all_installed(monkeypatch)
    s = Settings(
        vectorstore={"enabled": True, "backend": "chromadb"},  # type: ignore[arg-type]
        embeddings={"method": "sentence-transformers"},  # type: ignore[arg-type]
        pinecone_api_key="pc-secret",  # type: ignore[arg-type]
    )
    warnings, enabled = run_conflict_detection(s)
    assert enabled is True
    assert any("pinecone key set but chromadb backend" in w for w in warnings)


def test_c9_disabled_with_nondefault_vectorstore_config_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_extras_all_installed(monkeypatch)
    s = Settings(
        vectorstore={"enabled": False, "backend": "pinecone",  # type: ignore[arg-type]
                     "pinecone": {"index": "my-idx"}},
        embeddings={"method": "sentence-transformers"},  # type: ignore[arg-type]
        pinecone_api_key="pc-secret",  # type: ignore[arg-type]
    )
    warnings, enabled = run_conflict_detection(s)
    assert enabled is False
    assert any("dead config" in w for w in warnings)


# ---------------------------------------------------------------------------
# Zero-conflict / clean-config paths
# ---------------------------------------------------------------------------

def test_zero_setup_default_no_conflicts_keeps_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_extras_missing(monkeypatch)  # nothing missing → all extras present
    s = Settings(
        vectorstore={"enabled": True, "backend": "chromadb"},  # type: ignore[arg-type]
        embeddings={"method": "sentence-transformers"},  # type: ignore[arg-type]
    )
    warnings, enabled = run_conflict_detection(s)
    assert enabled is True
    assert warnings == []


# ---------------------------------------------------------------------------
# Field-validation (not conflict matrix)
# ---------------------------------------------------------------------------

def test_c10_nonpositive_timeout_is_field_validation_error_not_conflict() -> None:
    # C10 is enforced SOLELY by Field(gt=0.0) at field-validation time; it never
    # reaches the conflict matrix.
    with pytest.raises(pydantic.ValidationError):
        Settings(
            vectorstore={"enabled": True, "backend": "chromadb"},  # type: ignore[arg-type]
            embeddings={"method": "sentence-transformers"},  # type: ignore[arg-type]
            embed_timeout_seconds=0.0,
        )


# ---------------------------------------------------------------------------
# Settings model integration — conflict_warnings / vectorstore_enabled_post
# ---------------------------------------------------------------------------

def test_settings_stores_conflict_warnings_and_post_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_extras_all_installed(monkeypatch)
    s = Settings(
        vectorstore={"enabled": True, "backend": "chromadb"},  # type: ignore[arg-type]
        embeddings={"method": "openai"},  # type: ignore[arg-type]  # C2: no openai key -> disable
    )
    assert s.vectorstore_enabled_post is False
    assert any("PROFILE_PROJECT_OPENAI_API_KEY" in w for w in s.conflict_warnings)
    # The raw nested flag is NOT mutated by the validator (post value lives on the
    # dedicated field); only the resolved post-conflict value reflects auto-disable.
    assert s.vectorstore.enabled is True


def test_settings_clean_config_has_no_warnings_and_post_enabled_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_extras_all_installed(monkeypatch)
    s = Settings(
        vectorstore={"enabled": True, "backend": "chromadb"},  # type: ignore[arg-type]
        embeddings={"method": "sentence-transformers"},  # type: ignore[arg-type]
    )
    assert s.vectorstore_enabled_post is True
    assert s.conflict_warnings == []
