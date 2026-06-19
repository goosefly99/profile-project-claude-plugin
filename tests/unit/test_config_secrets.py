from __future__ import annotations

import pytest

from profile_project.config.settings import Settings


def test_secrets_masked_in_model_dump_python_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """model_dump() (python mode) must never expose real secret values."""
    monkeypatch.setenv("PROFILE_PROJECT_OPENAI_API_KEY", "sk-real-secret-value")
    monkeypatch.setenv("PROFILE_PROJECT_PINECONE_API_KEY", "pc-real-secret-value")
    settings = Settings()
    dumped = settings.model_dump()
    assert dumped["openai_api_key"] == "**********"
    assert dumped["pinecone_api_key"] == "**********"
    assert "sk-real-secret-value" not in repr(dumped)
    assert "pc-real-secret-value" not in repr(dumped)


def test_secrets_masked_in_model_dump_json_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """model_dump(mode='json') must also never expose real secret values."""
    monkeypatch.setenv("PROFILE_PROJECT_OPENAI_API_KEY", "sk-real-secret-value")
    monkeypatch.setenv("PROFILE_PROJECT_PINECONE_API_KEY", "pc-real-secret-value")
    settings = Settings()
    dumped = settings.model_dump(mode="json")
    assert dumped["openai_api_key"] == "**********"
    assert dumped["pinecone_api_key"] == "**********"
    assert "sk-real-secret-value" not in repr(dumped)
    assert "pc-real-secret-value" not in repr(dumped)


def test_secrets_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """When secrets are not configured, model_dump() returns None for both fields."""
    # Environment-independent: clear any exported secret vars before constructing.
    monkeypatch.delenv("PROFILE_PROJECT_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PROFILE_PROJECT_PINECONE_API_KEY", raising=False)
    settings = Settings()
    dumped = settings.model_dump()
    assert dumped["openai_api_key"] is None
    assert dumped["pinecone_api_key"] is None


def test_secrets_none_when_unset_json_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same None-when-unset invariant holds for json-mode dumps."""
    # Environment-independent: clear any exported secret vars before constructing.
    monkeypatch.delenv("PROFILE_PROJECT_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PROFILE_PROJECT_PINECONE_API_KEY", raising=False)
    settings = Settings()
    dumped = settings.model_dump(mode="json")
    assert dumped["openai_api_key"] is None
    assert dumped["pinecone_api_key"] is None
