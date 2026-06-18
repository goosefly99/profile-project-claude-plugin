from __future__ import annotations

import json
from pathlib import Path

import pytest

from profile_project.config.settings import CONFIG_FILENAME, Settings
from profile_project.config.sources import (
    FORBIDDEN_KEYS,
    ForbiddenSecretError,
    ProjectJsonConfigSettingsSource,
    load_settings,
    nested_set,
)


def _write_config(root: Path, payload: dict[str, object]) -> Path:
    path = root / CONFIG_FILENAME
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_source_returns_nested_dict(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        {"vectorstore": {"backend": "pinecone", "collection": "from-json"}},
    )
    source = ProjectJsonConfigSettingsSource(Settings, tmp_path)
    result = source()
    assert result == {
        "vectorstore": {"backend": "pinecone", "collection": "from-json"}
    }


def test_source_missing_file_returns_empty(tmp_path: Path) -> None:
    source = ProjectJsonConfigSettingsSource(Settings, tmp_path)
    assert source() == {}


def test_nested_set_expands_dotted_keys() -> None:
    assert nested_set({"vectorstore.backend": "pinecone"}) == {
        "vectorstore": {"backend": "pinecone"}
    }


def test_top_level_forbidden_key_raises(tmp_path: Path) -> None:
    _write_config(tmp_path, {"openai_api_key": "sk-should-never-be-here"})
    with pytest.raises(ForbiddenSecretError) as exc:
        ProjectJsonConfigSettingsSource(Settings, tmp_path)
    msg = str(exc.value)
    assert "openai_api_key" in msg
    assert "PROFILE_PROJECT_OPENAI_API_KEY" in msg
    assert "environment-only" in msg


def test_nested_forbidden_key_raises(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        {"vectorstore": {"pinecone": {"pinecone_api_key": "leaked"}}},
    )
    with pytest.raises(ForbiddenSecretError) as exc:
        ProjectJsonConfigSettingsSource(Settings, tmp_path)
    assert "pinecone_api_key" in str(exc.value)


def test_forbidden_secret_error_is_value_error() -> None:
    assert issubclass(ForbiddenSecretError, ValueError)


def test_forbidden_keys_constant_membership() -> None:
    assert FORBIDDEN_KEYS == frozenset(
        {
            "openai_api_key",
            "api_key",
            "pinecone_api_key",
            "OPENAI_API_KEY",
            "PINECONE_API_KEY",
        }
    )


def test_settings_customise_sources_order(tmp_path: Path) -> None:
    names: list[str] = []

    class _Probe(Settings):  # type: ignore[misc]
        @classmethod
        def settings_customise_sources(  # type: ignore[no-untyped-def]
            cls,
            settings_cls,
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        ):
            sources = Settings.settings_customise_sources(
                settings_cls,
                init_settings,
                env_settings,
                dotenv_settings,
                file_secret_settings,
            )
            names.extend(type(s).__name__ for s in sources)
            return sources

    _Probe()
    assert names[0] == "InitSettingsSource"
    assert names[1] == "ProjectJsonConfigSettingsSource"
    # Task 2's _TopLevelAliasSource sits at index 2 (between project-JSON and env)
    # so that init > project-JSON > alias-env > env > dotenv > secrets.
    assert names[2] == "_TopLevelAliasSource"
    assert "EnvSettingsSource" in names[3]


def test_json_overrides_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROFILE_PROJECT_VECTORSTORE__COLLECTION", "from-env")
    _write_config(
        tmp_path, {"vectorstore": {"collection": "from-json"}}
    )
    settings = load_settings(tmp_path)
    assert settings.vectorstore.collection == "from-json"


def test_env_used_when_json_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PROFILE_PROJECT_VECTORSTORE__COLLECTION", "from-env")
    settings = load_settings(tmp_path)
    assert settings.vectorstore.collection == "from-env"
