from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

from profile_project.config.settings import CONFIG_FILENAME

FORBIDDEN_KEYS: frozenset[str] = frozenset(
    {
        "openai_api_key",
        "api_key",
        "pinecone_api_key",
        "OPENAI_API_KEY",
        "PINECONE_API_KEY",
    }
)


class ForbiddenSecretError(ValueError):
    """Raised when a secret key appears anywhere in the project JSON config."""


def assert_no_forbidden_keys(raw: dict[str, Any], _path: str = "") -> None:
    """Reject secrets in project JSON, recursing into nested objects.

    Secrets are environment-only; any forbidden segment anywhere in the
    config raises a loud :class:`ForbiddenSecretError` naming the key and its
    flat env equivalent.
    """
    for key, value in raw.items():
        if key in FORBIDDEN_KEYS:
            flat = f"PROFILE_PROJECT_{key.upper()}"
            raise ForbiddenSecretError(
                f"Forbidden secret key '{key}' found in {CONFIG_FILENAME} "
                f"(at '{_path or key}'). Secrets are environment-only; set "
                f"'{flat}' in the environment, not in project JSON."
            )
        if isinstance(value, dict):
            child = f"{_path}.{key}" if _path else key
            assert_no_forbidden_keys(value, child)


def nested_set(flat: dict[str, object]) -> dict[str, Any]:
    """Expand a possibly-dotted/nested dict into a fully nested dict.

    Top-level keys may be dotted (``"vectorstore.backend"``) to line up with
    the nested sub-models and the ``__`` env delimiter; already-nested values
    are merged recursively.
    """
    result: dict[str, Any] = {}
    for dotted_key, value in flat.items():
        parts = dotted_key.split(".")
        cursor = result
        for part in parts[:-1]:
            existing = cursor.get(part)
            if not isinstance(existing, dict):
                existing = {}
                cursor[part] = existing
            cursor = existing
        if isinstance(value, dict):
            existing_leaf = cursor.get(parts[-1])
            if isinstance(existing_leaf, dict):
                existing_leaf.update(nested_set(value))
            else:
                cursor[parts[-1]] = nested_set(value)
        else:
            cursor[parts[-1]] = value
    return result


class ProjectJsonConfigSettingsSource(PydanticBaseSettingsSource):
    """Settings source reading .profile_project_config.json at the project root.

    Returns a nested dict (built via :func:`nested_set`) so it lines up with the
    nested sub-models and the ``__`` env delimiter. Sits above env in
    precedence, so project JSON overrides env defaults.
    """

    def __init__(
        self, settings_cls: type[BaseSettings], project_root: Path
    ) -> None:
        super().__init__(settings_cls)
        self.project_root = project_root
        self._data: dict[str, Any] = self._read()

    def _read(self) -> dict[str, Any]:
        path = self.project_root / CONFIG_FILENAME
        if not path.is_file():
            return {}
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(
                f"{CONFIG_FILENAME} must contain a JSON object, "
                f"got {type(raw).__name__}."
            )
        assert_no_forbidden_keys(raw)
        return nested_set(raw)

    def get_field_value(
        self, field: Any, field_name: str
    ) -> tuple[Any, str, bool]:
        if field_name in self._data:
            return self._data[field_name], field_name, False
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        return self._data


def load_settings(project_root: Path) -> Any:
    """Construct Settings for ``project_root`` with project JSON above env."""
    # Lazy import to break circular dependency:
    # sources.py -> settings.py (CONFIG_FILENAME)
    # settings.py -> sources.py (ProjectJsonConfigSettingsSource)
    from profile_project.config.settings import (  # noqa: PLC0415
        _PROJECT_ROOT,
    )
    from profile_project.config.settings import (
        Settings as _Settings,
    )

    token = _PROJECT_ROOT.set(project_root)
    try:
        return _Settings()
    finally:
        _PROJECT_ROOT.reset(token)
