from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError
from pydantic_settings.sources import EnvSettingsSource

from profile_project.config.conflicts import run_conflict_detection
from profile_project.config.init_gate import is_initialized, resolve_project_root
from profile_project.config.settings import (
    CONFIG_FILENAME,
    Settings,
    _TopLevelAliasSource,
)
from profile_project.config.sources import (
    ForbiddenSecretError,
    ProjectJsonConfigSettingsSource,
    load_settings,
)

# All dotted leaf fields tracked for provenance.
_PROVENANCE_FIELDS: tuple[str, ...] = (
    "embeddings.method",
    "vectorstore.backend",
    "vectorstore.enabled",
    "vectorstore.collection",
)


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    """Recursively flatten a nested dict to dotted keys."""
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten(child_prefix, child, out)
    else:
        out[prefix] = value


def _json_leaves(project_root: Path) -> dict[str, Any]:
    """Return flattened dotted-key leaves from the project JSON config."""
    cfg = project_root / CONFIG_FILENAME
    if not cfg.exists():
        return {}
    source = ProjectJsonConfigSettingsSource(Settings, project_root)
    leaves: dict[str, Any] = {}
    _flatten("", source(), leaves)
    return leaves


def _env_leaves() -> dict[str, Any]:
    """Return flattened dotted-key leaves from env (nested delimiter + alias source).

    We consult BOTH sources that contribute env-based overrides:
    - ``EnvSettingsSource`` handles ``PROFILE_PROJECT_EMBEDDINGS__METHOD`` (and
      other ``__``-delimited vars).
    - ``_TopLevelAliasSource`` handles the non-standard flat aliases such as
      ``PROFILE_PROJECT_DEFAULT_EMBEDDINGS_METHOD`` and
      ``PROFILE_PROJECT_CHROMADB__PATH``.

    The alias source is checked **after** the standard env source so its leaves
    only appear in provenance when the standard env var is not set (mirroring the
    actual source precedence: init > project_json > alias_source > env_source).
    When both set the same leaf the standard env source wins for provenance
    purposes (the alias source is lower precedence in the real source stack).
    """
    # Standard nested-delimiter env source
    env_source: EnvSettingsSource = EnvSettingsSource(
        Settings,
        env_prefix="PROFILE_PROJECT_",
        env_nested_delimiter="__",
    )
    leaves: dict[str, Any] = {}
    _flatten("", env_source(), leaves)

    # Top-level alias source (lower precedence, so only fills in missing leaves)
    alias_source = _TopLevelAliasSource(Settings)
    alias_leaves: dict[str, Any] = {}
    _flatten("", alias_source(), alias_leaves)
    for key, val in alias_leaves.items():
        if key not in leaves:
            leaves[key] = val

    return leaves


def compute_provenance(project_root: Path) -> dict[str, str]:
    """Return a dotted-leaf-name -> origin label mapping for tracked fields.

    Origin labels: ``"project_json"`` | ``"env"`` | ``"default"``.

    Precedence (highest → lowest): project JSON > env > default.
    Env attribution covers both the standard nested-delimiter source and the
    custom ``_TopLevelAliasSource`` (e.g.
    ``PROFILE_PROJECT_DEFAULT_EMBEDDINGS_METHOD``).
    """
    json_leaves = _json_leaves(project_root)
    env_leaves = _env_leaves()
    provenance: dict[str, str] = {}
    for field in _PROVENANCE_FIELDS:
        if field in json_leaves:
            provenance[field] = "project_json"
        elif field in env_leaves:
            provenance[field] = "env"
        else:
            provenance[field] = "default"
    return provenance


def resolve_field(settings: Settings, dotted_key: str) -> tuple[object, str]:
    """Resolve a single dotted leaf to ``(value, source)``.

    ``value`` is read off the constructed ``Settings`` (post-layering); ``source``
    is the provenance origin label from :func:`compute_provenance` for the same
    leaf, recomputed for the settings' project root.
    """
    obj: object = settings
    for part in dotted_key.split("."):
        obj = getattr(obj, part)
    project_root = settings.project_dir or Path.cwd()
    source = compute_provenance(project_root).get(dotted_key, "default")
    return obj, source


def validate_config(project_root: Path) -> dict[str, Any]:
    """Validate the project configuration and return the §6.6 shape.

    Returns::

        {
            "ok": bool,
            "config_path": str,
            "initialized": bool,
            "settings": dict,          # Settings.model_dump() with masked secrets
            "provenance": dict[str, str],
            "warnings": list[str],
            "vectorstore_enabled": bool,
            "errors": list[str],
        }

    Hard failures (``ok=False``): ``ForbiddenSecretError`` (secret in JSON) or
    ``pydantic.ValidationError`` (structurally invalid config).  Everything else
    is a soft warning that leaves ``ok=True``.

    C11: if the resolved project root does not exist, warn + disable vectorstore
    (``ok`` stays ``True``).
    """
    config_path = str(project_root / CONFIG_FILENAME)
    initialized = is_initialized(project_root)

    try:
        settings: Settings = load_settings(project_root)
    except (ForbiddenSecretError, ValueError, ValidationError) as exc:
        return {
            "ok": False,
            "config_path": config_path,
            "initialized": initialized,
            "settings": {},
            "provenance": {},
            "warnings": [],
            "vectorstore_enabled": False,
            "errors": [str(exc)],
        }

    warnings, vectorstore_enabled = run_conflict_detection(settings)

    # C11: the resolved project root must exist; validate_config holds the root.
    resolved_root = resolve_project_root(settings)
    if not resolved_root.exists():
        warnings.append(
            f"resolved project root {resolved_root} does not exist; "
            f"disabling vectorstore (no project to index)."
        )
        vectorstore_enabled = False

    return {
        "ok": True,
        "config_path": config_path,
        "initialized": initialized,
        "settings": settings.model_dump(mode="json"),
        "provenance": compute_provenance(project_root),
        "warnings": warnings,
        "vectorstore_enabled": vectorstore_enabled,
        "errors": [],
    }
