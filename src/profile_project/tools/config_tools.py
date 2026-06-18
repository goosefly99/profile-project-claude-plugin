from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from profile_project.config.files import (
    atomic_write_json,
    ensure_gitignore_entry,
    rewrite_root_prefix,
    transaction,
)
from profile_project.config.init_gate import (
    STAMP_DIRNAME,
    STAMP_FILENAME,
    build_init_stamp,
    detect_root_move,
    is_initialized,
    resolve_project_root,
)
from profile_project.config.provenance import (
    resolve_field,
    validate_config,
)
from profile_project.config.settings import CONFIG_FILENAME
from profile_project.config.sources import (
    FORBIDDEN_KEYS,
    ForbiddenSecretError,
    load_settings,
)
from profile_project.dag.run_state import PipelineError
from profile_project.tools._envelope import require_init, tool_envelope

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


_REQUIRED_SECRET_BY_SELECTOR: dict[tuple[str, str], str] = {
    ("embeddings.method", "openai"): "PROFILE_PROJECT_OPENAI_API_KEY",
    ("vectorstore.backend", "pinecone"): "PROFILE_PROJECT_PINECONE_API_KEY",
}

_SECRET_FIELDS: frozenset[str] = frozenset({"openai_api_key", "pinecone_api_key"})


def _assert_no_forbidden_secret(config: dict[str, object]) -> None:
    for key, value in config.items():
        if key in FORBIDDEN_KEYS:
            raise PipelineError(
                f"Forbidden secret key '{key}' present in config JSON; "
                f"set it via env (PROFILE_PROJECT_{key.upper()}) instead.",
                code="forbidden_secret",
                remedy="Move the secret out of the JSON and into the environment.",
            )
        if isinstance(value, dict):
            _assert_no_forbidden_secret(value)


def _required_env_secrets(config: dict[str, object]) -> list[str]:
    embeddings = config.get("embeddings")
    vectorstore = config.get("vectorstore")
    method = embeddings.get("method") if isinstance(embeddings, dict) else None
    backend = (
        vectorstore.get("backend") if isinstance(vectorstore, dict) else None
    )
    missing: list[str] = []
    for (selector, choice), env_var in _REQUIRED_SECRET_BY_SELECTOR.items():
        picked = method if selector == "embeddings.method" else backend
        if picked == choice and not os.environ.get(env_var):
            missing.append(env_var)
    return missing


def _nested_set(d: dict[str, object], dotted: str, value: object) -> None:
    parts = dotted.split(".")
    cursor: dict[str, object] = d
    for part in parts[:-1]:
        nxt = cursor.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[part] = nxt
        cursor = nxt
    cursor[parts[-1]] = value


def _mask_settings(settings: dict[str, object]) -> dict[str, object]:
    """Return a copy of the settings dict with secret fields removed."""
    return {k: v for k, v in settings.items() if k not in _SECRET_FIELDS}


def pp_config_path() -> dict[str, object]:
    root = resolve_project_root()
    return {
        "config_path": str(root / CONFIG_FILENAME),
        "initialized": is_initialized(root),
    }


def pp_config_show() -> dict[str, object]:
    root = resolve_project_root()
    result = validate_config(root)
    return {
        "settings": _mask_settings(result["settings"]),
        "vectorstore_enabled": result["vectorstore_enabled"],
    }


def pp_config_get(key: str) -> dict[str, object]:
    settings = load_settings(resolve_project_root())
    value, source = resolve_field(settings, key)
    return {"key": key, "value": value, "source": source}


def pp_config_validate(project_path: str | None = None) -> dict[str, object]:
    root = (
        Path(project_path) if project_path is not None else resolve_project_root()
    )
    return validate_config(root)


@tool_envelope
def pp_init_project(
    config: dict[str, object], force: bool = False
) -> dict[str, object]:
    root = resolve_project_root()
    config_path = root / CONFIG_FILENAME
    tree = root / STAMP_DIRNAME

    # 1. Candidate-level guards — no writes yet, so a refusal leaves zero residue.
    _assert_no_forbidden_secret(config)
    missing = _required_env_secrets(config)
    if missing:
        raise PipelineError(
            f"Required secret(s) not set in env: {', '.join(missing)}.",
            code="missing_secret",
            remedy="Export the listed PROFILE_PROJECT_* variable(s) and retry.",
        )

    moved, stamped_root = detect_root_move(root)
    created: list[str] = []

    # Snapshot the live config so an invalid candidate (or any later failure)
    # restores the prior file — the transaction only removes files it *creates*,
    # not an in-place overwrite of a pre-existing config.
    had_prior_config = config_path.exists()
    prior_config: object | None = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if had_prior_config
        else None
    )

    def _restore_prior_config() -> None:
        if had_prior_config and prior_config is not None:
            atomic_write_json(config_path, prior_config)
        else:
            config_path.unlink(missing_ok=True)

    validation: dict[str, Any]
    try:
        with transaction(root) as txn:
            # Stage the CANDIDATE, then validate the candidate now on disk.
            txn.write_json(config_path, config)
            try:
                validation = validate_config(root)
            except ForbiddenSecretError as exc:
                raise PipelineError(str(exc), code="forbidden_secret") from exc
            if not validation["ok"]:
                raise PipelineError(
                    "; ".join(validation["errors"]), code="invalid_config"
                )

            if not tree.is_dir():
                txn.mkdir(tree / "runs")
                txn.mkdir(tree / "artifacts")
                txn.mkdir(tree / "cache")
                txn.mkdir(tree / "chroma")
                created.append(".profile_project/")

            # Write the stamp through the transaction so a later failure rolls it
            # back too (raw write_init_stamp would leave residue in .profile_project).
            txn.write_json(
                tree / STAMP_FILENAME, build_init_stamp(root, config_path)
            )
            created.append(".initialized")

            if ensure_gitignore_entry(root, ".profile_project/"):
                created.append(".gitignore entry")

            if moved and force and stamped_root is not None:
                rewrite_root_prefix(tree / "runs", stamped_root, str(root))
    except (PipelineError, ForbiddenSecretError):
        _restore_prior_config()
        raise

    return {
        "config_path": str(config_path),
        "initialized": True,
        "created": created,
        "warnings": validation["warnings"],
    }


@tool_envelope
@require_init
def pp_config_set(key: str, value: object) -> dict[str, object]:
    if key in FORBIDDEN_KEYS:
        raise PipelineError(
            f"'{key}' is a secret; set it via env, not the JSON config.",
            code="forbidden_secret",
        )
    root = resolve_project_root()
    config_path = root / CONFIG_FILENAME
    with open(config_path, encoding="utf-8") as fh:
        current: dict[str, object] = json.load(fh)
    prior = copy.deepcopy(current)
    _nested_set(current, key, value)
    atomic_write_json(config_path, current)
    validation = validate_config(root)
    if not validation["ok"]:
        # Restore the prior config so a rejected set never corrupts the live file.
        atomic_write_json(config_path, prior)
        raise PipelineError(
            "; ".join(validation["errors"]), code="invalid_config"
        )
    return {"written": True, "warnings": validation["warnings"]}


def register_config_tools(mcp: FastMCP) -> None:
    mcp.tool()(pp_config_path)
    mcp.tool()(pp_config_show)
    mcp.tool()(pp_config_get)
    mcp.tool()(pp_config_set)
    mcp.tool()(pp_config_validate)
    mcp.tool()(pp_init_project)
