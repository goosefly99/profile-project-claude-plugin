from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from profile_project.config.init_gate import is_initialized, resolve_project_root
from profile_project.config.sources import load_settings
from profile_project.dag.run_state import list_runs
from profile_project.sources.index import build_source_index
from profile_project.sources.manifest import add_manifest_source
from profile_project.tools._envelope import ToolError, require_init, tool_envelope

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


def _source_index(root: Path) -> dict[str, object]:
    return build_source_index(root, load_settings(root))


def _persist_source_index(root: Path, index: dict[str, object]) -> None:
    from profile_project.artifacts.store import store_artifact

    runs_root = root / ".profile_project" / "runs"
    states = list_runs(runs_root) if runs_root.is_dir() else []
    run_id = states[-1].run_id if states else ""
    stamped = {**index, "run_id": run_id}
    store_artifact(root, run_id, "discover_context", "source-index", stamped)


@tool_envelope
def pp_discover_sources(persist: bool = False) -> dict[str, object]:
    root = resolve_project_root()
    index = _source_index(root)
    if persist:
        # In-handler gate: re-assert the init predicate BEFORE any write (§6b.3).
        if not is_initialized(root):
            raise ToolError(
                "not_initialized",
                "profile-project is not initialized for this project. "
                "Run /profile-project:init first.",
            )
        _persist_source_index(root, index)
    return {"source_index": index}


@tool_envelope
def pp_list_sources(kind: str | None = None) -> dict[str, object]:
    root = resolve_project_root()
    index = _source_index(root)
    raw = index["sources"]
    sources = raw if isinstance(raw, list) else []
    if kind is not None:
        sources = [s for s in sources if s["kind"] == kind]
    return {"sources": sources, "counts": index["counts"]}


@tool_envelope
def pp_get_source(source_id: str) -> dict[str, object]:
    root = resolve_project_root()
    index = _source_index(root)
    raw = index["sources"]
    sources = raw if isinstance(raw, list) else []
    match = next((s for s in sources if s["source_id"] == source_id), None)
    return {"source": match}


@tool_envelope
@require_init
def pp_add_source(path_or_url: str, kind: str | None = None) -> dict[str, object]:
    root = resolve_project_root()
    source = add_manifest_source(root, path_or_url, kind)
    return {"source": source}


def register_sources_tools(mcp: FastMCP) -> None:
    mcp.tool()(pp_discover_sources)
    mcp.tool()(pp_list_sources)
    mcp.tool()(pp_get_source)
    mcp.tool()(pp_add_source)
