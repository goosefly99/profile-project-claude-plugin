from __future__ import annotations

from typing import TYPE_CHECKING

from profile_project.tools.artifact_tools import register_artifact_tools
from profile_project.tools.config_tools import register_config_tools
from profile_project.tools.dag_tools import register_dag_tools
from profile_project.tools.sources_tools import register_sources_tools
from profile_project.tools.vectorstore_tools import register_vectorstore_tools

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


def register_tools(mcp: FastMCP) -> None:
    register_config_tools(mcp)
    register_dag_tools(mcp)
    register_sources_tools(mcp)
    register_artifact_tools(mcp)
    register_vectorstore_tools(mcp)
