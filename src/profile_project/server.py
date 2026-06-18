from __future__ import annotations

import logging
import sys

import structlog
from mcp.server.fastmcp import FastMCP

mcp: FastMCP = FastMCP("profile-project")

_LOG_HANDLER_TAG = "profile_project_stderr"


def configure_logging() -> None:
    """Configure structlog so all logs go to sys.stderr only.

    stdout is the JSON-RPC framing channel under stdio transport; writing
    anything to stdout corrupts the protocol stream (spec sec.4/sec.15).
    Idempotent: a tagged StreamHandler(sys.stderr) is installed at most once.
    """
    root = logging.getLogger()
    # Only this module's own tagged handler is de-duplicated; any pre-existing
    # foreign StreamHandler(sys.stderr) is intentionally left in place
    # (stderr is protocol-safe; we do not remove other libraries' handlers).
    already_installed = any(
        getattr(h, "_profile_project_tag", None) == _LOG_HANDLER_TAG
        for h in root.handlers
    )
    if not already_installed:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler._profile_project_tag = _LOG_HANDLER_TAG  # type: ignore[attr-defined]
        handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer(),
            )
        )
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )


def register_tools(mcp: FastMCP) -> None:
    """Register every pp_* tool onto the given FastMCP instance."""
    from profile_project.tools import register_tools as _register_tools  # noqa: PLC0415
    _register_tools(mcp)


def main() -> None:
    """Entry point: configure logging, register tools, run over stdio."""
    configure_logging()
    register_tools(mcp)
    mcp.run()
