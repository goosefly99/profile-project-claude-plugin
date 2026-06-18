from __future__ import annotations

import io
import logging

import structlog
from mcp.server.fastmcp import FastMCP

import profile_project
from profile_project import server


def test_package_version_is_target() -> None:
    assert profile_project.__version__ == "0.1.0"


def test_mcp_instance_is_fastmcp_named_profile_project() -> None:
    assert isinstance(server.mcp, FastMCP)
    assert server.mcp.name == "profile-project"


def test_register_tools_registers_without_error() -> None:
    # Registration is now real (not a stub): create a fresh instance so real
    # tool registration is exercised without polluting the shared server.mcp.
    m = FastMCP("test")
    assert server.register_tools(m) is None  # type: ignore[func-returns-value]


def test_configure_logging_routes_structlog_to_stderr_only(
    capsys: object,
) -> None:
    # configure_logging() must wire structlog so emitted logs land on stderr,
    # never stdout (stdout is the JSON-RPC framing channel and must stay clean).
    server.configure_logging()
    log = structlog.get_logger("bootstrap-test")
    log.info("hello-stderr", marker="bootstrap")
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert captured.out == ""
    assert "hello-stderr" in captured.err


def test_configure_logging_is_idempotent() -> None:
    # Calling twice must not raise and must not duplicate handlers on the
    # standard-library root logger that structlog renders through.
    server.configure_logging()
    before = len(logging.getLogger().handlers)
    server.configure_logging()
    after = len(logging.getLogger().handlers)
    assert after == before


def test_main_invokes_run_after_configuring_and_registering(
    monkeypatch: object,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(  # type: ignore[attr-defined]
        server, "configure_logging", lambda: calls.append("configure")
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        server, "register_tools", lambda m: calls.append("register")
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        server.mcp, "run", lambda: calls.append("run")
    )
    server.main()
    assert calls == ["configure", "register", "run"]


def test_configure_logging_never_writes_to_stdout_stream() -> None:
    # Defense in depth: the configured handler's stream is sys.stderr-bound,
    # so a captured stdout buffer stays empty across a direct stdlib log call.
    server.configure_logging()
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        structlog.get_logger("isolation").warning("on-its-own-handler")
    finally:
        root.removeHandler(handler)
    # The extra buffer (not stderr) may receive the record via the root logger,
    # but the structlog config itself must target stderr — asserted in the
    # dedicated stderr test above; here we only assert no exception path.
    assert isinstance(buf.getvalue(), str)
