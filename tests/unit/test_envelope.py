from __future__ import annotations

from pathlib import Path

from profile_project.dag.run_state import PipelineError
from profile_project.tools import _envelope
from profile_project.tools._envelope import ToolError, tool_envelope


def test_tool_envelope_wraps_success() -> None:
    @tool_envelope
    def handler(x: int) -> dict[str, object]:
        return {"value": x + 1}

    assert handler(1) == {"ok": True, "value": 2}


def test_tool_envelope_renders_pipeline_error() -> None:
    @tool_envelope
    def handler() -> dict[str, object]:
        raise PipelineError("no vectors yet", code="index_empty", retriable=True)

    result = handler()
    assert result == {
        "ok": False,
        "error": {
            "code": "index_empty",
            "message": "no vectors yet",
            "retriable": True,
            "remedy": "",
        },
    }


def test_tool_envelope_renders_tool_error() -> None:
    @tool_envelope
    def handler() -> dict[str, object]:
        raise ToolError("run_not_found", "no such run", retriable=False)

    result = handler()
    assert result == {
        "ok": False,
        "error": {
            "code": "run_not_found",
            "message": "no such run",
            "retriable": False,
        },
    }


def test_tool_envelope_renders_unexpected_exception() -> None:
    @tool_envelope
    def handler() -> dict[str, object]:
        raise ValueError("boom")

    result = handler()
    assert result["ok"] is False
    error = result["error"]
    assert isinstance(error, dict)
    assert error["code"] == "internal_error"
    assert error["retriable"] is False


def _patch_gate(
    monkeypatch: object,
    *,
    resolved_root: str,
    initialized: bool,
    moved: bool = False,
    stamped_root: str | None = None,
) -> None:
    monkeypatch.setattr(  # type: ignore[attr-defined]
        _envelope,
        "resolve_project_root",
        lambda settings=None: Path(resolved_root),
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        _envelope, "detect_root_move", lambda root: (moved, stamped_root)
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        _envelope, "is_initialized", lambda root: initialized
    )


def test_require_init_refuses_not_initialized(monkeypatch: object) -> None:
    _patch_gate(monkeypatch, resolved_root="/abs/proj", initialized=False)

    @_envelope.require_init
    def handler() -> dict[str, object]:
        raise AssertionError("handler must not run pre-init")

    result = handler()
    assert result["ok"] is False
    err = result["error"]
    assert isinstance(err, dict)
    assert err["code"] == "not_initialized"
    assert err["resolved_root"] == str(Path("/abs/proj"))
    assert err["remedy"] == "/profile-project:init"
    assert err["retriable"] is False


def test_require_init_refuses_moved_root(monkeypatch: object) -> None:
    _patch_gate(
        monkeypatch,
        resolved_root="/new/proj",
        initialized=False,
        moved=True,
        stamped_root="/old/proj",
    )

    @_envelope.require_init
    def handler() -> dict[str, object]:
        raise AssertionError("handler must not run after a move")

    result = handler()
    err = result["error"]
    assert isinstance(err, dict)
    assert err["code"] == "project_root_moved"
    assert err["stamped_root"] == "/old/proj"
    assert err["resolved_root"] == str(Path("/new/proj"))
    assert err["remedy"] == "/profile-project:init --reinit"


def test_require_init_passes_through_when_initialized(monkeypatch: object) -> None:
    _patch_gate(monkeypatch, resolved_root="/abs/proj", initialized=True)

    @_envelope.require_init
    def handler() -> dict[str, object]:
        return {"ran": True}

    assert handler() == {"ran": True}
