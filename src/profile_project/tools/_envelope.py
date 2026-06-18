from __future__ import annotations

import functools
from typing import Callable

import structlog

from profile_project.config.init_gate import (
    detect_root_move,
    is_initialized,
    not_initialized_error,
    project_root_moved_error,
    resolve_project_root,
)
from profile_project.dag.run_state import PipelineError

log = structlog.get_logger(__name__)


class ToolError(Exception):
    """A structured, tool-boundary error carrying an envelope code.

    Raised by handlers to signal a recoverable, user-facing condition
    (e.g. ``run_not_found``); rendered by ``tool_envelope`` into the
    ``{"ok": False, "error": {...}}`` shape without escaping the boundary.
    """

    def __init__(self, code: str, message: str, *, retriable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retriable = retriable


def tool_envelope(
    fn: Callable[..., dict[str, object]],
) -> Callable[..., dict[str, object]]:
    @functools.wraps(fn)
    def wrapper(*args: object, **kwargs: object) -> dict[str, object]:
        try:
            result = fn(*args, **kwargs)
        except PipelineError as exc:
            log.warning("tool_error", tool=fn.__name__, code=exc.code)
            return exc.envelope
        except ToolError as exc:
            log.warning("tool_error", tool=fn.__name__, code=exc.code)
            return {
                "ok": False,
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "retriable": exc.retriable,
                },
            }
        except Exception as exc:  # noqa: BLE001 - boundary must never raise out
            log.error("tool_internal_error", tool=fn.__name__, exc_info=exc)
            return {
                "ok": False,
                "error": {
                    "code": "internal_error",
                    "message": str(exc),
                    "retriable": False,
                },
            }
        return {"ok": True, **result}

    return wrapper


def require_init(
    fn: Callable[..., dict[str, object]],
) -> Callable[..., dict[str, object]]:
    @functools.wraps(fn)
    def wrapper(*args: object, **kwargs: object) -> dict[str, object]:
        root = resolve_project_root()
        moved, stamped_root = detect_root_move(root)
        if moved and stamped_root is not None:
            return project_root_moved_error(stamped_root, root)
        if not is_initialized(root):
            return not_initialized_error(root)
        return fn(*args, **kwargs)

    return wrapper
