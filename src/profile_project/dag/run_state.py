from __future__ import annotations


class PipelineError(Exception):
    """Structured pipeline/tool-boundary error carrying a result envelope.

    Rendered by ``tool_envelope`` into the ``{"ok": False, "error": {...}}``
    shape via the read-only :pyattr:`envelope` property.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str,
        remedy: str = "",
        retriable: bool = False,
        **extra: object,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.remedy = remedy
        self.retriable = retriable
        self.extra: dict[str, object] = dict(extra)

    @property
    def envelope(self) -> dict[str, object]:
        return {
            "ok": False,
            "error": {
                "code": self.code,
                "message": self.message,
                "retriable": self.retriable,
                "remedy": self.remedy,
                **self.extra,
            },
        }
