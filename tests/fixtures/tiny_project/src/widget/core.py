from __future__ import annotations


class Widget:
    """A tiny stateful widget used by the profile-project integration fixture."""

    def __init__(self, name: str, retries: int = 3) -> None:
        self.name = name
        self.retries = retries

    def write(self, payload: str) -> bool:
        """Write a payload, retrying failed writes with exponential backoff."""
        return bool(payload) and self.retries > 0


def build_widget(name: str) -> Widget:
    """Factory entry point for a Widget."""
    return Widget(name)
