from __future__ import annotations

import sys

from .core import build_widget


def main() -> int:
    widget = build_widget("demo")
    return 0 if widget.write("hello") else 1


if __name__ == "__main__":
    sys.exit(main())
