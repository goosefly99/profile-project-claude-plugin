# Widget Architecture

The widget package exposes a single `Widget` class plus a `build_widget` factory.

## Write reliability

Widget retries failed writes up to three times using exponential backoff.
The retry count defaults to three and is configurable per instance.
