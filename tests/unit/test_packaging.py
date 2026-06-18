from __future__ import annotations

from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[2]


def test_mcp_json_mirror_is_byte_identical() -> None:
    root_mcp = PLUGIN_ROOT / ".mcp.json"
    mirror_mcp = PLUGIN_ROOT / ".claude-plugin" / ".mcp.json"
    assert root_mcp.exists()
    assert mirror_mcp.exists()
    assert root_mcp.read_bytes() == mirror_mcp.read_bytes()
