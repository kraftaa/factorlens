#!/usr/bin/env python3
"""Unit tests for scripts/mcp/factorlens_mcp_server.py without external mcp dependency."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


class _DummyFastMCP:
    def __init__(self, *args, **kwargs):
        self.settings = types.SimpleNamespace(
            host=kwargs.get("host", "127.0.0.1"),
            port=kwargs.get("port", 8000),
            streamable_http_path=kwargs.get("streamable_http_path", "/mcp"),
        )

    def tool(self):
        def _decorator(fn):
            return fn

        return _decorator

    def run(self, *args, **kwargs):
        return None


def _load_module():
    mod_mcp = types.ModuleType("mcp")
    mod_server = types.ModuleType("mcp.server")
    mod_fastmcp = types.ModuleType("mcp.server.fastmcp")
    mod_fastmcp.FastMCP = _DummyFastMCP

    sys.modules["mcp"] = mod_mcp
    sys.modules["mcp.server"] = mod_server
    sys.modules["mcp.server.fastmcp"] = mod_fastmcp

    target = Path(__file__).resolve().parent / "factorlens_mcp_server.py"
    spec = importlib.util.spec_from_file_location("factorlens_mcp_server", target)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


class PeriodFlagsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module()

    def test_append_period_flags_appends_all_non_empty_args(self):
        cmd: list[str] = ["analyze"]
        self.mod._append_period_flags(
            cmd,
            date_column="date",
            time_grain="month",
            period="last",
            anchor_date="2026-03-31",
            current_start="2026-03-01",
            current_end="2026-03-31",
            previous_start="2026-02-01",
            previous_end="2026-02-28",
        )
        self.assertEqual(
            cmd,
            [
                "analyze",
                "--date-column",
                "date",
                "--time-grain",
                "month",
                "--period",
                "last",
                "--anchor-date",
                "2026-03-31",
                "--current-start",
                "2026-03-01",
                "--current-end",
                "2026-03-31",
                "--previous-start",
                "2026-02-01",
                "--previous-end",
                "2026-02-28",
            ],
        )

    def test_append_period_flags_skips_blank_values(self):
        cmd: list[str] = ["analyze"]
        self.mod._append_period_flags(
            cmd,
            date_column="",
            time_grain=None,
            period="  ",
            anchor_date=None,
            current_start=None,
            current_end="2026-03-31",
            previous_start=None,
            previous_end=None,
        )
        self.assertEqual(cmd, ["analyze", "--current-end", "2026-03-31"])


if __name__ == "__main__":
    unittest.main()
