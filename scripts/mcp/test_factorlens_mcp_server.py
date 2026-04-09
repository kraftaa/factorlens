#!/usr/bin/env python3
"""Unit tests for scripts/mcp/factorlens_mcp_server.py without external mcp dependency."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
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


class InvestigateToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module()

    def test_investigate_builds_expected_command(self):
        calls: dict[str, object] = {}
        mod = self.mod
        old_validate_read = mod._validate_read_path
        old_validate_write = mod._validate_write_path
        old_run = mod._run
        try:
            mod._validate_read_path = lambda p, _label: Path(f"/safe/read/{Path(p).name}")
            mod._validate_write_path = lambda p, _label: Path(
                f"/safe/write/{Path(p).name}"
            )

            def _fake_run(cmd: list[str], timeout_sec: int | None):
                calls["cmd"] = cmd
                calls["timeout_sec"] = timeout_sec
                return {"ok": True}

            mod._run = _fake_run
            raw = mod.investigate(
                question="Why did revenue change?",
                base="artifacts/base.json",
                new="artifacts/new.json",
                out="artifacts/investigate.md",
                mode="change_drivers",
                metric="revenue_usd",
                dimensions_csv="region,channel",
                drill_fields_csv="channel",
                max_depth=2,
                max_branches=2,
                min_contribution=7.5,
                min_score_improvement=1.25,
                min_slice_rows=10,
                top_movers=15,
                planner="llm",
                planner_backend="bedrock",
                planner_model="anthropic.claude-3-haiku-20240307-v1:0",
                verbose=True,
                trace=True,
                output_format="both",
                timeout_sec=45,
            )
            self.assertEqual(json.loads(raw), {"ok": True})
            self.assertEqual(calls["timeout_sec"], 45)
            self.assertEqual(
                calls["cmd"],
                [
                    "investigate",
                    "--question",
                    "Why did revenue change?",
                    "--out",
                    "/safe/write/investigate.md",
                    "--output-format",
                    "both",
                    "--max-depth",
                    "2",
                    "--max-branches",
                    "2",
                    "--min-contribution",
                    "7.5",
                    "--min-score-improvement",
                    "1.25",
                    "--min-slice-rows",
                    "10",
                    "--top-movers",
                    "15",
                    "--planner",
                    "llm",
                    "--planner-backend",
                    "bedrock",
                    "--base",
                    "/safe/read/base.json",
                    "--new",
                    "/safe/read/new.json",
                    "--metric",
                    "revenue_usd",
                    "--mode",
                    "change_drivers",
                    "--dimensions",
                    "region,channel",
                    "--drill-fields",
                    "channel",
                    "--planner-model",
                    "anthropic.claude-3-haiku-20240307-v1:0",
                    "--verbose",
                    "--trace",
                ],
            )
        finally:
            mod._validate_read_path = old_validate_read
            mod._validate_write_path = old_validate_write
            mod._run = old_run

    def test_investigate_query_mode_builds_expected_command(self):
        calls: dict[str, object] = {}
        mod = self.mod
        old_validate_read = mod._validate_read_path
        old_validate_write = mod._validate_write_path
        old_run = mod._run
        try:
            mod._validate_read_path = lambda p, _label: Path(f"/safe/read/{Path(p).name}")
            mod._validate_write_path = lambda p, _label: Path(
                f"/safe/write/{Path(p).name}"
            )

            def _fake_run(cmd: list[str], timeout_sec: int | None):
                calls["cmd"] = cmd
                calls["timeout_sec"] = timeout_sec
                return {"ok": True}

            mod._run = _fake_run
            raw = mod.investigate(
                question="Why did revenue change?",
                out="artifacts/investigate.md",
                query_file="sql/sales.sql",
                postgres_url="postgres://db",
                postgres_ssl_mode="require",
                postgres_ca_file="certs/ca.pem",
                date_column="order_date",
                time_grain="month",
                period="last",
                anchor_date="2026-04-15",
                current_start="2026-03-01",
                current_end="2026-03-31",
                previous_start="2026-02-01",
                previous_end="2026-02-28",
                output_format="both",
                timeout_sec=90,
            )
            self.assertEqual(json.loads(raw), {"ok": True})
            self.assertEqual(calls["timeout_sec"], 90)
            self.assertEqual(
                calls["cmd"],
                [
                    "investigate",
                    "--question",
                    "Why did revenue change?",
                    "--out",
                    "/safe/write/investigate.md",
                    "--output-format",
                    "both",
                    "--max-depth",
                    "2",
                    "--max-branches",
                    "1",
                    "--min-contribution",
                    "5.0",
                    "--min-score-improvement",
                    "0.0",
                    "--min-slice-rows",
                    "5",
                    "--top-movers",
                    "12",
                    "--planner",
                    "deterministic",
                    "--planner-backend",
                    "local",
                    "--postgres-url",
                    "postgres://db",
                    "--postgres-ssl-mode",
                    "require",
                    "--query-file",
                    "/safe/read/sales.sql",
                    "--postgres-ca-file",
                    "/safe/read/ca.pem",
                    "--date-column",
                    "order_date",
                    "--time-grain",
                    "month",
                    "--period",
                    "last",
                    "--anchor-date",
                    "2026-04-15",
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
        finally:
            mod._validate_read_path = old_validate_read
            mod._validate_write_path = old_validate_write
            mod._run = old_run

    def test_investigate_rejects_invalid_output_format(self):
        with self.assertRaisesRegex(ValueError, "output_format must be one of"):
            self.mod.investigate(
                question="q",
                base="artifacts/base.json",
                new="artifacts/new.json",
                out="artifacts/investigate.md",
                output_format="html",
            )

    def test_investigate_rejects_invalid_planner(self):
        with self.assertRaisesRegex(ValueError, "planner must be one of"):
            self.mod.investigate(
                question="q",
                base="artifacts/base.json",
                new="artifacts/new.json",
                out="artifacts/investigate.md",
                planner="auto",
            )

    def test_investigate_rejects_invalid_mode(self):
        with self.assertRaisesRegex(ValueError, "mode must be one of"):
            self.mod.investigate(
                question="q",
                base="artifacts/base.json",
                new="artifacts/new.json",
                out="artifacts/investigate.md",
                mode="auto",
            )

    def test_investigate_rejects_negative_min_score_improvement(self):
        with self.assertRaisesRegex(ValueError, "min_score_improvement must be >= 0"):
            self.mod.investigate(
                question="q",
                out="artifacts/investigate.md",
                base="artifacts/base.json",
                new="artifacts/new.json",
                min_score_improvement=-0.1,
            )

    def test_investigate_rejects_mixed_input_modes(self):
        with self.assertRaisesRegex(ValueError, "choose one input mode"):
            self.mod.investigate(
                question="q",
                out="artifacts/investigate.md",
                base="artifacts/base.json",
                new="artifacts/new.json",
                query="select 1",
            )

    def test_investigate_rejects_query_mode_without_query_or_file(self):
        with self.assertRaisesRegex(
            ValueError, "provide exactly one of query or query_file for query input mode"
        ):
            self.mod.investigate(
                question="q",
                out="artifacts/investigate.md",
            )

    def test_investigate_accepts_config_path(self):
        calls: dict[str, object] = {}
        mod = self.mod
        old_validate_read = mod._validate_read_path
        old_validate_write = mod._validate_write_path
        old_run = mod._run
        try:
            mod._validate_read_path = lambda p, _label: Path(f"/safe/read/{Path(p).name}")
            mod._validate_write_path = lambda p, _label: Path(
                f"/safe/write/{Path(p).name}"
            )

            def _fake_run(cmd: list[str], timeout_sec: int | None):
                calls["cmd"] = cmd
                calls["timeout_sec"] = timeout_sec
                return {"ok": True}

            mod._run = _fake_run
            mod.investigate(
                question="q",
                base="artifacts/base.json",
                new="artifacts/new.json",
                out="artifacts/investigate.md",
                config="profiles/investigate.example.toml",
            )
            self.assertIn("--config", calls["cmd"])
            cfg_idx = calls["cmd"].index("--config")
            self.assertEqual(calls["cmd"][cfg_idx + 1], "/safe/read/investigate.example.toml")
        finally:
            mod._validate_read_path = old_validate_read
            mod._validate_write_path = old_validate_write
            mod._run = old_run

    def test_investigate_accepts_profile_and_profile_config(self):
        calls: dict[str, object] = {}
        mod = self.mod
        old_validate_read = mod._validate_read_path
        old_validate_write = mod._validate_write_path
        old_run = mod._run
        try:
            mod._validate_read_path = lambda p, _label: Path(f"/safe/read/{Path(p).name}")
            mod._validate_write_path = lambda p, _label: Path(
                f"/safe/write/{Path(p).name}"
            )

            def _fake_run(cmd: list[str], timeout_sec: int | None):
                calls["cmd"] = cmd
                calls["timeout_sec"] = timeout_sec
                return {"ok": True}

            mod._run = _fake_run
            mod.investigate(
                question="q",
                base="artifacts/base.json",
                new="artifacts/new.json",
                out="artifacts/investigate.md",
                profile="default",
                profile_config="profiles/investigate.example.toml",
            )
            self.assertIn("--profile", calls["cmd"])
            profile_idx = calls["cmd"].index("--profile")
            self.assertEqual(calls["cmd"][profile_idx + 1], "default")
            self.assertIn("--profile-config", calls["cmd"])
            cfg_idx = calls["cmd"].index("--profile-config")
            self.assertEqual(calls["cmd"][cfg_idx + 1], "/safe/read/investigate.example.toml")
        finally:
            mod._validate_read_path = old_validate_read
            mod._validate_write_path = old_validate_write
            mod._run = old_run

    def test_investigate_rejects_mixed_config_and_profile(self):
        with self.assertRaisesRegex(ValueError, "use either config or profile/profile_config"):
            self.mod.investigate(
                question="q",
                base="artifacts/base.json",
                new="artifacts/new.json",
                out="artifacts/investigate.md",
                config="profiles/investigate.example.toml",
                profile="default",
            )


class ExplainAnalyzeToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module()

    def test_explain_analyze_builds_expected_command(self):
        calls: dict[str, object] = {}
        mod = self.mod
        old_validate_read = mod._validate_read_path
        old_run = mod._run
        try:
            mod._validate_read_path = lambda p, _label: Path(f"/safe/read/{Path(p).name}")

            def _fake_run(cmd: list[str], timeout_sec: int | None):
                calls["cmd"] = cmd
                calls["timeout_sec"] = timeout_sec
                return {"ok": True}

            mod._run = _fake_run
            raw = mod.explain_analyze(
                analysis_json="artifacts/analysis.json",
                question="What changed?",
                backend="local",
                model="/models/llama.gguf",
                strict_facts=True,
                max_bullets=4,
                timeout_sec=30,
            )
            self.assertEqual(json.loads(raw), {"ok": True})
            self.assertEqual(calls["timeout_sec"], 30)
            self.assertEqual(
                calls["cmd"],
                [
                    "explain-analyze",
                    "--backend",
                    "local",
                    "--model",
                    "/models/llama.gguf",
                    "--analysis-json",
                    "/safe/read/analysis.json",
                    "--question",
                    "What changed?",
                    "--max-bullets",
                    "4",
                    "--strict-facts",
                    "true",
                ],
            )
        finally:
            mod._validate_read_path = old_validate_read
            mod._run = old_run


class ReadArtifactToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module()

    def test_read_artifact_returns_content_and_truncation_flag(self):
        mod = self.mod
        old_validate_read = mod._validate_read_path
        try:
            with tempfile.TemporaryDirectory() as tmp:
                artifact = Path(tmp) / "report.md"
                artifact.write_text("0123456789ABCDEFGHIJ", encoding="utf-8")
                mod._validate_read_path = lambda _p, _label: artifact
                raw = mod.read_artifact("artifacts/report.md", max_chars=10)
                payload = json.loads(raw)
                self.assertEqual(payload["ok"], True)
                self.assertEqual(payload["path"], str(artifact))
                self.assertEqual(payload["truncated"], True)
                self.assertEqual(payload["content"], "0123456789")
        finally:
            mod._validate_read_path = old_validate_read

    def test_read_artifact_rejects_invalid_max_chars(self):
        with self.assertRaisesRegex(ValueError, "max_chars must be between 1 and"):
            self.mod.read_artifact("artifacts/report.md", max_chars=0)


class ServerInfoToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module()

    def test_server_info_reports_versions(self):
        mod = self.mod
        old_run = mod._run
        old_pkg_ver = mod._package_version
        try:
            mod._package_version = lambda _name: "1.27.0"
            mod._run = lambda _cmd, _timeout: {
                "ok": True,
                "return_code": 0,
                "cmd": ["/safe/bin/factorlens", "--version"],
                "stdout": "factorlens 4.1.4",
                "stderr": "",
                "timeout_sec": 20,
            }
            raw = mod.server_info(timeout_sec=20)
            payload = json.loads(raw)
            self.assertEqual(payload["ok"], True)
            self.assertEqual(payload["mcp_server_name"], "factorlens")
            self.assertEqual(payload["mcp_sdk_version"], "1.27.0")
            self.assertEqual(payload["factorlens_version"], "factorlens 4.1.4")
        finally:
            mod._run = old_run
            mod._package_version = old_pkg_ver

    def test_server_info_handles_missing_cli(self):
        mod = self.mod
        old_run = mod._run
        old_pkg_ver = mod._package_version
        try:
            mod._package_version = lambda _name: "1.27.0"

            def _raise(_cmd, _timeout):
                raise RuntimeError("factorlens binary not found")

            mod._run = _raise
            raw = mod.server_info(timeout_sec=20)
            payload = json.loads(raw)
            self.assertEqual(payload["ok"], False)
            self.assertEqual(payload["mcp_server_name"], "factorlens")
            self.assertEqual(payload["mcp_sdk_version"], "1.27.0")
            self.assertIn("factorlens binary not found", payload["factorlens_version_probe"]["stderr"])
        finally:
            mod._run = old_run
            mod._package_version = old_pkg_ver


if __name__ == "__main__":
    unittest.main()
