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


class AnalyzeInvestigateLegacyToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module()

    def test_analyze_investigate_legacy_builds_expected_command(self):
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
            raw = mod.analyze_investigate_legacy(
                input_csv="data/snapshot.csv",
                metric="revenue_usd",
                out="artifacts/legacy.md",
                drivers_csv="net_gmv,avg_order_value",
                driver_preset="mixed",
                auto_drivers="numeric-corr",
                dedup_drivers=False,
                driver_contrib="both",
                top_drivers=5,
                output_format="both",
                max_id_drivers=4,
                max_cat_drivers=3,
                max_num_drivers=2,
                date_column="order_date",
                time_grain="month",
                period="last",
                anchor_date="2026-04-15",
                timeout_sec=60,
            )
            self.assertEqual(json.loads(raw), {"ok": True})
            self.assertEqual(calls["timeout_sec"], 60)
            self.assertEqual(
                calls["cmd"],
                [
                    "analyze-investigate",
                    "--input",
                    "/safe/read/snapshot.csv",
                    "--metric",
                    "revenue_usd",
                    "--out",
                    "/safe/write/legacy.md",
                    "--output-format",
                    "both",
                    "--auto-drivers",
                    "numeric-corr",
                    "--dedup-drivers",
                    "false",
                    "--driver-contrib",
                    "both",
                    "--top-drivers",
                    "5",
                    "--max-id-drivers",
                    "4",
                    "--max-cat-drivers",
                    "3",
                    "--max-num-drivers",
                    "2",
                    "--drivers",
                    "net_gmv,avg_order_value",
                    "--driver-preset",
                    "mixed",
                    "--date-column",
                    "order_date",
                    "--time-grain",
                    "month",
                    "--period",
                    "last",
                    "--anchor-date",
                    "2026-04-15",
                ],
            )
        finally:
            mod._validate_read_path = old_validate_read
            mod._validate_write_path = old_validate_write
            mod._run = old_run

    def test_analyze_investigate_legacy_rejects_invalid_preset(self):
        with self.assertRaisesRegex(ValueError, "driver_preset must be one of"):
            self.mod.analyze_investigate_legacy(
                input_csv="data/snapshot.csv",
                metric="revenue_usd",
                out="artifacts/legacy.md",
                driver_preset="random",
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


class ListArtifactsToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module()

    def test_list_artifacts_returns_recent_files(self):
        mod = self.mod
        old_validate_read = mod._validate_read_path
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "artifacts"
                root.mkdir(parents=True, exist_ok=True)
                (root / "a.json").write_text("{}", encoding="utf-8")
                (root / "b.md").write_text("# report", encoding="utf-8")
                mod._validate_read_path = lambda _p, _label: root
                raw = mod.list_artifacts("artifacts", suffix=".json", limit=10)
                payload = json.loads(raw)
                self.assertEqual(payload["ok"], True)
                self.assertEqual(payload["count"], 1)
                self.assertEqual(payload["files"][0]["name"], "a.json")
        finally:
            mod._validate_read_path = old_validate_read

    def test_list_artifacts_rejects_invalid_limit(self):
        with self.assertRaisesRegex(ValueError, "limit must be between 1 and 5000"):
            self.mod.list_artifacts("artifacts", limit=0)


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


class SummarizeInvestigateToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module()

    def test_summarize_investigate_returns_strict_json_summary(self):
        mod = self.mod
        old_validate_read = mod._validate_read_path
        try:
            with tempfile.TemporaryDirectory() as tmp:
                artifact = Path(tmp) / "investigate.json"
                artifact.write_text(
                    json.dumps(
                        {
                            "question": "Why did revenue change?",
                            "mode": "change_drivers",
                            "steps": [
                                {
                                    "depth": 0,
                                    "dimension": "region",
                                    "primary_metric": "revenue_usd",
                                    "top1_concentration_base_pct": 40.0,
                                    "top1_concentration_new_pct": 45.0,
                                    "top1_concentration_delta_pp": 5.0,
                                    "top5_concentration_base_pct": 88.0,
                                    "top5_concentration_new_pct": 90.0,
                                    "top5_concentration_delta_pp": 2.0,
                                    "movers": [
                                        {
                                            "segment": "US",
                                            "base_primary_metric_value": 100.0,
                                            "new_primary_metric_value": 140.0,
                                            "delta_primary_metric_value": 40.0,
                                            "delta_share_pp": 3.2,
                                        }
                                    ],
                                },
                                {
                                    "depth": 1,
                                    "dimension": "channel",
                                    "scope": [["region", "US"]],
                                    "movers": [
                                        {
                                            "segment": "Direct",
                                            "delta_primary_metric_value": 25.0,
                                            "delta_share_pp": 2.1,
                                        }
                                    ],
                                },
                            ],
                            "major_global_changes": [
                                {
                                    "dimension": "organization_name",
                                    "segment": "Acme",
                                    "primary_metric": "revenue_usd",
                                    "delta_primary_metric_value": 120.0,
                                    "delta_share_pp": 1.1,
                                    "score": 120.0,
                                }
                            ],
                            "stopping_reason": "reached max depth 2",
                            "recommended_next_question": "Drill into provider",
                        }
                    ),
                    encoding="utf-8",
                )
                mod._validate_read_path = lambda _p, _label: artifact
                raw = mod.summarize_investigate("artifacts/investigate.json")
                payload = json.loads(raw)
                self.assertEqual(payload["ok"], True)
                self.assertEqual(payload["schema"], "investigate")
                self.assertEqual(payload["top_level"]["dimension"], "region")
                self.assertEqual(payload["top_level"]["delta_total"], 40.0)
                self.assertEqual(payload["major_global_changes"][0]["segment"], "Acme")
                self.assertEqual(payload["follow_up"][0]["strongest_segment"], "Direct")
        finally:
            mod._validate_read_path = old_validate_read

    def test_summarize_investigate_rejects_non_investigate_schema(self):
        mod = self.mod
        old_validate_read = mod._validate_read_path
        try:
            with tempfile.TemporaryDirectory() as tmp:
                artifact = Path(tmp) / "analysis.json"
                artifact.write_text(json.dumps({"records": 10}), encoding="utf-8")
                mod._validate_read_path = lambda _p, _label: artifact
                with self.assertRaisesRegex(ValueError, "does not look like investigate output"):
                    mod.summarize_investigate("artifacts/analysis.json")
        finally:
            mod._validate_read_path = old_validate_read


class ToolGuideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module()

    def test_tool_guide_contains_core_methods(self):
        payload = json.loads(self.mod.tool_guide())
        self.assertEqual(payload["ok"], True)
        methods = payload["methods"]
        self.assertIn("investigate", methods)
        self.assertIn("analyze_investigate_legacy", methods)
        self.assertIn("summarize_investigate", methods)
        self.assertIn("list_artifacts", methods)
        self.assertIn("read_artifact", methods)
        self.assertIn("legacy_mapping", payload)
        self.assertIn("analyze-investigate (legacy CLI)", payload["legacy_mapping"])
        self.assertIn("single_pass_flow_csv", payload["recommended_route"])
        self.assertIn("single_pass_flow_query", payload["recommended_route"])


if __name__ == "__main__":
    unittest.main()
