#!/usr/bin/env python3
"""FactorLens MCP server.

Production-oriented wrapper around the `factorlens` CLI with:
- argument validation
- path allowlists
- subprocess timeout + structured JSON responses
- optional Bedrock/local explanation tools

Environment variables:
- FACTORLENS_BIN: path to factorlens binary (default: "factorlens")
- FACTORLENS_ALLOWED_READ_DIRS: comma-separated readable roots
- FACTORLENS_ALLOWED_WRITE_DIRS: comma-separated writable roots
- FACTORLENS_CMD_TIMEOUT_SEC: default command timeout in seconds (default: 180)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got: {raw}") from exc


mcp = FastMCP(
    "factorlens",
    host=os.getenv("FASTMCP_HOST", "127.0.0.1"),
    port=_env_int("FASTMCP_PORT", 8000),
    mount_path=os.getenv("FASTMCP_MOUNT_PATH", "/"),
    sse_path=os.getenv("FASTMCP_SSE_PATH", "/sse"),
    message_path=os.getenv("FASTMCP_MESSAGE_PATH", "/messages/"),
    streamable_http_path=os.getenv("FASTMCP_STREAMABLE_HTTP_PATH", "/mcp"),
    log_level=os.getenv("FASTMCP_LOG_LEVEL", "INFO").upper(),
)


VALID_OUTPUT_FORMATS = {"md", "json", "both", "html"}
VALID_COMPARE_OUTPUT_FORMATS = {"md", "html", "json", "both"}
VALID_BACKENDS = {"local", "bedrock"}
VALID_POSTGRES_SSL_MODES = {"disable", "prefer", "require"}


def _split_roots(env_key: str, defaults: list[Path]) -> list[Path]:
    raw = os.getenv(env_key, "").strip()
    if not raw:
        return [p.resolve() for p in defaults]
    roots: list[Path] = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            roots.append(Path(item).expanduser().resolve())
    return roots


def _cwd() -> Path:
    return Path.cwd().resolve()


def _read_roots() -> list[Path]:
    c = _cwd()
    return _split_roots(
        "FACTORLENS_ALLOWED_READ_DIRS",
        [c / "data", c / "profiles", c / "artifacts", c],
    )


def _write_roots() -> list[Path]:
    c = _cwd()
    return _split_roots("FACTORLENS_ALLOWED_WRITE_DIRS", [c / "artifacts", c])


def _is_within(path: Path, roots: list[Path]) -> bool:
    rp = path.resolve()
    for root in roots:
        try:
            rp.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _validate_read_path(path_str: str, label: str) -> Path:
    p = Path(path_str).expanduser().resolve()
    if not _is_within(p, _read_roots()):
        raise ValueError(f"{label} must be inside allowed read dirs: {p}")
    return p


def _validate_write_path(path_str: str, label: str) -> Path:
    p = Path(path_str).expanduser().resolve()
    parent = p.parent
    if not _is_within(parent, _write_roots()):
        raise ValueError(f"{label} must be inside allowed write dirs: {p}")
    return p


def _factorlens_bin() -> str:
    configured = os.getenv("FACTORLENS_BIN", "factorlens").strip() or "factorlens"
    if configured == "factorlens":
        located = shutil.which("factorlens")
        if not located:
            raise RuntimeError(
                "factorlens binary not found on PATH. Set FACTORLENS_BIN or install factorlens."
            )
        return located
    p = Path(configured).expanduser().resolve()
    if not p.exists():
        raise RuntimeError(f"FACTORLENS_BIN does not exist: {p}")
    return str(p)


def _timeout_sec(requested: int | None) -> int:
    env_default = int(os.getenv("FACTORLENS_CMD_TIMEOUT_SEC", "180"))
    if requested is None:
        return env_default
    if requested <= 0:
        raise ValueError("timeout_sec must be > 0")
    return requested


def _run(args: list[str], timeout_sec: int | None = None) -> dict[str, Any]:
    timeout = _timeout_sec(timeout_sec)
    cmd = [_factorlens_bin(), *args]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "ok": proc.returncode == 0,
            "return_code": proc.returncode,
            "cmd": cmd,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "timeout_sec": timeout,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "return_code": 124,
            "cmd": cmd,
            "stdout": (exc.stdout or "").strip() if exc.stdout else "",
            "stderr": "command timed out",
            "timeout_sec": timeout,
        }


def _append_optional(cmd: list[str], flag: str, value: str | None) -> None:
    if value is not None and str(value).strip() != "":
        cmd.extend([flag, str(value)])


@mcp.tool()
def analyze_csv(
    input_csv: str,
    out: str,
    profile: str | None = None,
    profile_config: str | None = None,
    group_by_csv: str | None = None,
    metrics_csv: str | None = None,
    where_csv: str | None = None,
    rank_by: str | None = None,
    agg: str = "sum",
    percentiles_csv: str | None = None,
    min_records: int = 1,
    top: int = 20,
    count_only: bool = False,
    normalize_text_groups: bool = False,
    word_freq: bool = False,
    output_format: str = "both",
    timeout_sec: int | None = None,
) -> str:
    if output_format not in VALID_OUTPUT_FORMATS:
        raise ValueError(f"output_format must be one of {sorted(VALID_OUTPUT_FORMATS)}")
    if agg not in {"sum", "mean", "median"}:
        raise ValueError("agg must be one of: sum, mean, median")
    if min_records < 1:
        raise ValueError("min_records must be >= 1")
    if top < 1:
        raise ValueError("top must be >= 1")

    in_path = _validate_read_path(input_csv, "input_csv")
    out_path = _validate_write_path(out, "out")

    cmd = [
        "analyze",
        "--input",
        str(in_path),
        "--out",
        str(out_path),
        "--output-format",
        output_format,
        "--agg",
        agg,
        "--min-records",
        str(min_records),
        "--top",
        str(top),
    ]

    _append_optional(cmd, "--profile", profile)
    if profile_config:
        cfg = _validate_read_path(profile_config, "profile_config")
        cmd.extend(["--profile-config", str(cfg)])
    _append_optional(cmd, "--group-by", group_by_csv)
    _append_optional(cmd, "--metrics", metrics_csv)
    _append_optional(cmd, "--where", where_csv)
    _append_optional(cmd, "--rank-by", rank_by)
    _append_optional(cmd, "--percentiles", percentiles_csv)

    if count_only:
        cmd.append("--count-only")
    if normalize_text_groups:
        cmd.append("--normalize-text-groups")
    if word_freq:
        cmd.append("--word-freq")

    return json.dumps(_run(cmd, timeout_sec), ensure_ascii=True)


@mcp.tool()
def analyze_query(
    out: str,
    query: str | None = None,
    query_file: str | None = None,
    postgres_url: str | None = None,
    postgres_ssl_mode: str = "prefer",
    postgres_ca_file: str | None = None,
    profile: str | None = None,
    profile_config: str | None = None,
    group_by_csv: str | None = None,
    metrics_csv: str | None = None,
    where_csv: str | None = None,
    rank_by: str | None = None,
    agg: str = "sum",
    percentiles_csv: str | None = None,
    min_records: int = 1,
    top: int = 20,
    count_only: bool = False,
    normalize_text_groups: bool = False,
    word_freq: bool = False,
    output_format: str = "both",
    timeout_sec: int | None = None,
) -> str:
    if output_format not in VALID_OUTPUT_FORMATS:
        raise ValueError(f"output_format must be one of {sorted(VALID_OUTPUT_FORMATS)}")
    if agg not in {"sum", "mean", "median"}:
        raise ValueError("agg must be one of: sum, mean, median")
    if min_records < 1:
        raise ValueError("min_records must be >= 1")
    if top < 1:
        raise ValueError("top must be >= 1")
    if postgres_ssl_mode not in VALID_POSTGRES_SSL_MODES:
        raise ValueError(
            f"postgres_ssl_mode must be one of {sorted(VALID_POSTGRES_SSL_MODES)}"
        )
    if bool(query) == bool(query_file):
        raise ValueError("provide exactly one of query or query_file")

    out_path = _validate_write_path(out, "out")
    query_file_path: Path | None = None
    if query_file:
        query_file_path = _validate_read_path(query_file, "query_file")
    ca_file_path: Path | None = None
    if postgres_ca_file:
        ca_file_path = _validate_read_path(postgres_ca_file, "postgres_ca_file")

    cmd = [
        "analyze",
        "--out",
        str(out_path),
        "--output-format",
        output_format,
        "--agg",
        agg,
        "--min-records",
        str(min_records),
        "--top",
        str(top),
        "--postgres-ssl-mode",
        postgres_ssl_mode,
    ]

    _append_optional(cmd, "--postgres-url", postgres_url)
    if query:
        cmd.extend(["--query", query])
    if query_file_path:
        cmd.extend(["--query-file", str(query_file_path)])
    if ca_file_path:
        cmd.extend(["--postgres-ca-file", str(ca_file_path)])

    _append_optional(cmd, "--profile", profile)
    if profile_config:
        cfg = _validate_read_path(profile_config, "profile_config")
        cmd.extend(["--profile-config", str(cfg)])
    _append_optional(cmd, "--group-by", group_by_csv)
    _append_optional(cmd, "--metrics", metrics_csv)
    _append_optional(cmd, "--where", where_csv)
    _append_optional(cmd, "--rank-by", rank_by)
    _append_optional(cmd, "--percentiles", percentiles_csv)

    if count_only:
        cmd.append("--count-only")
    if normalize_text_groups:
        cmd.append("--normalize-text-groups")
    if word_freq:
        cmd.append("--word-freq")

    return json.dumps(_run(cmd, timeout_sec), ensure_ascii=True)


@mcp.tool()
def analyze_validate_csv(
    input_csv: str,
    profile: str | None = None,
    profile_config: str | None = None,
    group_by_csv: str | None = None,
    metrics_csv: str | None = None,
    where_csv: str | None = None,
    rank_by: str | None = None,
    agg: str = "sum",
    percentiles_csv: str | None = None,
    min_records: int = 1,
    top: int = 20,
    count_only: bool = False,
    normalize_text_groups: bool = False,
    word_freq: bool = False,
    alert_rule_csv: str | None = None,
    timeout_sec: int | None = None,
) -> str:
    if agg not in {"sum", "mean", "median"}:
        raise ValueError("agg must be one of: sum, mean, median")
    if min_records < 1:
        raise ValueError("min_records must be >= 1")
    if top < 1:
        raise ValueError("top must be >= 1")

    in_path = _validate_read_path(input_csv, "input_csv")

    cmd = [
        "analyze-validate",
        "--input",
        str(in_path),
        "--agg",
        agg,
        "--min-records",
        str(min_records),
        "--top",
        str(top),
    ]

    _append_optional(cmd, "--profile", profile)
    if profile_config:
        cfg = _validate_read_path(profile_config, "profile_config")
        cmd.extend(["--profile-config", str(cfg)])
    _append_optional(cmd, "--group-by", group_by_csv)
    _append_optional(cmd, "--metrics", metrics_csv)
    _append_optional(cmd, "--where", where_csv)
    _append_optional(cmd, "--rank-by", rank_by)
    _append_optional(cmd, "--percentiles", percentiles_csv)
    _append_optional(cmd, "--alert-rule", alert_rule_csv)

    if count_only:
        cmd.append("--count-only")
    if normalize_text_groups:
        cmd.append("--normalize-text-groups")
    if word_freq:
        cmd.append("--word-freq")

    return json.dumps(_run(cmd, timeout_sec), ensure_ascii=True)


@mcp.tool()
def analyze_validate_query(
    query: str | None = None,
    query_file: str | None = None,
    postgres_url: str | None = None,
    postgres_ssl_mode: str = "prefer",
    postgres_ca_file: str | None = None,
    profile: str | None = None,
    profile_config: str | None = None,
    group_by_csv: str | None = None,
    metrics_csv: str | None = None,
    where_csv: str | None = None,
    rank_by: str | None = None,
    agg: str = "sum",
    percentiles_csv: str | None = None,
    min_records: int = 1,
    top: int = 20,
    count_only: bool = False,
    normalize_text_groups: bool = False,
    word_freq: bool = False,
    alert_rule_csv: str | None = None,
    timeout_sec: int | None = None,
) -> str:
    if agg not in {"sum", "mean", "median"}:
        raise ValueError("agg must be one of: sum, mean, median")
    if min_records < 1:
        raise ValueError("min_records must be >= 1")
    if top < 1:
        raise ValueError("top must be >= 1")
    if postgres_ssl_mode not in VALID_POSTGRES_SSL_MODES:
        raise ValueError(
            f"postgres_ssl_mode must be one of {sorted(VALID_POSTGRES_SSL_MODES)}"
        )
    if bool(query) == bool(query_file):
        raise ValueError("provide exactly one of query or query_file")

    query_file_path: Path | None = None
    if query_file:
        query_file_path = _validate_read_path(query_file, "query_file")
    ca_file_path: Path | None = None
    if postgres_ca_file:
        ca_file_path = _validate_read_path(postgres_ca_file, "postgres_ca_file")

    cmd = [
        "analyze-validate",
        "--agg",
        agg,
        "--min-records",
        str(min_records),
        "--top",
        str(top),
        "--postgres-ssl-mode",
        postgres_ssl_mode,
    ]

    _append_optional(cmd, "--postgres-url", postgres_url)
    if query:
        cmd.extend(["--query", query])
    if query_file_path:
        cmd.extend(["--query-file", str(query_file_path)])
    if ca_file_path:
        cmd.extend(["--postgres-ca-file", str(ca_file_path)])

    _append_optional(cmd, "--profile", profile)
    if profile_config:
        cfg = _validate_read_path(profile_config, "profile_config")
        cmd.extend(["--profile-config", str(cfg)])
    _append_optional(cmd, "--group-by", group_by_csv)
    _append_optional(cmd, "--metrics", metrics_csv)
    _append_optional(cmd, "--where", where_csv)
    _append_optional(cmd, "--rank-by", rank_by)
    _append_optional(cmd, "--percentiles", percentiles_csv)
    _append_optional(cmd, "--alert-rule", alert_rule_csv)

    if count_only:
        cmd.append("--count-only")
    if normalize_text_groups:
        cmd.append("--normalize-text-groups")
    if word_freq:
        cmd.append("--word-freq")

    return json.dumps(_run(cmd, timeout_sec), ensure_ascii=True)


@mcp.tool()
def analyze_compare(
    base_json: str,
    new_json: str,
    out: str,
    output_format: str = "md",
    top_movers: int = 10,
    timeout_sec: int | None = None,
) -> str:
    if output_format not in VALID_COMPARE_OUTPUT_FORMATS:
        raise ValueError(
            f"output_format must be one of {sorted(VALID_COMPARE_OUTPUT_FORMATS)}"
        )
    if top_movers < 1:
        raise ValueError("top_movers must be >= 1")

    base_path = _validate_read_path(base_json, "base_json")
    new_path = _validate_read_path(new_json, "new_json")
    out_path = _validate_write_path(out, "out")

    cmd = [
        "analyze-compare",
        "--base",
        str(base_path),
        "--new",
        str(new_path),
        "--out",
        str(out_path),
        "--output-format",
        output_format,
        "--top-movers",
        str(top_movers),
    ]

    return json.dumps(_run(cmd, timeout_sec), ensure_ascii=True)


@mcp.tool()
def explain_analyze(
    analysis_json: str,
    question: str,
    backend: str = "bedrock",
    model: str = "anthropic.claude-3-haiku-20240307-v1:0",
    timeout_sec: int | None = None,
) -> str:
    if backend not in VALID_BACKENDS:
        raise ValueError(f"backend must be one of {sorted(VALID_BACKENDS)}")
    if not question.strip():
        raise ValueError("question must not be empty")

    analysis_path = _validate_read_path(analysis_json, "analysis_json")

    cmd = [
        "explain-analyze",
        "--backend",
        backend,
        "--model",
        model,
        "--analysis-json",
        str(analysis_path),
        "--question",
        question,
    ]

    return json.dumps(_run(cmd, timeout_sec), ensure_ascii=True)


@mcp.tool()
def healthcheck(timeout_sec: int | None = 20) -> str:
    return json.dumps(_run(["--help"], timeout_sec), ensure_ascii=True)


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio").strip().lower() or "stdio"
    if transport not in {"stdio", "sse", "streamable-http"}:
        raise RuntimeError(
            "MCP_TRANSPORT must be one of: stdio, sse, streamable-http"
        )
    if transport == "streamable-http":
        print(
            "Starting streamable-http MCP on "
            f"http://{mcp.settings.host}:{mcp.settings.port}{mcp.settings.streamable_http_path}"
        )
    elif transport == "sse":
        print(
            "Starting sse MCP on "
            f"http://{mcp.settings.host}:{mcp.settings.port}"
        )
    if transport == "sse":
        mcp.run("sse", mount_path=os.getenv("MCP_MOUNT_PATH"))
    else:
        mcp.run(transport)
