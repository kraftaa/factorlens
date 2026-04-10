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
from importlib import metadata as importlib_metadata
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
VALID_INVESTIGATE_OUTPUT_FORMATS = {"md", "json", "both"}
VALID_INVESTIGATE_PLANNERS = {"deterministic", "llm"}
VALID_INVESTIGATE_MODES = {
    "change_drivers",
    "concentration_drivers",
    "compare_snapshots",
    "recommend_next",
}
VALID_BACKENDS = {"local", "bedrock"}
VALID_POSTGRES_SSL_MODES = {"disable", "prefer", "require"}
MAX_READ_ARTIFACT_CHARS = 2_000_000


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
        [c / "data", c / "profiles", c / "artifacts", c / "certs"],
    )


def _write_roots() -> list[Path]:
    c = _cwd()
    return _split_roots("FACTORLENS_ALLOWED_WRITE_DIRS", [c / "artifacts"])


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
    env_default = _env_int("FACTORLENS_CMD_TIMEOUT_SEC", 180)
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


def _append_period_flags(
    cmd: list[str],
    date_column: str | None = None,
    time_grain: str | None = None,
    period: str | None = None,
    anchor_date: str | None = None,
    current_start: str | None = None,
    current_end: str | None = None,
    previous_start: str | None = None,
    previous_end: str | None = None,
) -> None:
    _append_optional(cmd, "--date-column", date_column)
    _append_optional(cmd, "--time-grain", time_grain)
    _append_optional(cmd, "--period", period)
    _append_optional(cmd, "--anchor-date", anchor_date)
    _append_optional(cmd, "--current-start", current_start)
    _append_optional(cmd, "--current-end", current_end)
    _append_optional(cmd, "--previous-start", previous_start)
    _append_optional(cmd, "--previous-end", previous_end)


def _package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except Exception:
        return "unknown"


def _extract_version_line(result: dict[str, Any]) -> str:
    stdout = str(result.get("stdout", "") or "").strip()
    if stdout:
        return stdout.splitlines()[0]
    stderr = str(result.get("stderr", "") or "").strip()
    if stderr:
        return stderr.splitlines()[0]
    return "unknown"


def _load_json_file(path: Path, label: str) -> Any:
    if not path.exists():
        raise ValueError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{label} must be a file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc


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
    date_column: str | None = None,
    time_grain: str | None = None,
    period: str | None = None,
    anchor_date: str | None = None,
    current_start: str | None = None,
    current_end: str | None = None,
    previous_start: str | None = None,
    previous_end: str | None = None,
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
    _append_period_flags(
        cmd,
        date_column=date_column,
        time_grain=time_grain,
        period=period,
        anchor_date=anchor_date,
        current_start=current_start,
        current_end=current_end,
        previous_start=previous_start,
        previous_end=previous_end,
    )

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
    date_column: str | None = None,
    time_grain: str | None = None,
    period: str | None = None,
    anchor_date: str | None = None,
    current_start: str | None = None,
    current_end: str | None = None,
    previous_start: str | None = None,
    previous_end: str | None = None,
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
    _append_period_flags(
        cmd,
        date_column=date_column,
        time_grain=time_grain,
        period=period,
        anchor_date=anchor_date,
        current_start=current_start,
        current_end=current_end,
        previous_start=previous_start,
        previous_end=previous_end,
    )

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
    date_column: str | None = None,
    time_grain: str | None = None,
    period: str | None = None,
    anchor_date: str | None = None,
    current_start: str | None = None,
    current_end: str | None = None,
    previous_start: str | None = None,
    previous_end: str | None = None,
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
    _append_period_flags(
        cmd,
        date_column=date_column,
        time_grain=time_grain,
        period=period,
        anchor_date=anchor_date,
        current_start=current_start,
        current_end=current_end,
        previous_start=previous_start,
        previous_end=previous_end,
    )
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
    date_column: str | None = None,
    time_grain: str | None = None,
    period: str | None = None,
    anchor_date: str | None = None,
    current_start: str | None = None,
    current_end: str | None = None,
    previous_start: str | None = None,
    previous_end: str | None = None,
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
    _append_period_flags(
        cmd,
        date_column=date_column,
        time_grain=time_grain,
        period=period,
        anchor_date=anchor_date,
        current_start=current_start,
        current_end=current_end,
        previous_start=previous_start,
        previous_end=previous_end,
    )
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
def investigate(
    question: str,
    out: str,
    base: str | None = None,
    new: str | None = None,
    query: str | None = None,
    query_file: str | None = None,
    postgres_url: str | None = None,
    postgres_ssl_mode: str = "prefer",
    postgres_ca_file: str | None = None,
    date_column: str | None = None,
    time_grain: str | None = None,
    period: str | None = None,
    anchor_date: str | None = None,
    current_start: str | None = None,
    current_end: str | None = None,
    previous_start: str | None = None,
    previous_end: str | None = None,
    config: str | None = None,
    profile: str | None = None,
    profile_config: str | None = None,
    mode: str | None = None,
    metric: str | None = None,
    dimensions_csv: str | None = None,
    drill_fields_csv: str | None = None,
    max_depth: int = 2,
    max_branches: int = 1,
    min_contribution: float = 5.0,
    min_score_improvement: float = 0.0,
    min_slice_rows: int = 5,
    top_movers: int = 12,
    planner: str = "deterministic",
    planner_backend: str = "local",
    planner_model: str | None = None,
    verbose: bool = False,
    trace: bool = False,
    output_format: str = "both",
    timeout_sec: int | None = None,
) -> str:
    if not question.strip():
        raise ValueError("question must not be empty")
    if output_format not in VALID_INVESTIGATE_OUTPUT_FORMATS:
        raise ValueError(
            f"output_format must be one of {sorted(VALID_INVESTIGATE_OUTPUT_FORMATS)}"
        )
    if mode is not None and mode not in VALID_INVESTIGATE_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_INVESTIGATE_MODES)}")
    if max_depth < 1:
        raise ValueError("max_depth must be >= 1")
    if max_branches < 1:
        raise ValueError("max_branches must be >= 1")
    if min_score_improvement < 0.0:
        raise ValueError("min_score_improvement must be >= 0")
    if min_slice_rows < 1:
        raise ValueError("min_slice_rows must be >= 1")
    if top_movers < 1:
        raise ValueError("top_movers must be >= 1")
    if planner not in VALID_INVESTIGATE_PLANNERS:
        raise ValueError(f"planner must be one of {sorted(VALID_INVESTIGATE_PLANNERS)}")
    if planner_backend not in VALID_BACKENDS:
        raise ValueError(f"planner_backend must be one of {sorted(VALID_BACKENDS)}")
    if postgres_ssl_mode not in VALID_POSTGRES_SSL_MODES:
        raise ValueError(
            f"postgres_ssl_mode must be one of {sorted(VALID_POSTGRES_SSL_MODES)}"
        )
    if config and profile:
        raise ValueError("use either config or profile/profile_config, not both")
    if profile_config and not profile:
        raise ValueError("profile_config requires profile")

    has_pair_mode = bool(base) or bool(new)
    has_query_mode = bool(query) or bool(query_file) or bool(postgres_url)
    if has_pair_mode and has_query_mode:
        raise ValueError(
            "choose one input mode: (--base and --new) OR (--query/--query-file with optional --postgres-url)"
        )

    if has_pair_mode:
        if not base or not new:
            raise ValueError("provide both base and new for file input mode")
    else:
        if bool(query) == bool(query_file):
            raise ValueError("provide exactly one of query or query_file for query input mode")

    base_path: Path | None = None
    new_path: Path | None = None
    query_file_path: Path | None = None
    ca_file_path: Path | None = None
    if base:
        base_path = _validate_read_path(base, "base")
    if new:
        new_path = _validate_read_path(new, "new")
    if query_file:
        query_file_path = _validate_read_path(query_file, "query_file")
    if postgres_ca_file:
        ca_file_path = _validate_read_path(postgres_ca_file, "postgres_ca_file")

    out_path = _validate_write_path(out, "out")
    config_path: Path | None = None
    profile_config_path: Path | None = None
    if config:
        config_path = _validate_read_path(config, "config")
    if profile_config:
        profile_config_path = _validate_read_path(profile_config, "profile_config")

    cmd = [
        "investigate",
        "--question",
        question,
        "--out",
        str(out_path),
        "--output-format",
        output_format,
        "--max-depth",
        str(max_depth),
        "--max-branches",
        str(max_branches),
        "--min-contribution",
        str(min_contribution),
        "--min-score-improvement",
        str(min_score_improvement),
        "--min-slice-rows",
        str(min_slice_rows),
        "--top-movers",
        str(top_movers),
        "--planner",
        planner,
        "--planner-backend",
        planner_backend,
    ]

    if base_path and new_path:
        cmd.extend(["--base", str(base_path), "--new", str(new_path)])
    else:
        _append_optional(cmd, "--postgres-url", postgres_url)
        cmd.extend(["--postgres-ssl-mode", postgres_ssl_mode])
        if query:
            cmd.extend(["--query", query])
        if query_file_path:
            cmd.extend(["--query-file", str(query_file_path)])
        if ca_file_path:
            cmd.extend(["--postgres-ca-file", str(ca_file_path)])

    _append_period_flags(
        cmd,
        date_column=date_column,
        time_grain=time_grain,
        period=period,
        anchor_date=anchor_date,
        current_start=current_start,
        current_end=current_end,
        previous_start=previous_start,
        previous_end=previous_end,
    )
    _append_optional(cmd, "--metric", metric)
    if config_path:
        cmd.extend(["--config", str(config_path)])
    _append_optional(cmd, "--profile", profile)
    if profile_config_path:
        cmd.extend(["--profile-config", str(profile_config_path)])
    _append_optional(cmd, "--mode", mode)
    _append_optional(cmd, "--dimensions", dimensions_csv)
    _append_optional(cmd, "--drill-fields", drill_fields_csv)
    _append_optional(cmd, "--planner-model", planner_model)
    if verbose:
        cmd.append("--verbose")
    if trace:
        cmd.append("--trace")

    return json.dumps(_run(cmd, timeout_sec), ensure_ascii=True)


@mcp.tool()
def explain_analyze(
    analysis_json: str,
    question: str,
    backend: str = "bedrock",
    model: str = "anthropic.claude-3-haiku-20240307-v1:0",
    strict_facts: bool = True,
    max_bullets: int = 5,
    timeout_sec: int | None = None,
) -> str:
    if backend not in VALID_BACKENDS:
        raise ValueError(f"backend must be one of {sorted(VALID_BACKENDS)}")
    if not question.strip():
        raise ValueError("question must not be empty")
    if max_bullets < 1:
        raise ValueError("max_bullets must be >= 1")

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
        "--max-bullets",
        str(max_bullets),
    ]
    if strict_facts:
        cmd.extend(["--strict-facts", "true"])
    else:
        cmd.extend(["--strict-facts", "false"])

    return json.dumps(_run(cmd, timeout_sec), ensure_ascii=True)


@mcp.tool()
def server_info(timeout_sec: int | None = 20) -> str:
    timeout = _timeout_sec(timeout_sec)
    try:
        version_probe = _run(["--version"], timeout)
    except Exception as exc:
        version_probe = {
            "ok": False,
            "return_code": 127,
            "cmd": [],
            "stdout": "",
            "stderr": str(exc),
            "timeout_sec": timeout,
        }

    payload = {
        "ok": bool(version_probe.get("ok", False)),
        "mcp_server_name": "factorlens",
        "mcp_sdk_version": _package_version("mcp"),
        "factorlens_version": _extract_version_line(version_probe),
        "factorlens_version_probe": version_probe,
    }
    return json.dumps(payload, ensure_ascii=True)


@mcp.tool()
def summarize_investigate(
    analysis_json: str,
    top_major_changes: int = 3,
    top_follow_up_steps: int = 5,
) -> str:
    if top_major_changes < 1:
        raise ValueError("top_major_changes must be >= 1")
    if top_follow_up_steps < 1:
        raise ValueError("top_follow_up_steps must be >= 1")

    analysis_path = _validate_read_path(analysis_json, "analysis_json")
    payload = _load_json_file(analysis_path, "analysis_json")
    if not isinstance(payload, dict):
        raise ValueError("analysis_json must contain a JSON object")

    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("analysis_json does not look like investigate output (missing steps)")

    step0 = steps[0] if isinstance(steps[0], dict) else {}
    movers0 = step0.get("movers", [])
    if not isinstance(movers0, list):
        movers0 = []
    base_total = sum(
        float(item.get("base_primary_metric_value", 0.0))
        for item in movers0
        if isinstance(item, dict)
    )
    new_total = sum(
        float(item.get("new_primary_metric_value", 0.0))
        for item in movers0
        if isinstance(item, dict)
    )
    top_mover = movers0[0] if movers0 and isinstance(movers0[0], dict) else {}

    major_raw = payload.get("major_global_changes", [])
    major_changes: list[dict[str, Any]] = []
    if isinstance(major_raw, list):
        for item in major_raw[:top_major_changes]:
            if not isinstance(item, dict):
                continue
            major_changes.append(
                {
                    "dimension": item.get("dimension"),
                    "segment": item.get("segment"),
                    "primary_metric": item.get("primary_metric"),
                    "delta_primary_metric_value": item.get("delta_primary_metric_value"),
                    "delta_share_pp": item.get("delta_share_pp"),
                    "score": item.get("score"),
                }
            )

    follow_up: list[dict[str, Any]] = []
    for step in steps[1 : 1 + top_follow_up_steps]:
        if not isinstance(step, dict):
            continue
        movers = step.get("movers", [])
        strongest = movers[0] if isinstance(movers, list) and movers and isinstance(movers[0], dict) else {}
        follow_up.append(
            {
                "depth": step.get("depth"),
                "dimension": step.get("dimension"),
                "scope": step.get("scope", []),
                "strongest_segment": strongest.get("segment"),
                "delta_primary_metric_value": strongest.get("delta_primary_metric_value"),
                "delta_share_pp": strongest.get("delta_share_pp"),
            }
        )

    response = {
        "ok": True,
        "schema": "investigate",
        "source_path": str(analysis_path),
        "question": payload.get("question"),
        "mode": payload.get("mode"),
        "top_level": {
            "dimension": step0.get("dimension"),
            "primary_metric": step0.get("primary_metric"),
            "base_total": base_total,
            "new_total": new_total,
            "delta_total": new_total - base_total,
            "strongest_segment": top_mover.get("segment"),
            "strongest_delta_primary_metric_value": top_mover.get("delta_primary_metric_value"),
            "strongest_delta_share_pp": top_mover.get("delta_share_pp"),
            "top1_concentration_base_pct": step0.get("top1_concentration_base_pct"),
            "top1_concentration_new_pct": step0.get("top1_concentration_new_pct"),
            "top1_concentration_delta_pp": step0.get("top1_concentration_delta_pp"),
            "top5_concentration_base_pct": step0.get("top5_concentration_base_pct"),
            "top5_concentration_new_pct": step0.get("top5_concentration_new_pct"),
            "top5_concentration_delta_pp": step0.get("top5_concentration_delta_pp"),
        },
        "major_global_changes": major_changes,
        "follow_up": follow_up,
        "stopping_reason": payload.get("stopping_reason"),
        "recommended_next_question": payload.get("recommended_next_question"),
    }
    return json.dumps(response, ensure_ascii=True)


@mcp.tool()
def read_artifact(path: str, max_chars: int = 200000) -> str:
    if max_chars < 1 or max_chars > MAX_READ_ARTIFACT_CHARS:
        raise ValueError(
            f"max_chars must be between 1 and {MAX_READ_ARTIFACT_CHARS}"
        )

    artifact_path = _validate_read_path(path, "path")
    if not artifact_path.exists():
        raise ValueError(f"path does not exist: {artifact_path}")
    if not artifact_path.is_file():
        raise ValueError(f"path must be a file: {artifact_path}")

    with artifact_path.open("r", encoding="utf-8", errors="replace") as handle:
        content = handle.read(max_chars + 1)
    truncated = len(content) > max_chars
    if truncated:
        content = content[:max_chars]

    return json.dumps(
        {
            "ok": True,
            "path": str(artifact_path),
            "truncated": truncated,
            "max_chars": max_chars,
            "content": content,
        },
        ensure_ascii=True,
    )


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
