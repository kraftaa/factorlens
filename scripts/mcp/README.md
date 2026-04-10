# FactorLens MCP Server

This folder provides a production-oriented MCP wrapper around the `factorlens` CLI.

## File

- `scripts/mcp/factorlens_mcp_server.py`

## What It Exposes

- `analyze_csv`: run `factorlens analyze` on CSV input
- `analyze_query`: run `factorlens analyze` on Postgres query input
- `analyze_compare`: compare two `analysis.json` snapshots (`md`, `html`, `json`, or `both`)
- `analyze_investigate_legacy`: legacy-compatible numeric-driver decomposition (`analyze-investigate`)
- `investigate`: guided drill-down across base/new snapshots (deterministic or LLM planner)
- `explain_analyze`: run `factorlens explain-analyze` (Bedrock or local)
- `server_info`: report MCP SDK version + factorlens CLI version
- `tool_guide`: quick "which tool to use when" routing guide for clients/agents
- `summarize_investigate`: deterministic JSON-only summary from investigate output JSON
- `read_artifact`: read generated `.md`/`.json` outputs from allowed paths
- `healthcheck`: quick CLI availability check

All tool responses are JSON strings with this shape:

```json
{
  "ok": true,
  "return_code": 0,
  "cmd": ["..."],
  "stdout": "...",
  "stderr": "...",
  "timeout_sec": 180
}
```

## Requirements

- `factorlens` binary available (`PATH` or `FACTORLENS_BIN`)
- Python package: `mcp`

Install:

```bash
pip install mcp
```

## Run

```bash
python scripts/mcp/factorlens_mcp_server.py
```

## Security Controls

The server validates filesystem paths against allowlists.

Environment variables:

- `FACTORLENS_BIN`: absolute path to factorlens binary (optional)
- `FACTORLENS_ALLOWED_READ_DIRS`: comma-separated readable roots
- `FACTORLENS_ALLOWED_WRITE_DIRS`: comma-separated writable roots
- `FACTORLENS_CMD_TIMEOUT_SEC`: default timeout per command
- `MCP_TRANSPORT`: `stdio` (default), `sse`, or `streamable-http`
- `MCP_MOUNT_PATH`: optional mount path when `MCP_TRANSPORT=sse`
- `FASTMCP_HOST`, `FASTMCP_PORT`, `FASTMCP_STREAMABLE_HTTP_PATH`: network settings for hosted MCP mode

Defaults (if env vars omitted):

- Read: `./data, ./profiles, ./artifacts, ./`
- Write: `./artifacts, ./`
- Timeout: `180`

Example strict setup:

```bash
export FACTORLENS_BIN=/usr/local/bin/factorlens
export FACTORLENS_ALLOWED_READ_DIRS=/path/to/data,/path/to/profiles,/path/to/artifacts
export FACTORLENS_ALLOWED_WRITE_DIRS=/path/to/artifacts
export FACTORLENS_CMD_TIMEOUT_SEC=180
python scripts/mcp/factorlens_mcp_server.py
```

## Example MCP Calls

Analyze CSV:
(These examples use placeholder paths.)
```json
{
  "input_csv": "/path/to/input.csv",
  "out": "/path/to/artifacts/report.md",
  "group_by_csv": "region,channel",
  "metrics_csv": "revenue_usd",
  "rank_by": "revenue_usd",
  "output_format": "both"
}
```

Explain analysis:

```json
{
  "analysis_json": "/path/to/artifacts/analysis.json",
  "backend": "bedrock",
  "model": "anthropic.claude-3-haiku-20240307-v1:0",
  "question": "What are the main drivers of revenue concentration?",
  "strict_facts": true,
  "max_bullets": 5
}
```

Investigate with config defaults:

```json
{
  "question": "Why did revenue change?",
  "base": "/path/to/feb.csv",
  "new": "/path/to/mar.csv",
  "out": "/path/to/artifacts/investigate.md",
  "profile": "default",
  "profile_config": "/path/to/profiles/investigate.example.toml",
  "mode": "change_drivers"
}
```

Legacy numeric-driver decomposition:

```json
{
  "input_csv": "/path/to/snapshot.csv",
  "metric": "revenue_usd",
  "drivers_csv": "net_gmv,avg_order_value",
  "driver_contrib": "both",
  "out": "/path/to/artifacts/investigate_legacy.md",
  "output_format": "both"
}
```

Investigate from Postgres query with period windows:

```json
{
  "question": "Why did revenue change?",
  "out": "/path/to/artifacts/investigate_query.md",
  "query": "select order_date, region, channel, product_line, plan_tier, revenue_usd from analytics.sales",
  "date_column": "order_date",
  "time_grain": "month",
  "period": "last",
  "anchor_date": "2026-04-15",
  "metric": "revenue_usd",
  "mode": "change_drivers",
  "planner": "deterministic",
  "output_format": "both"
}
```

Read investigate/analyze artifacts:

```json
{
  "path": "/path/to/artifacts/investigate_report.md",
  "max_chars": 120000
}
```

Report server/runtime versions:

```json
{
  "timeout_sec": 20
}
```

Get method guidance for clients/agents:

```json
{}
```

`tool_guide` also returns a `legacy_mapping` section so clients can route
legacy CLI workflows (for example `analyze-investigate`) to MCP-safe flows.

Summarize investigate JSON (no markdown rendering issues):

```json
{
  "analysis_json": "/path/to/artifacts/investigate_mcp.json",
  "top_major_changes": 3,
  "top_follow_up_steps": 5
}
```

Analyze Postgres query:

```json
{
  "out": "/path/to/artifacts/query_report.md",
  "query": "select region, channel, revenue_usd from analytics.sales limit 5000",
  "postgres_ssl_mode": "require",
  "postgres_ca_file": "/path/to/rds-global-bundle.pem",
  "group_by_csv": "region,channel",
  "metrics_csv": "revenue_usd",
  "rank_by": "revenue_usd",
  "output_format": "both"
}
```
