# FactorLens MCP Server

This folder provides a production-oriented MCP wrapper around the `factorlens` CLI.

## File

- `scripts/mcp/factorlens_mcp_server.py`

## What It Exposes

- `analyze_csv`: run `factorlens analyze` on CSV input
- `analyze_compare`: compare two `analysis.json` snapshots
- `explain_analyze`: run `factorlens explain-analyze` (Bedrock or local)
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

```json
{
  "input_csv": "/path/to/data/input.csv",
  "out": "/path/to/artifacts/demo_exec.html",
  "profile": "demo_exec",
  "profile_config": "/path/to/profiles/profile.toml",
  "count_only": true,
  "output_format": "html"
}
```

Explain analysis:

```json
{
  "analysis_json": "/path/to/artifacts/analysis.json",
  "backend": "bedrock",
  "model": "anthropic.claude-3-haiku-20240307-v1:0",
  "question": "Top concentration risks and 3 actions?"
}
```
