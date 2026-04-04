#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p "$ROOT_DIR/data" "$ROOT_DIR/profiles" "$ROOT_DIR/artifacts" "$ROOT_DIR/certs"

if [[ -z "${FACTORLENS_BIN:-}" ]]; then
  if [[ -x "$ROOT_DIR/target/release/factorlens" ]]; then
    FACTORLENS_BIN="$ROOT_DIR/target/release/factorlens"
  elif command -v factorlens >/dev/null 2>&1; then
    FACTORLENS_BIN="$(command -v factorlens)"
  else
    echo "factorlens binary not found." >&2
    echo "Build it first with: cargo build --release -p factor_cli" >&2
    exit 1
  fi
fi

if ! python3 -c 'import mcp' >/dev/null 2>&1; then
  echo "Python package 'mcp' is not installed." >&2
  echo "Install it with: pip install mcp awscli" >&2
  exit 1
fi

export FACTORLENS_BIN
export FACTORLENS_ALLOWED_READ_DIRS="${FACTORLENS_ALLOWED_READ_DIRS:-$ROOT_DIR/data,$ROOT_DIR/profiles,$ROOT_DIR/artifacts,$ROOT_DIR/certs}"
export FACTORLENS_ALLOWED_WRITE_DIRS="${FACTORLENS_ALLOWED_WRITE_DIRS:-$ROOT_DIR/artifacts}"
export FACTORLENS_CMD_TIMEOUT_SEC="${FACTORLENS_CMD_TIMEOUT_SEC:-300}"

export MCP_TRANSPORT="${MCP_TRANSPORT:-streamable-http}"
export FASTMCP_HOST="${FASTMCP_HOST:-127.0.0.1}"
export FASTMCP_PORT="${FASTMCP_PORT:-8010}"
export FASTMCP_MOUNT_PATH="${FASTMCP_MOUNT_PATH:-/}"
export FASTMCP_SSE_PATH="${FASTMCP_SSE_PATH:-/sse}"
export FASTMCP_MESSAGE_PATH="${FASTMCP_MESSAGE_PATH:-/messages/}"
export FASTMCP_STREAMABLE_HTTP_PATH="${FASTMCP_STREAMABLE_HTTP_PATH:-/mcp}"
export FASTMCP_LOG_LEVEL="${FASTMCP_LOG_LEVEL:-INFO}"

echo "Starting FactorLens MCP server"
echo "  root: $ROOT_DIR"
echo "  factorlens: $FACTORLENS_BIN"
echo "  transport: $MCP_TRANSPORT"
echo "  endpoint: http://$FASTMCP_HOST:$FASTMCP_PORT$FASTMCP_STREAMABLE_HTTP_PATH"

exec python3 "$ROOT_DIR/scripts/mcp/factorlens_mcp_server.py"
