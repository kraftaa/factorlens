# syntax=docker/dockerfile:1.7

FROM rust:1.86-bookworm AS builder
WORKDIR /src

COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

RUN cargo build --release -p factor_cli

FROM python:3.12-slim AS runtime
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "mcp<2" awscli

COPY --from=builder /src/target/release/factorlens /app/bin/factorlens
COPY scripts/mcp/factorlens_mcp_server.py /app/factorlens_mcp_server.py

ENV FACTORLENS_BIN=/app/bin/factorlens \
    FACTORLENS_ALLOWED_READ_DIRS=/data,/profiles,/artifacts \
    FACTORLENS_ALLOWED_WRITE_DIRS=/artifacts \
    FACTORLENS_CMD_TIMEOUT_SEC=300 \
    MCP_TRANSPORT=streamable-http \
    FASTMCP_HOST=0.0.0.0 \
    FASTMCP_PORT=8000 \
    FASTMCP_STREAMABLE_HTTP_PATH=/mcp

EXPOSE 8000

ENTRYPOINT ["python", "/app/factorlens_mcp_server.py"]
