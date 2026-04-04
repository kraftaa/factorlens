# FactorLens MCP Local Runbook (Docker + curl)

## 0) Build MCP image
# Builds container with:
# - factorlens Rust binary
# - Python MCP server wrapper
# - streamable HTTP transport support

docker build -f deploy/mcp/docker/Dockerfile -t ghcr.io/kraftaa/factorlens-mcp:0.3.8 .


## 1) Run MCP server locally in Docker
# Host port 8010 maps to container port 8010.
# Volumes:
# - /data: CSV inputs (read-only)
# - /profiles: profile TOML (read-only)
# - /artifacts: outputs (read-write)
# - /certs: CA bundle for TLS DB connections (read-only)

docker run --rm -p 8010:8010 \
  -e MCP_TRANSPORT=streamable-http \
  -e FASTMCP_HOST=0.0.0.0 \
  -e FASTMCP_PORT=8010 \
  -e FASTMCP_STREAMABLE_HTTP_PATH=/mcp \
  -e FACTORLENS_ALLOWED_READ_DIRS=/data,/profiles,/artifacts,/certs \
  -e FACTORLENS_ALLOWED_WRITE_DIRS=/artifacts \
  -v /path/to/factorlens/data:/data:ro \
  -v /path/to/factorlens/profiles:/profiles:ro \
  -v /path/to/factorlens/artifacts:/artifacts \
  -v /path/to/factorlens/certs:/certs:ro \
  ghcr.io/kraftaa/factorlens-mcp:0.3.8


## 2) Health check the MCP endpoint
# Note: GET /mcp can return empty/non-human response depending on transport behavior.
# Real protocol check is done via initialize POST below.

curl -i http://127.0.0.1:8010/mcp


## 3) Initialize MCP session
# Required for stateful streamable HTTP.
# Saves response headers to /tmp/mcp_headers.txt and extracts mcp-session-id.

curl -sS -D /tmp/mcp_headers.txt -o /tmp/mcp_init.json \
  -X POST http://127.0.0.1:8010/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  --data '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"curl","version":"0.1"}}}'

SESSION_ID=$(awk -F': ' 'tolower($1)=="mcp-session-id"{print $2}' /tmp/mcp_headers.txt | tr -d '\r')
echo "SESSION_ID=$SESSION_ID"


## 4) List available tools
# Confirms server is live and shows all MCP tools exposed by FactorLens.

curl -i -X POST http://127.0.0.1:8010/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION_ID" \
  --data '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'


## 5) Call analyze_csv tool via MCP
# Runs FactorLens analyze from MCP and writes:
# - /artifacts/demo_sales_mcp.md
# - /artifacts/demo_sales_mcp.json

curl -i -X POST http://127.0.0.1:8010/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION_ID" \
  --data '{
    "jsonrpc":"2.0",
    "id":3,
    "method":"tools/call",
    "params":{
      "name":"analyze_csv",
      "arguments":{
        "input_csv":"/data/factorlens_demo_sales.csv",
        "out":"/artifacts/demo_sales_mcp.md",
        "group_by_csv":"region,channel,product_line,plan_tier",
        "metrics_csv":"revenue_usd,cost_usd,orders",
        "rank_by":"revenue_usd",
        "agg":"sum",
        "top":12,
        "min_records":1,
        "output_format":"both"
      }
    }
  }'


## 6) Verify generated files on host
# These paths are on your Mac because /artifacts is volume-mounted.

ls -lh /path/to/factorlens/artifacts/demo_sales_mcp.md
ls -lh /path/to/factorlens/artifacts/demo_sales_mcp.json


## 7) Common issues
# "Empty reply from server":
# - Usually missing required headers or no initialize/session id.
# - Ensure Accept includes both application/json and text/event-stream.
#
# "Not Acceptable":
# - Add header: Accept: application/json, text/event-stream
#
# tools/list fails after initialize:
# - Re-extract SESSION_ID from /tmp/mcp_headers.txt and retry.


# Kubernetes (namespace: analytics)

## K1) Apply manifests

kubectl -n analytics apply -f deploy/mcp/k8s/configmap.yaml
kubectl -n analytics apply -f deploy/mcp/k8s/service.yaml
kubectl -n analytics apply -f deploy/mcp/k8s/deployment.yaml


## K2) Optional secrets (DB URL + RDS CA)
# DATABASE_URL for query mode:

kubectl -n analytics create secret generic factorlens-mcp-secrets \
  --from-literal=DATABASE_URL='postgres://USER:PASS@HOST:5432/DBNAME'

# RDS CA PEM for --postgres-ca-file:

kubectl -n analytics create secret generic factorlens-db-ca \
  --from-file=rds-global-bundle.pem=certs/rds-global-bundle.pem

# Restart deployment after creating/changing secrets:

kubectl -n analytics rollout restart deploy/factorlens-mcp
kubectl -n analytics rollout status deploy/factorlens-mcp
kubectl -n analytics get pods -l app=factorlens-mcp


## K3) Port-forward service to local 8010

kubectl -n analytics port-forward svc/factorlens-mcp 8010:80


## K4) Initialize session + tools/list (in another terminal)

curl -sS -D /tmp/mcp_headers.txt -o /tmp/mcp_init.json \
  -X POST http://127.0.0.1:8010/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  --data '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"curl","version":"0.1"}}}'

SESSION_ID=$(awk -F': ' 'tolower($1)=="mcp-session-id"{print $2}' /tmp/mcp_headers.txt | tr -d '\r')
echo "SESSION_ID=$SESSION_ID"

curl -i -X POST http://127.0.0.1:8010/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION_ID" \
  --data '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'


## K5) Validate query before full run

export PG_URL='postgres://USER:PASS@HOST:5432/DBNAME'
export Q_VALIDATE="select 'test'::text as group_key, 1::numeric as metric_value"

cat > /tmp/mcp_validate.json <<EOF
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "tools/call",
  "params": {
    "name": "analyze_validate_query",
    "arguments": {
      "postgres_url": $(python3 -c 'import json,os; print(json.dumps(os.environ["PG_URL"]))'),
      "query": $(python3 -c 'import json,os; print(json.dumps(os.environ["Q_VALIDATE"]))'),
      "group_by_csv": "group_key",
      "metrics_csv": "metric_value",
      "rank_by": "metric_value",
      "postgres_ssl_mode": "require",
      "postgres_ca_file": "/certs/rds-global-bundle.pem"
    }
  }
}
EOF

curl -i -X POST http://127.0.0.1:8010/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION_ID" \
  --data @/tmp/mcp_validate.json


## K6) Run full analysis query

export Q_ANALYZE='select category, customer, date, order_id, retail_amount from public.orders'

cat > /tmp/mcp_analyze.json <<EOF
{
  "jsonrpc": "2.0",
  "id": 5,
  "method": "tools/call",
  "params": {
    "name": "analyze_query",
    "arguments": {
      "postgres_url": $(python3 -c 'import json,os; print(json.dumps(os.environ["PG_URL"]))'),
      "query": $(python3 -c 'import json,os; print(json.dumps(os.environ["Q_ANALYZE"]))'),
      "postgres_ssl_mode": "require",
      "postgres_ca_file": "/certs/rds-global-bundle.pem",
      "group_by_csv": "category,customer",
      "metrics_csv": "retail_amount",
      "rank_by": "retail_amount",
      "out": "/artifacts/orders_mcp.md",
      "output_format": "both"
    }
  }
}
EOF

curl -i -X POST http://127.0.0.1:8010/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION_ID" \
  --data @/tmp/mcp_analyze.json


## K7) Explain analysis output

cat > /tmp/mcp_explain.json <<EOF
{
  "jsonrpc": "2.0",
  "id": 6,
  "method": "tools/call",
  "params": {
    "name": "explain_analyze",
    "arguments": {
      "backend": "bedrock",
      "model": "anthropic.claude-3-haiku-20240307-v1:0",
      "analysis_json": "/artifacts/orders_mcp.json",
      "question": "Top concentration risks and 3 actions?"
    }
  }
}
EOF

curl -i -X POST http://127.0.0.1:8010/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION_ID" \
  --data @/tmp/mcp_explain.json


# 1) run analysis from Postgres via MCP
curl -i -X POST http://127.0.0.1:8010/mcp \
-H "Content-Type: application/json" \
-H "Accept: application/json, text/event-stream" \
-H "mcp-session-id: $SESSION_ID" \
--data '{
"jsonrpc":"2.0",
"id":7,
"method":"tools/call",
"params":{
"name":"analyze_query",
"arguments":{
  "query":"select category, customer, date, order_id, retail_amount from public.orders",
  "postgres_ssl_mode":"require",
  "postgres_ca_file":"/certs/rds-global-bundle.pem",
  "group_by_csv":"category,customer",
  "metrics_csv":"retail_amount",
  "rank_by":"retail_amount",
  "out":"/artifacts/orders_mcp.md",
  "output_format":"both"
  }
 }
}'
curl -i -X POST http://127.0.0.1:8010/mcp \
-H "Content-Type: application/json" \
-H "Accept: application/json, text/event-stream" \
-H "mcp-session-id: $SESSION_ID" \
--data "{
\"jsonrpc\":\"2.0\",
\"id\":7,
\"method\":\"tools/call\",
\"params\":{
\"name\":\"analyze_query\",
\"arguments\":{
\"postgres_url\":\"$PG_URL\",
\"query\":\"select category, customer, date, order_id, retail_amount from public.orders\",
\"postgres_ssl_mode\":\"require\",
\"postgres_ca_file\":\"/certs/rds-global-bundle.pem\",
\"group_by_csv\":\"category,customer\",
\"metrics_csv\":\"retail_amount\",
\"rank_by\":\"retail_amount\",
\"out\":\"/artifacts/orders_mcp.md\",
\"output_format\":\"both\"
}
}
}"
