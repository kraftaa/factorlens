# FactorLens MCP Deployment

This folder lets you run FactorLens MCP as a shared service for multiple agents.

## What This Deploys

- FactorLens binary (`/app/bin/factorlens`)
- MCP wrapper (`scripts/mcp/factorlens_mcp_server.py`)
- FastMCP transport in `streamable-http` mode at `/mcp`

## 1) Build and Run with Docker

From repo root:

```bash
docker build -f deploy/mcp/docker/Dockerfile -t factorlens-mcp:latest .
```

Run locally:

```bash
docker run --rm -p 8000:8000 \
  -e MCP_TRANSPORT=streamable-http \
  -e FASTMCP_HOST=0.0.0.0 \
  -e FASTMCP_PORT=8000 \
  -e FASTMCP_STREAMABLE_HTTP_PATH=/mcp \
  -e FACTORLENS_ALLOWED_READ_DIRS=/data,/profiles,/artifacts \
  -e FACTORLENS_ALLOWED_WRITE_DIRS=/artifacts \
  -v "$PWD/data:/data:ro" \
  -v "$PWD/profiles:/profiles:ro" \
  -v "$PWD/artifacts:/artifacts" \
  factorlens-mcp:latest
```

Shared endpoint:

- `http://localhost:8000/mcp`

## 2) Deploy to Kubernetes

1. Update image in `deploy/mcp/k8s/deployment.yaml`.
2. Update namespace/PVC names to your cluster values.
3. Apply manifests:

```bash
kubectl apply -f deploy/mcp/k8s/configmap.yaml
kubectl apply -f deploy/mcp/k8s/secret.example.yaml
kubectl apply -f deploy/mcp/k8s/deployment.yaml
kubectl apply -f deploy/mcp/k8s/service.yaml
```

Service endpoint inside cluster:

- `http://factorlens-mcp.analytics.svc.cluster.local/mcp`

Ingress:

- Not required for in-cluster clients.
- Use ingress only if clients are outside the cluster/VPC.
- If needed, use `deploy/mcp/k8s/ingress.example.yaml` with a real DNS host and TLS secret.

## 3) Multi-Agent Access (Gateway Pattern)

`streamable-http` is the shared transport/gateway layer.

- Agents that support remote MCP over HTTP can connect directly to `/mcp`.
- Clients that only support local `stdio` need a local bridge/proxy that forwards to this hosted endpoint.

## Security Notes

- Keep service internal if possible.
- Restrict ingress by CIDR/authn/authz.
- Use minimal `FACTORLENS_ALLOWED_READ_DIRS` and `FACTORLENS_ALLOWED_WRITE_DIRS`.
- Prefer IRSA/Workload Identity for AWS over static keys.
