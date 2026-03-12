#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="artifacts/demo"
mkdir -p "$OUT_DIR"

if command -v factorlens >/dev/null 2>&1; then
  FL=(factorlens)
else
  FL=(cargo run -p factor_cli --)
fi

echo "[1/4] Analyze baseline snapshot (100 rows)..."
"${FL[@]}" analyze \
  --input data/factorlens_demo_sales_100.csv \
  --group-by region,channel,product_line,plan_tier \
  --metrics revenue_usd,cost_usd,orders \
  --rank-by revenue_usd \
  --out "$OUT_DIR/demo_sales_100.md"

echo "[2/4] Analyze new snapshot (150 rows)..."
"${FL[@]}" analyze \
  --input data/factorlens_demo_sales_150.csv \
  --group-by region,channel,product_line,plan_tier \
  --metrics revenue_usd,cost_usd,orders \
  --rank-by revenue_usd \
  --out "$OUT_DIR/demo_sales_150.md"

echo "[3/4] Compare snapshots..."
"${FL[@]}" analyze-compare \
  --base "$OUT_DIR/demo_sales_100.json" \
  --new "$OUT_DIR/demo_sales_150.json" \
  --output-format both \
  --out "$OUT_DIR/demo_compare.md"

echo "[4/4] Optional Bedrock summary..."
if [[ "${RUN_BEDROCK:-0}" == "1" ]]; then
  : "${AWS_REGION:?AWS_REGION is required when RUN_BEDROCK=1}"
  MODEL_ID="${MODEL_ID:-anthropic.claude-3-haiku-20240307-v1:0}"
  "${FL[@]}" explain-analyze \
    --backend bedrock \
    --model "$MODEL_ID" \
    --analysis-json "$OUT_DIR/demo_sales_150.json" \
    --question "What are the top concentration risks and what 3 actions should we take in the next 30 days?"
else
  echo "Skipping Bedrock. Set RUN_BEDROCK=1 to enable."
fi

echo "Done. Outputs:"
ls -1 "$OUT_DIR" | sed 's/^/ - /'
