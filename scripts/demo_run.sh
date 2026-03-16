#!/usr/bin/env bash
set -euo pipefail

# One-command local demo runner.
# Optional:
#   RUN_BEDROCK=1 AWS_REGION=us-east-1 MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0 ./scripts/demo_run.sh
#   DATASET_A_INPUT=data/your_dataset_a.csv
#   DATASET_A_PROFILE=your_profile_a
#   DATASET_A_PROFILE_CONFIG=profiles/your_profiles_a.toml
#   DATASET_B_INPUT=data/your_dataset_b.csv
#   DATASET_B_GROUP_BY=title
#   DATASET_B_METRICS=revenue_metric
#   DATASET_B_PROFILE=your_profile_b
#   DATASET_B_PROFILE_CONFIG=profiles/your_profiles_b.toml

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="artifacts/demo"
mkdir -p "$OUT_DIR"

if command -v factorlens >/dev/null 2>&1; then
  FL=(factorlens)
else
  FL=(cargo run -p factor_cli --)
fi

DATASET_A_INPUT="${DATASET_A_INPUT:-data/your_dataset_a.csv}"
DATASET_A_PROFILE="${DATASET_A_PROFILE:-exec}"
DATASET_A_PROFILE_CONFIG="${DATASET_A_PROFILE_CONFIG:-profiles/profiles.example.toml}"
DATASET_A_OUT="${DATASET_A_OUT:-$OUT_DIR/dataset_a_exec.md}"

DATASET_B_INPUT="${DATASET_B_INPUT:-data/your_dataset_b.csv}"
DATASET_B_OUT="${DATASET_B_OUT:-$OUT_DIR/dataset_b_text.md}"
DATASET_B_GROUP_BY="${DATASET_B_GROUP_BY:-title}"
DATASET_B_METRICS="${DATASET_B_METRICS:-revenue_metric}"
DATASET_B_PROFILE="${DATASET_B_PROFILE:-}"
DATASET_B_PROFILE_CONFIG="${DATASET_B_PROFILE_CONFIG:-}"

echo "[1/3] Dataset A executive analysis..."
"${FL[@]}" analyze \
  --input "$DATASET_A_INPUT" \
  --profile "$DATASET_A_PROFILE" \
  --profile-config "$DATASET_A_PROFILE_CONFIG" \
  --out "$DATASET_A_OUT"

echo "[2/3] Dataset B distribution + text insights..."
if [[ -n "$DATASET_B_PROFILE" && -n "$DATASET_B_PROFILE_CONFIG" ]]; then
  "${FL[@]}" analyze \
    --input "$DATASET_B_INPUT" \
    --profile "$DATASET_B_PROFILE" \
    --profile-config "$DATASET_B_PROFILE_CONFIG" \
    --agg median \
    --percentiles p50,p90 \
    --normalize-text-groups \
    --word-freq \
    --min-records 5 \
    --out "$DATASET_B_OUT"
else
  "${FL[@]}" analyze \
    --input "$DATASET_B_INPUT" \
    --group-by "$DATASET_B_GROUP_BY" \
    --metrics "$DATASET_B_METRICS" \
    --agg median \
    --percentiles p50,p90 \
    --normalize-text-groups \
    --word-freq \
    --min-records 5 \
    --out "$DATASET_B_OUT"
fi

echo "[3/3] Done. Outputs:"
ls -1 "$OUT_DIR" | sed 's/^/ - /'

if [[ "${RUN_BEDROCK:-0}" == "1" ]]; then
  : "${AWS_REGION:?AWS_REGION is required when RUN_BEDROCK=1}"
  MODEL_ID="${MODEL_ID:-anthropic.claude-3-haiku-20240307-v1:0}"
  REPORT="$(sed -n '1,260p' "$DATASET_A_OUT")"
  PROMPT="You are a business analyst. Summarize key risks/opportunities in 5 bullets and 3 actions.

${REPORT}"
  export PROMPT

  cat > /tmp/bedrock_analysis.json <<EOF
{
  "messages": [
    {
      "role": "user",
      "content": [
        { "text": $(python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))' <<< "$PROMPT") }
      ]
    }
  ],
  "inferenceConfig": {
    "maxTokens": 700,
    "temperature": 0.1
  }
}
EOF

  echo "[Bedrock] Summarizing analysis report..."
  aws bedrock-runtime converse \
    --region "$AWS_REGION" \
    --model-id "$MODEL_ID" \
    --cli-input-json file:///tmp/bedrock_analysis.json \
    --query 'output.message.content[0].text' \
    --output text
fi
