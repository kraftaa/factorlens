# FactorLens Demo (Wednesday Runbook)

## Goal
Show that FactorLens turns raw operational data into repeatable, executive-ready insights with optional Bedrock explanation.

## Demo Story (10 minutes)

1. Dataset A executive concentration view (profile-based)
2. Dataset B distribution + text insights (median + p50/p90 + word frequency)
3. Postgres read path (TLS + RDS CA bundle)
4. Optional Bedrock narrative from generated markdown

## Prerequisites

- `factorlens` installed and runnable
- Data files present:
  - `data/your_dataset_a.csv`
  - `data/your_dataset_b.csv`
- Profile files present:
  - `profiles/your_profiles_a.toml`
  - `profiles/your_profiles_b.toml`
- Optional Bedrock:
  - `aws` CLI installed
  - `AWS_REGION` set
  - model access enabled

## Step 1: Dataset A Executive View

```bash
factorlens analyze \
  --input data/your_dataset_a.csv \
  --profile your_profile_a \
  --profile-config profiles/your_profiles_a.toml \
  --out artifacts/demo/dataset_a_exec.md
```

Show:
- `artifacts/demo/dataset_a_exec.md`
- Explain top concentration and top 5 segment names.

## Step 2: Dataset B Robust Distribution + Text Signals

```bash
factorlens analyze \
  --input data/your_dataset_b.csv \
  --group-by title \
  --metrics revenue_metric \
  --agg median \
  --percentiles p50,p90 \
  --normalize-text-groups \
  --word-freq \
  --min-records 5 \
  --out artifacts/demo/dataset_b_text.md
```

Show:
- `artifacts/demo/dataset_b_text.md`
- `Top Words`
- `p50/p90` columns

## Step 3: Postgres Path (Production Readiness)

If DB TLS works with CA:

```bash
factorlens analyze \
  --query "select * from schema.table_a limit 5000" \
  --postgres-ssl-mode require \
  --postgres-ca-file /path/to/rds-global-bundle.pem \
  --profile your_profile_a \
  --profile-config /path/to/profiles/your_profiles_a.toml \
  --out /path/to/artifacts/demo/dataset_a_pg.md
```

Fallback (always reliable):

```bash
psql "$DATABASE_URL" -c "\copy (select * from schema.table_a limit 5000) to '/path/to/artifacts/demo/table_a_tmp.csv' with (format csv, header true)"

factorlens analyze \
  --input /path/to/artifacts/demo/table_a_tmp.csv \
  --profile your_profile_a \
  --profile-config /path/to/profiles/your_profiles_a.toml \
  --out /path/to/artifacts/demo/dataset_a_pg.md
```

## Step 4: Optional Bedrock Narrative

```bash
export AWS_REGION="${AWS_REGION:-eu-central-1}"
REPORT="$(sed -n '1,260p' artifacts/demo/dataset_a_exec.md)"
PROMPT="You are a business analyst. Summarize key risks/opportunities in 5 bullets and 3 actions.

${REPORT}"
export PROMPT

cat > /tmp/bedrock_marketplace.json <<EOF
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

aws bedrock-runtime converse \
  --region "$AWS_REGION" \
  --model-id anthropic.claude-3-haiku-20240307-v1:0 \
  --cli-input-json file:///tmp/bedrock_marketplace.json \
  --query 'output.message.content[0].text' \
  --output text
```

## Key Talking Points

- Deterministic analytics first, LLM explanation second.
- Profile-driven design: new datasets onboarded without code changes.
- Outputs for both audiences:
  - Markdown for leaders
  - JSON for automation/pipelines
- Works with local models or Bedrock.

## If Something Breaks During Live Demo

1. Use already-generated files in `artifacts/demo/*.md`.
2. Skip Bedrock and narrate from markdown output.
3. Use CSV fallback instead of direct Postgres query.
