# Demo Checklist

## Goal
Show FactorLens end-to-end in under 5 minutes:
1. analyze
2. analyze-compare
3. explain-analyze

Use only public-safe demo data.

## Pre-Demo (10-15 min before)

- Open terminal in repo root.
- Set clean prompt:

```bash
export PS1='$ '
```

- Confirm binary:

```bash
./target/debug/factorlens --help
```

- Confirm AWS region (Bedrock path):

```bash
export AWS_REGION=eu-central-1
```

- Optional: open outputs directory in Finder:

```bash
open artifacts
```

## Live Demo Commands

### 1) Baseline snapshot

```bash
./target/debug/factorlens analyze \
  --input data/factorlens_demo_sales_100.csv \
  --group-by region,channel,product_line,plan_tier \
  --metrics revenue_usd,cost_usd,orders \
  --rank-by revenue_usd \
  --top-insights 3 \
  --out artifacts/demo_sales_100.md
```

### 2) New snapshot

```bash
./target/debug/factorlens analyze \
  --input data/factorlens_demo_sales_150.csv \
  --group-by region,channel,product_line,plan_tier \
  --metrics revenue_usd,cost_usd,orders \
  --rank-by revenue_usd \
  --top-insights 3 \
  --out artifacts/demo_sales_150.md
```

### 3) Compare

```bash
./target/debug/factorlens analyze-compare \
  --base artifacts/demo_sales_100.json \
  --new artifacts/demo_sales_150.json \
  --output-format html \
  --out artifacts/demo_compare.html
```

### 4) Explain (Bedrock)

```bash
./target/debug/factorlens explain-analyze \
  --backend bedrock \
  --model anthropic.claude-3-haiku-20240307-v1:0 \
  --analysis-json artifacts/demo_sales_150.json \
  --question "What are the top concentration risks and what 3 actions should we take in the next 30 days?"
```

## Optional: Auto-Suggest Profile

```bash
./target/debug/factorlens analyze-suggest \
  --input data/factorlens_demo_sales_150.csv \
  --out artifacts/demo_suggest.md \
  --out-profile artifacts/demo_exec_profile.toml \
  --profile-name demo_exec \
  --sample-mode random \
  --sample-rows 100 \
  --sample-seed 42
```

## Fallback Plan (if Bedrock fails)

- Skip `explain-analyze` and show deterministic insights from:
  - `artifacts/demo_sales_150.md` (`Top Insights` section)
  - `artifacts/demo_compare.html` (biggest movers)

## Talking Points (short)

- "Math first, AI second."
- "Rust computes attribution; LLM explains computed artifacts."
- "Same flow works for CSV and Postgres."
- "Outputs are markdown/json/html for people and pipelines."

## Expected Output Files

- `artifacts/demo_sales_100.md`
- `artifacts/demo_sales_100.json`
- `artifacts/demo_sales_150.md`
- `artifacts/demo_sales_150.json`
- `artifacts/demo_compare.html`

## Backup One-Command Run

```bash
./scripts/demo_sales.sh
```

With Bedrock:

```bash
RUN_BEDROCK=1 AWS_REGION=eu-central-1 ./scripts/demo_sales.sh
```
