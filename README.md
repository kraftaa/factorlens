# FactorLens

FactorLens is an offline-first factor attribution assistant in Rust.

It computes statistical factors (PCA) from price history, writes artifacts, and supports explainability through a pluggable LLM backend interface (`local` and `bedrock`).

## MVP Features

- Price ingestion from CSV
- PCA factor model fitting
- Portfolio factor attribution
- Residual outlier detection
- Artifact outputs (`json` + `csv`)
- Markdown report generation
- Explain command using a local llama.cpp backend (`llama-cli`) with a Bedrock-ready backend contract

## Workspace Layout

- `crates/factor_core`: Returns, PCA, attribution math
- `crates/factor_io`: CSV IO and artifact writing
- `crates/factor_cli`: CLI binary (`factorlens`)
- `crates/llm_local`: `LLMClient` trait + local/bedrock backends
- `crates/report`: Markdown report generation

## Build Instructions

For advanced build/release details, see `BUILD_INSTRUCTIONS.md`.

Quick local build:

```bash
cargo build -p factor_cli
cargo build -p factor_cli --release
```

## Input Formats

`prices.csv`

- `date` (YYYY-MM-DD)
- `ticker`
- `close`

`portfolio.csv` (optional)

- `ticker`
- `weight`

`holdings.csv` (optional alternative to `portfolio.csv`)

- `ticker`
- either `market_value` or both `shares` and `price`

`factors.csv` (for known-factor regression mode)

- `date` (YYYY-MM-DD)
- one or more numeric factor columns (for example: `MKT`, `SMB`, `HML`)

## Quick Start

```bash
cargo run -p factor_cli -- factors fit \
  --prices data/prices.csv \
  --k 3 \
  --out artifacts/ \
  --portfolio data/portfolio.csv

# safer residual analysis: auto-pick k (< number of assets)
cargo run -p factor_cli -- factors fit \
  --prices data/prices.csv \
  --k-auto \
  --out artifacts/ \
  --portfolio data/portfolio.csv

# alternative: derive weights automatically from holdings
cargo run -p factor_cli -- factors fit \
  --prices data/prices.csv \
  --k 3 \
  --out artifacts/ \
  --holdings data/holdings.csv

cargo run -p factor_cli -- report \
  --artifacts artifacts/ \
  --format markdown \
  --out artifacts/report.md

# known-factor regression mode (MKT/SMB/HML-style)
cargo run -p factor_cli -- factors regress \
  --prices data/prices.csv \
  --factors data/factors.csv \
  --out artifacts/ \
  --portfolio data/portfolio.csv

cargo run -p factor_cli -- explain \
  --backend local \
  --model models/llama.gguf \
  --artifacts artifacts/ \
  --question "What drove the largest drawdown?"
```

## Notes

- `explain --backend local` expects `llama-cli` on your PATH.
- `explain --backend bedrock` uses AWS Bedrock via AWS CLI (`aws bedrock-runtime converse`).
- This project is designed for explainability of computed analytics, not market prediction.

## Explainability Notes

- `factors fit` excludes weekend dates by default.
- Pass `--include-weekends` if your dataset intentionally includes weekend trading.
- `explain` supports focused analysis with `--focus-factors`.

Examples:

```bash
cargo run -p factor_cli -- factors fit --prices data/prices.csv --k 3 --out artifacts/ --portfolio data/portfolio.csv
cargo run -p factor_cli -- factors fit --prices data/prices.csv --k 3 --out artifacts/ --portfolio data/portfolio.csv --include-weekends

cargo run -p factor_cli -- explain --backend local --model models/llama_instruct.gguf --artifacts artifacts/ --question "What drove the largest drawdown?" --focus-factors factor_1,factor_2
```

### Custom Factor Names

By default, FactorLens auto-generates factor names from your dataset loadings
(top positive and negative loading tickers per factor), so it works on any dataset.

You can still override labels with a CSV or TSV file via `--factor-labels`.

Example `data/factor_labels.csv`:

```csv
factor,label
factor_1_contrib,Broad Market Beta
factor_2_contrib,Growth vs Value Rotation
factor_3_contrib,Idiosyncratic Spread
```

Use in `explain`:

```bash
cargo run -p factor_cli -- explain --backend local --model models/llama_instruct.gguf --artifacts artifacts/ --question "What drove the largest drawdown?" --factor-labels data/factor_labels.csv
```

Notes:
- Factor keys may be `factor_1`, `factor_1_contrib`, or just `1`.
- `#` comment lines are ignored.

## Suggested Questions

- What was the worst modeled drawdown day, and what factors drove it?
- On the worst day, what percentage came from each factor?
- Which factor is my largest average downside contributor over the full sample?
- Which dates had the biggest positive factor-driven gains?
- Which 5 days had the largest residuals (moves not explained by factors)?
- Did my risk concentration increase in the last month?
- Is my portfolio dominated by one factor or diversified across factors?
- How stable are exposures across time windows?
- Which factor changed direction most often?
- Which factor contributed most to volatility, not just returns?
- If I remove `factor_1`, how much modeled downside is left?
- Compare drawdown drivers with and without weekends included.
- Using only `factor_1,factor_2`, what drove the drawdown?
- Which assets are most aligned with `factor_1` loadings?
- Which assets increased my exposure to downside factors most?

## Generic Table Analysis

Analyze any CSV table by grouping columns and numeric metrics you choose:

```bash
cargo run -p factor_cli -- analyze \
  --input data/your_file.csv \
  --group-by region,product_line,channel \
  --out artifacts/analysis.md

# profile-based quick starts
cargo run -p factor_cli -- analyze \
  --input data/your_file.csv \
  --profile exec \
  --out artifacts/analysis_exec.md

cargo run -p factor_cli -- analyze \
  --input data/your_file.csv \
  --profile segment \
  --out artifacts/analysis_segment.md

cargo run -p factor_cli -- analyze \
  --input data/your_file.csv \
  --profile supplier \
  --out artifacts/analysis_supplier.md

# custom profile config (recommended for private/domain fields)
cargo run -p factor_cli -- analyze \
  --input data/your_file.csv \
  --profile exec_custom \
  --profile-config profiles/profiles.example.toml \
  --out artifacts/analysis.md

# filtered + ranked view
cargo run -p factor_cli -- analyze \
  --input data/your_file.csv \
  --where region=US \
  --rank-by revenue_usd \
  --agg median \
  --percentiles p50,p90 \
  --alert-top5-share 60 \
  --alert-blank-share 10 \
  --top 10 \
  --min-records 20 \
  --out artifacts/analysis_filtered_ranked.md

# text normalization for name/title grouping + JSON-only output
cargo run -p factor_cli -- analyze \
  --input data/your_file.csv \
  --group-by title \
  --metrics revenue_usd \
  --normalize-text-groups \
  --word-freq \
  --output-format html \
  --out artifacts/analysis_title.html
```

Auto-detect useful grouping columns (if `--group-by` is omitted):

```bash
cargo run -p factor_cli -- analyze \
  --input data/your_file.csv \
  --out artifacts/analysis_auto.md
```

Or analyze directly from Postgres:

```bash
# option 1: inline query
factorlens analyze \
  --postgres-url "$DATABASE_URL" \
  --query "SELECT region, channel, revenue_usd, cost_usd FROM analytics.sales" \
  --postgres-ssl-mode require \
  --postgres-ca-file /path/to/rds-ca-bundle.pem \
  --profile exec_custom \
  --profile-config profiles/profiles.example.toml \
  --out artifacts/analysis.md

# option 2: query file
factorlens analyze \
  --postgres-url "$DATABASE_URL" \
  --query-file sql/sales_analysis.sql \
  --profile exec_custom \
  --profile-config profiles/profiles.example.toml \
  --out artifacts/analysis.md

# option 3: AWS RDS/Aurora TLS with explicit CA bundle (recommended in pods)
mkdir -p /path/to/certs
curl -fL "https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem" \
  -o /path/to/rds-global-bundle.pem

factorlens analyze \
  --query "SELECT * FROM schema.table_a LIMIT 5000" \
  --postgres-ssl-mode require \
  --postgres-ca-file /path/to/rds-global-bundle.pem \
  --profile exec_custom \
  --profile-config profiles/profiles.example.toml \
  --out artifacts/analysis.md
```

Notes:
- Outputs both markdown and JSON (`<out>.json`).
- If `--metrics` is omitted, numeric metrics are auto-detected from the input file.
- `--profile` built-ins (`exec`, `segment`, `supplier`) are generic (no hardcoded domain columns).
- Use `--profile-config <path.toml>` for your own private, file-specific profile mappings.
- Input source is exclusive: use either `--input <csv>` or `--postgres-url` + (`--query` or `--query-file`).
- `--postgres-url` can be omitted if `DATABASE_URL` env var is set.
- `--postgres-ssl-mode` supports `prefer` (default), `require`, or `disable`.
- `--postgres-ca-file` optionally adds PEM CA certificates for DB TLS verification.
- For AWS RDS/Aurora in containers/pods, pass explicit RDS CA bundle via `--postgres-ca-file` if TLS handshake fails with system certs.
- Recommended layout: commit `profiles/profiles.example.toml`, keep private variants as `profiles/*.local.toml` or `profiles/*.private.toml` (gitignored).
- `--where` accepts comma-separated `column=value` filters (AND semantics).
- `--rank-by` ranks groups by a chosen metric (default ranking is by count).
- `--agg` controls metric aggregation: `sum` (default), `mean`, or `median`.
- `--percentiles` adds optional metric columns (`p50`, `p90`) per metric.
- `--count-only` disables numeric metric aggregation and reports concentration using records only.
- `--alert-top5-share` and `--alert-blank-share` add threshold-based alerts to report output.
- `--top` controls how many groups are listed in the report.
- `--normalize-text-groups` normalizes group values for columns like `name`/`title` (lowercase + punctuation cleanup).
- `--word-freq` adds a Top Words section/counts for `name`/`title`-style grouping columns.
- `--output-format` supports `md`, `json`, `both` (default), or `html`.
- `--min-records` drops tiny segments before ranking (useful to avoid one-record outliers).

Example `--profile-config` file:

```toml
[profiles.exec_custom]
group_by = ["region", "channel"]
metrics = ["revenue_usd"]
rank_by = "revenue_usd"
top = 12
min_records = 20
auto_group_k = 3
```

### pip Package Usage

Install from PyPI:

For packaging/build/publish details, see `BUILD_INSTRUCTIONS.md`.

```bash
pip install --upgrade factorlens==0.1.3
factorlens --help
```

Local model:

```bash
factorlens explain \
  --backend local \
  --model /path/to/model.gguf \
  --artifacts /path/to/artifacts \
  --question "What drove the largest drawdown?"
```

Bedrock:

```bash
export AWS_REGION=us-east-1
factorlens explain \
  --backend bedrock \
  --model anthropic.claude-3-5-sonnet-20240620-v1:0 \
  --artifacts /path/to/artifacts \
  --question "What drove the largest drawdown?"
```

Explain from generic table analysis output (`analysis.json`):

```bash
factorlens explain-analyze \
  --backend bedrock \
  --model anthropic.claude-3-haiku-20240307-v1:0 \
  --analysis-json /path/to/analysis.json \
  --question "What are the top concentration risks and 3 actions?"
```

### MCP Server (Optional)

If you want to call FactorLens as tools from an MCP client, use:

- `scripts/mcp/factorlens_mcp_server.py`
- `scripts/mcp/README.md`

Quick start:

```bash
pip install mcp
python scripts/mcp/factorlens_mcp_server.py
```

### What Bedrock Step Is Doing

`factorlens explain --backend bedrock` does **not** compute analytics. It only explains
already-computed artifacts.

Step-by-step:

1. You run analytics first (`factors fit` or `analyze`) to produce artifacts.
2. `explain` loads artifact context (for factor mode: `factors.json`, `attribution.csv`, `outliers.csv`).
3. FactorLens builds a constrained prompt from that context.
4. FactorLens calls AWS Bedrock through AWS CLI (`aws bedrock-runtime converse`).
5. Bedrock returns plain-text explanation grounded in the provided artifact context.

Important:
- `analyze` command = pure Rust analytics, no LLM used.
- `explain` command = LLM narrative layer over artifacts.
- For table-analysis markdown (`analysis.md`), you can optionally call Bedrock directly with AWS CLI by passing report text as prompt.
