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

### Build Rust CLI (local)

```bash
cargo build -p factor_cli
```

Release binary:

```bash
cargo build -p factor_cli --release
```

### Build Python wheel (local)

```bash
python -m pip install --upgrade maturin
maturin build --release --manifest-path crates/factor_cli/Cargo.toml
```

Install built wheel:

```bash
python -m pip install target/wheels/factorlens-*.whl
```

### Build + publish wheels via GitHub Actions (recommended for cross-platform)

```bash
# tag-based release build/publish
git tag v0.1.3
git push origin v0.1.3

# or manual workflow trigger
gh workflow run release.yml -f publish_to_pypi=true -f ref=main
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

## Python (pip) Package

FactorLens is published as a platform-specific binary wheel via `maturin`.

Build/install locally:

```bash
python -m pip install --upgrade maturin
maturin build --release --manifest-path crates/factor_cli/Cargo.toml
python -m pip install target/wheels/factorlens-*.whl
```

Run:

```bash
factorlens factors fit --prices data/prices.csv --k 3 --out artifacts/
```

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

## Marketplace Analysis

Analyze generic marketplace CSVs by group columns you choose:

```bash
cargo run -p factor_cli -- marketplace analyze \
  --input data/analytics.sales5000.csv \
  --group-by discipline,category,subcategory,ware_name,advantage_plan \
  --out artifacts/market_report.md

# filtered + ranked view
cargo run -p factor_cli -- marketplace analyze \
  --input data/analytics.sales5000.csv \
  --where advantage_plan=1 \
  --rank-by net_gmv \
  --top 10 \
  --out artifacts/market_filtered_ranked.md
```

Auto-detect useful grouping columns (if `--group-by` is omitted):

```bash
cargo run -p factor_cli -- marketplace analyze \
  --input data/analytics.sales5000.csv \
  --out artifacts/market_auto.md
```

Notes:
- `ware_name` alias maps to `quote_group_ware_name`.
- Outputs both markdown and JSON (`<out>.json`).
- Default metrics: `net_gmv`, `customer_purchase_order_retail_total_price_usd`, `provider_purchase_order_wholesale_total_price_usd`.
- `--where` accepts comma-separated `column=value` filters (AND semantics).
- `--rank-by` ranks groups by a chosen metric (default ranking is by count).
- `--top` controls how many groups are listed in the report.
- For cleaner executive analytics on this dataset, start with `--group-by discipline,advantage_plan`.

## PyPI Publishing (Rustream-Style)

FactorLens uses the same publishing pattern as `rustream`: `maturin` + GitHub Actions
to build platform wheels (Linux/macOS/Windows) and publish to PyPI.

### Release from macOS via CLI

1. Bump version in `pyproject.toml`.
2. Commit and push to `main`.
3. Create and push a release tag:

```bash
git tag v0.1.3
git push origin v0.1.3
```

This triggers `.github/workflows/release.yml`, which:
- builds platform-specific wheels via `maturin`
- publishes to PyPI using `PYPI_API_TOKEN`
- attaches wheels to GitHub Release

To manually trigger from CLI without a tag:

```bash
gh workflow run release.yml -f publish_to_pypi=true -f ref=main
gh run list --workflow release.yml
gh run view <run-id> --log
```

### Jupyter Usage

Install from PyPI in Jupyter:

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
