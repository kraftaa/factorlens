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

## Input Formats

`prices.csv`

- `date` (YYYY-MM-DD)
- `ticker`
- `close`

`portfolio.csv` (optional)

- `ticker`
- `weight`

## Quick Start

```bash
cargo run -p factor_cli -- factors fit \
  --prices data/prices.csv \
  --k 3 \
  --out artifacts/ \
  --portfolio data/portfolio.csv

cargo run -p factor_cli -- report \
  --artifacts artifacts/ \
  --format markdown \
  --out artifacts/report.md

cargo run -p factor_cli -- explain \
  --backend local \
  --model models/llama.gguf \
  --artifacts artifacts/ \
  --question "What drove the largest drawdown?"
```

## Notes

- `explain --backend local` expects `llama-cli` on your PATH.
- `explain --backend bedrock` is scaffolded through the shared interface, but runtime invocation is intentionally left as the next implementation step.
- This project is designed for explainability of computed analytics, not market prediction.

## Python (pip) Package

A Python wrapper is included so you can distribute a `pip` package and run FactorLens via `factorlens-py`.

Build/install locally:

```bash
python -m pip install --upgrade build
python -m build
python -m pip install dist/factorlens-0.1.0-py3-none-any.whl
```

Run:

```bash
factorlens-py factors fit --prices data/prices.csv --k 3 --out artifacts/
```

Binary resolution order used by `factorlens-py`:

1. `FACTORLENS_BIN` env var
2. `factorlens` on `PATH`
3. `cargo run -p factor_cli --` (dev fallback)

If you want to publish to PyPI, run:

```bash
python -m pip install --upgrade twine
python -m twine upload dist/*
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

You can pass custom labels in a CSV or TSV file via `--factor-labels`.

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
