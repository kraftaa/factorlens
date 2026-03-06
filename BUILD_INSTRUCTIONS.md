# Build Instructions

This document contains advanced build and release notes for FactorLens.

## Local Rust Build

```bash
cargo check -p factor_cli
cargo build -p factor_cli
cargo build -p factor_cli --release
```

## Python Wheel Build (maturin)

```bash
python -m pip install --upgrade build maturin
maturin build --release --manifest-path crates/factor_cli/Cargo.toml
python -m pip install target/wheels/factorlens-*.whl
```

## PyPI Publish (manual)

```bash
python -m pip install --upgrade twine
python -m twine upload dist/*
```

Notes:
- PyPI versions are immutable; bump version before each upload.
- Avoid mixing manual and CI uploads for the same version.

## GitHub Actions Release Flow

1. Bump version in `pyproject.toml`.
2. Commit and push to `main`.
3. Tag and push:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

This triggers wheel builds and publish workflow (if configured).

## Postgres TLS Notes

For `factorlens analyze --query ...`:
- `--postgres-ssl-mode prefer|require|disable`
- `--postgres-ca-file /path/to/ca.pem` to add custom CA bundle

Example:

```bash
factorlens analyze \
  --query "select * from schema.table limit 5000" \
  --postgres-ssl-mode require \
  --postgres-ca-file /path/to/rds-ca-bundle.pem \
  --out artifacts/analysis.md
```

If TLS still fails in constrained environments, export query to CSV with `psql \copy` and run `--input`.

## Bedrock Backend Notes

`factorlens explain --backend bedrock` shells out to AWS CLI. Ensure:

```bash
aws --version
aws sts get-caller-identity
```

Set region:

```bash
export AWS_REGION=us-east-1
```

Run:

```bash
factorlens explain \
  --backend bedrock \
  --model anthropic.claude-3-haiku-20240307-v1:0 \
  --artifacts /path/to/factor_artifacts \
  --question "What are the main concentration risks?"
```
