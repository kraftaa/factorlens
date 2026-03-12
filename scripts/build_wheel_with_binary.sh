#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BIN_DIR="$ROOT_DIR/factorlens/bin"
mkdir -p "$BIN_DIR"

cleanup() {
  rm -f "$BIN_DIR/factorlens" "$BIN_DIR/factorlens.exe"
}
trap cleanup EXIT

echo "[1/4] Building Rust CLI (release)..."
cargo build -p factor_cli --release

if [[ -f "$ROOT_DIR/target/release/factorlens" ]]; then
  cp "$ROOT_DIR/target/release/factorlens" "$BIN_DIR/factorlens"
  chmod +x "$BIN_DIR/factorlens"
elif [[ -f "$ROOT_DIR/target/release/factorlens.exe" ]]; then
  cp "$ROOT_DIR/target/release/factorlens.exe" "$BIN_DIR/factorlens.exe"
else
  echo "Could not find built factorlens binary in target/release" >&2
  exit 1
fi

echo "[2/4] Cleaning previous dist artifacts..."
rm -rf dist/

echo "[3/4] Building Python wheel/sdist..."
python -m build --no-isolation

echo "[4/4] Done. Artifacts in dist/"
ls -lh dist/
