#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <input.json> <output.json>" >&2
  exit 2
fi

INPUT_JSON="$1"
OUTPUT_JSON="$2"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
DISKANN_RS="$REPO_ROOT/DiskANN-rs"

abspath() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    printf '%s\n' "$p"
  else
    printf '%s\n' "$REPO_ROOT/$p"
  fi
}

INPUT_ABS="$(abspath "$INPUT_JSON")"
OUTPUT_ABS="$(abspath "$OUTPUT_JSON")"

mkdir -p "$(dirname "$OUTPUT_ABS")"

cd "$REPO_ROOT"

# diskann-benchmark has empty default features; enable what we need.
cargo run --release --manifest-path "$DISKANN_RS/Cargo.toml" --package diskann-benchmark \
  --features product-quantization,spherical-quantization \
  -- run --input-file "$INPUT_ABS" --output-file "$OUTPUT_ABS"

echo "Wrote output: $OUTPUT_JSON"
