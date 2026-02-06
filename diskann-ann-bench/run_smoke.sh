#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$HERE/../.." && pwd)"
ANN_BENCH_ROOT="${ANN_BENCH_ROOT:-"$WORKSPACE_ROOT/ann-benchmark-epeshared"}"

cd "$ANN_BENCH_ROOT"

python run.py \
  --dataset glove-25-angular \
  --algorithm diskann-rs-smoke \
  -k 10 \
  --runs 1 \
  --timeout 900 \
  --parallelism 1
