#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$HERE/../.." && pwd)"
ANN_BENCH_ROOT="${ANN_BENCH_ROOT:-"$WORKSPACE_ROOT/ann-benchmark-epeshared"}"

cd "$ANN_BENCH_ROOT"

echo "==> Vendoring DiskANN-rs into ann-benchmark (recommended)"
bash ann_benchmarks/algorithms/diskann_rs/sync_diskann_rs.sh

echo "==> Building ann-benchmarks base image (may need network)"
docker build --network=host -t ann-benchmarks -f ann_benchmarks/algorithms/base/Dockerfile .

echo "==> Building diskann_rs algorithm image"
docker build --network=host -t ann-benchmarks-diskann_rs -f ann_benchmarks/algorithms/diskann_rs/Dockerfile .

echo "OK: built ann-benchmarks and ann-benchmarks-diskann_rs"