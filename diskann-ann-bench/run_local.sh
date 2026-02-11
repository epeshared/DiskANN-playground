#!/usr/bin/env bash
set -euo pipefail

# One-click local run for diskann-ann-bench (host runner).
# - Binds the run to CPU cores 0-16 (clamped to available CPUs)
# - Writes cpu-bind.txt into the run folder so the web UI shows the binding
#
# Usage:
#   bash DiskANN-playground/diskann-ann-bench/run_local.sh [--hdf5 /path/to/file.hdf5]

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PLAYGROUND_DIR="$(realpath "$SCRIPT_DIR/..")"
WORKSPACE_ROOT="$(realpath "$PLAYGROUND_DIR/..")"

# You can override via --hdf5.
HDF5="/mnt/nvme2n1p1/xtang/ann-data/dbpedia-openai-1000k-angular.hdf5"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hdf5)
      HDF5="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$HDF5" ]]; then
  echo "ERROR: hdf5 not found: $HDF5" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found" >&2
  exit 1
fi

# Build the native extension (cargo debug build).
NATIVE_DIR="$WORKSPACE_ROOT/ann-benchmark-epeshared/ann_benchmarks/algorithms/diskann_rs/native"
if [[ ! -d "$NATIVE_DIR" ]]; then
  echo "ERROR: native crate not found: $NATIVE_DIR" >&2
  exit 1
fi

echo "==> cargo build (native extension)"
( cd "$NATIVE_DIR" && cargo build )

# Compute CPU bind string and clamp if needed.
CPU_BIND="0-16"
if command -v nproc >/dev/null 2>&1; then
  NPROC="$(nproc)"
  if [[ "$NPROC" -lt 17 ]]; then
    CPU_BIND="0-$((NPROC-1))"
    echo "WARN: only $NPROC CPUs available; using CPU_BIND=$CPU_BIND" >&2
  fi
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
DATASET="$(basename "$HDF5")"
DATASET="${DATASET%.hdf5}"

RUNS_DIR="$PLAYGROUND_DIR/extend-rabitq/ann-harness/runs"
WORK_DIR="$RUNS_DIR/$DATASET/$RUN_ID"

echo "==> running split runner (host)"
echo "    cpu bind: $CPU_BIND"
echo "    work dir: $WORK_DIR"

affinity_prefix=()
if command -v numactl >/dev/null 2>&1; then
  affinity_prefix=(numactl --physcpubind="$CPU_BIND" --)
elif command -v taskset >/dev/null 2>&1; then
  affinity_prefix=(taskset -c "$CPU_BIND")
else
  echo "WARN: neither numactl nor taskset found; run will not be CPU-affined" >&2
fi

"${affinity_prefix[@]}" python3 "$SCRIPT_DIR/run_diskann_rs_split.py" \
  --runner host \
  --hdf5 "$HDF5" \
  --dataset "$DATASET" \
  --run-id "$RUN_ID" \
  --work-dir "$WORK_DIR" \
  --cpu-bind "$CPU_BIND" \
  --metric l2 \
  --l-build 64 \
  --max-outdegree 32 \
  --alpha 1.2 \
  -k 10 \
  --l-search 50 \
  --reps 2

echo "==> done"
echo "    dataset: $DATASET"
echo "    run_id:  $RUN_ID"
echo "    work:    $WORK_DIR"
echo
echo "Next: start web server:"
echo "  bash DiskANN-playground/diskann-ann-bench/run_web.sh"
