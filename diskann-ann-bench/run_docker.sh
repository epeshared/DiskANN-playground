#!/usr/bin/env bash
set -euo pipefail

# Run the framework_entry workflow inside the ann-benchmarks-diskann_rs Docker image.
# This supports split build/search via --stage {all,build,search}.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PLAYGROUND_DIR="$(realpath "$SCRIPT_DIR/..")"

HDF5=""
METRIC=""
ALGO="fp"
COMPARE=0
STAGE="all"
RUN_ID_OVERRIDE=""
INDEX_DIR=""
IMAGE="ann-benchmarks-diskann_rs"

L_BUILD=64
MAX_OUTDEGREE=32
ALPHA=1.2
K=10
L_SEARCH=50
REPS=2

NUM_PQ_CHUNKS=""
SPHERICAL_NBITS=2

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hdf5)
      HDF5="$2"; shift 2 ;;
    --metric)
      METRIC="$2"; shift 2 ;;
    --algo)
      ALGO="$2"; shift 2 ;;
    --compare)
      COMPARE=1; shift 1 ;;
    --stage)
      STAGE="$2"; shift 2 ;;
    --run-id)
      RUN_ID_OVERRIDE="$2"; shift 2 ;;
    --index-dir)
      INDEX_DIR="$2"; shift 2 ;;
    --image)
      IMAGE="$2"; shift 2 ;;
    --l-build)
      L_BUILD="$2"; shift 2 ;;
    --max-outdegree)
      MAX_OUTDEGREE="$2"; shift 2 ;;
    --alpha)
      ALPHA="$2"; shift 2 ;;
    -k)
      K="$2"; shift 2 ;;
    --l-search)
      L_SEARCH="$2"; shift 2 ;;
    --reps)
      REPS="$2"; shift 2 ;;
    --num-pq-chunks)
      NUM_PQ_CHUNKS="$2"; shift 2 ;;
    --spherical-nbits)
      SPHERICAL_NBITS="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
 done

if [[ -z "$HDF5" ]]; then
  echo "ERROR: --hdf5 is required" >&2
  exit 2
fi
if [[ ! -f "$HDF5" ]]; then
  echo "ERROR: hdf5 not found: $HDF5" >&2
  exit 1
fi
if [[ -z "$METRIC" ]]; then
  echo "ERROR: --metric is required (cosine|l2 or angular|euclidean)" >&2
  exit 2
fi
if [[ "$STAGE" != "all" && "$STAGE" != "build" && "$STAGE" != "search" ]]; then
  echo "ERROR: invalid --stage=$STAGE (expected: all|build|search)" >&2
  exit 2
fi

if [[ -n "$RUN_ID_OVERRIDE" ]]; then
  RUN_ID="$RUN_ID_OVERRIDE"
else
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi

DATASET="$(basename "$HDF5")"
DATASET="${DATASET%.hdf5}"

DEFAULT_RUNS_DIR="$PLAYGROUND_DIR/extend-rabitq/ann-harness/runs"
RUNS_DIR="${RUNS_DIR:-$DEFAULT_RUNS_DIR}"

# Mount dataset and run folder.
HDF5_ABS="$(realpath "$HDF5")"
RUNS_DIR_ABS="$(realpath "$RUNS_DIR")"
SCRIPT_DIR_ABS="$(realpath "$SCRIPT_DIR")"

# Container paths
C_HDF5="/data/$(basename "$HDF5_ABS")"
C_RUNS_DIR="/runs"
C_WORK_DIR="$C_RUNS_DIR/$DATASET/$RUN_ID"

if [[ -n "$INDEX_DIR" ]]; then
  mkdir -p "$INDEX_DIR"
fi

run_one() {
  local algo="$1"; shift
  local suffix="$1"; shift
  local c_work_dir="$C_RUNS_DIR/$DATASET/${RUN_ID}${suffix}"
  local host_work_dir="$RUNS_DIR/$DATASET/${RUN_ID}${suffix}"

  mkdir -p "$host_work_dir/outputs"
  echo "ann_bench_diskann_rs" > "$host_work_dir/mode.txt"

  extra_args=(--algo "$algo")
  if [[ "$algo" == "pq" ]]; then
    if [[ -z "$NUM_PQ_CHUNKS" ]]; then
      echo "ERROR: --num-pq-chunks is required for algo=pq (or --compare)" >&2
      exit 2
    fi
    extra_args+=(--num-pq-chunks "$NUM_PQ_CHUNKS")
  fi
  if [[ "$algo" == "spherical" ]]; then
    extra_args+=(--spherical-nbits "$SPHERICAL_NBITS")
  fi

  echo "==> docker run: algo=$algo work_dir=$RUNS_DIR/$DATASET/${RUN_ID}${suffix}" >&2

  docker_args=(
    run --rm
    -v "$RUNS_DIR_ABS":"$C_RUNS_DIR"
    -v "$HDF5_ABS":"$C_HDF5":ro
    -v "$SCRIPT_DIR_ABS":"/diskann-ann-bench":ro
    -w "/diskann-ann-bench"
  )

  if [[ -n "$INDEX_DIR" ]]; then
    docker_args+=( -v "$(realpath "$INDEX_DIR")":"/index" )
  fi

  docker "${docker_args[@]}" "$IMAGE" \
    python3 /diskann-ann-bench/framework_entry.py \
      --work-dir "$c_work_dir" \
      --hdf5 "$C_HDF5" \
      --dataset "$DATASET" \
      --metric "$METRIC" \
      --stage "$STAGE" \
      --l-build "$L_BUILD" \
      --max-outdegree "$MAX_OUTDEGREE" \
      --alpha "$ALPHA" \
      -k "$K" \
      --l-search "$L_SEARCH" \
      --reps "$REPS" \
      ${INDEX_DIR:+--index-dir /index} \
      "${extra_args[@]}"
}

if [[ "$COMPARE" -eq 1 ]]; then
  run_one pq "-pq"
  run_one spherical "-spherical"
else
  run_one "$ALGO" ""
fi
