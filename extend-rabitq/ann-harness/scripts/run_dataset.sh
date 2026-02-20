#!/usr/bin/env bash
set -euo pipefail

HDF5_PATH="${HDF5_PATH:-/mnt/nvme2n1p1/xtang/ann-data/dbpedia-openai-1000k-angular.hdf5}"
DATASET="${DATASET:-dbpedia-openai-1000k-angular}"
DISTANCE="${DISTANCE:-cosine}"
SEARCH_N="${SEARCH_N:-100}"
SEARCH_L="${SEARCH_L:-100,200,400,800}"
# Run mode:
# - pq_vs_spherical: current (PQ vs spherical quantization) benchmark config
# - mem_fp: async full-precision in-memory index
# - disk_index: disk index build+search (requires diskann-benchmark feature disk-index)
MODE="${MODE:-pq_vs_spherical}"
# Controls how many times each search run repeats the full query set (passed to run_dataset.py --loop).
LOOP="${LOOP:-1}"
REBUILD_INDEX="${REBUILD_INDEX:-false}"
EMON_ENABLE="${EMON_ENABLE:-false}"

# Optional CPU binding via numactl.
# Example:
#   NUMACTL_CPUS=0-31 NUMACTL_MEMBIND=0 bash run_dataset.sh
NUMACTL_CPUS="${NUMACTL_CPUS:-0-15}"
NUMACTL_MEMBIND="${NUMACTL_MEMBIND:-0}"

# Optional: set CPU_BIND_DESC directly (for recording only). remote-test uses this.
CPU_BIND_DESC="${CPU_BIND_DESC:-}"
RUN_PREFIX=()
if [[ -z "$CPU_BIND_DESC" && ( -n "$NUMACTL_CPUS" || -n "$NUMACTL_MEMBIND" ) ]]; then
  if ! command -v numactl >/dev/null 2>&1; then
    echo "ERROR: NUMACTL_CPUS/NUMACTL_MEMBIND set but numactl not found on PATH" >&2
    exit 2
  fi
  RUN_PREFIX+=(numactl)
  if [[ -n "$NUMACTL_CPUS" ]]; then
    RUN_PREFIX+=(-C "$NUMACTL_CPUS")
    CPU_BIND_DESC="numactl -C $NUMACTL_CPUS"
  fi
  if [[ -n "$NUMACTL_MEMBIND" ]]; then
    RUN_PREFIX+=(-m "$NUMACTL_MEMBIND")
    if [[ -n "$CPU_BIND_DESC" ]]; then
      CPU_BIND_DESC="$CPU_BIND_DESC -m $NUMACTL_MEMBIND"
    else
      CPU_BIND_DESC="numactl -m $NUMACTL_MEMBIND"
    fi
  fi
fi

print_cmd() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
}

CMD=(
  "${RUN_PREFIX[@]}" python run_dataset.py
  --hdf5 "$HDF5_PATH"
  --dataset "$DATASET"
  --distance "$DISTANCE"
  --mode "$MODE"
  --loop "$LOOP"
  --search-n "$SEARCH_N"
  --search-l "$SEARCH_L"
  --rebuild-index "$REBUILD_INDEX"
  --emon-enable "$EMON_ENABLE"
  --cpu-bind "${CPU_BIND_DESC}"
)

print_cmd "${CMD[@]}"
"${CMD[@]}"