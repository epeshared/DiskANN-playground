#!/usr/bin/env bash
set -euo pipefail

# --- Demo parameters (tune here) ---
DATA_TYPE="float"      # demo 固定用 float/f32（benchmark job 里也写死为 float32）
METRIC="l2"            # demo 固定用 l2（benchmark job 里用 squared_l2）
NDIMS=32
NPTS_BASE=10000
NPTS_QUERY=200
GT_K=10

# Graph build params (DiskANN-ish)
MAX_DEGREE=32
L_BUILD=50
ALPHA=1.2
BACKEDGE_RATIO=1.0
NUM_THREADS_BUILD=4

# Search params
REPS=3
SEARCH_N=10
RECALL_K=10
SEARCH_LS=(20 40 60)
NUM_THREADS_SEARCH=(1 2 4)

# --- Paths ---
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
DISKANN_RS_DIR="${WORKSPACE_ROOT}/DiskANN-rs"

DATA_DIR="${SCRIPT_DIR}/data"
OUT_DIR="${SCRIPT_DIR}/output"

BASE_FILE="${DATA_DIR}/base.fbin"
QUERY_FILE="${DATA_DIR}/query.fbin"
GT_FILE="${DATA_DIR}/gt"          # current tool writes exactly this path

RUN_FILE="${OUT_DIR}/benchmark_run.json"
RESULT_FILE="${OUT_DIR}/benchmark_output.json"

mkdir -p "${DATA_DIR}" "${OUT_DIR}"

if [[ "${DATA_TYPE}" != "float" ]]; then
  echo "ERROR: This demo currently supports DATA_TYPE=float only." >&2
  exit 1
fi
if [[ "${METRIC}" != "l2" ]]; then
  echo "ERROR: This demo currently supports METRIC=l2 only." >&2
  exit 1
fi

# --- Conda env (required for your workflow) ---
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH. Please install conda or run inside your configured environment." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diskann-rs

cd "${DISKANN_RS_DIR}"

echo "[1/5] Building required binaries (release)..."
# These will also respect rust-toolchain.toml and auto-install Rust 1.92 via rustup if needed.
cargo build --release -p diskann-tools -p diskann-benchmark >/dev/null

echo "[2/5] Generating random base/query data..."
# base
cargo run --release -p diskann-tools --bin random_data_generator -- \
  --data_type "${DATA_TYPE}" \
  --output_file "${BASE_FILE}" \
  --ndims "${NDIMS}" \
  --npts "${NPTS_BASE}" \
  --norm 150 >/dev/null

# query
cargo run --release -p diskann-tools --bin random_data_generator -- \
  --data_type "${DATA_TYPE}" \
  --output_file "${QUERY_FILE}" \
  --ndims "${NDIMS}" \
  --npts "${NPTS_QUERY}" \
  --norm 150 >/dev/null

echo "[3/5] Computing ground truth (K=${GT_K})..."
# NOTE: pass prefix without .bin (tool appends .bin in the no-filter case)
cargo run --release -p diskann-tools --bin compute_groundtruth -- \
  --data_type "${DATA_TYPE}" \
  --dist_fn "${METRIC}" \
  --base_file "${BASE_FILE}" \
  --query_file "${QUERY_FILE}" \
  --gt_file "${GT_FILE}" \
  --recall_at "${GT_K}" >/dev/null

if [[ ! -f "${GT_FILE}" ]]; then
  echo "ERROR: ground truth file not found: ${GT_FILE}" >&2
  exit 1
fi

echo "[4/5] Writing benchmark runner input JSON..."
# Build JSON arrays from bash arrays
SEARCH_L_JSON="$(printf '%s\n' "${SEARCH_LS[@]}" | awk 'BEGIN{printf "["} {printf (NR==1?"%s":" ,%s"), $1} END{printf "]"}')"
THREADS_JSON="$(printf '%s\n' "${NUM_THREADS_SEARCH[@]}" | awk 'BEGIN{printf "["} {printf (NR==1?"%s":" ,%s"), $1} END{printf "]"}')"

cat > "${RUN_FILE}" <<EOF
{
  "search_directories": [
    "${DATA_DIR}"
  ],
  "output_directory": "${OUT_DIR}",
  "jobs": [
    {
      "type": "async-index-build",
      "content": {
        "source": {
          "index-source": "Build",
          "data_type": "float32",
          "data": "${BASE_FILE}",
          "distance": "squared_l2",
          "max_degree": ${MAX_DEGREE},
          "l_build": ${L_BUILD},
          "alpha": ${ALPHA},
          "backedge_ratio": ${BACKEDGE_RATIO},
          "num_threads": ${NUM_THREADS_BUILD},
          "insert_retry": null,
          "multi_insert": {
            "batch_size": 128,
            "batch_parallelism": 32,
            "intra_batch_candidates": "none"
          },
          "save_path": null,
          "start_point_strategy": "medoid"
        },
        "search_phase": {
          "search-type": "topk",
          "queries": "${QUERY_FILE}",
          "groundtruth": "${GT_FILE}",
          "reps": ${REPS},
          "num_threads": ${THREADS_JSON},
          "runs": [
            {
              "search_n": ${SEARCH_N},
              "search_l": ${SEARCH_L_JSON},
              "recall_k": ${RECALL_K}
            }
          ]
        }
      }
    }
  ]
}
EOF

echo "[5/5] Running benchmark..."
# This prints a human-readable summary to stdout and saves JSON to RESULT_FILE.
cargo run --release --package diskann-benchmark -- \
  run --input-file "${RUN_FILE}" --output-file "${RESULT_FILE}"

echo
echo "Done."
echo "- Runner input:   ${RUN_FILE}"
echo "- Benchmark out:  ${RESULT_FILE}"
echo "- Data dir:       ${DATA_DIR}"
