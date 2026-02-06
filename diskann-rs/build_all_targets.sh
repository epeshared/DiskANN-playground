#!/usr/bin/env bash
set -euo pipefail

# Build the whole DiskANN-rs workspace with --all-targets.
#
# This script is meant for new shells/sessions: it activates conda env `diskann-rs`
# (as requested in your workflow) and then runs cargo commands inside DiskANN-rs/.

usage() {
  cat <<'EOF'
Usage:
  ./build_all_targets.sh [options]

Options:
  --release           Build in release mode (default)
  --debug             Build in debug mode
  --fmt               Run `cargo fmt --check`
  --clippy            Run `cargo clippy --workspace --all-targets -- -D warnings`
  --test              Run `cargo test --workspace`
  --test-all-targets  Run `cargo test --workspace --all-targets`
  --clean             Run `cargo clean` before anything else
  --help              Show this help

Environment variables:
  DISKANN_RS_DIR       Path to DiskANN-rs (default: ../DiskANN-rs relative to workspace root)
  CARGO_TARGET_DIR     Optional cargo target dir (useful for sharing build cache)

Examples:
  ./build_all_targets.sh
  ./build_all_targets.sh --release --fmt --clippy
  ./build_all_targets.sh --debug --test
EOF
}

MODE="release"
DO_FMT=0
DO_CLIPPY=0
DO_TEST=0
DO_TEST_ALL_TARGETS=0
DO_CLEAN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --release)
      MODE="release";
      shift
      ;;
    --debug)
      MODE="debug";
      shift
      ;;
    --fmt)
      DO_FMT=1;
      shift
      ;;
    --clippy)
      DO_CLIPPY=1;
      shift
      ;;
    --test)
      DO_TEST=1;
      shift
      ;;
    --test-all-targets)
      DO_TEST_ALL_TARGETS=1;
      shift
      ;;
    --clean)
      DO_CLEAN=1;
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

DISKANN_RS_DIR_DEFAULT="${WORKSPACE_ROOT}/DiskANN-rs"
DISKANN_RS_DIR="${DISKANN_RS_DIR:-${DISKANN_RS_DIR_DEFAULT}}"

if [[ ! -d "${DISKANN_RS_DIR}" ]]; then
  echo "ERROR: DiskANN-rs directory not found: ${DISKANN_RS_DIR}" >&2
  echo "Set DISKANN_RS_DIR to override." >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diskann-rs

cd "${DISKANN_RS_DIR}"

echo "== DiskANN-rs: mode=${MODE} fmt=${DO_FMT} clippy=${DO_CLIPPY} test=${DO_TEST} test_all_targets=${DO_TEST_ALL_TARGETS} clean=${DO_CLEAN} =="

if [[ ${DO_CLEAN} -eq 1 ]]; then
  echo "[clean] cargo clean"
  cargo clean
fi

if [[ ${DO_FMT} -eq 1 ]]; then
  echo "[fmt] cargo fmt --check"
  cargo fmt --check
fi

BUILD_ARGS=("--workspace" "--all-targets")
if [[ "${MODE}" == "release" ]]; then
  BUILD_ARGS=("--release" "${BUILD_ARGS[@]}")
fi

echo "[build] cargo build ${BUILD_ARGS[*]}"
cargo build "${BUILD_ARGS[@]}"

if [[ ${DO_CLIPPY} -eq 1 ]]; then
  CLIPPY_ARGS=("--workspace" "--all-targets" "--" "-D" "warnings")
  echo "[clippy] cargo clippy ${CLIPPY_ARGS[*]}"
  cargo clippy "${CLIPPY_ARGS[@]}"
fi

if [[ ${DO_TEST_ALL_TARGETS} -eq 1 ]]; then
  TEST_ARGS=("--workspace" "--all-targets")
  echo "[test] cargo test ${TEST_ARGS[*]}"
  cargo test "${TEST_ARGS[@]}"
elif [[ ${DO_TEST} -eq 1 ]]; then
  TEST_ARGS=("--workspace")
  echo "[test] cargo test ${TEST_ARGS[*]}"
  cargo test "${TEST_ARGS[@]}"
fi

echo "Done."
