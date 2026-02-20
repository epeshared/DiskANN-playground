#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_LOCAL="$SCRIPT_DIR/run_local.sh"

if [[ ! -f "$RUN_LOCAL" ]]; then
	echo "ERROR: run_local.sh not found: $RUN_LOCAL" >&2
	exit 1
fi

usage() {
	cat >&2 <<'EOF'
Usage:
	bash DiskANN-playground/diskann-ann-bench/run-ann-bench-diskann.sh [opts] [-- <run_local.sh args...>]

Wrapper around run_local.sh with sensible defaults for diskann_rs.

Wrapper opts:
	--threads N          Sets DISKANN_RS_NUM_THREADS (default: 16)
	--fit-batch-size N   Sets DISKANN_RS_FIT_BATCH_SIZE (default: 20000)
	--profile P          Sets DISKANN_RS_NATIVE_PROFILE (debug|release; default: release)
	-h, --help           Show this help

Everything after '--' (or any unknown flag) is passed through to run_local.sh.
Example:
	bash DiskANN-playground/diskann-ann-bench/run-ann-bench-diskann.sh \
		--threads 16 --fit-batch-size 10000 -- \
		--stage build --algo fp --hdf5 /path/to/data.hdf5
EOF
}

: "${DISKANN_RS_NUM_THREADS:=16}"
: "${DISKANN_RS_FIT_BATCH_SIZE:=20000}"
: "${DISKANN_RS_NATIVE_PROFILE:=release}"

passthrough=()
while [[ $# -gt 0 ]]; do
	case "$1" in
		--threads)
			DISKANN_RS_NUM_THREADS="$2"; shift 2 ;;
		--fit-batch-size)
			DISKANN_RS_FIT_BATCH_SIZE="$2"; shift 2 ;;
		--profile)
			DISKANN_RS_NATIVE_PROFILE="$2"; shift 2 ;;
		-h|--help)
			usage
			exit 0
			;;
		--)
			shift
			passthrough+=("$@")
			break
			;;
		*)
			passthrough+=("$1")
			shift
			;;
	esac
done

export DISKANN_RS_NUM_THREADS
export DISKANN_RS_FIT_BATCH_SIZE
export DISKANN_RS_NATIVE_PROFILE

echo "==> diskann_rs wrapper"
echo "    DISKANN_RS_NUM_THREADS=$DISKANN_RS_NUM_THREADS"
echo "    DISKANN_RS_FIT_BATCH_SIZE=$DISKANN_RS_FIT_BATCH_SIZE"
echo "    DISKANN_RS_NATIVE_PROFILE=$DISKANN_RS_NATIVE_PROFILE"

exec bash "$RUN_LOCAL" "${passthrough[@]}"