#!/usr/bin/env bash
set -euo pipefail

# Simple launcher for diskann-ann-bench/web.
# Usage:
#   bash DiskANN-playground/diskann-ann-bench/web/run_web.sh --host 127.0.0.1 --port 8081

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec python3 "$SCRIPT_DIR/app.py" "$@"
