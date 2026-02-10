#!/usr/bin/env bash
set -euo pipefail

# Convenience launcher that runs the web server inside the conda env "diskann-rs".
# Usage:
#   bash DiskANN-playground/extend-rabitq/web/run_web.sh --host 0.0.0.0 --port 8080

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found on PATH" >&2
  exit 1
fi

exec conda run -n diskann-rs python "$SCRIPT_DIR/app.py" "$@"
