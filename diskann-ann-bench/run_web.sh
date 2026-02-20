#!/usr/bin/env bash
set -euo pipefail

# Start the diskann-ann-bench web UI.
# Usage:
#   bash DiskANN-playground/diskann-ann-bench/run_web.sh [--host 127.0.0.1] [--port 8081]
#
# If RUNS_DIR is not set, defaults to DiskANN-playground/diskann-ann-bench/result.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" && pwd)"
PLAYGROUND_DIR="$(realpath "$SCRIPT_DIR/..")"

HOST="0.0.0.0"
PORT="8081"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="$2"; shift 2 ;;
    --port)
      PORT="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found" >&2
  exit 1
fi

# Ensure web deps exist.
if ! python3 -c 'import fastapi, uvicorn, jinja2, markdown_it' >/dev/null 2>&1; then
  echo "ERROR: missing web deps. Install with:" >&2
  echo "  python3 -m pip install -r DiskANN-playground/diskann-ann-bench/web/requirements.txt" >&2
  exit 1
fi

if [[ -z "${RUNS_DIR:-}" ]]; then
  export RUNS_DIR="$SCRIPT_DIR/result"
fi

URL_BASE="http://$HOST:$PORT"
echo "==> starting web server"
echo "    runs_dir: $RUNS_DIR"
echo "    Open: $URL_BASE/"
echo "    (Ctrl-C to stop)"

exec python3 "$SCRIPT_DIR/web/app.py" --host "$HOST" --port "$PORT"
