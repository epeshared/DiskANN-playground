#!/usr/bin/env bash
set -euo pipefail

# One-click local run for diskann-ann-bench (host runner).
# - Builds the native extension (cargo) and runs via the ann-benchmarks adapter
# - Binds the run to CPU cores 0-16 (clamped to available CPUs)
# - Writes mode.txt + cpu-bind.txt into the run folder so the web UI can filter
#
# Usage:
#   bash DiskANN-playground/diskann-ann-bench/run_local.sh [--conf conf/job-conf.yml]
#
# Notes:
# - This script is config-file driven (YAML default; JSON/TOML also supported).
# - All run parameters come from the job config file (defaults can be overridden there).
# - Only the --conf flag is accepted; any other CLI flags are rejected.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PLAYGROUND_DIR="$(realpath "$SCRIPT_DIR/..")"
WORKSPACE_ROOT="$(realpath "$PLAYGROUND_DIR/..")"

CONF_DIR="$SCRIPT_DIR/conf"
JOB_CONF="$CONF_DIR/job-conf.yml"

# You can override via --hdf5.
HDF5="/mnt/nvme2n1p1/xtang/ann-data/dbpedia-openai-1000k-angular.hdf5"

# If not provided, metric will be auto-detected from HDF5 attrs['distance'].
METRIC=""
ALGO="fp"
# Run mode:
# - COMPARE=1 (default): run pq + spherical compare workflow.
# - COMPARE=0: run a single algorithm workflow (controlled by --algo/--name/--run-group).
COMPARE=1
STAGE="all"
# If not provided, a timestamp-based run_id will be generated.
RUN_ID_OVERRIDE=""
INDEX_DIR=""
RESUME_RUN_ID=""
INDEX_RUN_ID=""

# Index cleanup control:
# - auto: delete indexes only for auto timestamp-based runs (default)
# - yes: delete created indexes under the run folder
# - no: keep indexes
DELETE_INDEX_AFTER_RUN="auto"
AUTO_RUN_ID=0

# Defaults are set to reproduce the dbpedia-openai-1000k-angular sweep results via config.yml presets.
RUN_ALL=1
NAME=""
NAME_PQ="diskann-rs-pq-memory"
NAME_SPHERICAL="diskann-rs-spherical-memory"

# When running --run-all (default), only execute run_groups matching these prefixes.
RUN_GROUP_PREFIX_PQ="pq_100_32_1-2_"
RUN_GROUP_PREFIX_SPHERICAL="spherical_100_32_1-2_"

L_BUILD=100
MAX_OUTDEGREE=32
ALPHA=1.2
K=100
L_SEARCH=200
REPS=2

# Query execution mode:
# - 0: per-query loop (algo.query for each query)
# - 1: batch mode (algo.batch_query on all queries)
BATCH=0

# CPU binding for the host runner.
# Default: 0-16 (clamped to available CPUs).
CPU_BIND="${CPU_BIND:-0-16}"

NUM_PQ_CHUNKS=""
SPHERICAL_NBITS=2

L_SEARCH_LIST=""
NUM_PQ_CHUNKS_LIST=""
SPHERICAL_NBITS_LIST=""

# Pass-through to src/framework_entry.py (tri-state: unset/true/false).
TRANSLATE_TO_CENTER=""

CONFIG_YML=""
RUN_GROUP=""
RUN_GROUP_PQ=""
RUN_GROUP_SPHERICAL=""

# Native build profile for diskann_rs_native (debug|release).
# You can override by exporting DISKANN_RS_NATIVE_PROFILE before running this script.
: "${DISKANN_RS_NATIVE_PROFILE:=release}"
export DISKANN_RS_NATIVE_PROFILE

# Fit batching (passed to diskann_rs_native via env var).
# You can override by exporting DISKANN_RS_FIT_BATCH_SIZE before running this script.
: "${DISKANN_RS_FIT_BATCH_SIZE:=20000}"
export DISKANN_RS_FIT_BATCH_SIZE

usage() {
  cat >&2 <<EOF
Usage:
  bash DiskANN-playground/diskann-ann-bench/run_local.sh [--conf conf/job-conf.yml]

Config:
  Default config path: $JOB_CONF
  Copy from:           $CONF_DIR/job-conf.example.yml
EOF
}

if [[ $# -gt 0 ]]; then
  if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    usage
    exit 0
  fi
  if [[ "$1" == "--conf" && $# -eq 2 ]]; then
    JOB_CONF="$2"
  else
    echo "ERROR: this script is config-file driven; only --conf is accepted." >&2
    echo "Got args: $*" >&2
    usage
    exit 2
  fi
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found" >&2
  exit 1
fi

pick_python_bin() {
  # Allow overrides (useful for custom envs / CI).
  if [[ -n "${DISKANN_ANN_BENCH_PYTHON:-}" ]]; then
    if command -v "${DISKANN_ANN_BENCH_PYTHON}" >/dev/null 2>&1; then
      printf '%s' "${DISKANN_ANN_BENCH_PYTHON}"
      return 0
    fi
    echo "ERROR: DISKANN_ANN_BENCH_PYTHON not found: ${DISKANN_ANN_BENCH_PYTHON}" >&2
    return 1
  fi

  # Prefer system Python on remote machines where shell rc files may inject conda/pyenv.
  local sys_py="/usr/bin/python3"
  if [[ -x "$sys_py" ]]; then
    if "$sys_py" - <<'PY' >/dev/null 2>&1
import yaml  # noqa: F401
PY
    then
      printf '%s' "$sys_py"
      return 0
    fi
  fi

  if python3 - <<'PY' >/dev/null 2>&1
import yaml  # noqa: F401
PY
  then
    printf '%s' "python3"
    return 0
  fi

  echo "ERROR: python cannot import required module: yaml (PyYAML)." >&2
  echo "- For Ubuntu: install apt package: python3-yaml" >&2
  echo "- Or set DISKANN_ANN_BENCH_PYTHON to a venv python with PyYAML" >&2
  return 1
}

PYTHON_BIN="$(pick_python_bin)"

ensure_framework_py_deps() {
  if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import yaml  # noqa: F401
import h5py  # noqa: F401
import numpy  # noqa: F401
PY
  then
    return 0
  fi
  echo "ERROR: python cannot import required modules for framework_entry (yaml, h5py, numpy)." >&2
  echo "- For Ubuntu (offline-friendly): install apt packages: python3-yaml python3-numpy python3-h5py" >&2
  echo "- Or set DISKANN_ANN_BENCH_PYTHON to a venv python with those deps" >&2
  return 1
}

load_job_conf_as_bash() {
  local conf="$1"
  "$PYTHON_BIN" - "$conf" <<'PY'
import json
import shlex
import sys
from pathlib import Path

def load_config(path: Path):
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix in (".yml", ".yaml"):
        try:
            import yaml  # type: ignore
        except Exception as e:  # pragma: no cover
            raise SystemExit(
                f"failed to import PyYAML for {path} (install pyyaml or use .json): {type(e).__name__}"
            )
        return yaml.safe_load(text)
    if suffix == ".json":
        return json.loads(text)
    if suffix == ".toml":
        try:
            import tomllib
        except Exception as e:  # pragma: no cover
            raise SystemExit(
                f"failed to import tomllib for {path} (use Python 3.11+ or install tomli): {type(e).__name__}"
            )
        return tomllib.loads(text)

    # Fallback: try JSON then YAML.
    try:
        return json.loads(text)
    except Exception:
        try:
            import yaml  # type: ignore
        except Exception as e:  # pragma: no cover
            raise SystemExit(
                f"unknown config extension {suffix!r} and PyYAML unavailable: {path}: {type(e).__name__}"
            )
        return yaml.safe_load(text)

p = Path(sys.argv[1]).expanduser().resolve()
if not p.is_file():
    raise SystemExit(f"job conf not found: {p}")

obj = load_config(p)
if not isinstance(obj, dict):
  raise SystemExit(f"job conf must be a mapping/object: {p}")

def emit(name: str, value) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        v = "1" if value else "0"
    else:
        v = str(value)
    print(f"{name}={shlex.quote(v)}")

emit("HDF5", obj.get("hdf5"))
emit("METRIC", obj.get("metric"))
emit("STAGE", obj.get("stage"))
emit("RUN_ID_OVERRIDE", obj.get("run_id"))
emit("RESUME_RUN_ID", obj.get("resume_runid"))
emit("INDEX_DIR", obj.get("index_dir"))
emit("DELETE_INDEX_AFTER_RUN", obj.get("delete_index_after_run"))

emit("RUN_ALL", obj.get("run_all"))
emit("COMPARE", obj.get("compare"))

emit("ALGO", obj.get("algo"))
emit("NAME", obj.get("name"))
emit("NAME_PQ", obj.get("name_pq"))
emit("NAME_SPHERICAL", obj.get("name_spherical"))

emit("RUN_GROUP_PREFIX_PQ", obj.get("run_group_prefix_pq"))
emit("RUN_GROUP_PREFIX_SPHERICAL", obj.get("run_group_prefix_spherical"))

emit("L_BUILD", obj.get("l_build"))
emit("MAX_OUTDEGREE", obj.get("max_outdegree"))
emit("ALPHA", obj.get("alpha"))
emit("K", obj.get("k"))
emit("L_SEARCH", obj.get("l_search"))
emit("REPS", obj.get("reps"))

emit("BATCH", obj.get("batch"))
emit("CPU_BIND", obj.get("cpu_bind"))

emit("NUM_PQ_CHUNKS", obj.get("num_pq_chunks"))
emit("SPHERICAL_NBITS", obj.get("spherical_nbits"))

emit("L_SEARCH_LIST", obj.get("l_search_list"))
emit("NUM_PQ_CHUNKS_LIST", obj.get("num_pq_chunks_list"))
emit("SPHERICAL_NBITS_LIST", obj.get("spherical_nbits_list"))

ttc = obj.get("translate_to_center")
if ttc is True:
    emit("TRANSLATE_TO_CENTER", "true")
elif ttc is False:
    emit("TRANSLATE_TO_CENTER", "false")

emit("CONFIG_YML", obj.get("config_yml"))

emit("RUN_GROUP", obj.get("run_group"))
emit("RUN_GROUP_PQ", obj.get("run_group_pq"))
emit("RUN_GROUP_SPHERICAL", obj.get("run_group_spherical"))

emit("DISKANN_RS_NATIVE_PROFILE", obj.get("diskann_rs_native_profile"))
emit("DISKANN_RS_FIT_BATCH_SIZE", obj.get("fit_batch_size"))
emit("RUNS_DIR", obj.get("runs_dir"))
PY
}

eval "$(load_job_conf_as_bash "$JOB_CONF")"

export DISKANN_RS_NATIVE_PROFILE
export DISKANN_RS_FIT_BATCH_SIZE

csv_to_lines() {
  # Split comma-separated values into newline-separated tokens.
  # Trims leading/trailing whitespace and drops empty tokens.
  printf '%s\n' "$1" \
    | tr ',' '\n' \
    | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e '/^$/d'
}

validate_uint_list() {
  local label="$1"; shift
  local v
  for v in "$@"; do
    if [[ ! "$v" =~ ^[0-9]+$ ]]; then
      echo "ERROR: invalid $label value: $v" >&2
      exit 2
    fi
  done
}

L_SEARCH_VALUES=()
NUM_PQ_CHUNKS_VALUES=()
SPHERICAL_NBITS_VALUES=()

if [[ -n "$L_SEARCH_LIST" ]]; then
  mapfile -t L_SEARCH_VALUES < <(csv_to_lines "$L_SEARCH_LIST")
else
  L_SEARCH_VALUES=("$L_SEARCH")
fi

if [[ -n "$NUM_PQ_CHUNKS_LIST" ]]; then
  mapfile -t NUM_PQ_CHUNKS_VALUES < <(csv_to_lines "$NUM_PQ_CHUNKS_LIST")
else
  if [[ -n "$NUM_PQ_CHUNKS" ]]; then
    NUM_PQ_CHUNKS_VALUES=("$NUM_PQ_CHUNKS")
  fi
fi

if [[ -n "$SPHERICAL_NBITS_LIST" ]]; then
  mapfile -t SPHERICAL_NBITS_VALUES < <(csv_to_lines "$SPHERICAL_NBITS_LIST")
else
  SPHERICAL_NBITS_VALUES=("$SPHERICAL_NBITS")
fi

validate_uint_list "--l-search" "${L_SEARCH_VALUES[@]}"
if [[ ${#NUM_PQ_CHUNKS_VALUES[@]} -gt 0 ]]; then
  validate_uint_list "--num-pq-chunks" "${NUM_PQ_CHUNKS_VALUES[@]}"
fi
validate_uint_list "--spherical-nbits" "${SPHERICAL_NBITS_VALUES[@]}"

if [[ -n "$RESUME_RUN_ID" ]]; then
  if [[ -n "$RUN_ID_OVERRIDE" && "$RUN_ID_OVERRIDE" != "$RESUME_RUN_ID" ]]; then
    echo "ERROR: --resume-runid cannot be combined with a different --run-id" >&2
    exit 2
  fi
  RUN_ID_OVERRIDE="$RESUME_RUN_ID"
fi

# Support a convenience mode: in job-conf.yml, `index_dir` may be set to a *run_id*
# (instead of a filesystem path) to reuse the saved index artifacts from that run.
#
# Behavior:
# - Only supported for `--stage search`.
# - We do NOT pass a global --index-dir (which would be shared); instead we map each case
#   to: <runs_dir>/<dataset>/<index_run_id>/cases/<case>/index
# - We validate index presence (by expected prefix files) before executing.
if [[ -n "$INDEX_DIR" && ! -d "$INDEX_DIR" ]]; then
  # Heuristic: if it's not an existing directory and it doesn't look like a path, treat as run_id.
  # Allow "remote_build_diskann_rs_pq_memory" style ids.
  if [[ "$INDEX_DIR" != *"/"* ]]; then
    INDEX_RUN_ID="$INDEX_DIR"
    INDEX_DIR=""
  fi
fi

default_config_yml() {
  echo "$WORKSPACE_ROOT/ann-benchmark-epeshared/ann_benchmarks/algorithms/diskann_rs/config.yml"
}

metric_to_distance() {
  local m="$1"
  case "$m" in
    cosine|angular) echo "angular" ;;
    l2|euclidean) echo "euclidean" ;;
    *) echo "$m" ;;
  esac
}

detect_distance_from_hdf5() {
  local hdf5_path="$1"
  "$PYTHON_BIN" - "$hdf5_path" <<'PY'
import sys
from pathlib import Path

try:
    import h5py  # type: ignore
except Exception:
    print("")
    raise SystemExit(0)

p = Path(sys.argv[1])
try:
    with h5py.File(p, "r") as f:
        d = str(f.attrs.get("distance", "")).strip()
        print(d)
except Exception:
    print("")
PY
}

get_run_groups_by_name() {
  local algo_name="$1"
  local metric="$2"
  local cfg_yml="$3"

  local distance
  distance="$(metric_to_distance "$metric")"

  "$PYTHON_BIN" - "$cfg_yml" "$distance" "$algo_name" <<'PY'
import sys
from pathlib import Path

import yaml

cfg_path = Path(sys.argv[1]).expanduser().resolve()
distance = sys.argv[2]
name = sys.argv[3]

cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
float_cfg = (cfg or {}).get("float")
if not isinstance(float_cfg, dict):
    raise SystemExit(f"invalid config.yml (missing float): {cfg_path}")

dist_cfg = float_cfg.get(distance)
if not isinstance(dist_cfg, list):
    raise SystemExit(f"config.yml has no float/{distance} section: {cfg_path}")

for entry in dist_cfg:
    if not isinstance(entry, dict):
        continue
    if str(entry.get("name", "")) != name:
        continue
    run_groups = entry.get("run_groups")
    if not isinstance(run_groups, dict) or not run_groups:
        raise SystemExit(f"no run_groups found for name={name!r} in float/{distance}: {cfg_path}")
    for k in run_groups.keys():
        print(k)
    raise SystemExit(0)

raise SystemExit(f"name={name!r} not found in float/{distance}: {cfg_path}")
PY
}

filter_run_groups_by_prefix() {
  local prefix="$1"; shift
  if [[ -z "$prefix" ]]; then
    printf '%s\n' "$@"
    return 0
  fi
  local v
  for v in "$@"; do
    if [[ "$v" == "$prefix"* ]]; then
      printf '%s\n' "$v"
    fi
  done
}

STAGE="${STAGE,,}"
if [[ "$STAGE" != "all" && "$STAGE" != "build" && "$STAGE" != "search" && "$STAGE" != "native" ]]; then
  echo "ERROR: invalid --stage=$STAGE (expected: all|build|search|native)" >&2
  exit 2
fi

if [[ -n "$INDEX_RUN_ID" && "$STAGE" != "search" ]]; then
  echo "ERROR: index_dir-as-runid is only supported with --stage search" >&2
  exit 2
fi

if [[ "$STAGE" == "search" && -z "$RUN_ID_OVERRIDE" ]]; then
  if [[ -z "$INDEX_DIR" && -z "$INDEX_RUN_ID" ]]; then
    echo "ERROR: --stage search requires either --run-id/--resume-runid <existing_run_id> or --index-dir <index_dir>" >&2
    exit 2
  fi
fi


if [[ "$RUN_ALL" -eq 1 ]]; then
  if [[ -n "$RUN_GROUP" || -n "$RUN_GROUP_PQ" || -n "$RUN_GROUP_SPHERICAL" ]]; then
    echo "ERROR: --run-all cannot be combined with --run-group/--run-group-pq/--run-group-spherical" >&2
    exit 2
  fi

  if [[ "$COMPARE" -eq 1 ]]; then
    if [[ -z "$NAME_PQ" || -z "$NAME_SPHERICAL" ]]; then
      echo "ERROR: --run-all with --compare requires --name-pq and --name-spherical" >&2
      exit 2
    fi
  else
    if [[ -z "$NAME" ]]; then
      echo "ERROR: --run-all requires --name <algo_name_from_config_yml>" >&2
      exit 2
    fi
  fi
fi

HAS_LIST_SWEEP=0
if [[ -n "$L_SEARCH_LIST" || -n "$NUM_PQ_CHUNKS_LIST" || -n "$SPHERICAL_NBITS_LIST" ]]; then
  HAS_LIST_SWEEP=1
fi

if [[ "$HAS_LIST_SWEEP" -eq 1 ]]; then
  if [[ "$RUN_ALL" -eq 1 || -n "$RUN_GROUP" || -n "$RUN_GROUP_PQ" || -n "$RUN_GROUP_SPHERICAL" ]]; then
    echo "ERROR: list sweeps (--*-list) cannot be combined with --run-all/--run-group presets" >&2
    exit 2
  fi
  if [[ -n "$INDEX_DIR" ]]; then
    echo "ERROR: list sweeps (--*-list) cannot be combined with --index-dir (would overwrite shared indexes)" >&2
    exit 2
  fi
fi

if [[ -n "$RUN_GROUP" && "$COMPARE" -eq 1 ]]; then
  echo "ERROR: --run-group cannot be combined with --compare. Use --run-group-pq/--run-group-spherical." >&2
  exit 2
fi

if [[ "$COMPARE" -eq 1 ]]; then
  if [[ -n "$RUN_GROUP_PQ" && -z "$RUN_GROUP_SPHERICAL" ]] || [[ -z "$RUN_GROUP_PQ" && -n "$RUN_GROUP_SPHERICAL" ]]; then
    echo "ERROR: when using presets in compare mode, you must provide both --run-group-pq and --run-group-spherical" >&2
    exit 2
  fi
fi

if [[ -n "$RUN_ID_OVERRIDE" && -n "$INDEX_DIR" ]]; then
  echo "ERROR: --run-id and --index-dir are mutually exclusive (choose one)" >&2
  exit 2
fi

# index_dir-as-runid is compatible with setting a new output run_id.
# But it is NOT compatible with resume_runid (which implies reusing an existing run folder)
# and we disallow using the same id for both output and index source to avoid accidental overwrites.
if [[ -n "$RESUME_RUN_ID" && -n "$INDEX_RUN_ID" ]]; then
  echo "ERROR: --resume-runid cannot be combined with index_dir-as-runid" >&2
  exit 2
fi
if [[ -n "$RUN_ID_OVERRIDE" && -n "$INDEX_RUN_ID" && "$RUN_ID_OVERRIDE" == "$INDEX_RUN_ID" ]]; then
  echo "ERROR: run_id must differ from index_dir-as-runid (would write outputs into the index source run folder)" >&2
  echo "- Either set run_id to a new value, or clear index_dir and use resume_runid=${RUN_ID_OVERRIDE}" >&2
  exit 2
fi

if [[ -n "$INDEX_DIR" && "$RUN_ALL" -eq 1 ]]; then
  echo "ERROR: --index-dir cannot be combined with --run-all (multiple run_groups would overwrite indexes)" >&2
  exit 2
fi

if [[ "$STAGE" != "native" ]]; then
  ensure_framework_py_deps

  if [[ ! -f "$HDF5" ]]; then
    echo "ERROR: hdf5 not found: $HDF5" >&2
    exit 1
  fi

  # Auto-detect metric from HDF5 when not specified.
  if [[ -z "$METRIC" ]]; then
    detected_distance="$(detect_distance_from_hdf5 "$HDF5" | tr -d '\r' | head -n1)"
    if [[ -n "$detected_distance" ]]; then
      METRIC="$detected_distance"
    else
      # Heuristic fallback based on filename.
      hdf5_base="$(basename "$HDF5")"
      if [[ "$hdf5_base" == *"angular"* || "$hdf5_base" == *"cosine"* ]]; then
        METRIC="angular"
      elif [[ "$hdf5_base" == *"l2"* || "$hdf5_base" == *"euclidean"* ]]; then
        METRIC="l2"
      else
        METRIC="angular"
        echo "WARN: could not detect metric from HDF5 attrs['distance']; defaulting to angular" >&2
      fi
    fi
  fi
fi

# The native extension build requires cargo (Rust toolchain).
if ! command -v cargo >/dev/null 2>&1; then
  echo "ERROR: cargo not found in PATH" >&2
  echo "Hint: install Rust (rustup) and/or run setup_remote.sh." >&2
  echo "Hint (remote): source \$HOME/.cargo/env" >&2
  exit 1
fi

# Build the native extension (cargo debug build).
NATIVE_DIR="$WORKSPACE_ROOT/ann-benchmark-epeshared/ann_benchmarks/algorithms/diskann_rs/native"
if [[ ! -d "$NATIVE_DIR" ]]; then
  echo "ERROR: native crate not found: $NATIVE_DIR" >&2
  exit 1
fi

echo "==> cargo build (native extension; profile=$DISKANN_RS_NATIVE_PROFILE)"
if [[ "$DISKANN_RS_NATIVE_PROFILE" == "release" ]]; then
  ( cd "$NATIVE_DIR" && cargo build --release )
else
  ( cd "$NATIVE_DIR" && cargo build )
fi

if [[ "$STAGE" == "native" ]]; then
  echo "==> native build-only complete" >&2
  exit 0
fi

# Compute CPU bind string and clamp if needed.
if command -v nproc >/dev/null 2>&1; then
  NPROC="$(nproc)"
  if [[ "$NPROC" =~ ^[0-9]+$ ]] && [[ "$NPROC" -gt 0 ]]; then
    max_cpu="$((NPROC - 1))"
    if [[ "$CPU_BIND" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      lo="${BASH_REMATCH[1]}"
      hi="${BASH_REMATCH[2]}"
      if [[ "$lo" =~ ^[0-9]+$ ]] && [[ "$hi" =~ ^[0-9]+$ ]]; then
        if [[ "$lo" -gt "$max_cpu" ]]; then
          echo "WARN: CPU_BIND=$CPU_BIND starts beyond available CPUs (nproc=$NPROC); using 0-$max_cpu" >&2
          CPU_BIND="0-$max_cpu"
        elif [[ "$hi" -gt "$max_cpu" ]]; then
          echo "WARN: CPU_BIND=$CPU_BIND exceeds available CPUs (nproc=$NPROC); clamping to $lo-$max_cpu" >&2
          CPU_BIND="$lo-$max_cpu"
        fi
      fi
    elif [[ "$NPROC" -lt 17 && "$CPU_BIND" == "0-16" ]]; then
      # Backwards-compatible behavior: clamp the default range on small machines.
      CPU_BIND="0-$max_cpu"
      echo "WARN: only $NPROC CPUs available; using CPU_BIND=$CPU_BIND" >&2
    fi
  fi
fi

if [[ -n "$RUN_ID_OVERRIDE" ]]; then
  RUN_ID="$RUN_ID_OVERRIDE"
else
  RUN_ID="$(date +%Y%m%d-%H%M%S)-$$"
  AUTO_RUN_ID=1
fi
DATASET="$(basename "$HDF5")"
DATASET="${DATASET%.hdf5}"

DEFAULT_RUNS_DIR="$SCRIPT_DIR/result"
RUNS_DIR="${RUNS_DIR:-$DEFAULT_RUNS_DIR}"
WORK_DIR="$RUNS_DIR/$DATASET/$RUN_ID"

ROOT_CASES_DIR="$WORK_DIR/cases"

sanitize_case_id() {
  # Keep names URL-friendly for the web UI path validator.
  # Replace any disallowed characters with '_'.
  echo "$1" | sed -E 's/[^A-Za-z0-9._-]+/_/g'
}

case_done_marker_path() {
  local case_dir="$1"
  local stage="${2:-$STAGE}"
  echo "$case_dir/.done.$stage"
}

case_is_done() {
  local case_dir="$1"
  local stage="${2:-$STAGE}"
  local marker
  marker="$(case_done_marker_path "$case_dir" "$stage")"
  if [[ -f "$marker" ]]; then
    return 0
  fi

  # Heuristic fallback if marker is missing.
  if [[ "$stage" == "build" ]]; then
    [[ -f "$case_dir/outputs/output.build.json" ]]
    return $?
  fi
  if [[ "$stage" == "search" ]]; then
    [[ -f "$case_dir/outputs/summary.tsv" ]]
    return $?
  fi
  # all
  [[ -f "$case_dir/outputs/output.build.json" && -f "$case_dir/outputs/summary.tsv" ]]
  return $?
}

merge_root_outputs() {
  "$PYTHON_BIN" - "$WORK_DIR" <<'PY'
import sys
import json
import csv
import re
from pathlib import Path

work_dir = Path(sys.argv[1]).resolve()
cases_dir = work_dir / "cases"
out_dir = work_dir / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)

summary_rows = []
summary_header = None
details_parts = ["# diskann-ann-bench details (aggregated)", ""]

build_cases = {}
search_cases = {}

# For shared-index sweeps, many cases are search-only and don't have a build JSON.
# Reuse build params from another case that points at the same index_prefix.
build_params_by_index_prefix = {}

csv_rows_pq = []
csv_rows_spherical = []

COMMON_FIELDS = [
  "case",
  "distance",
  "k",
  "l_search",
  "reps",
  "peak_rss_gib",
  "peak_tps",
  "peak_kB_read_s",
  "peak_kB_wrtn_s",
  "recall_at_k",
  "qps_mean",
  "lat_mean_us",
  "lat_p50_us",
  "lat_p95_us",
  "lat_p99_us",
  "load_index_s",
  "l_build",
  "max_outdegree",
  "alpha",
  "index_prefix",
]

PQ_FIELDS = [
  *COMMON_FIELDS,
  "num_pq_chunks",
  "translate_to_center",
  "num_centers",
  "max_k_means_reps",
  "pq_seed",
]

SPHERICAL_FIELDS = [
  *COMMON_FIELDS,
  "nbits",
  "spherical_seed",
]

PERF_2DP_FIELDS = {
  "peak_rss_gib",
  "peak_tps",
  "peak_kB_read_s",
  "peak_kB_wrtn_s",
  "recall_at_k",
  "qps_mean",
  "lat_mean_us",
  "lat_p50_us",
  "lat_p95_us",
  "lat_p99_us",
  "load_index_s",
}


def _as_str(v):
  if v is None:
    return ""
  if isinstance(v, bool):
    return "true" if v else "false"
  return str(v)


def _get(search_obj, build_obj, key: str):
  if isinstance(search_obj, dict) and key in search_obj:
    return search_obj.get(key)
  if isinstance(build_obj, dict) and key in build_obj:
    return build_obj.get(key)
  return None


def _fmt_2dp(v: str) -> str:
  s = (v or "").strip()
  if not s:
    return ""
  try:
    return f"{float(s):.2f}"
  except Exception:
    return s


_re_chunks = re.compile(r"chunks(?P<chunks>[0-9]+)")
_re_spherical_bits = re.compile(r"_(?P<bits>[0-9]+)b_")

if cases_dir.is_dir():
    for case_dir in sorted([p for p in cases_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        case_id = case_dir.name
        summary_path = case_dir / "outputs" / "summary.tsv"
        if summary_path.is_file():
            lines = summary_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if lines:
                header = lines[0].split("\t")
                if summary_header is None:
                    summary_header = ["case"] + header
                for line in lines[1:]:
                    if not line.strip():
                        continue
                    summary_rows.append([case_id] + line.split("\t"))

        details_path = case_dir / "outputs" / "details.md"
        if details_path.is_file():
            details = details_path.read_text(encoding="utf-8", errors="replace").rstrip() + "\n"
            details_parts.extend([f"## {case_id}", "", details, ""])

        build_obj = None
        build_path = case_dir / "outputs" / "output.build.json"
        if build_path.is_file():
            try:
                build_obj = json.loads(build_path.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                build_cases[case_id] = {"_raw": build_path.read_text(encoding="utf-8", errors="replace")}
                build_obj = None
            else:
                build_cases[case_id] = build_obj

        search_path = case_dir / "outputs" / "output.search.json"
        if search_path.is_file():
            try:
                search_obj = json.loads(search_path.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                search_cases[case_id] = {"_raw": search_path.read_text(encoding="utf-8", errors="replace")}
                continue

            search_cases[case_id] = search_obj

            algo = (search_obj.get("algo") or "").strip().lower()
            distance = _get(search_obj, build_obj, "distance")

            index_prefix_val = _as_str(_get(search_obj, build_obj, "index_prefix"))

            common_row = {
              "case": case_id,
              "distance": _as_str(distance),
              "k": _as_str(_get(search_obj, build_obj, "k")),
              "l_search": _as_str(_get(search_obj, build_obj, "l_search")),
              "reps": _as_str(_get(search_obj, build_obj, "reps")),
              "peak_rss_gib": _as_str(_get(search_obj, build_obj, "peak_rss_gib")),
              "peak_tps": _as_str(_get(search_obj, build_obj, "peak_tps")),
              "peak_kB_read_s": _as_str(_get(search_obj, build_obj, "peak_kB_read_s")),
              "peak_kB_wrtn_s": _as_str(_get(search_obj, build_obj, "peak_kB_wrtn_s")),
              "recall_at_k": _as_str(_get(search_obj, build_obj, "recall_at_k")),
              "qps_mean": _as_str(_get(search_obj, build_obj, "qps_mean")),
              "lat_mean_us": _as_str(_get(search_obj, build_obj, "lat_mean_us")),
              "lat_p50_us": _as_str(_get(search_obj, build_obj, "lat_p50_us")),
              "lat_p95_us": _as_str(_get(search_obj, build_obj, "lat_p95_us")),
              "lat_p99_us": _as_str(_get(search_obj, build_obj, "lat_p99_us")),
              "load_index_s": _as_str(_get(search_obj, build_obj, "load_index_s")),
              "l_build": _as_str(_get(search_obj, build_obj, "l_build")),
              "max_outdegree": _as_str(_get(search_obj, build_obj, "max_outdegree")),
              "alpha": _as_str(_get(search_obj, build_obj, "alpha")),
              "index_prefix": index_prefix_val,
            }

            if index_prefix_val:
              # Fill missing build params for search-only cases.
              prior = build_params_by_index_prefix.get(index_prefix_val)
              if prior:
                for key in ("l_build", "max_outdegree", "alpha"):
                  if not common_row.get(key):
                    common_row[key] = prior.get(key, "")
              # Record build params if we have them.
              if common_row.get("l_build") or common_row.get("max_outdegree") or common_row.get("alpha"):
                slot = build_params_by_index_prefix.setdefault(index_prefix_val, {})
                for key in ("l_build", "max_outdegree", "alpha"):
                  v = common_row.get(key)
                  if v:
                    slot[key] = v

            if algo == "pq":
              chunks = _get(search_obj, build_obj, "num_pq_chunks")
              if chunks is None:
                m = _re_chunks.search(case_id)
                if m:
                  try:
                    chunks = int(m.group("chunks"))
                  except Exception:
                    chunks = None

              translate = _get(search_obj, build_obj, "translate_to_center")
              if translate is None and isinstance(distance, str):
                d = distance.strip().lower()
                if d in ("angular", "cosine"):
                  translate = False
                elif d in ("euclidean", "l2"):
                  translate = True

              row = {k: "" for k in PQ_FIELDS}
              row.update(common_row)
              row.update(
                {
                  "num_pq_chunks": _as_str(chunks),
                  "translate_to_center": _as_str(translate),
                  "num_centers": _as_str(_get(search_obj, build_obj, "num_centers")),
                  "max_k_means_reps": _as_str(_get(search_obj, build_obj, "max_k_means_reps")),
                  "pq_seed": _as_str(_get(search_obj, build_obj, "pq_seed")),
                }
              )
              csv_rows_pq.append(row)

            elif algo == "spherical":
              nbits = _get(search_obj, build_obj, "nbits")
              if nbits is None:
                m = _re_spherical_bits.search(case_id)
                if m:
                  try:
                    nbits = int(m.group("bits"))
                  except Exception:
                    nbits = None

              row = {k: "" for k in SPHERICAL_FIELDS}
              row.update(common_row)
              row.update(
                {
                  "nbits": _as_str(nbits),
                  "spherical_seed": _as_str(_get(search_obj, build_obj, "spherical_seed")),
                }
              )
              csv_rows_spherical.append(row)

if summary_header is not None:
    out = ["\t".join(summary_header)]
    out += ["\t".join(row) for row in summary_rows]
    (out_dir / "summary.tsv").write_text("\n".join(out) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows, fieldnames):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row = dict(r)
            for key in PERF_2DP_FIELDS:
                if key in row:
                    row[key] = _fmt_2dp(str(row.get(key, "")))
            w.writerow(row)


_write_csv(out_dir / "summary.pq.csv", csv_rows_pq, PQ_FIELDS)
_write_csv(out_dir / "summary.spherical.csv", csv_rows_spherical, SPHERICAL_FIELDS)

(out_dir / "details.md").write_text("\n".join(details_parts).rstrip() + "\n", encoding="utf-8")

if build_cases:
    (out_dir / "output.build.json").write_text(json.dumps({"cases": build_cases}, indent=2) + "\n", encoding="utf-8")
if search_cases:
    (out_dir / "output.search.json").write_text(json.dumps({"cases": search_cases}, indent=2) + "\n", encoding="utf-8")
PY
}

on_exit() {
  local status=$?

  # Always attempt to merge outputs from finished cases so the web UI can show partial progress.
  merge_root_outputs || true

  # Delete created indexes after a successful full run (stage=all) to control disk usage.
  # Safety: never delete external --index-dir content.
  local should_delete=0
  if [[ "$status" -eq 0 && "$STAGE" == "all" && -z "$INDEX_DIR" ]]; then
    if [[ "$DELETE_INDEX_AFTER_RUN" == "yes" ]]; then
      should_delete=1
    elif [[ "$DELETE_INDEX_AFTER_RUN" == "no" ]]; then
      should_delete=0
    else
      # auto
      if [[ "$AUTO_RUN_ID" -eq 1 ]]; then
        should_delete=1
      fi
    fi
  fi

  if [[ "$should_delete" -eq 1 ]]; then
    echo "==> deleting created indexes (to save disk)" >&2
    rm -rf "$WORK_DIR/shared_indexes" 2>/dev/null || true
    if [[ -d "$ROOT_CASES_DIR" ]]; then
      local idx
      for idx in "$ROOT_CASES_DIR"/*/index; do
        if [[ -d "$idx" ]]; then
          rm -rf "$idx" 2>/dev/null || true
        fi
      done
    fi
    echo "==> index deletion done" >&2
  fi

  return "$status"
}

trap on_exit EXIT

echo "==> running framework_entry (host)"
echo "    cpu bind: $CPU_BIND"
echo "    metric:   $METRIC"
echo "    fit batch size: $DISKANN_RS_FIT_BATCH_SIZE"
echo "    query mode: $([[ "$BATCH" -eq 1 ]] && echo batch || echo per-query)"
if [[ "$COMPARE" -eq 1 ]]; then
  echo "    mode:     compare (pq + spherical)"
else
  echo "    algo:     $ALGO"
fi
echo "    work dir: $WORK_DIR"

mkdir -p "$ROOT_CASES_DIR" "$WORK_DIR/outputs"
echo "ann_bench_diskann_rs" > "$WORK_DIR/mode.txt"
echo "$CPU_BIND" > "$WORK_DIR/cpu-bind.txt"
echo "$BATCH" > "$WORK_DIR/batch.txt"

if command -v lscpu >/dev/null 2>&1; then
  if [[ ! -f "$WORK_DIR/lscpu.txt" ]]; then
    lscpu > "$WORK_DIR/lscpu.txt" 2>/dev/null || true
  fi
fi

# Best-effort total memory capture (for web UI display).
if [[ ! -f "$WORK_DIR/memory.txt" ]]; then
  mem_kb=""
  if [[ -r /proc/meminfo ]]; then
    mem_kb="$(awk '/^MemTotal:/ {print $2; exit}' /proc/meminfo 2>/dev/null || true)"
  fi
  if [[ "$mem_kb" =~ ^[0-9]+$ ]] && command -v awk >/dev/null 2>&1; then
    awk -v kb="$mem_kb" 'BEGIN {printf "%.2f GiB\n", kb/1024/1024}' > "$WORK_DIR/memory.txt" 2>/dev/null || true
  fi
fi

affinity_prefix=()
if command -v numactl >/dev/null 2>&1; then
  affinity_prefix=(numactl --physcpubind="$CPU_BIND" --)
elif command -v taskset >/dev/null 2>&1; then
  affinity_prefix=(taskset -c "$CPU_BIND")
else
  echo "WARN: neither numactl nor taskset found; run will not be CPU-affined" >&2
fi

base_args=(
  --hdf5 "$HDF5"
  --dataset "$DATASET"
  --metric "$METRIC"
  -k "$K"
  --reps "$REPS"
)

if [[ "$BATCH" -eq 1 ]]; then
  base_args+=(--batch)
fi

if [[ "$TRANSLATE_TO_CENTER" == "true" ]]; then
  base_args+=(--translate-to-center)
elif [[ "$TRANSLATE_TO_CENTER" == "false" ]]; then
  base_args+=(--no-translate-to-center)
fi

if [[ -z "$CONFIG_YML" ]]; then
  CONFIG_YML="$(default_config_yml)"
fi

base_args+=(--config-yml "$CONFIG_YML")

if [[ -n "$INDEX_DIR" ]]; then
  if [[ ! -d "$INDEX_DIR" ]]; then
    echo "ERROR: index dir not found: $INDEX_DIR" >&2
    exit 2
  fi
fi

index_prefix_glob_for_algo() {
  local algo="$1"
  case "$algo" in
    fp) echo "diskann_rs" ;;
    pq) echo "diskann_rs_pq" ;;
    spherical) echo "diskann_rs_spherical" ;;
    *) echo "diskann_rs" ;;
  esac
}

index_dir_from_runid_for_case() {
  local index_run_id="$1"; shift
  local case_id="$1"; shift
  # Uses the standard run output layout.
  echo "$RUNS_DIR/$DATASET/$index_run_id/cases/$case_id/index"
}

validate_index_dir_for_case() {
  local algo="$1"; shift
  local dir="$1"; shift
  local glob_prefix
  glob_prefix="$(index_prefix_glob_for_algo "$algo")"
  if [[ ! -d "$dir" ]]; then
    echo "ERROR: missing index dir: $dir" >&2
    return 2
  fi
  # Ensure at least one artifact exists for the expected prefix.
  shopt -s nullglob
  local matches=("$dir/${glob_prefix}"*)
  shopt -u nullglob
  if [[ ${#matches[@]} -eq 0 ]]; then
    echo "ERROR: index dir present but no artifacts found for prefix ${glob_prefix}*: $dir" >&2
    return 2
  fi
  return 0
}

if [[ -n "$INDEX_RUN_ID" ]]; then
  INDEX_RUN_DIR="$RUNS_DIR/$DATASET/$INDEX_RUN_ID"
  if [[ ! -d "$INDEX_RUN_DIR" ]]; then
    echo "ERROR: index run_id not found under runs_dir: $INDEX_RUN_DIR" >&2
    exit 2
  fi

  # Preflight validation for the common case (single-algo, run_all preset): validate all cases
  # before running anything.
  if [[ "$COMPARE" -eq 0 && "$RUN_ALL" -eq 1 ]]; then
    if [[ -z "$NAME" ]]; then
      echo "ERROR: run_all requires name when using index_dir-as-runid" >&2
      exit 2
    fi
    mapfile -t _groups < <(get_run_groups_by_name "$NAME" "$METRIC" "$CONFIG_YML")
    missing=0
    for rg in "${_groups[@]}"; do
      cid="$(sanitize_case_id "$ALGO-$rg")"
      idir="$(index_dir_from_runid_for_case "$INDEX_RUN_ID" "$cid")"
      if ! validate_index_dir_for_case "$ALGO" "$idir"; then
        missing=1
      fi
    done
    if [[ "$missing" -ne 0 ]]; then
      echo "ERROR: one or more required indexes are missing under index run_id=$INDEX_RUN_ID; aborting." >&2
      exit 2
    fi
  fi
fi

run_one() {
  local algo="$1"; shift
  local case_id_raw="$1"; shift
  local run_group_for_algo="${1:-}"; shift || true
  local stage_override="${1:-}"; shift || true
  local index_dir_override="${1:-}"; shift || true
  local case_id
  case_id="$(sanitize_case_id "$case_id_raw")"
  local work_dir_full="$ROOT_CASES_DIR/$case_id"

  local stage_to_use
  stage_to_use="${stage_override:-$STAGE}"

  # For user-invoked --stage search, require an existing case dir to avoid accidental
  # searches against missing/incorrect build artifacts.
  # For internal shared-index searches (stage_override provided with an explicit index dir),
  # we allow creating a new case directory and writing outputs there.
  if [[ "$stage_to_use" == "search" && -z "$stage_override" && -n "$RUN_ID_OVERRIDE" && -z "$INDEX_DIR" && -z "$INDEX_RUN_ID" && ! -d "$work_dir_full" ]]; then
    echo "ERROR: case dir not found for --stage search: $work_dir_full" >&2
    echo "Hint: resume with --stage build (or all) using the same --resume-runid." >&2
    exit 2
  fi

  if case_is_done "$work_dir_full" "$stage_to_use"; then
    echo "==> skip (already done): case=$case_id stage=$stage_to_use" >&2
    return 0
  fi

  local invocation_args=()
  invocation_args+=("${base_args[@]}")
  invocation_args+=(--stage "$stage_to_use")

  local extra_args=()
  if [[ -n "$run_group_for_algo" ]]; then
    invocation_args+=(--run-group "$run_group_for_algo")
  else
    invocation_args+=(--l-build "$L_BUILD" --max-outdegree "$MAX_OUTDEGREE" --alpha "$ALPHA" --l-search "$L_SEARCH")
    extra_args+=(--algo "$algo")
    if [[ "$algo" == "pq" ]]; then
      if [[ -z "$NUM_PQ_CHUNKS" ]]; then
        echo "ERROR: --num-pq-chunks is required for algo=pq (or --compare), unless provided via --run-group" >&2
        exit 2
      fi
      extra_args+=(--num-pq-chunks "$NUM_PQ_CHUNKS")
    fi
    if [[ "$algo" == "spherical" ]]; then
      extra_args+=(--spherical-nbits "$SPHERICAL_NBITS")
    fi
  fi

  mkdir -p "$work_dir_full/outputs"
  echo "ann_bench_diskann_rs" > "$work_dir_full/mode.txt"
  echo "$CPU_BIND" > "$work_dir_full/cpu-bind.txt"
  echo "$BATCH" > "$work_dir_full/batch.txt"

  # Make diskann_rs_native importable from the cargo output folder.
  target_subdir="debug"
  if [[ "$DISKANN_RS_NATIVE_PROFILE" == "release" ]]; then
    target_subdir="release"
  fi
  native_target_dir="$NATIVE_DIR/target/$target_subdir"
  if [[ -f "$native_target_dir/libdiskann_rs_native.so" ]]; then
    ln -sf "libdiskann_rs_native.so" "$native_target_dir/diskann_rs_native.so"
  fi

  export PYTHONPATH="$native_target_dir:$WORKSPACE_ROOT/ann-benchmark-epeshared:${PYTHONPATH:-}"

  echo "==> run: algo=$algo case=$case_id stage=$stage_to_use" >&2
  cmd=("${affinity_prefix[@]}" "$PYTHON_BIN" "$SCRIPT_DIR/src/framework_entry.py")
  cmd+=(--work-dir "$work_dir_full")
  cmd+=("${invocation_args[@]}")
  if [[ -n "$index_dir_override" ]]; then
    mkdir -p "$index_dir_override"
    cmd+=(--index-dir "$index_dir_override")
  elif [[ -n "$INDEX_DIR" ]]; then
    cmd+=(--index-dir "$INDEX_DIR")
  elif [[ -n "$INDEX_RUN_ID" && "$stage_to_use" == "search" ]]; then
    # Reuse per-case index artifacts from an existing run.
    prior_index_dir="$(index_dir_from_runid_for_case "$INDEX_RUN_ID" "$case_id")"
    validate_index_dir_for_case "$algo" "$prior_index_dir"
    cmd+=(--index-dir "$prior_index_dir")
  fi
  cmd+=("${extra_args[@]}")
  "${cmd[@]}"

  # Mark completion and refresh aggregated outputs.
  touch "$(case_done_marker_path "$work_dir_full" "$stage_to_use")"
  merge_root_outputs || true
}

if [[ "$COMPARE" -eq 1 ]]; then
  if [[ "$RUN_ALL" -eq 1 ]]; then
    mapfile -t pq_groups < <(get_run_groups_by_name "$NAME_PQ" "$METRIC" "$CONFIG_YML")
    mapfile -t spherical_groups < <(get_run_groups_by_name "$NAME_SPHERICAL" "$METRIC" "$CONFIG_YML")

    if [[ -n "$RUN_GROUP_PREFIX_PQ" ]]; then
      mapfile -t pq_groups < <(filter_run_groups_by_prefix "$RUN_GROUP_PREFIX_PQ" "${pq_groups[@]}")
    fi
    if [[ -n "$RUN_GROUP_PREFIX_SPHERICAL" ]]; then
      mapfile -t spherical_groups < <(filter_run_groups_by_prefix "$RUN_GROUP_PREFIX_SPHERICAL" "${spherical_groups[@]}")
    fi

    if [[ ${#pq_groups[@]} -eq 0 ]]; then
      echo "ERROR: no pq run_groups matched (name=$NAME_PQ prefix=$RUN_GROUP_PREFIX_PQ metric=$METRIC)" >&2
      exit 2
    fi
    if [[ ${#spherical_groups[@]} -eq 0 ]]; then
      echo "ERROR: no spherical run_groups matched (name=$NAME_SPHERICAL prefix=$RUN_GROUP_PREFIX_SPHERICAL metric=$METRIC)" >&2
      exit 2
    fi

    # Shared-index mode: if run_group names end with _L<digits>, reuse one index per build key
    # (base name without the _L suffix) and only vary l_search via run_group.
    pq_shared=0
    for rg in "${pq_groups[@]}"; do
      if [[ "$rg" =~ _L[0-9]+$ ]]; then
        pq_shared=1
        break
      fi
    done
    spherical_shared=0
    for rg in "${spherical_groups[@]}"; do
      if [[ "$rg" =~ _L[0-9]+$ ]]; then
        spherical_shared=1
        break
      fi
    done

    if [[ "$pq_shared" -eq 1 ]]; then
      declare -A pq_map
      for rg in "${pq_groups[@]}"; do
        base="${rg%_L*}"
        l="${rg##*_L}"
        pq_map["$base"]+="$l\t$rg\n"
      done
      for base in "${!pq_map[@]}"; do
        index_dir="$WORK_DIR/shared_indexes/pq/$base"
        # Sort by L so the smallest L builds+searches first in stage=all.
        mapfile -t rows < <(printf "%b" "${pq_map[$base]}" | sort -n -k1,1)
        if [[ "$STAGE" == "build" ]]; then
          rg0="${rows[0]#*$'\t'}"
          run_one pq "pq-$base-build" "$rg0" build "$index_dir"
        elif [[ "$STAGE" == "search" ]]; then
          for row in "${rows[@]}"; do
            rgx="${row#*$'\t'}"
            run_one pq "pq-$rgx" "$rgx" search "$index_dir"
          done
        else
          # all
          rg0="${rows[0]#*$'\t'}"
          run_one pq "pq-$rg0" "$rg0" all "$index_dir"
          for row in "${rows[@]:1}"; do
            rgx="${row#*$'\t'}"
            run_one pq "pq-$rgx" "$rgx" search "$index_dir"
          done
        fi
      done
    else
      for rg in "${pq_groups[@]}"; do
        run_one pq "pq-$rg" "$rg"
      done
    fi

    if [[ "$spherical_shared" -eq 1 ]]; then
      declare -A spherical_map
      for rg in "${spherical_groups[@]}"; do
        base="${rg%_L*}"
        l="${rg##*_L}"
        spherical_map["$base"]+="$l\t$rg\n"
      done
      for base in "${!spherical_map[@]}"; do
        index_dir="$WORK_DIR/shared_indexes/spherical/$base"
        mapfile -t rows < <(printf "%b" "${spherical_map[$base]}" | sort -n -k1,1)
        if [[ "$STAGE" == "build" ]]; then
          rg0="${rows[0]#*$'\t'}"
          run_one spherical "spherical-$base-build" "$rg0" build "$index_dir"
        elif [[ "$STAGE" == "search" ]]; then
          for row in "${rows[@]}"; do
            rgx="${row#*$'\t'}"
            run_one spherical "spherical-$rgx" "$rgx" search "$index_dir"
          done
        else
          rg0="${rows[0]#*$'\t'}"
          run_one spherical "spherical-$rg0" "$rg0" all "$index_dir"
          for row in "${rows[@]:1}"; do
            rgx="${row#*$'\t'}"
            run_one spherical "spherical-$rgx" "$rgx" search "$index_dir"
          done
        fi
      done
    else
      for rg in "${spherical_groups[@]}"; do
        run_one spherical "spherical-$rg" "$rg"
      done
    fi
  else
    if [[ -n "$RUN_GROUP_PQ" && -n "$RUN_GROUP_SPHERICAL" ]]; then
      run_one pq "pq" "$RUN_GROUP_PQ"
      run_one spherical "spherical" "$RUN_GROUP_SPHERICAL"
    else
      if [[ "$HAS_LIST_SWEEP" -eq 1 ]]; then
        if [[ ${#NUM_PQ_CHUNKS_VALUES[@]} -eq 0 ]]; then
          echo "ERROR: --compare with list sweeps requires --num-pq-chunks (or --num-pq-chunks-list)" >&2
          exit 2
        fi

        for pq_chunks in "${NUM_PQ_CHUNKS_VALUES[@]}"; do
          for l in "${L_SEARCH_VALUES[@]}"; do
            NUM_PQ_CHUNKS="$pq_chunks"
            L_SEARCH="$l"
            run_one pq "pq_chunks${pq_chunks}_L${l}"
          done
        done

        for nbits in "${SPHERICAL_NBITS_VALUES[@]}"; do
          for l in "${L_SEARCH_VALUES[@]}"; do
            SPHERICAL_NBITS="$nbits"
            L_SEARCH="$l"
            run_one spherical "spherical_${nbits}b_L${l}"
          done
        done
      else
        run_one pq "pq"
        run_one spherical "spherical"
      fi
    fi
  fi
else
  if [[ "$RUN_ALL" -eq 1 ]]; then
    mapfile -t groups < <(get_run_groups_by_name "$NAME" "$METRIC" "$CONFIG_YML")
    for rg in "${groups[@]}"; do
      run_one "$ALGO" "$ALGO-$rg" "$rg"
    done
  else
    if [[ -n "$RUN_GROUP" ]]; then
      run_one "$ALGO" "$ALGO" "$RUN_GROUP"
    else
      if [[ "$HAS_LIST_SWEEP" -eq 1 ]]; then
        if [[ "$ALGO" == "pq" ]]; then
          if [[ ${#NUM_PQ_CHUNKS_VALUES[@]} -eq 0 ]]; then
            echo "ERROR: --num-pq-chunks is required for algo=pq (or provide --num-pq-chunks-list)" >&2
            exit 2
          fi
          for pq_chunks in "${NUM_PQ_CHUNKS_VALUES[@]}"; do
            for l in "${L_SEARCH_VALUES[@]}"; do
              NUM_PQ_CHUNKS="$pq_chunks"
              L_SEARCH="$l"
              run_one pq "pq_chunks${pq_chunks}_L${l}"
            done
          done
        elif [[ "$ALGO" == "spherical" ]]; then
          for nbits in "${SPHERICAL_NBITS_VALUES[@]}"; do
            for l in "${L_SEARCH_VALUES[@]}"; do
              SPHERICAL_NBITS="$nbits"
              L_SEARCH="$l"
              run_one spherical "spherical_${nbits}b_L${l}"
            done
          done
        else
          for l in "${L_SEARCH_VALUES[@]}"; do
            L_SEARCH="$l"
            run_one fp "fp_L${l}"
          done
        fi
      else
        run_one "$ALGO" "$ALGO"
      fi
    fi
  fi
fi

echo "==> done"
echo "    dataset: $DATASET"
echo "    run_id:  $RUN_ID"
echo "    run dir: $RUNS_DIR/$DATASET/$RUN_ID"
echo
echo "Tip: resume this run (skip completed cases):"
echo "  bash DiskANN-playground/diskann-ann-bench/run_local.sh --resume-runid $RUN_ID [same args...]"
echo
echo "Tip: reuse saved fit/index (no rebuild, same run):"
echo "  bash DiskANN-playground/diskann-ann-bench/run_local.sh --stage search --resume-runid $RUN_ID [same args...]"
echo
echo "Next: start web server:"
echo "  bash DiskANN-playground/diskann-ann-bench/run_web.sh"
