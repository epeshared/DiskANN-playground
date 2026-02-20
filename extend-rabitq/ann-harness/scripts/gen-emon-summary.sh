#!/usr/bin/env bash
set -euo pipefail

# Make sort/comm ordering consistent.
export LC_ALL=C

# Generate EMON Excel summaries from ann-harness run directories.
#
# Given one or more run timestamp directories (e.g. .../runs/<dataset>/<timestamp>/),
# this script finds all emon.dat files under them and runs:
#   emon -process-pyedp /opt/intel/sep/config/edp/pyedp_config.txt
# in each emon.dat directory to produce an Excel output.
#
# It then deletes *newly created* intermediate files/directories, keeping:
#   - the original emon.dat
#   - any .xlsx files produced by the processing step
#
# Usage:
#   ./gen-emon-summary.sh /path/to/runs/<dataset>/<timestamp>
#   ./gen-emon-summary.sh /path/to/runs/<dataset>/<ts1> /path/to/runs/<dataset>/<ts2>
#
# Environment:
#   EMON_BIN         (default: emon)
#   PYEDP_CONFIG     (default: /opt/intel/sep/config/edp/pyedp_config.txt)
#

EMON_BIN="${EMON_BIN:-emon}"
PYEDP_CONFIG="${PYEDP_CONFIG:-/opt/intel/sep/config/edp/pyedp_config.txt}"

print_cmd() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

if [[ $# -lt 1 ]]; then
  cat >&2 <<'EOF'
Usage:
  gen-emon-summary.sh RUN_DIR [RUN_DIR ...]

Example:
  gen-emon-summary.sh ../runs/dbpedia-openai-1000k-angular/20260211T012941Z
EOF
  exit 2
fi

if ! command -v "$EMON_BIN" >/dev/null 2>&1; then
  die "'$EMON_BIN' not found on PATH (set EMON_BIN or load Intel SEP environment)"
fi
if [[ ! -f "$PYEDP_CONFIG" ]]; then
  die "PYEDP config not found: $PYEDP_CONFIG (set PYEDP_CONFIG)"
fi

# Build a unique list of emon.dat paths across all input run dirs.
emon_dats=()
for run_dir in "$@"; do
  if [[ ! -d "$run_dir" ]]; then
    die "run dir not found: $run_dir"
  fi
  while IFS= read -r -d '' p; do
    emon_dats+=("$p")
  done < <(find "$run_dir" -type f -name 'emon.dat' -print0)
done

if [[ ${#emon_dats[@]} -eq 0 ]]; then
  echo "No emon.dat found under: $*" >&2
  exit 0
fi

# De-dup while preserving order.
uniq_emon_dats=()
declare -A seen
for p in "${emon_dats[@]}"; do
  if [[ -z "${seen[$p]+x}" ]]; then
    seen[$p]=1
    uniq_emon_dats+=("$p")
  fi
done

ok_count=0
fail_count=0

echo "Found ${#uniq_emon_dats[@]} emon.dat files"

for emon_dat in "${uniq_emon_dats[@]}"; do
  dir="$(dirname "$emon_dat")"

  echo ""
  echo "==> Processing: $emon_dat"

  # Record directory contents before processing so we only clean up newly-created artifacts.
  before_list="$(mktemp)"
  after_list="$(mktemp)"
  created_list="$(mktemp)"
  cleanup_list="$(mktemp)"

  # List regular files + directories (relative paths) before processing.
  (cd "$dir" && find . -mindepth 1 -maxdepth 1 -print | LC_ALL=C sort) >"$before_list"

  # Run emon processing in-place.
  if ! (
    cd "$dir"
    print_cmd "$EMON_BIN" -process-pyedp "$PYEDP_CONFIG"
    "$EMON_BIN" -process-pyedp "$PYEDP_CONFIG"
  ); then
    echo "WARN: emon processing failed for: $emon_dat" >&2
    fail_count=$((fail_count + 1))
    rm -f "$before_list" "$after_list" "$created_list" "$cleanup_list"
    continue
  fi

  # List after.
  (cd "$dir" && find . -mindepth 1 -maxdepth 1 -print | LC_ALL=C sort) >"$after_list"

  # Determine newly-created items.
  comm -13 "$before_list" "$after_list" >"$created_list" || true

  # Decide what to delete: newly-created items EXCEPT emon.dat and *.xlsx
  while IFS= read -r rel; do
    rel_no_prefix="${rel#./}"
    # Always keep the original emon.dat
    if [[ "$rel_no_prefix" == "emon.dat" ]]; then
      continue
    fi
    # Keep any excel output
    if [[ "$rel_no_prefix" == *.xlsx ]]; then
      continue
    fi
    echo "$rel_no_prefix" >>"$cleanup_list"
  done <"$created_list"

  # If no xlsx exists, warn (but still do cleanup).
  xlsx_count=$(find "$dir" -maxdepth 1 -type f -name '*.xlsx' | wc -l | tr -d ' ')
  if [[ "$xlsx_count" -eq 0 ]]; then
    echo "WARN: no .xlsx found in $dir after processing" >&2
  fi

  # Perform cleanup.
  if [[ -s "$cleanup_list" ]]; then
    echo "Cleaning up intermediate artifacts in $dir (keeping emon.dat and *.xlsx)"
    while IFS= read -r item; do
      # item is relative to dir.
      target="$dir/$item"
      if [[ -e "$target" || -L "$target" ]]; then
        print_cmd rm -rf -- "$target"
        rm -rf -- "$target"
      fi
    done <"$cleanup_list"
  else
    echo "No intermediate artifacts to clean in $dir"
  fi

  ok_count=$((ok_count + 1))
  rm -f "$before_list" "$after_list" "$created_list" "$cleanup_list"

done

echo ""
echo "Done: ok=$ok_count fail=$fail_count"
exit 0
