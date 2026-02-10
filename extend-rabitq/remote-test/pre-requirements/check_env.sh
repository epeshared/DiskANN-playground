#!/usr/bin/env bash
set -euo pipefail

check() {
  local name="$1"
  if command -v "$name" >/dev/null 2>&1; then
    echo "OK: $name -> $(command -v "$name")"
    "$name" --version 2>/dev/null | head -n 1 || true
  else
    echo "MISSING: $name"
  fi
}

echo "== basic =="
check git
check tar
check bash
check python3

echo

echo "== cpu/numa =="
check taskset
check numactl

echo

echo "== conda =="
if command -v conda >/dev/null 2>&1; then
  echo "OK: conda -> $(command -v conda)"
  conda --version || true
  conda info --base || true
  conda env list || true
else
  echo "MISSING: conda"
fi
