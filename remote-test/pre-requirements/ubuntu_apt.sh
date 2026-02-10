#!/usr/bin/env bash
set -euo pipefail

if ! command -v apt-get >/dev/null 2>&1; then
  echo "ERROR: apt-get not found; this script is for Ubuntu/Debian." >&2
  exit 1
fi

if [[ "$(id -u)" -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

echo "+ ${SUDO} apt-get update"
${SUDO} apt-get update

echo "+ ${SUDO} apt-get install -y ..."
${SUDO} apt-get install -y \
  git \
  tar \
  curl \
  ca-certificates \
  numactl \
  util-linux \
  build-essential \
  pkg-config \
  libssl-dev \
  cmake \
  clang \
  python3 \
  python3-pip

echo "Done."
