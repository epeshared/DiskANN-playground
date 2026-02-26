#!/usr/bin/env bash
set -euo pipefail

# 远端环境初始化脚本
# 用法:
#   bash setup_remote.sh <remote_workspace_dir>
#   bash setup_remote.sh <remote_workspace_dir> --mode build
#
# mode:
#   full  - install system deps + Rust + Python venv deps (default)
#   build - install system deps + Rust only (skip pip installs)

REMOTE_DIR="$HOME/diskann-workspace"
MODE="full"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="${2:-}"
            shift 2
            ;;
        --mode=*)
            MODE="${1#*=}"
            shift
            ;;
        --remote-dir)
            REMOTE_DIR="${2:-}"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 <remote_workspace_dir> [--mode full|build]" >&2
            exit 0
            ;;
        --*)
            echo "ERROR: unknown option: $1" >&2
            exit 2
            ;;
        *)
            REMOTE_DIR="$1"
            shift
            ;;
    esac
done

if [[ -z "$REMOTE_DIR" ]]; then
    echo "ERROR: remote workspace dir is empty" >&2
    exit 2
fi

case "${MODE,,}" in
    full|build) ;;
    *)
        echo "ERROR: invalid --mode=$MODE (expected: full|build)" >&2
        exit 2
        ;;
esac

SUDO=""
if [[ "$(id -u)" -ne 0 ]]; then
    if command -v sudo &>/dev/null; then
        SUDO="sudo"
    else
        echo "ERROR: not running as root and sudo not found" >&2
        exit 1
    fi
fi

VENV_DIR="$REMOTE_DIR/.venv"

echo "==> [Remote] Updating apt and installing system dependencies..."
# 避免 apt-get 交互式弹窗
export DEBIAN_FRONTEND=noninteractive

# Avoid post-install interactive prompts (e.g., needrestart) when running under a TTY.
export NEEDRESTART_MODE=a
export NEEDRESTART_SUSPEND=1

# Ensure predictable tool output even if some locales are missing.
export LC_ALL=C.UTF-8

# Make apt/dpkg non-interactive even when running under a TTY.
# - Dpkg::Options: avoid conffile prompts
# - Dpkg::Use-Pty: disables fancy TTY progress that can break log streaming
APT_GET=(
    $SUDO apt-get
    -o Acquire::Retries=3
    -o Dpkg::Options::=--force-confdef
    -o Dpkg::Options::=--force-confold
    -o Dpkg::Use-Pty=0
)

"${APT_GET[@]}" update
"${APT_GET[@]}" install -y \
    python3 python3-pip python3-venv python3-dev \
    python3-yaml python3-numpy python3-h5py \
    curl ca-certificates build-essential pkg-config libssl-dev \
    htop numactl \
    rsync

if ! command -v cargo &> /dev/null; then
    # Prefer distro Rust (often more reliable than rustup in locked-down networks).
    if "${APT_GET[@]}" install -y cargo rustc; then
        :
    else
        echo "==> [Remote] Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    fi
fi

# 加载 cargo 环境变量
source "$HOME/.cargo/env" || true

if [[ "${MODE,,}" == "build" ]]; then
    echo "==> [Remote] Skipping Python dependency install (mode=build)."
else
    echo "==> [Remote] Installing Python dependencies..."
    if [ -f "$REMOTE_DIR/ann-benchmark-epeshared/requirements.txt" ]; then
        export PIP_DISABLE_PIP_VERSION_CHECK=1
        export PIP_PROGRESS_BAR=off

        # Ubuntu 22.04+/24.04 may enforce PEP 668 (externally-managed-environment),
        # which blocks system/user-site pip installs. Use a venv under REMOTE_DIR.
        echo "==> [Remote] Creating venv at $VENV_DIR"
        python3 -m venv "$VENV_DIR"

        echo "==> [Remote] Upgrading pip in venv"
        "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel --progress-bar off

        echo "==> [Remote] Installing requirements into venv"
        "$VENV_DIR/bin/python" -m pip install -r "$REMOTE_DIR/ann-benchmark-epeshared/requirements.txt" --progress-bar off
    else
        echo "WARN: requirements.txt not found at $REMOTE_DIR/ann-benchmark-epeshared/requirements.txt"
    fi
fi

echo "==> [Remote] Setup complete."
