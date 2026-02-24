#!/usr/bin/env bash
set -euo pipefail

# 远端环境初始化脚本
# 用法: bash setup_remote.sh <remote_workspace_dir>

REMOTE_DIR="${1:-$HOME/diskann-workspace}"

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
$SUDO apt-get update
$SUDO apt-get install -y \
    python3 python3-pip python3-venv python3-dev \
    curl build-essential pkg-config libssl-dev \
    htop numactl \
    rsync

if ! command -v cargo &> /dev/null; then
    echo "==> [Remote] Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi

# 加载 cargo 环境变量
source "$HOME/.cargo/env" || true

echo "==> [Remote] Installing Python dependencies..."
if [ -f "$REMOTE_DIR/ann-benchmark-epeshared/requirements.txt" ]; then
    # Ubuntu 22.04+/24.04 may enforce PEP 668 (externally-managed-environment),
    # which blocks system/user-site pip installs. Use a venv under REMOTE_DIR.
    echo "==> [Remote] Creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"

    echo "==> [Remote] Upgrading pip in venv"
    "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel

    echo "==> [Remote] Installing requirements into venv"
    "$VENV_DIR/bin/python" -m pip install -r "$REMOTE_DIR/ann-benchmark-epeshared/requirements.txt"
else
    echo "WARN: requirements.txt not found at $REMOTE_DIR/ann-benchmark-epeshared/requirements.txt"
fi

echo "==> [Remote] Setup complete."
