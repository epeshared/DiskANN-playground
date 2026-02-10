#!/usr/bin/env bash
set -euo pipefail

# Installs Miniforge (conda-forge) into $HOME/miniforge3 by default.
# Safe to re-run: will update existing install with -u.

INSTALL_DIR="${INSTALL_DIR:-$HOME/miniforge3}"
ARCH="$(uname -m)"
OS="$(uname -s)"

if [[ "$OS" != "Linux" ]]; then
  echo "ERROR: This script currently supports Linux only (got: $OS)" >&2
  exit 1
fi

case "$ARCH" in
  x86_64|amd64)
    INSTALLER_NAME="Miniforge3-Linux-x86_64.sh"
    ;;
  aarch64|arm64)
    INSTALLER_NAME="Miniforge3-Linux-aarch64.sh"
    ;;
  *)
    echo "ERROR: Unsupported arch for Miniforge installer: $ARCH" >&2
    exit 1
    ;;
 esac

URL="https://github.com/conda-forge/miniforge/releases/latest/download/${INSTALLER_NAME}"

if command -v conda >/dev/null 2>&1; then
  echo "INFO: conda already on PATH: $(command -v conda)"
  if conda --version >/dev/null 2>&1 && conda info --base >/dev/null 2>&1; then
    echo "INFO: conda appears functional; skipping Miniforge install."
    exit 0
  fi
  echo "WARN: conda is on PATH but seems broken; proceeding to install Miniforge..." >&2
fi

if [[ -x "$INSTALL_DIR/bin/conda" ]]; then
  echo "INFO: Found existing conda at $INSTALL_DIR/bin/conda; skipping reinstall."
  export PATH="$INSTALL_DIR/bin:$PATH"

  echo "+ $INSTALL_DIR/bin/conda --version"
  "$INSTALL_DIR/bin/conda" --version

  bashrc="$HOME/.bashrc"
  line="export PATH=$INSTALL_DIR/bin:\$PATH"
  if [[ -f "$bashrc" ]] && grep -Fqx "$line" "$bashrc"; then
    echo "INFO: ~/.bashrc already contains Miniforge PATH entry."
  else
    echo "# Miniforge" >> "$bashrc"
    echo "$line" >> "$bashrc"
    echo "INFO: Added Miniforge PATH entry to ~/.bashrc"
  fi

  echo "Done. Open a new shell or run: source ~/.bashrc"
  exit 0
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
installer="$tmp_dir/$INSTALLER_NAME"

if command -v wget >/dev/null 2>&1; then
  echo "+ wget -O $installer $URL"
  wget -q -O "$installer" "$URL"
elif command -v curl >/dev/null 2>&1; then
  echo "+ curl -L -o $installer $URL"
  curl -fsSL -o "$installer" "$URL"
else
  echo "ERROR: Need wget or curl to download Miniforge installer." >&2
  exit 1
fi

chmod a+x "$installer"

mkdir -p "$(dirname "$INSTALL_DIR")"

echo "+ bash $installer -b -u -p $INSTALL_DIR"
# -b: batch mode, -u: update an existing installation, -p: prefix
bash "$installer" -b -u -p "$INSTALL_DIR"

# Ensure conda is available for the current shell.
export PATH="$INSTALL_DIR/bin:$PATH"

echo "+ $INSTALL_DIR/bin/conda --version"
"$INSTALL_DIR/bin/conda" --version

# Persist PATH in ~/.bashrc (idempotent)
bashrc="$HOME/.bashrc"
line="export PATH=$INSTALL_DIR/bin:\$PATH"
if [[ -f "$bashrc" ]] && grep -Fqx "$line" "$bashrc"; then
  echo "INFO: ~/.bashrc already contains Miniforge PATH entry."
else
  echo "# Miniforge" >> "$bashrc"
  echo "$line" >> "$bashrc"
  echo "INFO: Added Miniforge PATH entry to ~/.bashrc"
fi

echo "Done. Open a new shell or run: source ~/.bashrc"
