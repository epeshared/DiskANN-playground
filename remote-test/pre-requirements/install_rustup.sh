#!/usr/bin/env bash
set -euo pipefail

# Installs Rust toolchain via rustup (provides cargo/rustc).
# Safe to re-run.

if command -v cargo >/dev/null 2>&1; then
  echo "INFO: cargo already on PATH: $(command -v cargo)"
  cargo --version || true
  exit 0
fi

if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
  echo "ERROR: Need curl or wget to download rustup-init." >&2
  exit 1
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

if command -v curl >/dev/null 2>&1; then
  echo "+ curl https://sh.rustup.rs"
  curl -fsSL https://sh.rustup.rs -o "$tmp_dir/rustup-init.sh"
else
  echo "+ wget https://sh.rustup.rs"
  wget -q -O "$tmp_dir/rustup-init.sh" https://sh.rustup.rs
fi

chmod +x "$tmp_dir/rustup-init.sh"

echo "+ sh rustup-init.sh -y"
"$tmp_dir/rustup-init.sh" -y

# Load cargo for current shell
if [[ -f "$HOME/.cargo/env" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "ERROR: cargo still not found after rustup install." >&2
  exit 1
fi

cargo --version
rustc --version || true

# Ensure login shells can see cargo.
profile="$HOME/.profile"
line='if [ -f "$HOME/.cargo/env" ]; then . "$HOME/.cargo/env"; fi'
if [[ -f "$profile" ]] && grep -Fqx "$line" "$profile"; then
  echo "INFO: ~/.profile already sources ~/.cargo/env"
else
  echo "# Rust" >> "$profile"
  echo "$line" >> "$profile"
  echo "INFO: Added rustup env sourcing to ~/.profile"
fi

echo "Done. Open a new shell or run: source ~/.cargo/env"
