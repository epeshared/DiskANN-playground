# pre-requirements

Scripts in this folder are meant to be executed on the **remote** machine to install/check prerequisites.

They are optional and controlled by `pre_requirements` in `remote-test/config.json`.

## Scripts

- `ubuntu_apt.sh`
  - Installs common packages via `apt-get` (git, tar, numactl/taskset tools, build essentials, python3, pip, cmake, clang).

- `install_miniforge.sh`
  - Installs Miniforge (conda) to `$HOME/miniforge3` (override with `INSTALL_DIR=/path`).
  - Adds `export PATH=<install>/bin:$PATH` to `~/.bashrc` (idempotent).
  - Requires `wget` or `curl`.

- `install_rustup.sh`
  - Installs Rust toolchain via rustup (adds `cargo`/`rustc`).
  - Adds a line to `~/.profile` to source `~/.cargo/env` for login shells.

- `check_env.sh`
  - Prints versions/availability of common tools (git, tar, conda, numactl, taskset).

## Notes

- These scripts intentionally avoid installing CUDA or other heavyweight components.
- If you need conda on remote and it is not present, install it first (manually or extend these scripts for your environment).
- The runner executes scripts as `bash <script>`, so executable bit is not required.
