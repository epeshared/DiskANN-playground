# remote-test

Password-based SSH remote test runner.

What it does:
1. Reads a JSON config with remote `host/username/password` and remote paths.
2. Uploads local sources (excluding `.gitignore`d files) to the remote machine.
   - Uploads `DiskANN-playground/`.
   - Uploads `DiskANN-rs/` too (optional, but required for `diskann-rs/build_all_targets.sh` by default).
3. Ensures the remote C++ DiskANN repo exists at the configured directory (clones if missing):
   - https://github.com/microsoft/DiskANN.git
4. Runs:
   - `DiskANN-playground/diskann-rs/build_all_targets.sh` (compiles DiskANN-rs)
   - `DiskANN-playground/extend-rabitq/ann-harness/scripts/run_dataset.sh` (runs the benchmark)
5. Downloads the newest run output directory back into local:
   - `DiskANN-playground/extend-rabitq/ann-harness/runs/`

## Prereqs

Local:
- Python 3
- `git`
- `pip install paramiko`

Remote:
- `bash`, `git`, `tar`
- The environment expected by `DiskANN-playground/diskann-rs/build_all_targets.sh` (notably `conda` + env `diskann-rs`).
- The dataset file referenced by the benchmark must exist on the remote machine.

If you don't want to pre-place the dataset on the remote machine, use `data_copy` (see below).

## Remote prerequirements (自动安装依赖)

`remote-test/pre-requirements/` contains optional scripts that can be executed on the remote machine.

In config:
- `pre_requirements.enable`: whether to run prereq scripts after upload/extract
- `pre_requirements.scripts`: list of script paths relative to `DiskANN-playground/`

Example:
- `"scripts": ["remote-test/pre-requirements/ubuntu_apt.sh", "remote-test/pre-requirements/check_env.sh"]`

## Conda environment (远端 conda 环境)

You can configure an optional conda environment. If configured, the runner will:
1. Check `conda` exists on remote
2. If `conda env` is missing and `create_if_missing=true`, run `create_command`
3. For subsequent commands, run `conda activate <env_name>` before executing build/test

Config fields:
- `conda.env_name`: environment name (set to null to disable conda integration)
- `conda.auto_activate`: whether to `conda activate` for commands (default true)
- `conda.create_if_missing`: whether to create env when missing
- `conda.create_command`: command to create env (executed under `paths.remote_workspace_root`)

## Usage

1. Copy and edit config:

- Start from `remote-test/config.example.json`

If you do not want to store the SSH password in JSON, set `remote.password` to `null` (or omit it) and the script will prompt interactively.

If the HDF5 dataset path in `extend-rabitq/ann-harness/scripts/run_dataset.sh` does not exist on the remote machine, set `paths.remote_hdf5_path` in the config to a valid remote path. The runner will pass it via the `HDF5_PATH` env var.

## Remote results dir (远端 results 独立目录)

If you want to delete/clean the remote playground directory without losing benchmark results, set:
- `paths.remote_results_dir`: e.g. `/home/xtang/diskann-results`

When configured, the runner will:
- Create that directory on the remote
- Replace `.../extend-rabitq/ann-harness/runs` with a symlink pointing to `remote_results_dir`

So results persist even if `options.clean_remote_playground=true`.

## Data copy (远端没有就拷贝数据)

Use `data_copy` to upload the local HDF5 file to the remote machine when it is missing.

Config fields:
- `data_copy.enable`: enable/disable
- `data_copy.if_missing_only`: if true (default), only upload when remote file is missing
- `data_copy.local_hdf5_path`: local absolute path to the `.hdf5`
- `data_copy.remote_hdf5_path`: remote absolute path to store the `.hdf5` (also used as `HDF5_PATH`)

Notes:
- If you already set `paths.remote_hdf5_path`, you can omit `data_copy.remote_hdf5_path`.

2. Dry run:

- `python3 remote-test/remote_test.py --config remote-test/config.json --dry-run`

3. Run:

- `python3 remote-test/remote_test.py --config remote-test/config.json`

## CPU binding (绑核)

Configure `cpu_bind` in the JSON config. This is implemented as a command prefix on the remote side.

Examples:

- Bind test to cores 0-31 using taskset:
   - `"cpu_bind": {"apply_to": ["test"], "taskset_cpus": "0-31"}`

- Bind test to NUMA node 0 cores + memory:
   - `"cpu_bind": {"apply_to": ["test"], "numactl_physcpubind": "0-31", "numactl_membind": "0"}`

If you want binding to apply to both build and test:
- `"apply_to": ["build", "test"]`

## Notes

- By default `allow_unknown_host_keys=true` (convenient, but less strict). Set to `false` to require known_hosts.
- If the runner can't infer the dataset name from `run_dataset.sh` (either `DATASET=...` default or `--dataset ...`), set `paths.dataset_name` in the config.
