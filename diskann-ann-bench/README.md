# diskann-ann-bench

This folder documents how to **build + run ann-benchmark-epeshared** with the Rust DiskANN integration (`diskann_rs`).

Repo layout assumption (matches this workspace):

- `DiskANN-playground/` (this repo)
- `ann-benchmark-epeshared/` (ann-benchmarks fork)
- `DiskANN-rs/` (Rust DiskANN workspace; must not be modified)

## What’s integrated

- ann-benchmark algorithm folder: `ann_benchmarks/algorithms/diskann_rs/`
- Python adapter: `DiskANNRS` in `module.py` (implements `BaseANN`)
- Native extension: `diskann_rs_native` (pyo3 + maturin)
- Docker image: `ann-benchmarks-diskann_rs`

Notes:

- In `install.py`, `--algorithm` refers to the **folder name** (here: `diskann_rs`).
- In `run.py`, `--algorithm` refers to the **algorithm "name"** from `config.yml` (e.g. `diskann-rs`).

## Prerequisites

- Docker
- Python 3.10+ (for running `install.py`/`run.py`)
- Internet access from inside Docker builds (or use `--network=host`)

## Step -1: install ann-benchmark Python deps (host)

`run.py` uses the Docker Python SDK (`import docker`), so you need the repo requirements installed on the host:

```bash
cd ../../ann-benchmark-epeshared
python -m pip install -r requirements.txt
```

## Step 0: (Recommended) vendor DiskANN-rs into ann-benchmark

This avoids cloning during the Docker build and keeps everything reproducible/offline-friendly.

```bash
cd ../../ann-benchmark-epeshared
bash ann_benchmarks/algorithms/diskann_rs/sync_diskann_rs.sh
```

This copies `../DiskANN-rs/` into:

- `ann_benchmarks/algorithms/diskann_rs/third_party/DiskANN-rs/`

Tip: this folder includes helper scripts:

- `./build_images.sh` (vendors DiskANN-rs + builds base + algo images with `--network=host`)

## Step 1: build Docker images

### Option A: build via the ann-benchmark installer (no `--network` control)

```bash
cd ../../ann-benchmark-epeshared
python install.py --algorithm diskann_rs
```

### Option B (recommended when pip networking is flaky): build manually with host networking

```bash
cd ../../ann-benchmark-epeshared

# Base image
docker build --network=host -t ann-benchmarks \
  -f ann_benchmarks/algorithms/base/Dockerfile .

# DiskANN-RS algorithm image
docker build --network=host -t ann-benchmarks-diskann_rs \
  -f ann_benchmarks/algorithms/diskann_rs/Dockerfile .
```

If your environment is fully offline:

- you still need the Python wheels used by the base image (see `requirements.txt`)
- consider pre-populating a wheelhouse and adjusting the base Dockerfile accordingly

## Step 2: full run (same integration, larger sweep)

```bash
cd ../../ann-benchmark-epeshared
python run.py --dataset glove-100-angular --algorithm diskann-rs -k 10 --runs 3 --timeout 7200 --parallelism 1
```

## Split build/search + loop (harness-style, framework-based)

If you want an explicit **build stage** (build + save index once) and then a **search stage** that can loop the full query set (`--reps`), use:

- `DiskANN-playground/diskann-ann-bench/src/framework_entry.py`

This harness runs **diskann-rs via the ann-benchmarks adapters** in:

- [ann-benchmark-epeshared/ann_benchmarks/algorithms/diskann_rs/module.py](../../ann-benchmark-epeshared/ann_benchmarks/algorithms/diskann_rs/module.py)

Stages:

- `--stage build`: `index_action=build_and_save`
- `--stage search`: `index_action=load`
- `--stage all`: build then search

Memory/RSS note:

- If you care about **search-stage peak RSS** (e.g. `peak_rss_gib`) being representative of **index + query** (not build-time allocations), prefer running build and search as **two separate processes**:
  - `--stage build` (process exits, memory released)
  - then `--stage search` (fresh process loads the index and runs queries)
- `--stage all` runs build then search in the same Python process. Even if the build objects are freed, the process RSS may stay high due to allocator behavior, which can make search RSS look larger than it “should”.
- For `--stage search` runs, this harness avoids loading the full training matrix into memory (only test + neighbors are loaded), so `peak_rss_gib` is closer to “index + query” memory.

Disk index note (PQ disk):

- A “disk index” still needs an explicit **load/open** step in the search process to initialize the index (metadata + file handles, possibly mmap). It does **not** reuse in-memory build state unless you keep the same process and the same index object alive.

Outputs:

- `outputs/output.build.json` (if stage includes build)
- `outputs/output.search.json` (if stage includes search)
- `outputs/summary.tsv`
- `outputs/details.md`

Example:

```bash
cd ../../DiskANN-playground/diskann-ann-bench

python3 src/framework_entry.py \
  --work-dir /tmp/runs/glove-25-angular/001 \
  --hdf5 ../../ann-benchmark-epeshared/data/glove-25-angular.hdf5 \
  --dataset glove-25-angular \
  --metric cosine \
  --stage all \
  --run-group diskann_rs_125_64_1-2 \
  -k 10 \
  --reps 3

```

If you want a cleaner search RSS (recommended when tracking `peak_rss_gib`), split it into two commands:

```bash
cd ../../DiskANN-playground/diskann-ann-bench

# 1) Build + save index
python3 src/framework_entry.py \
  --work-dir /tmp/runs/glove-25-angular/001 \
  --hdf5 ../../ann-benchmark-epeshared/data/glove-25-angular.hdf5 \
  --dataset glove-25-angular \
  --metric cosine \
  --stage build \
  --run-group diskann_rs_125_64_1-2 \
  -k 10

# 2) Fresh process: load index + search
python3 src/framework_entry.py \
  --work-dir /tmp/runs/glove-25-angular/001 \
  --hdf5 ../../ann-benchmark-epeshared/data/glove-25-angular.hdf5 \
  --dataset glove-25-angular \
  --metric cosine \
  --stage search \
  --run-group diskann_rs_125_64_1-2 \
  -k 10 \
  --reps 3
```

Notes:

- `--run-group` loads build/search parameters from:
  - `ann-benchmark-epeshared/ann_benchmarks/algorithms/diskann_rs/config.yml`
- You can still run without `--run-group` by specifying `--algo` and the build/search parameters explicitly.

### Quantized modes (PQ / spherical)

Supported algos:

- `--algo fp` → `DiskANNRS` (full precision)
- `--algo pq` → `DiskANNRS_PQ` (Product Quantization)
- `--algo spherical` → `DiskANNRS_Spherical` (spherical quantization / extended RaBitQ)

Examples:

```bash
# PQ (num_pq_chunks is provided via run-group)
python3 src/framework_entry.py \
  --work-dir /tmp/runs/tmp_sanity_small/001 \
  --hdf5 ../../tmp_sanity_small.hdf5 \
  --dataset tmp_sanity_small \
  --metric cosine \
  --stage all \
  --run-group diskann_rs_pq_125_64_1-2_14 \
  -k 10 --reps 2

# Spherical (nbits is provided via run-group)
python3 src/framework_entry.py \
  --work-dir /tmp/runs/tmp_sanity_small/002 \
  --hdf5 ../../tmp_sanity_small.hdf5 \
  --dataset tmp_sanity_small \
  --metric cosine \
  --stage all \
  --run-group diskann_rs_spherical_125_64_1-2_2b \
  -k 10 --reps 2
```

## Quick local run + web

If you want a one-command local run (host runner + CPU binding recorded) and then start the local web UI:

```bash
cd DiskANN-playground/diskann-ann-bench

# one-time: create a local job config (DO NOT commit it)
cp conf/job-conf.example.yml conf/job-conf.yml

# edit conf/job-conf.yml (set: hdf5, dataset, metric, etc.)

bash run_local.sh
bash DiskANN-playground/diskann-ann-bench/run_web.sh --host 127.0.0.1 --port 8081
```

If you want build/search in separate processes (recommended for cleaner search RSS):

```bash
# Edit conf/job-conf.yml:
#   - set stage="build" and run_id="<some_id>" then run: bash run_local.sh
#   - then set stage="search" and resume_runid="<same_id>" then run: bash run_local.sh

# (Config format note) The harness supports YAML/JSON/TOML.
# YAML is the default because it supports real comments.
```

CPU binding:

- Default is `cpu_bind: "0-16"` (clamped to available CPUs).
- Override: set `cpu_bind` in `conf/job-conf.yml`.
- Override: set `cpu_bind` in `conf/job-conf.yml`.

Batch query mode (uses ann-benchmarks `batch_query` path, passing all queries at once):

```bash
# By default, rayon uses all available logical CPU cores.
# You can control the number of threads using the RAYON_NUM_THREADS environment variable.
RAYON_NUM_THREADS=8 bash DiskANN-playground/diskann-ann-bench/run_local.sh
```

Enable batch mode by setting `batch: true` in `conf/job-conf.yml`.

## Remote run (SSH password login)

`run_remote.sh` can sync the workspace to a remote machine, run `run_local.sh` there, then sync `result/` back so the local web UI can browse it.

Configuration files (in `DiskANN-playground/diskann-ann-bench/`):

All configs live under `conf/`:

- `conf/remote-conf.yml`: remote host/user/port/remote_dir, plus optional `connect` mode.
- `conf/proxy-conf.yml`: optional per-remote proxy mapping (keyed by remote IP/host).
- `conf/password`: local file containing SSH password(s) (this file is gitignored).
- `conf/job-conf.yml`: job config including local `hdf5` and `remote_hdf5_dir`.

Notes:

- YAML is the default, but `run_local.sh`/`run_remote.sh` also accept `.json` / `.toml`.

Local tool requirements:

- `sshpass` (password auth)
- If using proxy: `ncat` (preferred) or `nc` (netcat)

Example:

```bash
cd DiskANN-playground/diskann-ann-bench

# one-time: create configs (DO NOT commit them)
cp conf/job-conf.example.yml conf/job-conf.yml
cp conf/remote-conf.example.yml conf/remote-conf.yml
cp conf/proxy-conf.example.yml conf/proxy-conf.yml   # optional

# one-time: create password file (DO NOT commit it)
printf '%s\n' '<ssh-password>' > conf/password

# edit conf/job-conf.yml:
#   - set hdf5 to a local path
#   - set remote_hdf5_dir to a remote directory (default: <remote_dir>/data)
#   - optionally set remote_setup=true (to run setup_remote.sh)

# run remotely (sync code + dataset, optional setup, run, then fetch result/)
./run_remote.sh
```

Run mode is controlled via `conf/job-conf.yml` (e.g., `compare`, `algo`, `name`, `run_group*`).

### Proxy support

`run_remote.sh` can connect via a SOCKS5 / HTTP proxy using SSH `ProxyCommand`.

1) In `conf/remote-conf.yml`, set `connect`:

- `auto` (default): use proxy if there is an entry for `host` in `conf/proxy-conf.yml`, otherwise direct SSH.
- `ssh`: always direct SSH (ignore proxy-conf).
- `socks` / `http`: force proxy (requires proxy-conf entry for this host).

Alias:

- `socks5`, `socket5`, and `sock5` are accepted as aliases for `socks`.

Example `conf/remote-conf.yml`:

```yml
host: 101.43.139.29
user: ubuntu
port: 22
remote_dir: ~/diskann-workspace
connect: auto
```

2) Create `conf/proxy-conf.yml` (keyed by the remote `host` IP/name):

```yml
101.43.139.29:
  type: socks
  host: 127.0.0.1
  port: 1080
```

3) Dry-run to verify parsing and planned commands:

```bash
# set remote_dry_run=true in conf/job-conf.yml, then:
./run_remote.sh
```

If proxy is enabled, the script will print the resolved `ProxyCommand` (using `ncat` if available, otherwise `nc`).

### Output layout (single run_id + cases)

`run_local.sh` writes results under:

- `DiskANN-playground/diskann-ann-bench/result/<dataset>/<run_id>/`

Within one run:

- `cases/<case_id>/...` contains per-case build/search outputs.
- `outputs/*` at the run root is an aggregated view so the web UI can browse a run as a single unit.

Example:

```
result/<dataset>/<run_id>/
  mode.txt
  cpu-bind.txt
  memory.txt
  batch.txt
  outputs/
    summary.tsv
    details.md
    output.build.json
    output.search.json
  cases/
    <case_id>/
      batch.txt
      outputs/
        summary.tsv
        details.md
        output.build.json
        output.search.json
```

Notes:

- `batch.txt` is written by `run_local.sh` (both at the run root and per-case). Values:
  - `1` → batch mode
  - `0` (or missing file) → single-query mode

To continue a partially-finished run (skips completed cases):

```bash
bash DiskANN-playground/diskann-ann-bench/run_local.sh \
  --resume-runid <run_id> \
  --hdf5 /path/to/dataset.hdf5 \
  --metric cosine \
  --stage all \
  --compare \
  --run-group-pq diskann_rs_pq_125_64_1-2_14 \
  --run-group-spherical diskann_rs_spherical_125_64_1-2_2b \
  -k 10 --reps 2
```

### Running PQ / spherical with run_local.sh

`run_local.sh` is a convenience wrapper around `src/framework_entry.py`.
It builds the native extension (via cargo), sets `PYTHONPATH` so the adapter can import it, and then runs the split build/search workflow.

Run modes:

- **Compare mode** (default): runs both PQ + spherical in one `run_id`.
- **PQ-only / single-algo mode**: add `--no-compare` and select one algorithm (`--algo`) plus one config name (`--name`) or one `--run-group`.

Examples:

```bash
# PQ-only (disk), run all matching run_groups under the name
bash DiskANN-playground/diskann-ann-bench/run_local.sh \
  --hdf5 /path/to/dataset.hdf5 \
  --metric angular \
  --no-compare \
  --algo pq \
  --run-all \
  --name diskann-rs-pq-disk \
  --batch

# Compare (PQ + spherical)
bash DiskANN-playground/diskann-ann-bench/run_local.sh \
  --hdf5 /path/to/dataset.hdf5 \
  --metric angular \
  --compare \
  --run-all \
  --name-pq diskann-rs-pq-disk \
  --name-spherical diskann-rs-spherical-memory \
  --batch
```

Full precision:

> Note: the next three `--run-group` examples are legacy-style single-case commands.
> With current `run_local.sh` defaults (`--compare` + `--run-all`), prefer the name-based mode examples below,
> or explicitly switch to single-algo mode (`--no-compare`) with a non-`run-all` setup.

```bash
bash DiskANN-playground/diskann-ann-bench/run_local.sh \
  --hdf5 /path/to/dataset.hdf5 \
  --metric cosine \
  --run-group diskann_rs_125_64_1-2 \
  -k 10 --reps 2
```

Drop-in replacement (current style):

```bash
bash DiskANN-playground/diskann-ann-bench/run_local.sh \
  --hdf5 /path/to/dataset.hdf5 \
  --metric cosine \
  --no-compare \
  --algo fp \
  --run-all \
  --name diskann-rs-memory \
  -k 10 --reps 2
```

PQ:

```bash
bash DiskANN-playground/diskann-ann-bench/run_local.sh \
  --hdf5 /path/to/dataset.hdf5 \
  --metric cosine \
  --run-group diskann_rs_pq_125_64_1-2_14 \
  -k 10 --reps 2
```

Drop-in replacement (current style):

```bash
bash DiskANN-playground/diskann-ann-bench/run_local.sh \
  --hdf5 /path/to/dataset.hdf5 \
  --metric cosine \
  --no-compare \
  --algo pq \
  --run-all \
  --name diskann-rs-pq-memory \
  -k 10 --reps 2
```

Spherical:

```bash
bash DiskANN-playground/diskann-ann-bench/run_local.sh \
  --hdf5 /path/to/dataset.hdf5 \
  --metric cosine \
  --run-group diskann_rs_spherical_125_64_1-2_2b \
  -k 10 --reps 2
```

Drop-in replacement (current style):

```bash
bash DiskANN-playground/diskann-ann-bench/run_local.sh \
  --hdf5 /path/to/dataset.hdf5 \
  --metric cosine \
  --no-compare \
  --algo spherical \
  --run-all \
  --name diskann-rs-spherical-memory \
  -k 10 --reps 2
```

Compare mode (runs PQ + spherical in a single `run_id` and writes them as separate cases under `cases/`):

```bash
bash DiskANN-playground/diskann-ann-bench/run_local.sh \
  --hdf5 /path/to/dataset.hdf5 \
  --metric cosine \
  --compare \
  --run-group-pq diskann_rs_pq_125_64_1-2_14 \
  --run-group-spherical diskann_rs_spherical_125_64_1-2_2b \
  -k 10 --reps 2
```

#### Using diskann_rs/config.yml presets (recommended)

If you want to avoid specifying build/search parameters on the command line, you can pull them from:

- `ann-benchmark-epeshared/ann_benchmarks/algorithms/diskann_rs/config.yml`

Single run-group:

> Note: this is also a legacy single-case `--run-group` pattern.
> For current defaults, use the `--run-all --name ...` pattern shown below.

```bash
bash DiskANN-playground/diskann-ann-bench/run_local.sh \
  --hdf5 /path/to/dataset.hdf5 \
  --metric cosine \
  --run-group diskann_rs_125_64_1-2 \
  -k 10 --reps 2
```

Drop-in replacement (current style):

```bash
bash DiskANN-playground/diskann-ann-bench/run_local.sh \
  --hdf5 /path/to/dataset.hdf5 \
  --metric cosine \
  --no-compare \
  --algo fp \
  --run-all \
  --name diskann-rs-memory \
  -k 10 --reps 2
```

Compare with presets:

```bash
bash DiskANN-playground/diskann-ann-bench/run_local.sh \
  --hdf5 /path/to/dataset.hdf5 \
  --metric cosine \
  --compare \
  --run-group-pq diskann_rs_pq_125_64_1-2_14 \
  --run-group-spherical diskann_rs_spherical_125_64_1-2_2b \
  -k 10 --reps 2
```

To see all available `run_groups`, open:

- `ann-benchmark-epeshared/ann_benchmarks/algorithms/diskann_rs/config.yml`

#### Run all run_groups under a config `name:`

Sometimes you want to sweep *every* `run_group` under a specific config entry (selected by `name:`), without listing run-group keys yourself.

Single algorithm name sweep:

```bash
bash DiskANN-playground/diskann-ann-bench/run_local.sh \
  --hdf5 /path/to/dataset.hdf5 \
  --metric cosine \
  --run-all \
  --name diskann-rs-pq-memory \
  -k 10 --reps 2
```

Compare sweep (PQ name + spherical name):

```bash
bash DiskANN-playground/diskann-ann-bench/run_local.sh \
  --hdf5 /path/to/dataset.hdf5 \
  --metric cosine \
  --compare \
  --run-all \
  --name-pq diskann-rs-pq-memory \
  --name-spherical diskann-rs-spherical-memory \
  -k 10 --reps 2
```

What `run_local.sh` does:

- Runs `cargo build` in `ann-benchmark-epeshared/ann_benchmarks/algorithms/diskann_rs/native`
- Chooses a CPU bind range (default `0-16`, clamped to `0-(nproc-1)` if fewer cores)
- Uses `numactl --physcpubind=...` (preferred) or `taskset -c ...` to **actually bind** the benchmark process
- Calls `src/framework_entry.py` and writes `cpu-bind.txt` so the binding is shown in the web UI

Notes:

- By default it builds the native extension in **release** mode. Override with `DISKANN_RS_NATIVE_PROFILE=debug`.
- This is “ann-benchmarks framework” in the sense that the adapters are the same `BaseANN` implementations, but the workflow is the split harness (build/save once, then search loop), not `python run.py`.

What `run_web.sh` does:

- Starts the standalone web UI in `DiskANN-playground/diskann-ann-bench/web/`
- If `RUNS_DIR` is not set, it defaults to `DiskANN-playground/diskann-ann-bench/result`
- The UI shows `cpu-bind.txt` and the derived core count on the dataset/run pages

### Remote mode (ssh + sync run folder back)

If your **index build/search must run on a remote machine** (e.g. server with more cores/DRAM) but you still want the results to show up in the existing web UI locally, use `src/run_remote.py`.

Example:

> Note: this forwards a single `--run-group` case.
> If your remote `run_local.sh` uses default compare/sweep mode, prefer forwarding `--no-compare` + name-based args.

```bash
cd ../../DiskANN-playground/diskann-ann-bench

python3 src/run_remote.py \
  --remote-host myserver \
  --remote-user ubuntu \
  --ssh-opts "-i ~/.ssh/id_rsa -o StrictHostKeyChecking=no" \
  --remote-workspace-root /data/work/diskann-workspace \
  --remote-copy-hdf5 \
  -- \
  --hdf5 /path/to/dataset.hdf5 \
  --metric l2 \
  --stage all \
  --run-group diskann_rs_100_64_1-2 \
  -k 10 --reps 2
```

Drop-in replacement (current style):

```bash
cd ../../DiskANN-playground/diskann-ann-bench

python3 src/run_remote.py \
  --remote-host myserver \
  --remote-user ubuntu \
  --ssh-opts "-i ~/.ssh/id_rsa -o StrictHostKeyChecking=no" \
  --remote-workspace-root /data/work/diskann-workspace \
  --remote-copy-hdf5 \
  -- \
  --hdf5 /path/to/dataset.hdf5 \
  --metric l2 \
  --stage all \
  --no-compare \
  --algo fp \
  --run-all \
  --name diskann-rs-memory \
  -k 10 --reps 2
```

What happens:

- Runs `run_local.sh` on the remote host via ssh.
- `rsync`s the resulting run folder back into your local `diskann-ann-bench/result/<dataset>/`.

### Docker mode

To run the same split build/search workflow inside Docker:

> Note: this is a legacy single-case `--run-group` example.
> For current defaults, prefer name-based sweep mode (or explicitly switch to single-algo mode).

```bash
bash DiskANN-playground/diskann-ann-bench/run_docker.sh \
  --hdf5 /path/to/dataset.hdf5 \
  --metric cosine \
  --stage all \
  --run-group diskann_rs_125_64_1-2 \
  -k 10 --reps 3
```

Drop-in replacement (current style):

```bash
bash DiskANN-playground/diskann-ann-bench/run_docker.sh \
  --hdf5 /path/to/dataset.hdf5 \
  --metric cosine \
  --stage all \
  --no-compare \
  --algo fp \
  --run-all \
  --name diskann-rs-memory \
  -k 10 --reps 3
```

## Useful commands

List available algorithms:

```bash
cd ../../ann-benchmark-epeshared
python run.py --list-algorithms
```

Run only algorithms belonging to a docker image tag:

```bash
python run.py --dataset glove-25-angular --docker-tag ann-benchmarks-diskann_rs
```

## Troubleshooting

### Docker build fails with `Network is unreachable` while pip installing

Use manual builds with host networking:

- `docker build --network=host ...`

### The runner says the docker image is missing

Make sure the image exists locally:

```bash
docker images | grep ann-benchmarks
```

### DiskANN-rs sources missing inside the algorithm build

Either:

- run the sync step (`sync_diskann_rs.sh`), or
- allow the algorithm Dockerfile to `git clone` the upstream repo (requires network)
