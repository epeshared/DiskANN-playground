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

## Split build/search + loop (harness-style)

If you want an explicit **build stage** (build + save index once) and then a **search stage** that can loop the full query set (`--reps`), use:

What `run_diskann_rs_split.py` does:

- Creates a run folder under `DiskANN-playground/extend-rabitq/ann-harness/runs/<dataset>/<run_id>/` (unless `--work-dir` is provided)
- Writes `mode.txt = ann_bench_diskann_rs` (used by the web UI for filtering)
- Optional: writes `cpu-bind.txt` if you pass `--cpu-bind` (record-only; does not enforce affinity)
- Build stage:
  - reads `train` vectors from the HDF5
  - builds the index and saves it under `index/`
  - writes `outputs/output.build.json`
- Search stage:
  - loads the saved index
  - runs search over the full `test` query set
  - repeats the full query set `--reps` times (harness-style loop)
  - computes recall vs `neighbors` and latency/QPS stats
  - writes `outputs/output.search.json`
- Produces web-friendly summary artifacts:
  - `outputs/summary.tsv`
  - `outputs/details.md`

```bash
cd ../../DiskANN-playground/diskann-ann-bench

# Requires the Docker image to exist: ann-benchmarks-diskann_rs
python run_diskann_rs_split.py \
  --hdf5 ../../ann-benchmark-epeshared/data/glove-25-angular.hdf5 \
  --metric cosine \
  --l-build 128 \
  --max-outdegree 64 \
  --alpha 1.2 \
  -k 10 \
  --l-search 100 \
  --reps 3
```

If you don't have Docker build network access (pip/rustup/git clone fail), you can still run the split workflow on the host:

```bash
cd ../../ann-benchmark-epeshared/ann_benchmarks/algorithms/diskann_rs/native
cargo build

cd ../../DiskANN-playground/diskann-ann-bench
python run_diskann_rs_split.py --runner host \
  --hdf5 ../../tmp_sanity_small.hdf5 \
  --metric l2 -k 10 --l-search 50 --reps 2
```

The host runner auto-creates a `diskann_rs_native.so` symlink next to the cargo-built `libdiskann_rs_native.so` and sets `PYTHONPATH` so Python can import it.

## Quick local run + web

If you want a one-command local run (host runner + CPU binding recorded) and then start the local web UI:

```bash
bash DiskANN-playground/diskann-ann-bench/run_local.sh --hdf5 /path/to/dataset.hdf5
bash DiskANN-playground/diskann-ann-bench/run_web.sh --host 127.0.0.1 --port 8081
```

What `run_local.sh` does:

- Runs `cargo build` in `ann-benchmark-epeshared/ann_benchmarks/algorithms/diskann_rs/native`
- Chooses a CPU bind range (default `0-16`, clamped to `0-(nproc-1)` if fewer cores)
- Uses `numactl --physcpubind=...` (preferred) or `taskset -c ...` to **actually bind** the benchmark process
- Calls `run_diskann_rs_split.py --runner host` and passes `--cpu-bind` so the binding is written into `cpu-bind.txt`

What `run_web.sh` does:

- Starts the standalone web UI in `DiskANN-playground/diskann-ann-bench/web/`
- If `RUNS_DIR` is not set, it defaults to `DiskANN-playground/extend-rabitq/ann-harness/runs`
- The UI shows `cpu-bind.txt` and the derived core count on the dataset/run pages

### Remote mode (ssh + sync run folder back)

If your **index build/search must run on a remote machine** (e.g. server with more cores/DRAM) but you still want the results to show up in the existing web UI locally, use `--runner remote`.

Assumptions:

- The remote machine has the same repo layout (or you can point to it explicitly):
  - `<remote-workspace-root>/DiskANN-playground/`
  - `<remote-workspace-root>/ann-benchmark-epeshared/`
- The remote host can access the dataset path you provide (or use `--remote-copy-hdf5`).
- The remote host has `cargo` available if you use `--remote-build-native`.

Example:

```bash
cd ../../DiskANN-playground/diskann-ann-bench

python run_diskann_rs_split.py --runner remote \
  --hdf5 ../../tmp_sanity_small.hdf5 \
  --metric l2 -k 10 --l-search 50 --reps 2 \
  --remote-host myserver \
  --remote-user ubuntu \
  --ssh-opts "-i ~/.ssh/id_rsa -o StrictHostKeyChecking=no" \
  --remote-workspace-root /data/work/diskann-workspace \
  --remote-inner-runner host \
  --remote-build-native \
  --remote-copy-hdf5
```

What happens:

- Runs the split runner on the remote host (build + save index, then search).
- Pulls the entire remote run folder back into your local `extend-rabitq/ann-harness/runs/<dataset>/<run-id>/` so the web UI can browse it.

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
