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
- In `run.py`, `--algorithm` refers to the **algorithm "name"** from `config.yml` (e.g. `diskann-rs` or `diskann-rs-smoke`).

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
- `./run_smoke.sh` (runs a quick `diskann-rs-smoke` benchmark)

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

## Step 2: smoke test (small-ish)

The `diskann_rs/config.yml` defines a small `diskann-rs-smoke` algorithm entry.

```bash
cd ../../ann-benchmark-epeshared

# Quick-ish run: one algorithm, one run, small k
python run.py \
  --dataset glove-25-angular \
  --algorithm diskann-rs-smoke \
  -k 10 \
  --runs 1 \
  --timeout 900 \
  --parallelism 1
```

What this does:

- downloads the dataset into `ann-benchmark-epeshared/data/` if missing
- runs the algorithm inside Docker and writes JSON results under `ann-benchmark-epeshared/results/`

## Step 3: full run (same integration, larger sweep)

```bash
cd ../../ann-benchmark-epeshared
python run.py --dataset glove-100-angular --algorithm diskann-rs -k 10 --runs 3 --timeout 7200 --parallelism 1
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
