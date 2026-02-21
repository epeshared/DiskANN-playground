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

- `DiskANN-playground/diskann-ann-bench/framework_entry.py`

This harness runs **diskann-rs via the ann-benchmarks adapters** in:

- [ann-benchmark-epeshared/ann_benchmarks/algorithms/diskann_rs/module.py](../../ann-benchmark-epeshared/ann_benchmarks/algorithms/diskann_rs/module.py)

Stages:

- `--stage build`: `index_action=build_and_save`
- `--stage search`: `index_action=load`
- `--stage all`: build then search

Outputs:

- `outputs/output.build.json` (if stage includes build)
- `outputs/output.search.json` (if stage includes search)
- `outputs/summary.tsv`
- `outputs/details.md`

Example:

```bash
cd ../../DiskANN-playground/diskann-ann-bench

python3 framework_entry.py \
  --work-dir /tmp/runs/glove-25-angular/001 \
  --hdf5 ../../ann-benchmark-epeshared/data/glove-25-angular.hdf5 \
  --dataset glove-25-angular \
  --metric cosine \
  --stage all \
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
python3 framework_entry.py \
  --work-dir /tmp/runs/tmp_sanity_small/001 \
  --hdf5 ../../tmp_sanity_small.hdf5 \
  --dataset tmp_sanity_small \
  --metric cosine \
  --stage all \
  --run-group diskann_rs_pq_125_64_1-2_14 \
  -k 10 --reps 2

# Spherical (nbits is provided via run-group)
python3 framework_entry.py \
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
bash DiskANN-playground/diskann-ann-bench/run_local.sh --hdf5 /path/to/dataset.hdf5
bash DiskANN-playground/diskann-ann-bench/run_web.sh --host 127.0.0.1 --port 8081
```

Batch query mode (uses ann-benchmarks `batch_query` path, passing all queries at once):

```bash
# By default, rayon uses all available logical CPU cores.
# You can control the number of threads using the RAYON_NUM_THREADS environment variable.
RAYON_NUM_THREADS=8 bash DiskANN-playground/diskann-ann-bench/run_local.sh --batch --hdf5 /path/to/dataset.hdf5
```

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

`run_local.sh` is a convenience wrapper around `framework_entry.py`.
It builds the native extension (via cargo), sets `PYTHONPATH` so the adapter can import it, and then runs the split build/search workflow.

Full precision:

```bash
bash DiskANN-playground/diskann-ann-bench/run_local.sh \
  --hdf5 /path/to/dataset.hdf5 \
  --metric cosine \
  --run-group diskann_rs_125_64_1-2 \
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

Spherical:

```bash
bash DiskANN-playground/diskann-ann-bench/run_local.sh \
  --hdf5 /path/to/dataset.hdf5 \
  --metric cosine \
  --run-group diskann_rs_spherical_125_64_1-2_2b \
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

```bash
bash DiskANN-playground/diskann-ann-bench/run_local.sh \
  --hdf5 /path/to/dataset.hdf5 \
  --metric cosine \
  --run-group diskann_rs_125_64_1-2 \
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
  --name diskann-rs-pq \
  -k 10 --reps 2
```

Compare sweep (PQ name + spherical name):

```bash
bash DiskANN-playground/diskann-ann-bench/run_local.sh \
  --hdf5 /path/to/dataset.hdf5 \
  --metric cosine \
  --compare \
  --run-all \
  --name-pq diskann-rs-pq \
  --name-spherical diskann-rs-spherical \
  -k 10 --reps 2
```

What `run_local.sh` does:

- Runs `cargo build` in `ann-benchmark-epeshared/ann_benchmarks/algorithms/diskann_rs/native`
- Chooses a CPU bind range (default `0-16`, clamped to `0-(nproc-1)` if fewer cores)
- Uses `numactl --physcpubind=...` (preferred) or `taskset -c ...` to **actually bind** the benchmark process
- Calls `framework_entry.py` and writes `cpu-bind.txt` so the binding is shown in the web UI

Notes:

- By default it builds the native extension in **release** mode. Override with `DISKANN_RS_NATIVE_PROFILE=debug`.
- This is “ann-benchmarks framework” in the sense that the adapters are the same `BaseANN` implementations, but the workflow is the split harness (build/save once, then search loop), not `python run.py`.

What `run_web.sh` does:

- Starts the standalone web UI in `DiskANN-playground/diskann-ann-bench/web/`
- If `RUNS_DIR` is not set, it defaults to `DiskANN-playground/diskann-ann-bench/result`
- The UI shows `cpu-bind.txt` and the derived core count on the dataset/run pages

### Remote mode (ssh + sync run folder back)

If your **index build/search must run on a remote machine** (e.g. server with more cores/DRAM) but you still want the results to show up in the existing web UI locally, use `run_remote.py`.

Example:

```bash
cd ../../DiskANN-playground/diskann-ann-bench

python3 run_remote.py \
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

What happens:

- Runs `run_local.sh` on the remote host via ssh.
- `rsync`s the resulting run folder back into your local `diskann-ann-bench/result/<dataset>/`.

### Docker mode

To run the same split build/search workflow inside Docker:

```bash
bash DiskANN-playground/diskann-ann-bench/run_docker.sh \
  --hdf5 /path/to/dataset.hdf5 \
  --metric cosine \
  --stage all \
  --run-group diskann_rs_125_64_1-2 \
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
