# diskann-ann-bench web

A standalone web UI for browsing **diskann-ann-bench split runner** outputs.

It reads benchmark runs under a `runs/` directory:

- `<runs_dir>/<dataset>/<run_id>/outputs/summary.tsv`
- `<runs_dir>/<dataset>/<run_id>/outputs/details.md` (optional)
- `<runs_dir>/<dataset>/<run_id>/outputs/output.build.json` (optional)
- `<runs_dir>/<dataset>/<run_id>/outputs/output.search.json` (optional)
- `<runs_dir>/<dataset>/<run_id>/mode.txt` (used for filtering)
- `<runs_dir>/<dataset>/<run_id>/cpu-bind.txt` (used for filtering/display)
- `<runs_dir>/<dataset>/<run_id>/lscpu.txt` (optional; used for CPU model/info display)
- `<runs_dir>/<dataset>/<run_id>/batch.txt` (optional; used for query mode display/filter)

Default filter:

- Only shows runs whose `mode.txt` is `ann_bench_diskann_rs`.

Query mode:

- If `batch.txt` is truthy (`1`, `true`, `yes`, `batch`), the UI shows `query: batch`.
- If missing or falsey (`0`, `false`, `no`), the UI shows `query: single`.
- In batch mode, the number of threads used by the Rust native extension can be controlled via the `RAYON_NUM_THREADS` environment variable.

Compare exports:

- Compare `.xlsx` includes `query mode` in the `server-info` sheet.
- Compare `.csv` includes `a_query_mode` / `b_query_mode` columns.

## Install

Use your existing Python env:

```bash
python3 -m pip install -r DiskANN-playground/diskann-ann-bench/web/requirements.txt
```

## Run

Default (reads `DiskANN-playground/diskann-ann-bench/result`):

```bash
bash DiskANN-playground/diskann-ann-bench/web/run_web.sh --host 127.0.0.1 --port 8081
```

Custom runs dir:

```bash
RUNS_DIR=/path/to/runs_dir \
  bash DiskANN-playground/diskann-ann-bench/web/run_web.sh --host 0.0.0.0 --port 8081
```

Open:

- `http://<host>:<port>/`

## Tips

- If you used `run_local.sh` or `run_remote.py`, they write runs under `DiskANN-playground/diskann-ann-bench/result` by default, so they should appear automatically.
- In compare/sweep runs, a single `run_id` can contain multiple cases under `cases/`, but the run root still has `outputs/*` as an aggregated view for browsing.
