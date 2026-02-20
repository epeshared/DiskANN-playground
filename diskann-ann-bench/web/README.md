# diskann-ann-bench web

A standalone web UI for browsing **diskann-ann-bench split runner** outputs.

It reads benchmark runs under a `runs/` directory:

- `<runs_dir>/<dataset>/<run_id>/outputs/summary.tsv`
- `<runs_dir>/<dataset>/<run_id>/outputs/details.md` (optional)
- `<runs_dir>/<dataset>/<run_id>/outputs/output.build.json` (optional)
- `<runs_dir>/<dataset>/<run_id>/outputs/output.search.json` (optional)
- `<runs_dir>/<dataset>/<run_id>/mode.txt` (used for filtering)

Default filter:

- Only shows runs whose `mode.txt` is `ann_bench_diskann_rs`.

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
