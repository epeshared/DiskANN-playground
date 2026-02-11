# diskann-ann-bench web

A standalone web UI for browsing **diskann-ann-bench split runner** outputs.

It reads benchmark runs under a `runs/` directory in the same format as `extend-rabitq/ann-harness`:

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

Default (reads `DiskANN-playground/extend-rabitq/ann-harness/runs`):

```bash
bash DiskANN-playground/diskann-ann-bench/web/run_web.sh --host 127.0.0.1 --port 8081
```

Custom runs dir:

```bash
RUNS_DIR=/path/to/DiskANN-playground/extend-rabitq/ann-harness/runs \
  bash DiskANN-playground/diskann-ann-bench/web/run_web.sh --host 0.0.0.0 --port 8081
```

Open:

- `http://<host>:<port>/`

## Tips

- If you used `run_diskann_rs_split.py` (local or remote), it already writes the run folder into the harness runs tree by default, so it should appear automatically.
