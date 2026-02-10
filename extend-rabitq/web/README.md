# extend-rabitq web

A small web server to browse benchmark runs under:
- `DiskANN-playground/extend-rabitq/ann-harness/runs/`

## Install

Recommended (use the existing conda env `diskann-rs`):

```bash
conda install -n diskann-rs -y -c conda-forge fastapi uvicorn jinja2 markdown-it-py
```

Alternative (pip, but still inside your chosen Python env):

```bash
python3 -m pip install -r DiskANN-playground/extend-rabitq/web/requirements.txt
```

## Run

Default (bind to localhost):

```bash
bash DiskANN-playground/extend-rabitq/web/run_web.sh --host 127.0.0.1 --port 8080
```

Bind to all interfaces:

```bash
bash DiskANN-playground/extend-rabitq/web/run_web.sh --host 0.0.0.0 --port 8080
```

Open:
- `http://<host>:<port>/`

## Configuration

All settings can be provided via CLI args or environment variables:

- `--host` / `WEB_HOST` (default: `127.0.0.1`)
- `--port` / `WEB_PORT` (default: `8080`)
- `--runs-dir` / `RUNS_DIR` (default: `../ann-harness/runs` relative to this file)

Example:

```bash
export WEB_HOST=0.0.0.0
export WEB_PORT=8080
export RUNS_DIR=/path/to/DiskANN-playground/extend-rabitq/ann-harness/runs
python3 DiskANN-playground/extend-rabitq/web/app.py
```

## What you can browse

- Dataset list
- Per-dataset run list (timestamp directories)
- Run detail page:
  - `outputs/summary.tsv` rendered as a table
  - `outputs/details.md` rendered as HTML (if present)
  - download links for common artifacts (json/tsv/md)
