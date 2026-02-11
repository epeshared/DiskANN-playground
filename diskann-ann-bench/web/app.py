#!/usr/bin/env python3

from __future__ import annotations

import argparse
import html
import json
import os
import re
import math
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.requests import Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markdown_it import MarkdownIt


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    runs_dir: Path


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v is not None and v.strip() else default


def _safe_name(s: str) -> str:
    # Keep URLs simple and prevent traversal. Dataset and run_id are directory names.
    if not re.fullmatch(r"[A-Za-z0-9._-]+", s or ""):
        raise HTTPException(status_code=400, detail=f"Invalid path component: {s!r}")
    return s


def _safe_join(base: Path, *parts: str) -> Path:
    p = base
    for part in parts:
        p = p / _safe_name(part)
    rp = p.resolve()
    rb = base.resolve()
    if rb != rp and rb not in rp.parents:
        raise HTTPException(status_code=400, detail="Path traversal blocked")
    return rp


def _list_dirs(path: Path) -> list[str]:
    if not path.exists():
        return []
    out: list[str] = []
    for child in path.iterdir():
        if child.is_dir():
            out.append(child.name)
    out.sort()
    return out


def _list_dirs_sorted_desc(path: Path) -> list[str]:
    xs = _list_dirs(path)
    xs.sort(reverse=True)
    return xs


def _read_text_if_exists(path: Path, *, max_bytes: int = 2_000_000) -> str | None:
    try:
        if not path.is_file():
            return None
        if path.stat().st_size > max_bytes:
            return path.read_text(encoding="utf-8", errors="replace")[:max_bytes] + "\n<truncated>\n"
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def _read_tsv(path: Path) -> tuple[list[str], list[list[str]]]:
    if not path.is_file():
        return ([], [])
    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not raw:
        return ([], [])
    headers = raw[0].split("\t")
    rows = [line.split("\t") for line in raw[1:] if line.strip()]
    return (headers, rows)


def _parse_cpu_bind_to_set(cpu_bind: str) -> set[int]:
    s = (cpu_bind or '').strip()
    if not s:
        return set()

    # Accept:
    # - "0-16"
    # - "0,1,2,3"
    # - "0 1 2"
    # - "physcpubind: 0 1 2"
    s = s.replace('physcpubind:', ' ')
    toks = re.split(r"[\s,]+", s)
    out: set[int] = set()
    for t in toks:
        t = t.strip()
        if not t:
            continue
        if '-' in t:
            a, b = t.split('-', 1)
            if a.strip().isdigit() and b.strip().isdigit():
                lo = int(a)
                hi = int(b)
                if lo > hi:
                    lo, hi = hi, lo
                for x in range(lo, hi + 1):
                    out.add(x)
                continue
        if t.isdigit():
            out.add(int(t))
    return out


def _breadcrumbs(*items: tuple[str, str]) -> str:
    parts: list[str] = []
    for label, href in items:
        parts.append(f'<a href="{html.escape(href, quote=True)}">{html.escape(label)}</a>')
    return " / ".join(parts)


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title="diskann-ann-bench web")

    templates_dir = Path(__file__).resolve().parent / "templates"
    templates = Jinja2Templates(directory=str(templates_dir))
    templates.env.filters["urlencode"] = lambda s: urllib.parse.quote(str(s), safe="")

    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    md = MarkdownIt("commonmark")

    default_mode = "ann_bench_diskann_rs"

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "runs_dir": str(settings.runs_dir)}

    @app.get("/api/datasets")
    def api_datasets() -> JSONResponse:
        return JSONResponse(_list_dirs(settings.runs_dir))

    @app.get("/api/runs/{dataset}")
    def api_runs(dataset: str) -> JSONResponse:
        dataset_dir = _safe_join(settings.runs_dir, dataset)
        return JSONResponse(_list_dirs_sorted_desc(dataset_dir))

    @app.get("/api/file/{dataset}/{run_id}/{rel_path:path}")
    def api_file(dataset: str, run_id: str, rel_path: str):
        # Lock downloads to files under the run directory.
        run_dir = _safe_join(settings.runs_dir, dataset, run_id)
        # Disallow weird components in rel_path.
        rel_path = rel_path.lstrip("/")
        if any(x in rel_path for x in ["..", "\\", "//"]):
            raise HTTPException(status_code=400, detail="Invalid rel_path")
        path = (run_dir / rel_path).resolve()
        if run_dir != path and run_dir not in path.parents:
            raise HTTPException(status_code=400, detail="Path traversal blocked")
        if not path.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(str(path))

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        datasets = _list_dirs(settings.runs_dir)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "title": "diskann-ann-bench",
                "datasets": datasets,
                "runs_dir": str(settings.runs_dir),
                "breadcrumbs": None,
            },
        )

    @app.get("/dataset/{dataset}", response_class=HTMLResponse)
    def dataset_page(dataset: str, request: Request):
        dataset_dir = _safe_join(settings.runs_dir, dataset)
        run_ids = _list_dirs_sorted_desc(dataset_dir)

        q = (request.query_params.get("q") or "").strip().lower() or None
        mode = request.query_params.get("mode")
        mode = (mode.strip() if isinstance(mode, str) else "")
        mode_norm = (mode.lower().strip() if mode else "") or default_mode

        runs = []
        for run_id in run_ids:
            run_dir = dataset_dir / run_id
            mode_txt = _read_text_if_exists(run_dir / "mode.txt", max_bytes=2000)
            run_mode = (mode_txt or "").strip() or None

            cpu_bind_txt = _read_text_if_exists(run_dir / "cpu-bind.txt", max_bytes=5000)
            cpu_bind = (cpu_bind_txt or '').strip() or None
            cpu_cores = len(_parse_cpu_bind_to_set(cpu_bind or '')) if cpu_bind else None

            if mode_norm and (run_mode or "").strip().lower() != mode_norm:
                continue

            if q:
                hay = " ".join([run_id, run_mode or ""]).lower()
                if q not in hay:
                    continue

            runs.append(
                {
                    "id": run_id,
                    "mode": run_mode,
                    "cpu_bind": cpu_bind,
                    "cpu_cores": cpu_cores,
                    "has_summary": (run_dir / "outputs" / "summary.tsv").is_file(),
                    "has_details": (run_dir / "outputs" / "details.md").is_file(),
                    "has_build": (run_dir / "outputs" / "output.build.json").is_file(),
                    "has_search": (run_dir / "outputs" / "output.search.json").is_file(),
                }
            )

        return templates.TemplateResponse(
            "dataset.html",
            {
                "request": request,
                "title": f"{dataset}",
                "dataset": dataset,
                "runs": runs,
                "q": q or "",
                "mode": mode,
                "default_mode": default_mode,
                "runs_dir": str(settings.runs_dir),
                "breadcrumbs": _breadcrumbs(("datasets", "/"), (dataset, f"/dataset/{urllib.parse.quote(dataset, safe='')}")),
            },
        )

    @app.get("/run/{dataset}/{run_id}", response_class=HTMLResponse)
    def run_page(dataset: str, run_id: str, request: Request):
        run_dir = _safe_join(settings.runs_dir, dataset, run_id)

        mode_txt = _read_text_if_exists(run_dir / "mode.txt", max_bytes=2000)
        run_mode = (mode_txt or "").strip() or None

        cpu_bind_txt = _read_text_if_exists(run_dir / "cpu-bind.txt", max_bytes=5000)
        cpu_bind = (cpu_bind_txt or '').strip() or None
        cpu_cores = len(_parse_cpu_bind_to_set(cpu_bind or '')) if cpu_bind else None

        headers, rows = _read_tsv(run_dir / "outputs" / "summary.tsv")
        details_md = _read_text_if_exists(run_dir / "outputs" / "details.md")
        details_html = md.render(details_md) if details_md else None

        build_json_obj = None
        search_json_obj = None

        build_raw = _read_text_if_exists(run_dir / "outputs" / "output.build.json")
        if build_raw:
            try:
                build_json_obj = json.dumps(json.loads(build_raw), indent=2)
            except Exception:
                build_json_obj = build_raw

        search_raw = _read_text_if_exists(run_dir / "outputs" / "output.search.json")
        if search_raw:
            try:
                search_json_obj = json.dumps(json.loads(search_raw), indent=2)
            except Exception:
                search_json_obj = search_raw

        return templates.TemplateResponse(
            "run.html",
            {
                "request": request,
                "title": f"{dataset}/{run_id}",
                "dataset": dataset,
                "run_id": run_id,
                "mode": run_mode,
                "cpu_bind": cpu_bind,
                "cpu_cores": cpu_cores,
                "summary": {"headers": headers, "rows": rows},
                "details_html": details_html,
                "build_json": build_json_obj,
                "search_json": search_json_obj,
                "runs_dir": str(settings.runs_dir),
                "breadcrumbs": _breadcrumbs(
                    ("datasets", "/"),
                    (dataset, f"/dataset/{urllib.parse.quote(dataset, safe='')}"),
                    (run_id, f"/run/{urllib.parse.quote(dataset, safe='')}/{urllib.parse.quote(run_id, safe='')}"),
                ),
            },
        )

    @app.get("/api/run/{dataset}/{run_id}/summary")
    def api_run_summary(dataset: str, run_id: str) -> JSONResponse:
        run_dir = _safe_join(settings.runs_dir, dataset, run_id)
        headers, rows = _read_tsv(run_dir / "outputs" / "summary.tsv")
        return JSONResponse({"headers": headers, "rows": rows})

    @app.get("/api/run/{dataset}/{run_id}/mode")
    def api_run_mode(dataset: str, run_id: str) -> PlainTextResponse:
        run_dir = _safe_join(settings.runs_dir, dataset, run_id)
        mode_txt = _read_text_if_exists(run_dir / "mode.txt", max_bytes=2000) or ""
        return PlainTextResponse(mode_txt)

    return app


def _default_runs_dir() -> Path:
    # Default to DiskANN-playground/extend-rabitq/ann-harness/runs relative to this file.
    here = Path(__file__).resolve()
    # .../DiskANN-playground/diskann-ann-bench/web/app.py
    playground = here.parents[2]
    return (playground / "extend-rabitq" / "ann-harness" / "runs").resolve()


def main() -> int:
    ap = argparse.ArgumentParser(description="diskann-ann-bench web")
    ap.add_argument("--host", default=_env("WEB_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(_env("WEB_PORT", "8081")))
    ap.add_argument("--runs-dir", default=_env("RUNS_DIR", str(_default_runs_dir())))
    args = ap.parse_args()

    settings = Settings(host=args.host, port=int(args.port), runs_dir=Path(args.runs_dir).expanduser().resolve())

    app = create_app(settings)

    import uvicorn  # type: ignore

    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
