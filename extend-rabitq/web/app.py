#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import html
import os
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

from markdown_it import MarkdownIt


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    runs_dir: Path


def _default_runs_dir() -> Path:
    here = Path(__file__).resolve()
    extend_rabitq_dir = here.parent.parent
    return (extend_rabitq_dir / "ann-harness" / "runs").resolve()


def _safe_join(root: Path, *parts: str) -> Path:
    out = root
    for p in parts:
        # Prevent absolute paths.
        if p.startswith("/") or p.startswith("\\"):
            raise HTTPException(status_code=400, detail="invalid path")
        out = out / p
    out = out.resolve()
    try:
        out.relative_to(root)
    except Exception:
        raise HTTPException(status_code=400, detail="path traversal blocked")
    return out


def _list_dirs(p: Path) -> list[str]:
    if not p.exists():
        return []
    out: list[str] = []
    for child in p.iterdir():
        if child.is_dir():
            out.append(child.name)
    out.sort()
    return out


def _list_dirs_sorted_desc(p: Path) -> list[str]:
    if not p.exists():
        return []
    dirs = [c for c in p.iterdir() if c.is_dir()]
    dirs.sort(key=lambda x: x.name, reverse=True)
    return [d.name for d in dirs]


def _read_text_if_exists(path: Path, *, max_bytes: int = 2_000_000) -> str | None:
    try:
        if not path.is_file():
            return None
        data = path.read_bytes()
        if len(data) > max_bytes:
            return f"<file too large: {len(data)} bytes>"
        return data.decode("utf-8", errors="replace")
    except Exception:
        return None


def _read_tsv(path: Path, *, max_rows: int = 5000) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        return ([], [])
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        headers = reader.fieldnames or []
        rows: list[dict[str, str]] = []
        for i, row in enumerate(reader):
            if i >= max_rows:
                rows.append({h: f"<truncated after {max_rows} rows>" for h in headers})
                break
            rows.append({k: (v if v is not None else "") for k, v in row.items()})
        return (headers, rows)


def _markdown_to_html(md_text: str) -> str:
    # Keep it simple and safe-ish:
    # - markdown-it-py does not execute JS.
    # - We still escape raw HTML by disabling html.
    md = MarkdownIt("commonmark", {"html": False, "linkify": True, "typographer": False})
    rendered = md.render(md_text)
    return rendered


def _breadcrumbs(*items: tuple[str, str]) -> str:
    # items: (label, href)
    parts: list[str] = []
    for label, href in items:
        parts.append(f'<a href="{html.escape(href, quote=True)}">{html.escape(label)}</a>')
    return " / ".join(parts)


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title="extend-rabitq web")

    templates_dir = Path(__file__).resolve().parent / "templates"
    templates = Jinja2Templates(directory=str(templates_dir))
    templates.env.filters["urlencode"] = lambda s: urllib.parse.quote(str(s), safe="")

    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "runs_dir": str(settings.runs_dir),
        }

    @app.get("/api/datasets")
    def api_datasets() -> JSONResponse:
        return JSONResponse(_list_dirs(settings.runs_dir))

    @app.get("/api/runs/{dataset}")
    def api_runs(dataset: str) -> JSONResponse:
        dataset_dir = _safe_join(settings.runs_dir, dataset)
        return JSONResponse(_list_dirs_sorted_desc(dataset_dir))

    @app.get("/api/run/{dataset}/{run_id}/summary")
    def api_run_summary(dataset: str, run_id: str) -> JSONResponse:
        run_dir = _safe_join(settings.runs_dir, dataset, run_id)
        summary_path = run_dir / "outputs" / "summary.tsv"
        headers, rows = _read_tsv(summary_path)
        return JSONResponse({"headers": headers, "rows": rows})

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        datasets = _list_dirs(settings.runs_dir)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "title": "DiskANN runs",
                "datasets": datasets,
                "runs_dir": str(settings.runs_dir),
                "breadcrumbs": None,
            },
        )

    @app.get("/dataset/{dataset}", response_class=HTMLResponse)
    def dataset_page(dataset: str, request: Request):
        dataset_dir = _safe_join(settings.runs_dir, dataset)
        runs = _list_dirs_sorted_desc(dataset_dir)
        bc = _breadcrumbs(("datasets", "/"), (dataset, f"/dataset/{dataset}"))
        return templates.TemplateResponse(
            "dataset.html",
            {
                "request": request,
                "title": f"{dataset} runs",
                "dataset": dataset,
                "runs": runs,
                "runs_dir": str(settings.runs_dir),
                "breadcrumbs": bc,
            },
        )

    @app.get("/run/{dataset}/{run_id}", response_class=HTMLResponse)
    def run_page(dataset: str, run_id: str, request: Request):
        run_dir = _safe_join(settings.runs_dir, dataset, run_id)
        if not run_dir.is_dir():
            raise HTTPException(status_code=404, detail="run not found")

        summary_path = run_dir / "outputs" / "summary.tsv"
        headers, rows = _read_tsv(summary_path)

        details_text = _read_text_if_exists(run_dir / "outputs" / "details.md")
        details_html = _markdown_to_html(details_text) if details_text else None

        server_info = _read_text_if_exists(run_dir / "server-info.txt")

        downloads = []
        for rel, label in [
            ("outputs/summary.tsv", "summary.tsv"),
            ("outputs/details.md", "details.md"),
            ("outputs/output.json", "output.json"),
            ("outputs/output.search.json", "output.search.json"),
            ("outputs/output.build.json", "output.build.json"),
            ("configs/pq-vs-spherical.json", "config"),
            ("server-info.txt", "server-info.txt"),
        ]:
            p = run_dir / rel
            if p.is_file():
                downloads.append({
                    "label": label,
                    "href": f"/file/{dataset}/{run_id}/{rel}",
                })

        bc = _breadcrumbs(
            ("datasets", "/"),
            (dataset, f"/dataset/{dataset}"),
            (run_id, f"/run/{dataset}/{run_id}"),
        )
        return templates.TemplateResponse(
            "run.html",
            {
                "request": request,
                "title": f"{dataset}/{run_id}",
                "dataset": dataset,
                "run_id": run_id,
                "run_path": str(run_dir),
                "summary_headers": headers,
                "summary_rows": rows,
                "details_html": details_html,
                "server_info": server_info,
                "downloads": downloads,
                "runs_dir": str(settings.runs_dir),
                "breadcrumbs": bc,
            },
        )

    @app.get("/file/{dataset}/{run_id}/{rel_path:path}")
    def get_file(dataset: str, run_id: str, rel_path: str):
        run_dir = _safe_join(settings.runs_dir, dataset, run_id)
        file_path = _safe_join(run_dir, rel_path)
        if not file_path.is_file():
            raise HTTPException(status_code=404, detail="file not found")
        return FileResponse(str(file_path), filename=file_path.name)

    @app.get("/raw/{dataset}/{run_id}/{rel_path:path}")
    def get_raw(dataset: str, run_id: str, rel_path: str):
        run_dir = _safe_join(settings.runs_dir, dataset, run_id)
        file_path = _safe_join(run_dir, rel_path)
        if not file_path.is_file():
            raise HTTPException(status_code=404, detail="file not found")
        text = _read_text_if_exists(file_path)
        if text is None:
            raise HTTPException(status_code=500, detail="failed to read file")
        return PlainTextResponse(text)

    return app


def _parse_args() -> Settings:
    ap = argparse.ArgumentParser(description="Web UI to browse ann-harness runs")
    ap.add_argument("--host", default=os.environ.get("WEB_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("WEB_PORT", "8080")))
    ap.add_argument("--runs-dir", default=os.environ.get("RUNS_DIR", str(_default_runs_dir())))
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    return Settings(host=str(args.host), port=int(args.port), runs_dir=runs_dir)


def main() -> int:
    settings = _parse_args()
    if not settings.runs_dir.exists():
        raise SystemExit(f"RUNS_DIR does not exist: {settings.runs_dir}")

    app = create_app(settings)

    import uvicorn  # type: ignore

    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
