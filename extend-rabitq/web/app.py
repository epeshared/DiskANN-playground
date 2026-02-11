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


def _extract_cpu_bind_from_server_info(server_info: str | None) -> str | None:
    if not server_info:
        return None
    for line in server_info.splitlines():
        # run_dataset.py writes: cpu_bind: <desc>
        if line.startswith("cpu_bind:"):
            v = line.split(":", 1)[1].strip()
            return v or None
    return None


def _extract_cpu_model_from_server_info(server_info: str | None) -> str | None:
    if not server_info:
        return None
    for line in server_info.splitlines():
        # lscpu line format on Linux:
        #   Model name:                           Intel(R) Xeon(R) ...
        if line.lower().startswith("model name:"):
            v = line.split(":", 1)[1].strip()
            return v or None
    return None


def _extract_numactl_cpulist(cpu_bind: str | None) -> str | None:
    if not cpu_bind:
        return None
    # Expect patterns like: "numactl -C 0-15 -m 0" (tokens) or "numactl -C0-15 -m0".
    tokens = cpu_bind.split()
    for i, tok in enumerate(tokens):
        if tok == "-C" and i + 1 < len(tokens):
            return tokens[i + 1].strip() or None
        if tok.startswith("-C") and len(tok) > 2:
            return tok[2:].strip() or None
        if tok.startswith("--physcpubind="):
            v = tok.split("=", 1)[1].strip()
            return v or None
    return None


def _count_cpulist(cpulist: str | None) -> int | None:
    if not cpulist:
        return None
    total = 0
    for part in cpulist.split(","):
        p = part.strip()
        if not p:
            continue
        if "-" in p:
            a, b, *rest = p.split("-")
            if rest:
                return None
            try:
                start = int(a)
                end = int(b)
            except ValueError:
                return None
            if end < start:
                return None
            total += (end - start + 1)
        else:
            try:
                int(p)
            except ValueError:
                return None
            total += 1
    return total


def _parse_sort_spec(spec: str | None) -> list[tuple[str, bool]]:
    # Returns list[(field, reverse)]
    # Supported fields: timestamp, cpu_cores, cpu_model
    default = [
        ("timestamp", True),
        ("cpu_cores", True),
        ("cpu_model", False),
    ]
    if not spec:
        return default

    allowed = {"timestamp", "cpu_cores", "cpu_model"}
    out: list[tuple[str, bool]] = []
    for raw in str(spec).split(","):
        item = raw.strip()
        if not item:
            continue
        reverse = item.startswith("-")
        field = item[1:] if reverse else item
        if field not in allowed:
            continue
        out.append((field, reverse))
    return out or default


def _to_int_maybe(v: str | None) -> int | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _to_float_maybe(v: str | None) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() == "none":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _fmt_delta(v: float | None) -> str:
    if v is None:
        return "-"
    # Keep small deltas readable.
    return f"{v:+.4g}"


def _fmt_ratio_pct(numer: float | None, denom: float | None) -> str:
    # Return percent change: ((numer/denom) - 1) * 100%.
    # Example: +3.1% means numer is 3.1% higher than denom.
    if numer is None or denom is None:
        return "-"
    if denom == 0:
        return "-"
    return f"{((numer / denom) - 1.0) * 100.0:+.2f}%"


def _run_meta(run_dir: Path) -> dict[str, Any]:
    server_info_head = _read_text_if_exists(run_dir / "server-info.txt", max_bytes=300_000)
    cpu_bind = _read_text_if_exists(run_dir / "cpu-bind.txt")
    if not cpu_bind:
        cpu_bind = _extract_cpu_bind_from_server_info(server_info_head)

    cpu_model = _extract_cpu_model_from_server_info(server_info_head)
    cpu_cores = _count_cpulist(_extract_numactl_cpulist(cpu_bind))

    return {
        "cpu_bind": (cpu_bind.strip() if isinstance(cpu_bind, str) and cpu_bind.strip() else None),
        "cpu_model": cpu_model,
        "cpu_cores": cpu_cores,
        "server_info": server_info_head,
        "has_cpu_bind_file": (run_dir / "cpu-bind.txt").is_file(),
        "run_path": str(run_dir),
    }


def _index_rows(rows: list[dict[str, str]], key_fields: list[str]) -> dict[tuple[str, ...], dict[str, str]]:
    out: dict[tuple[str, ...], dict[str, str]] = {}
    for r in rows:
        k = tuple((r.get(f) or "").strip() for f in key_fields)
        out[k] = r
    return out


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
        run_ids = _list_dirs_sorted_desc(dataset_dir)
        sort_spec_raw = request.query_params.get("sort")
        sort_spec = _parse_sort_spec(sort_spec_raw)
        runs = []
        for run_id in run_ids:
            run_dir = dataset_dir / run_id
            server_info_head = _read_text_if_exists(run_dir / "server-info.txt", max_bytes=200_000)
            cpu_bind = _read_text_if_exists(run_dir / "cpu-bind.txt")
            if not cpu_bind:
                cpu_bind = _extract_cpu_bind_from_server_info(server_info_head)

            cpu_model = _extract_cpu_model_from_server_info(server_info_head)
            cpu_cores = _count_cpulist(_extract_numactl_cpulist(cpu_bind))

            runs.append(
                {
                    "id": run_id,
                    "timestamp": run_id,
                    "has_cpu_bind": (run_dir / "cpu-bind.txt").is_file(),
                    "cpu_bind": (cpu_bind.strip() if isinstance(cpu_bind, str) and cpu_bind.strip() else None),
                    "cpu_cores": cpu_cores,
                    "cpu_model": cpu_model,
                }
            )

        # Apply multi-key sort. We sort last key first (stable sort) to honor sort_spec order.
        for field, reverse in reversed(sort_spec):
            if field == "timestamp":
                runs.sort(key=lambda r: r.get("timestamp") or "", reverse=reverse)
            elif field == "cpu_cores":
                if reverse:
                    runs.sort(
                        key=lambda r: (
                            r.get("cpu_cores") is None,
                            -(int(r.get("cpu_cores")) if r.get("cpu_cores") is not None else 0),
                        )
                    )
                else:
                    runs.sort(
                        key=lambda r: (
                            r.get("cpu_cores") is None,
                            int(r.get("cpu_cores")) if r.get("cpu_cores") is not None else 0,
                        )
                    )
            elif field == "cpu_model":
                runs.sort(key=lambda r: (r.get("cpu_model") or ""), reverse=reverse)
                # Keep unknown CPU model (None) last regardless of direction.
                runs.sort(key=lambda r: (r.get("cpu_model") is None))
        bc = _breadcrumbs(("datasets", "/"), (dataset, f"/dataset/{dataset}"))
        return templates.TemplateResponse(
            "dataset.html",
            {
                "request": request,
                "title": f"{dataset} runs",
                "dataset": dataset,
                "runs": runs,
                "sort": sort_spec_raw or "-timestamp,-cpu_cores,cpu_model",
                "runs_dir": str(settings.runs_dir),
                "breadcrumbs": bc,
            },
        )

    @app.get("/run/{dataset}/{run_id}", response_class=HTMLResponse)
    def run_page(dataset: str, run_id: str, request: Request):
        run_dir = _safe_join(settings.runs_dir, dataset, run_id)
        if not run_dir.is_dir():
            raise HTTPException(status_code=404, detail="run not found")

        cpu_bind_path = run_dir / "cpu-bind.txt"
        has_cpu_bind_file = cpu_bind_path.is_file()

        cpu_bind = _read_text_if_exists(cpu_bind_path)

        summary_path = run_dir / "outputs" / "summary.tsv"
        headers, rows = _read_tsv(summary_path)

        details_text = _read_text_if_exists(run_dir / "outputs" / "details.md")
        details_html = _markdown_to_html(details_text) if details_text else None

        server_info = _read_text_if_exists(run_dir / "server-info.txt")
        if not cpu_bind:
            cpu_bind = _extract_cpu_bind_from_server_info(server_info)

        downloads = []
        for rel, label in [
            ("cpu-bind.txt", "cpu-bind.txt"),
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
                "has_cpu_bind_file": has_cpu_bind_file,
                "cpu_bind": (cpu_bind.strip() if isinstance(cpu_bind, str) and cpu_bind.strip() else None),
                "summary_headers": headers,
                "summary_rows": rows,
                "details_html": details_html,
                "server_info": server_info,
                "downloads": downloads,
                "runs_dir": str(settings.runs_dir),
                "breadcrumbs": bc,
            },
        )

    @app.get("/compare/{dataset}", response_class=HTMLResponse)
    def compare_page(dataset: str, request: Request):
        a = request.query_params.get("a")
        b = request.query_params.get("b")
        mode = (request.query_params.get("mode") or "b_over_a").strip().lower()
        if mode in {"ba", "b/a", "b_over_a"}:
            mode = "b_over_a"
        elif mode in {"ab", "a/b", "a_over_b"}:
            mode = "a_over_b"
        else:
            mode = "b_over_a"

        delta_label = "Δ = ((B/A) - 1) × 100%" if mode == "b_over_a" else "Δ = ((A/B) - 1) × 100%"
        if not a or not b:
            raise HTTPException(status_code=400, detail="missing query params: a, b")
        if a == b:
            raise HTTPException(status_code=400, detail="a and b must be different")

        dataset_dir = _safe_join(settings.runs_dir, dataset)
        run_a_dir = _safe_join(dataset_dir, a)
        run_b_dir = _safe_join(dataset_dir, b)
        if not run_a_dir.is_dir() or not run_b_dir.is_dir():
            raise HTTPException(status_code=404, detail="run not found")

        meta_a = _run_meta(run_a_dir)
        meta_b = _run_meta(run_b_dir)

        headers_a, rows_a = _read_tsv(run_a_dir / "outputs" / "summary.tsv")
        headers_b, rows_b = _read_tsv(run_b_dir / "outputs" / "summary.tsv")

        # Compare only on common columns.
        common_headers = [h for h in headers_a if h in set(headers_b)]
        key_fields = [f for f in ["job", "detail", "tasks", "L", "N"] if f in common_headers]
        metric_fields = [h for h in common_headers if h not in key_fields]

        # Show ALL metrics; order a few important ones first.
        preferred_metrics = [
            "recall(avg)",
            "QPS(mean)",
            "lat_mean_us(mean)",
            "lat_p99_us(mean)",
        ]
        metrics = [m for m in preferred_metrics if m in metric_fields]
        metrics.extend([m for m in metric_fields if m not in set(metrics)])

        idx_a = _index_rows(rows_a, key_fields)
        idx_b = _index_rows(rows_b, key_fields)
        all_keys = sorted(set(idx_a.keys()) | set(idx_b.keys()))

        def sort_key(k: tuple[str, ...]) -> tuple[Any, ...]:
            # job/detail lexicographic; tasks/L/N numeric when possible.
            parts: list[Any] = []
            for name, v in zip(key_fields, k):
                if name in {"tasks", "L", "N"}:
                    parts.append(_to_int_maybe(v) if _to_int_maybe(v) is not None else v)
                else:
                    parts.append(v)
            return tuple(parts)

        all_keys.sort(key=sort_key)

        compare_rows: list[dict[str, Any]] = []
        for k in all_keys:
            ra = idx_a.get(k)
            rb = idx_b.get(k)
            out: dict[str, Any] = {}
            for i, f in enumerate(key_fields):
                out[f] = k[i] if i < len(k) else ""
            out["present_a"] = ra is not None
            out["present_b"] = rb is not None

            metric_rows: list[dict[str, Any]] = []
            for m in metrics:
                va = (ra.get(m) if ra else None)
                vb = (rb.get(m) if rb else None)
                da = _to_float_maybe(va)
                db = _to_float_maybe(vb)
                delta = _fmt_ratio_pct(db, da) if mode == "b_over_a" else _fmt_ratio_pct(da, db)
                metric_rows.append(
                    {
                        "name": m,
                        "a": (va if va is not None and str(va).strip() else "-"),
                        "b": (vb if vb is not None and str(vb).strip() else "-"),
                        "ratio_pct": delta,
                    }
                )
            out["metrics"] = metric_rows
            compare_rows.append(out)

        bc = _breadcrumbs(
            ("datasets", "/"),
            (dataset, f"/dataset/{dataset}"),
            ("compare", f"/compare/{dataset}?a={urllib.parse.quote(a)}&b={urllib.parse.quote(b)}&mode={urllib.parse.quote(mode)}"),
        )
        return templates.TemplateResponse(
            "compare.html",
            {
                "request": request,
                "title": f"compare {dataset}",
                "dataset": dataset,
                "a": a,
                "b": b,
                "meta_a": meta_a,
                "meta_b": meta_b,
                "mode": mode,
                "delta_label": delta_label,
                "key_fields": key_fields,
                "metrics": metrics,
                "rows": compare_rows,
                "wide": True,
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
