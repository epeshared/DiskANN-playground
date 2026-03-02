#!/usr/bin/env python3
"""Verify whether a DiskANN-RS PQ run used disk index artifacts.

Hard evidence lives in the index directory:
- disk index: *.pq.disk.meta.json
- memory PQ:  *.pq.meta.json (but not *.pq.disk.meta.json)

This script scans a run folder and prints a small per-case report.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CaseReport:
    case_id: str
    index_dir: Path
    disk_meta_count: int
    mem_meta_count: int
    build_disk_index: bool | None
    search_disk_index: bool | None

    @property
    def verdict(self) -> str:
        if self.disk_meta_count > 0:
            return "disk"
        if self.mem_meta_count > 0:
            return "mem"
        if self.index_dir.is_dir():
            return "index_dir_empty"
        return "missing"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"failed to parse json: {path} ({exc})") from exc


def _get_disk_index_flag(obj: dict[str, Any] | None) -> bool | None:
    if obj is None:
        return None
    value = obj.get("disk_index")
    if value is None:
        return None
    return bool(value)


def _is_run_dir(path: Path) -> bool:
    return (path / "cases").is_dir() and (path / "outputs").is_dir()


def _resolve_run_dir(path: Path) -> Path:
    p = path.expanduser().resolve()
    if _is_run_dir(p):
        return p
    raise SystemExit(
        f"Not a run dir: {p}\n"
        "Expected a directory containing ./cases and ./outputs (e.g. result/<dataset>/<run_id>)."
    )


def scan_run(run_dir: Path) -> list[CaseReport]:
    cases_dir = run_dir / "cases"
    reports: list[CaseReport] = []

    for case_dir in sorted([p for p in cases_dir.iterdir() if p.is_dir()]):
        index_dir = case_dir / "index"
        disk_meta = list(index_dir.glob("*.pq.disk.meta.json")) if index_dir.is_dir() else []
        mem_meta = []
        if index_dir.is_dir():
            for p in index_dir.glob("*.pq.meta.json"):
                if p.name.endswith(".pq.disk.meta.json"):
                    continue
                mem_meta.append(p)

        build_obj = _read_json(case_dir / "outputs" / "output.build.json")
        search_obj = _read_json(case_dir / "outputs" / "output.search.json")

        reports.append(
            CaseReport(
                case_id=case_dir.name,
                index_dir=index_dir,
                disk_meta_count=len(disk_meta),
                mem_meta_count=len(mem_meta),
                build_disk_index=_get_disk_index_flag(build_obj),
                search_disk_index=_get_disk_index_flag(search_obj),
            )
        )

    return reports


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Verify disk-index evidence for a diskann-ann-bench run folder."
    )
    ap.add_argument(
        "run_dir",
        type=Path,
        help="Run directory (must contain ./cases and ./outputs)",
    )
    ap.add_argument(
        "--only",
        choices=["disk", "mem", "missing", "index_dir_empty", "all"],
        default="all",
        help="Filter printed cases by verdict",
    )
    args = ap.parse_args(argv)

    run_dir = _resolve_run_dir(args.run_dir)
    reports = scan_run(run_dir)

    if not reports:
        print(f"No cases found under: {run_dir / 'cases'}", file=sys.stderr)
        return 2

    # Header
    print(
        "case_id\tverdict\tdisk_meta\tmem_meta\tbuild.disk_index\tsearch.disk_index\tindex_dir"
    )

    wanted = args.only
    for r in reports:
        if wanted != "all" and r.verdict != wanted:
            continue
        b = "" if r.build_disk_index is None else str(r.build_disk_index).lower()
        s = "" if r.search_disk_index is None else str(r.search_disk_index).lower()
        print(
            f"{r.case_id}\t{r.verdict}\t{r.disk_meta_count}\t{r.mem_meta_count}\t{b}\t{s}\t{r.index_dir}"
        )

    # Simple summary
    disk = sum(1 for r in reports if r.verdict == "disk")
    mem = sum(1 for r in reports if r.verdict == "mem")
    missing = sum(1 for r in reports if r.verdict in ("missing", "index_dir_empty"))
    print(
        f"\nSummary: cases={len(reports)} disk={disk} mem={mem} missing_or_empty={missing}",
        file=sys.stderr,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
