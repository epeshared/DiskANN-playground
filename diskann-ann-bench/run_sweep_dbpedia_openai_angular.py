#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BuildCfg:
    algo: str  # pq|spherical
    name: str
    l_build: int
    max_outdegree: int
    alpha: float
    num_pq_chunks: int | None = None
    spherical_nbits: int | None = None


def _workspace_root() -> Path:
    # .../DiskANN-playground/diskann-ann-bench/run_sweep_dbpedia_openai_angular.py
    return Path(__file__).resolve().parents[2]


def _native_target_dir(*, profile: str) -> Path:
    ws = _workspace_root()
    native_dir = (
        ws
        / "ann-benchmark-epeshared"
        / "ann_benchmarks"
        / "algorithms"
        / "diskann_rs"
        / "native"
        / "target"
        / profile
    )
    return native_dir


def _ensure_native_importable(*, profile: str) -> None:
    native_target_dir = _native_target_dir(profile=profile)
    so = native_target_dir / "libdiskann_rs_native.so"
    if not so.is_file():
        raise FileNotFoundError(f"missing native library: {so}")
    # The Python adapter imports diskann_rs_native (module name), so provide a stable filename.
    stable = native_target_dir / "diskann_rs_native.so"
    try:
        if stable.exists() or stable.is_symlink():
            stable.unlink()
        stable.symlink_to(so.name)
    except Exception:
        # Best-effort; if the symlink can't be created, import may still succeed depending on loader behavior.
        pass


def _env_with_pythonpath(*, profile: str) -> dict[str, str]:
    ws = _workspace_root()
    native_target_dir = _native_target_dir(profile=profile)
    existing = os.environ.get("PYTHONPATH", "")
    parts = [str(native_target_dir), str(ws / "ann-benchmark-epeshared")]
    if existing:
        parts.append(existing)
    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join(parts)
    return env


def _sanitize_case_id(s: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def _merge_root_outputs(*, work_dir: Path) -> None:
    cases_dir = work_dir / "cases"
    out_dir = work_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[list[str]] = []
    summary_header: list[str] | None = None
    details_parts = ["# diskann-ann-bench details (aggregated)", ""]

    build_cases: dict[str, object] = {}
    search_cases: dict[str, object] = {}

    if cases_dir.is_dir():
        for case_dir in sorted([p for p in cases_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            case_id = case_dir.name
            summary_path = case_dir / "outputs" / "summary.tsv"
            if summary_path.is_file():
                lines = summary_path.read_text(encoding="utf-8", errors="replace").splitlines()
                if lines:
                    header = lines[0].split("\t")
                    if summary_header is None:
                        summary_header = ["case"] + header
                    for line in lines[1:]:
                        if not line.strip():
                            continue
                        summary_rows.append([case_id] + line.split("\t"))

            details_path = case_dir / "outputs" / "details.md"
            if details_path.is_file():
                details = details_path.read_text(encoding="utf-8", errors="replace").rstrip() + "\n"
                details_parts.extend([f"## {case_id}", "", details, ""]) 

            build_path = case_dir / "outputs" / "output.build.json"
            if build_path.is_file():
                try:
                    build_cases[case_id] = json.loads(build_path.read_text(encoding="utf-8", errors="replace"))
                except Exception:
                    build_cases[case_id] = {"_raw": build_path.read_text(encoding="utf-8", errors="replace")}

            search_path = case_dir / "outputs" / "output.search.json"
            if search_path.is_file():
                try:
                    search_cases[case_id] = json.loads(search_path.read_text(encoding="utf-8", errors="replace"))
                except Exception:
                    search_cases[case_id] = {"_raw": search_path.read_text(encoding="utf-8", errors="replace")}

    if summary_header is not None:
        out = ["\t".join(summary_header)]
        out += ["\t".join(row) for row in summary_rows]
        (out_dir / "summary.tsv").write_text("\n".join(out) + "\n", encoding="utf-8")

    (out_dir / "details.md").write_text("\n".join(details_parts).rstrip() + "\n", encoding="utf-8")

    if build_cases:
        (out_dir / "output.build.json").write_text(
            json.dumps({"cases": build_cases}, indent=2) + "\n", encoding="utf-8"
        )
    if search_cases:
        (out_dir / "output.search.json").write_text(
            json.dumps({"cases": search_cases}, indent=2) + "\n", encoding="utf-8"
        )


def _run_framework_entry(
    *,
    framework_entry: Path,
    env: dict[str, str],
    args: list[str],
) -> None:
    cmd = [sys.executable, str(framework_entry)] + args
    print("==>", " ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a small sweep for dbpedia-openai-1000k-angular (host runner)")
    ap.add_argument(
        "--hdf5",
        default="/mnt/nvme2n1p1/xtang/ann-data/dbpedia-openai-1000k-angular.hdf5",
        help="Path to dataset HDF5",
    )
    ap.add_argument("--dataset", default="dbpedia-openai-1000k-angular")
    ap.add_argument("--metric", default="cosine", choices=["cosine", "angular", "l2", "euclidean"])
    ap.add_argument("-k", type=int, default=100)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--batch", action="store_true", default=True)
    ap.add_argument("--no-batch", action="store_false", dest="batch")

    ap.add_argument("--l-build", type=int, default=125)
    ap.add_argument("--max-outdegree", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=1.2)

    ap.add_argument("--pq-chunks", default="96,128,192")
    ap.add_argument("--l-search-list", default="200,300,500")
    ap.add_argument(
        "--spherical-nbits",
        type=int,
        default=4,
        choices=[1, 2, 4],
        help="Back-compat single value (use --spherical-nbits-list for multiple)",
    )
    ap.add_argument(
        "--spherical-nbits-list",
        default=None,
        help="Comma-separated spherical nbits list to sweep (e.g. 2,4). Overrides --spherical-nbits if set.",
    )

    ap.add_argument(
        "--runs-dir",
        default=None,
        help="Override runs dir (default: DiskANN-playground/diskann-ann-bench/result)",
    )
    ap.add_argument("--run-id", default=None, help="Override run id (default: sweep_<unix_ts>)")
    ap.add_argument("--profile", default=os.environ.get("DISKANN_RS_NATIVE_PROFILE", "release"))
    ap.add_argument(
        "--cleanup-indexes",
        action="store_true",
        default=True,
        help="Delete shared index directories after finishing each build config (default: true)",
    )
    ap.add_argument(
        "--no-cleanup-indexes",
        action="store_false",
        dest="cleanup_indexes",
        help="Keep shared index directories (WARNING: can be very large)",
    )
    args = ap.parse_args()

    ws = _workspace_root()
    framework_entry = ws / "DiskANN-playground" / "diskann-ann-bench" / "framework_entry.py"
    if not framework_entry.is_file():
        raise FileNotFoundError(str(framework_entry))

    runs_dir = Path(args.runs_dir).expanduser().resolve() if args.runs_dir else (framework_entry.parent / "result")
    run_id = args.run_id or f"sweep_{int(time.time())}"
    dataset = str(args.dataset)
    work_dir = (runs_dir / dataset / run_id).resolve()
    cases_dir = work_dir / "cases"
    indexes_dir = work_dir / "shared_indexes"
    (work_dir / "outputs").mkdir(parents=True, exist_ok=True)
    cases_dir.mkdir(parents=True, exist_ok=True)
    indexes_dir.mkdir(parents=True, exist_ok=True)

    work_dir.joinpath("mode.txt").write_text("ann_bench_diskann_rs\n", encoding="utf-8")
    work_dir.joinpath("cpu-bind.txt").write_text("0-16\n", encoding="utf-8")

    pq_chunks = [int(x) for x in str(args.pq_chunks).split(",") if x.strip()]
    l_search_list = [int(x) for x in str(args.l_search_list).split(",") if x.strip()]

    if args.spherical_nbits_list is not None:
        spherical_nbits_list = [int(x) for x in str(args.spherical_nbits_list).split(",") if x.strip()]
    else:
        spherical_nbits_list = [int(args.spherical_nbits)]

    for nbits in spherical_nbits_list:
        if nbits not in (1, 2, 4):
            raise ValueError(f"invalid spherical nbits={nbits}; expected one of 1,2,4")

    _ensure_native_importable(profile=str(args.profile))
    env = _env_with_pythonpath(profile=str(args.profile))

    common = [
        "--hdf5",
        str(Path(args.hdf5).expanduser().resolve()),
        "--dataset",
        dataset,
        "--metric",
        str(args.metric),
        "-k",
        str(int(args.k)),
        "--reps",
        str(int(args.reps)),
        "--l-build",
        str(int(args.l_build)),
        "--max-outdegree",
        str(int(args.max_outdegree)),
        "--alpha",
        str(float(args.alpha)),
    ]
    if bool(args.batch):
        common.append("--batch")

    build_cfgs: list[BuildCfg] = [
        *[
            BuildCfg(
                algo="pq",
                name=f"pq_chunks_{chunks}",
                l_build=int(args.l_build),
                max_outdegree=int(args.max_outdegree),
                alpha=float(args.alpha),
                num_pq_chunks=int(chunks),
            )
            for chunks in pq_chunks
        ],
        *[
            BuildCfg(
                algo="spherical",
                name=f"spherical_{int(nbits)}b",
                l_build=int(args.l_build),
                max_outdegree=int(args.max_outdegree),
                alpha=float(args.alpha),
                spherical_nbits=int(nbits),
            )
            for nbits in spherical_nbits_list
        ],
    ]

    print(f"Run dir: {work_dir}")
    print(f"PQ chunks: {pq_chunks} | L values: {l_search_list} | spherical nbits: {spherical_nbits_list}")

    try:
        for b in build_cfgs:
            index_dir = indexes_dir / b.name
            build_case_id = _sanitize_case_id(f"build_{b.name}")
            build_work_dir = cases_dir / build_case_id
            (build_work_dir / "outputs").mkdir(parents=True, exist_ok=True)

            build_json = build_work_dir / "outputs" / "output.build.json"

            index_looks_present = index_dir.is_dir() and any(index_dir.iterdir())
            if build_json.is_file() and index_looks_present:
                print(f"==> skip build (already exists): {b.name}", flush=True)
            else:
                build_args = [
                    "--work-dir",
                    str(build_work_dir),
                    "--index-dir",
                    str(index_dir),
                    "--stage",
                    "build",
                    "--algo",
                    b.algo,
                ]
                if b.algo == "pq":
                    assert b.num_pq_chunks is not None
                    build_args += ["--num-pq-chunks", str(int(b.num_pq_chunks))]
                if b.algo == "spherical":
                    assert b.spherical_nbits is not None
                    build_args += ["--spherical-nbits", str(int(b.spherical_nbits))]

                _run_framework_entry(framework_entry=framework_entry, env=env, args=build_args + common)
            if not build_json.is_file():
                raise RuntimeError(f"build stage did not produce {build_json}")

            for l_search in l_search_list:
                case_id = _sanitize_case_id(f"{b.algo}_{b.name}_L{int(l_search)}")
                case_dir = cases_dir / case_id
                (case_dir / "outputs").mkdir(parents=True, exist_ok=True)

                summary_path = case_dir / "outputs" / "summary.tsv"
                if summary_path.is_file():
                    print(f"==> skip search (already exists): {case_id}", flush=True)
                    continue

                search_args = [
                    "--work-dir",
                    str(case_dir),
                    "--index-dir",
                    str(index_dir),
                    "--stage",
                    "search",
                    "--algo",
                    b.algo,
                    "--l-search",
                    str(int(l_search)),
                ]
                if b.algo == "pq":
                    search_args += ["--num-pq-chunks", str(int(b.num_pq_chunks or 0))]
                if b.algo == "spherical":
                    search_args += ["--spherical-nbits", str(int(b.spherical_nbits or 0))]

                _run_framework_entry(framework_entry=framework_entry, env=env, args=search_args + common)

                # Make per-(chunks,L) cases web-friendly: include the build metadata alongside search results.
                try:
                    (case_dir / "outputs" / "output.build.json").write_text(
                        build_json.read_text(encoding="utf-8", errors="replace"),
                        encoding="utf-8",
                    )
                except Exception:
                    pass

                _merge_root_outputs(work_dir=work_dir)

            if bool(args.cleanup_indexes):
                try:
                    if index_dir.is_dir():
                        print(f"==> cleanup index dir: {index_dir}", flush=True)
                        import shutil

                        shutil.rmtree(index_dir, ignore_errors=True)
                except Exception:
                    pass

    finally:
        _merge_root_outputs(work_dir=work_dir)

    print(f"Done. Run dir: {work_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
