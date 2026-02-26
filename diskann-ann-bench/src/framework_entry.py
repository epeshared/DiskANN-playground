#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

import h5py
import numpy as np

from ann_benchmarks.algorithms.diskann_rs.module import (
    DiskANNRS,
    DiskANNRS_PQ,
    DiskANNRS_Spherical,
)
from ann_benchmarks.distance import dataset_transform


try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


def _find_workspace_root() -> Path:
    """Find the workspace root containing both DiskANN-playground/ and ann-benchmark-epeshared/.

    This script lives under DiskANN-playground/diskann-ann-bench/src/.
    Avoid relying on a fixed number of parent directories so refactors don't break path logic.
    """

    here = Path(__file__).resolve()
    for p in (here.parent,) + tuple(here.parents):
        if (p / "DiskANN-playground").is_dir() and (p / "ann-benchmark-epeshared").is_dir():
            return p
    # Fallback to the historical assumption (best effort).
    return here.parents[3]


def _rss_bytes_self() -> int | None:
    """Best-effort RSS (resident set size) for the current process in bytes."""
    if psutil is not None:
        try:
            return int(psutil.Process().memory_info().rss)
        except Exception:
            pass

    # Fallback: parse /proc/self/status (Linux).
    try:
        with open("/proc/self/status", "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # e.g. "VmRSS:\t  123456 kB"
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        return int(parts[1]) * 1024
    except Exception:
        return None
    return None


class _PeakRssSampler:
    """Samples RSS in the background and tracks the max during an interval."""

    def __init__(self, interval_s: float = 0.02):
        self._interval_s = float(interval_s)
        self._stop = threading.Event()
        self._peak: int | None = None
        self._thread: threading.Thread | None = None

    @property
    def peak_bytes(self) -> int | None:
        return self._peak

    def _sample_once(self) -> None:
        v = _rss_bytes_self()
        if v is None:
            return
        if self._peak is None or v > self._peak:
            self._peak = int(v)

    def __enter__(self) -> "_PeakRssSampler":
        # Prime an initial sample so very short intervals still get a value.
        self._sample_once()

        def _run() -> None:
            while not self._stop.is_set():
                self._sample_once()
                # Event-aware sleep to stop promptly.
                self._stop.wait(self._interval_s)

        self._thread = threading.Thread(target=_run, name="peak-rss-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            try:
                self._thread.join(timeout=1.0)
            except Exception:
                pass
        # Final sample at the end of the interval.
        self._sample_once()


def _disk_io_totals() -> tuple[int, int, int, int] | None:
    if psutil is not None:
        try:
            io = psutil.disk_io_counters(perdisk=False)
            if io is not None:
                return (
                    int(getattr(io, "read_count", 0)),
                    int(getattr(io, "write_count", 0)),
                    int(getattr(io, "read_bytes", 0)),
                    int(getattr(io, "write_bytes", 0)),
                )
        except Exception:
            pass

    dev_pat = re.compile(r"^(sd[a-z]+|hd[a-z]+|vd[a-z]+|xvd[a-z]+|nvme\d+n\d+|mmcblk\d+|dm-\d+|md\d+)$")
    total_read_ios = 0
    total_write_ios = 0
    total_read_bytes = 0
    total_write_bytes = 0

    try:
        with open("/proc/diskstats", "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 14:
                    continue
                dev = parts[2]
                if not dev_pat.match(dev):
                    continue

                read_ios = int(parts[3])
                read_sectors = int(parts[5])
                write_ios = int(parts[7])
                write_sectors = int(parts[9])

                total_read_ios += read_ios
                total_write_ios += write_ios
                total_read_bytes += read_sectors * 512
                total_write_bytes += write_sectors * 512
    except Exception:
        return None

    return (total_read_ios, total_write_ios, total_read_bytes, total_write_bytes)


class _PeakDiskIoSampler:
    def __init__(self, interval_s: float = 0.1):
        self._interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self._last_totals: tuple[int, int, int, int] | None = None
        self._last_ts: float | None = None

        self._peak_tps: float | None = None
        self._peak_kb_read_s: float | None = None
        self._peak_kb_wrtn_s: float | None = None

    @property
    def peak_tps(self) -> float | None:
        return self._peak_tps

    @property
    def peak_kb_read_s(self) -> float | None:
        return self._peak_kb_read_s

    @property
    def peak_kb_wrtn_s(self) -> float | None:
        return self._peak_kb_wrtn_s

    def _sample_once(self) -> None:
        totals = _disk_io_totals()
        ts = time.perf_counter()
        if totals is None:
            return

        if self._last_totals is None or self._last_ts is None:
            self._last_totals = totals
            self._last_ts = ts
            return

        dt = ts - self._last_ts
        if dt <= 0:
            self._last_totals = totals
            self._last_ts = ts
            return

        prev_r_ios, prev_w_ios, prev_r_b, prev_w_b = self._last_totals
        cur_r_ios, cur_w_ios, cur_r_b, cur_w_b = totals

        d_ios = max(0, cur_r_ios - prev_r_ios) + max(0, cur_w_ios - prev_w_ios)
        d_r_b = max(0, cur_r_b - prev_r_b)
        d_w_b = max(0, cur_w_b - prev_w_b)

        tps = float(d_ios) / float(dt)
        kb_read_s = float(d_r_b) / float(1024.0 * dt)
        kb_wrtn_s = float(d_w_b) / float(1024.0 * dt)

        if self._peak_tps is None or tps > self._peak_tps:
            self._peak_tps = tps
        if self._peak_kb_read_s is None or kb_read_s > self._peak_kb_read_s:
            self._peak_kb_read_s = kb_read_s
        if self._peak_kb_wrtn_s is None or kb_wrtn_s > self._peak_kb_wrtn_s:
            self._peak_kb_wrtn_s = kb_wrtn_s

        self._last_totals = totals
        self._last_ts = ts

    def __enter__(self) -> "_PeakDiskIoSampler":
        self._sample_once()

        def _run() -> None:
            while not self._stop.is_set():
                self._sample_once()
                self._stop.wait(self._interval_s)

        self._thread = threading.Thread(target=_run, name="peak-disk-io-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            try:
                self._thread.join(timeout=1.0)
            except Exception:
                pass
        self._sample_once()



@dataclass(frozen=True)
class StageOutputs:
    build_json: Path
    search_json: Path
    summary_tsv: Path
    details_md: Path


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def _pctl(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs2 = sorted(xs)
    if q <= 0:
        return xs2[0]
    if q >= 1:
        return xs2[-1]
    i = int(round(q * (len(xs2) - 1)))
    return xs2[max(0, min(len(xs2) - 1, i))]


def _recall_at_k(results: Iterable[list[int]], truth: np.ndarray, k: int) -> float:
    # truth: (n_queries, >=k)
    if k <= 0:
        return 0.0
    total = 0.0
    n = 0
    for i, ids in enumerate(results):
        gt = truth[i, :k]
        gt_set = set(int(x) for x in gt if int(x) >= 0)
        if not gt_set:
            continue
        hit = 0
        for x in ids[:k]:
            if int(x) in gt_set:
                hit += 1
        total += hit / float(k)
        n += 1
    return total / float(max(1, n))


def _run_queries(
    *,
    algo: Any,
    X_test: np.ndarray,
    k: int,
    reps: int,
    batch: bool,
) -> tuple[list[float], list[list[int]]]:
    latencies_s: list[float] = []
    ids_per_query: list[list[int]] = []

    if reps <= 0:
        raise ValueError("reps must be >= 1")

    for _ in range(reps):
        if batch:
            t0 = time.perf_counter()
            algo.batch_query(X_test, k)
            t1 = time.perf_counter()
            res = algo.get_batch_results()

            ids_per_query = [list(map(int, row)) for row in res]

            per_query = (t1 - t0) / float(max(1, len(ids_per_query)))
            latencies_s.extend([per_query] * len(ids_per_query))
        else:
            ids_per_query = []
            for v in X_test:
                t0 = time.perf_counter()
                ids = algo.query(v, k)
                t1 = time.perf_counter()
                latencies_s.append(t1 - t0)
                ids_per_query.append([int(x) for x in ids])

    return latencies_s, ids_per_query


def _algo_ctor_and_prefix(algo: str, index_dir: Path) -> tuple[type, Path]:
    if algo == "fp":
        return DiskANNRS, (index_dir / "diskann_rs")
    if algo == "pq":
        return DiskANNRS_PQ, (index_dir / "diskann_rs_pq")
    if algo == "spherical":
        return DiskANNRS_Spherical, (index_dir / "diskann_rs_spherical")
    raise ValueError(f"unsupported algo={algo!r}")


def _default_diskann_config_yml() -> Path:
    # Resolve the workspace root that contains both DiskANN-playground/ and ann-benchmark-epeshared/.
    workspace_root = _find_workspace_root()
    return (
        workspace_root
        / "ann-benchmark-epeshared"
        / "ann_benchmarks"
        / "algorithms"
        / "diskann_rs"
        / "config.yml"
    )


def _load_run_group_preset(
    *,
    config_yml: Path,
    distance: str,
    run_group: str,
) -> dict[str, Any]:
    """Load params from ann-benchmarks algorithm config.yml.

    Returns a dict containing:
      - algo: one of fp|pq|spherical
      - params: dict to merge into the adapter param
      - l_search: optional int derived from query_args
    """
    cfg = yaml.safe_load(config_yml.read_text(encoding="utf-8"))
    float_cfg = (cfg or {}).get("float")
    if not isinstance(float_cfg, dict):
        raise ValueError(f"invalid config.yml (missing float: ...): {config_yml}")

    # ann-benchmarks uses distance keys angular/euclidean.
    dist_cfg = float_cfg.get(distance)
    if not isinstance(dist_cfg, list):
        raise ValueError(
            f"config.yml has no float/{distance} section (expected list): {config_yml}"
        )

    for entry in dist_cfg:
        if not isinstance(entry, dict):
            continue
        run_groups = entry.get("run_groups")
        if not isinstance(run_groups, dict):
            continue
        if run_group not in run_groups:
            continue

        rg = run_groups[run_group]
        if not isinstance(rg, dict):
            raise ValueError(f"invalid run_group shape for {run_group!r} in {config_yml}")

        args_list = rg.get("args")
        if not isinstance(args_list, list) or not args_list or not isinstance(args_list[0], dict):
            raise ValueError(f"run_group {run_group!r} is missing args: [{{...}}]")

        ctor = str(entry.get("constructor", ""))
        if ctor == "DiskANNRS":
            algo = "fp"
        elif ctor == "DiskANNRS_PQ":
            algo = "pq"
        elif ctor == "DiskANNRS_Spherical":
            algo = "spherical"
        else:
            raise ValueError(f"unsupported constructor in config.yml: {ctor!r}")

        # query_args in config.yml maps to adapter set_query_arguments(l_search).
        l_search: int | None = None
        qargs = rg.get("query_args")
        if isinstance(qargs, list) and qargs and isinstance(qargs[0], list) and qargs[0]:
            try:
                l_search = int(qargs[0][0])
            except Exception:
                l_search = None

        return {"algo": algo, "params": dict(args_list[0]), "l_search": l_search}

    raise ValueError(
        f"run_group {run_group!r} not found for float/{distance} in {config_yml}"
    )


def _ensure_dataset_symlink(*, work_dir: Path, dataset: str, hdf5_path: Path) -> Path:
    data_dir = work_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    dst = data_dir / f"{dataset}.hdf5"
    if dst.exists() or dst.is_symlink():
        try:
            dst.unlink()
        except Exception:
            pass
    dst.symlink_to(hdf5_path)
    return dst


def _load_hdf5_arrays(
    *,
    hdf5_path: Path,
    train_key: str,
    test_key: str,
    neighbors_key: str,
    need_train: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    with h5py.File(hdf5_path, "r") as f:
        distance = str(f.attrs.get("distance", "angular"))
        if f.attrs.get("type", "dense") == "sparse":
            # This harness currently assumes dense float32 matrices.
            # ann-benchmarks supports sparse datasets (list-of-arrays), but diskann-rs adapters here do not.
            raise ValueError(f"sparse HDF5 datasets are not supported by this harness: {hdf5_path}")

        if need_train:
            X_train_raw, X_test_raw = dataset_transform(f)
            X_train = np.asarray(X_train_raw, dtype=np.float32, order="C")
            X_test = np.asarray(X_test_raw, dtype=np.float32, order="C")
        else:
            # For search-only runs (index_action=load), we don't need the training vectors.
            # Avoid loading them to keep RSS representative of index+query, not dataset size.
            X_train = np.empty((0, 0), dtype=np.float32)
            X_test = np.asarray(np.array(f[test_key]), dtype=np.float32, order="C")
        neighbors = np.asarray(f[neighbors_key], dtype=np.int32, order="C")
    return X_train, X_test, neighbors, distance


def _outputs(work_dir: Path) -> StageOutputs:
    out_dir = work_dir / "outputs"
    return StageOutputs(
        build_json=out_dir / "output.build.json",
        search_json=out_dir / "output.search.json",
        summary_tsv=out_dir / "summary.tsv",
        details_md=out_dir / "details.md",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Run diskann-rs via ann-benchmarks framework (split build/search)")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--dataset", required=True)

    ap.add_argument(
        "--metric",
        default=None,
        choices=["angular", "euclidean", "cosine", "l2"],
        help="Override dataset distance/metric (default: use HDF5 attrs['distance'])",
    )

    ap.add_argument("--train-key", default="train")
    ap.add_argument("--test-key", default="test")
    ap.add_argument("--neighbors-key", default="neighbors")

    ap.add_argument(
        "--index-dir",
        default=None,
        help="Directory to store/reuse index files (default: <work-dir>/index)",
    )

    ap.add_argument("--stage", choices=["all", "build", "search"], default="all")
    ap.add_argument("--algo", choices=["fp", "pq", "spherical"], default="fp")

    ap.add_argument(
        "--config-yml",
        default=None,
        help="Path to ann-benchmarks diskann_rs/config.yml (only used with --run-group)",
    )
    ap.add_argument(
        "--run-group",
        default=None,
        help="Run-group name from diskann_rs/config.yml; fills l_build/max_outdegree/alpha/(pq|spherical) params",
    )

    ap.add_argument("--l-build", type=int, default=None)
    ap.add_argument("--max-outdegree", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None)

    ap.add_argument("--num-pq-chunks", type=int, default=None)
    ap.add_argument("--num-centers", type=int, default=256)
    ap.add_argument("--max-k-means-reps", type=int, default=12)
    ap.add_argument(
        "--translate-to-center",
        action="store_true",
        default=None,
        help="PQ only. If omitted, defaults to True for euclidean/l2 and False for angular/cosine.",
    )
    ap.add_argument("--no-translate-to-center", action="store_false", dest="translate_to_center")
    ap.add_argument("--pq-seed", type=int, default=0)

    ap.add_argument("--spherical-nbits", type=int, default=2, choices=[1, 2, 4])
    ap.add_argument("--spherical-seed", type=int, default=0)

    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--l-search", type=int, default=None)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--batch", action="store_true", help="Use ann-benchmarks batch_query path")

    args = ap.parse_args()

    work_dir = Path(args.work_dir).expanduser().resolve()
    hdf5_path = Path(args.hdf5).expanduser().resolve()
    dataset = str(args.dataset)

    if not hdf5_path.is_file():
        raise FileNotFoundError(str(hdf5_path))

    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "outputs").mkdir(parents=True, exist_ok=True)
    index_dir = (Path(args.index_dir).expanduser().resolve() if args.index_dir else (work_dir / "index"))
    index_dir.mkdir(parents=True, exist_ok=True)

    # Keep all framework relative paths contained under work_dir.
    os.chdir(work_dir)

    # Make the dataset available to ann_benchmarks.datasets.get_dataset() under ./data/<dataset>.hdf5
    _ensure_dataset_symlink(work_dir=work_dir, dataset=dataset, hdf5_path=hdf5_path)

    do_build = args.stage in ("all", "build")
    do_search = args.stage in ("all", "search")

    # Load raw arrays (also used for recall ground truth).
    # For search-only runs we intentionally avoid loading train vectors to keep RSS meaningful.
    t_load0 = time.perf_counter()
    X_train_raw, X_test_raw, neighbors, distance = _load_hdf5_arrays(
        hdf5_path=hdf5_path,
        train_key=args.train_key,
        test_key=args.test_key,
        neighbors_key=args.neighbors_key,
        need_train=do_build,
    )
    t_load1 = time.perf_counter()

    if args.metric is not None:
        metric = str(args.metric)
        if metric in {"cosine", "angular"}:
            distance = "angular"
        elif metric in {"l2", "euclidean"}:
            distance = "euclidean"
        else:
            distance = metric

    # Optional: load build/search params from ann-benchmarks config.yml run_group.
    if args.run_group is not None:
        config_yml = (
            Path(args.config_yml).expanduser().resolve()
            if args.config_yml
            else _default_diskann_config_yml()
        )
        preset = _load_run_group_preset(config_yml=config_yml, distance=distance, run_group=str(args.run_group))

        preset_algo = str(preset["algo"])
        if args.algo and args.algo != preset_algo:
            # Keep behavior predictable: preset decides algo type.
            args.algo = preset_algo
        else:
            args.algo = preset_algo

        p = dict(preset["params"])
        if args.l_build is None and "l_build" in p:
            args.l_build = int(p["l_build"])
        if args.max_outdegree is None and "max_outdegree" in p:
            args.max_outdegree = int(p["max_outdegree"])
        if args.alpha is None and "alpha" in p:
            args.alpha = float(p["alpha"])

        if args.algo == "pq" and args.num_pq_chunks is None and "num_pq_chunks" in p:
            args.num_pq_chunks = int(p["num_pq_chunks"])
        if args.algo == "pq" and args.translate_to_center is None and "translate_to_center" in p:
            v = p["translate_to_center"]
            if isinstance(v, bool):
                args.translate_to_center = v
            elif isinstance(v, str):
                vv = v.strip().lower()
                if vv in {"1", "true", "t", "yes", "y"}:
                    args.translate_to_center = True
                elif vv in {"0", "false", "f", "no", "n"}:
                    args.translate_to_center = False
            else:
                args.translate_to_center = bool(v)
        if args.algo == "spherical" and "nbits" in p:
            args.spherical_nbits = int(p["nbits"])

        if args.l_search is None and preset.get("l_search") is not None:
            args.l_search = int(preset["l_search"])

    # Fill remaining defaults (CLI values win over preset).
    if args.l_build is None:
        args.l_build = 64
    if args.max_outdegree is None:
        args.max_outdegree = 32
    if args.alpha is None:
        args.alpha = 1.2
    if args.l_search is None:
        args.l_search = 100

    if int(args.l_search) < int(args.k):
        print(
            f"WARN: l_search ({int(args.l_search)}) < k ({int(args.k)}); bumping l_search to {int(args.k)}",
            file=sys.stderr,
        )
        args.l_search = int(args.k)

    X_train = np.asarray(X_train_raw, dtype=np.float32, order="C")
    X_test = np.asarray(X_test_raw, dtype=np.float32, order="C")

    alg_cls, index_prefix = _algo_ctor_and_prefix(args.algo, index_dir)

    if args.algo == "pq" and args.num_pq_chunks is None:
        raise ValueError("--num-pq-chunks is required when --algo pq (or supply it via --run-group)")

    outputs = _outputs(work_dir)

    translate_to_center: bool | None = args.translate_to_center
    if args.algo == "pq" and translate_to_center is None:
        # PQ centering is compatible with L2. For cosine/angular, centering destroys
        # similarity semantics for unit-normalized vectors.
        translate_to_center = distance == "euclidean"

    # Build stage
    if do_build:
        param: dict[str, Any] = {
            "l_build": int(args.l_build),
            "max_outdegree": int(args.max_outdegree),
            "alpha": float(args.alpha),
            "index_action": "build_and_save",
            "index_prefix": str(index_prefix),
        }
        if args.algo == "pq":
            param.update(
                {
                    "num_pq_chunks": int(args.num_pq_chunks),
                    "num_centers": int(args.num_centers),
                    "max_k_means_reps": int(args.max_k_means_reps),
                    "translate_to_center": bool(translate_to_center),
                    "rng_seed": int(args.pq_seed),
                }
            )
        if args.algo == "spherical":
            param.update({"nbits": int(args.spherical_nbits), "rng_seed": int(args.spherical_seed)})

        algo = alg_cls(distance, param)
        try:
            t_fit0 = time.perf_counter()
            algo.fit(X_train)
            t_fit1 = time.perf_counter()
        finally:
            algo.done()

        build_obj: dict[str, Any] = {
            "algo": args.algo,
            "distance": distance,
            "l_build": int(args.l_build),
            "max_outdegree": int(args.max_outdegree),
            "alpha": float(args.alpha),
            "n_points": int(X_train.shape[0]),
            "dim": int(X_train.shape[1]),
            "load_s": float(t_load1 - t_load0),
            "fit_s": float(t_fit1 - t_fit0),
            "index_prefix": str(index_prefix),
        }
        if args.algo == "pq":
            build_obj.update(
                {
                    "num_pq_chunks": int(args.num_pq_chunks),
                    "num_centers": int(args.num_centers),
                    "max_k_means_reps": int(args.max_k_means_reps),
                    "translate_to_center": bool(translate_to_center),
                    "pq_seed": int(args.pq_seed),
                }
            )
        if args.algo == "spherical":
            build_obj.update({"nbits": int(args.spherical_nbits), "spherical_seed": int(args.spherical_seed)})
        _write_json(outputs.build_json, build_obj)

    # Search stage
    if do_search:
        param: dict[str, Any] = {
            "l_build": int(args.l_build),
            "max_outdegree": int(args.max_outdegree),
            "alpha": float(args.alpha),
            "index_action": "load",
            "index_prefix": str(index_prefix),
        }
        if args.algo == "pq":
            param.update(
                {
                    "num_pq_chunks": int(args.num_pq_chunks),
                    "num_centers": int(args.num_centers),
                    "max_k_means_reps": int(args.max_k_means_reps),
                    "translate_to_center": bool(translate_to_center),
                    "rng_seed": int(args.pq_seed),
                }
            )
        if args.algo == "spherical":
            param.update({"nbits": int(args.spherical_nbits), "rng_seed": int(args.spherical_seed)})

        algo = alg_cls(distance, param)
        try:
            # fit() will load the index. Also apply l_search.
            algo.set_query_arguments(int(args.l_search))
            t_load_index0 = time.perf_counter()
            algo.fit(X_train)
            t_load_index1 = time.perf_counter()

            with _PeakRssSampler() as rss_sampler:
                with _PeakDiskIoSampler() as disk_io_sampler:
                    latencies, ids_only = _run_queries(
                        algo=algo,
                        X_test=X_test,
                        k=int(args.k),
                        reps=int(args.reps),
                        batch=bool(args.batch),
                    )
        finally:
            algo.done()

        recall = _recall_at_k(ids_only, neighbors, int(args.k))

        mean_lat_s = float(sum(latencies) / max(1, len(latencies)))
        qps = float(1.0 / mean_lat_s) if mean_lat_s > 0 else 0.0

        peak_rss_gib: float | None = None
        if rss_sampler.peak_bytes is not None and rss_sampler.peak_bytes > 0:
            peak_rss_gib = float(rss_sampler.peak_bytes) / float(1024**3)

        search_obj: dict[str, Any] = {
            "algo": args.algo,
            "distance": distance,
            "l_build": int(args.l_build),
            "max_outdegree": int(args.max_outdegree),
            "alpha": float(args.alpha),
            "k": int(args.k),
            "l_search": int(args.l_search),
            "reps": int(args.reps),
            "batch": bool(args.batch),
            "n_queries": int(X_test.shape[0]),
            "recall_at_k": float(recall),
            "peak_rss_gib": (float(peak_rss_gib) if peak_rss_gib is not None else None),
            "peak_tps": (float(disk_io_sampler.peak_tps) if disk_io_sampler.peak_tps is not None else None),
            "peak_kB_read_s": (
                float(disk_io_sampler.peak_kb_read_s)
                if disk_io_sampler.peak_kb_read_s is not None
                else None
            ),
            "peak_kB_wrtn_s": (
                float(disk_io_sampler.peak_kb_wrtn_s)
                if disk_io_sampler.peak_kb_wrtn_s is not None
                else None
            ),
            "qps_mean": float(qps),
            "lat_mean_us": float(mean_lat_s * 1e6),
            "lat_p50_us": float(_pctl(latencies, 0.50) * 1e6),
            "lat_p95_us": float(_pctl(latencies, 0.95) * 1e6),
            "lat_p99_us": float(_pctl(latencies, 0.99) * 1e6),
            "index_prefix": str(index_prefix),
            "load_index_s": float(t_load_index1 - t_load_index0),
        }
        if args.algo == "pq":
            search_obj.update(
                {
                    "num_pq_chunks": int(args.num_pq_chunks),
                    "num_centers": int(args.num_centers),
                    "max_k_means_reps": int(args.max_k_means_reps),
                    "translate_to_center": bool(translate_to_center),
                    "pq_seed": int(args.pq_seed),
                }
            )
        if args.algo == "spherical":
            search_obj.update({"nbits": int(args.spherical_nbits), "spherical_seed": int(args.spherical_seed)})
        _write_json(outputs.search_json, search_obj)

        # Web-friendly summary
        headers = ["algo", "distance", "k", "l_search", "reps", "recall@k", "qps_mean", "lat_mean_us", "lat_p99_us"]
        row = [
            str(args.algo),
            str(distance),
            str(int(args.k)),
            str(int(args.l_search)),
            str(int(args.reps)),
            f"{recall:.4f}",
            f"{qps:.1f}",
            f"{search_obj['lat_mean_us']:.1f}",
            f"{search_obj['lat_p99_us']:.1f}",
        ]
        _write_text(outputs.summary_tsv, "\t".join(headers) + "\n" + "\t".join(row) + "\n")

        details = [
            "# diskann-ann-bench details",
            "",
            f"- algo: {args.algo}",
            f"- distance: {distance}",
            f"- k: {int(args.k)}",
            f"- l_search: {int(args.l_search)}",
            f"- reps: {int(args.reps)}",
            f"- batch: {bool(args.batch)}",
            f"- recall@k: {recall:.6f}",
            f"- peak_rss_gib (query only): {(peak_rss_gib if peak_rss_gib is not None else 'n/a')}",
            f"- peak_tps (query only): {(search_obj['peak_tps'] if search_obj.get('peak_tps') is not None else 'n/a')}",
            f"- peak_kB_read/s (query only): {(search_obj['peak_kB_read_s'] if search_obj.get('peak_kB_read_s') is not None else 'n/a')}",
            f"- peak_kB_wrtn/s (query only): {(search_obj['peak_kB_wrtn_s'] if search_obj.get('peak_kB_wrtn_s') is not None else 'n/a')}",
            f"- qps_mean: {qps:.3f}",
            f"- lat_mean_us: {search_obj['lat_mean_us']:.3f}",
            f"- lat_p99_us: {search_obj['lat_p99_us']:.3f}",
            "",
            "## Timings",
            "",
            f"- load_hdf5_s: {t_load1 - t_load0:.3f}",
            f"- load_index_s: {t_load_index1 - t_load_index0:.3f}",
        ]
        _write_text(outputs.details_md, "\n".join(details) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
