#!/usr/bin/env python3

import argparse
import json
import struct
from pathlib import Path


def _read_fbin_dim(path: Path) -> int:
    with path.open("rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError(f"Invalid fbin header (too short): {path}")
        _npts, dim = struct.unpack("<II", header)
        return int(dim)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate diskann-benchmark config: disk index (build + search)")
    ap.add_argument("--data-dir", required=True, help="Directory containing train.fbin/test.fbin/neighbors.ibin")
    ap.add_argument("--out", required=True, help="Output JSON path")

    ap.add_argument(
        "--distance",
        default="cosine",
        choices=["squared_l2", "inner_product", "cosine", "cosine_normalized"],
        help="Distance metric; cosine_normalized is mapped to cosine for disk-index",
    )

    ap.add_argument("--max-degree", type=int, default=64)
    ap.add_argument("--l-build", type=int, default=128)
    ap.add_argument("--build-threads", type=int, default=32)

    ap.add_argument("--search-threads", type=int, default=32)
    ap.add_argument("--beam-width", type=int, default=16)
    ap.add_argument("--recall-k", type=int, default=10)
    ap.add_argument("--search-l", type=str, default="64,128,256,512", help="Comma-separated list")

    ap.add_argument("--pq-chunks", type=int, default=64)
    ap.add_argument("--build-ram-gb", type=float, default=16.0)
    ap.add_argument(
        "--is-flat-search",
        action="store_true",
        help="Use flat search on disk index (usually slower; for debugging)",
    )

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)

    train = data_dir / "train.fbin"
    test = data_dir / "test.fbin"
    gt = data_dir / "neighbors.ibin"

    for p in [train, test, gt]:
        if not p.is_file():
            raise FileNotFoundError(str(p))

    dim = _read_fbin_dim(train)

    search_l = [int(x) for x in args.search_l.split(",") if x.strip()]
    if not search_l:
        search_l = [64]

    # disk-index does not accept cosine_normalized; use cosine.
    distance = "cosine" if args.distance == "cosine_normalized" else args.distance

    if args.pq_chunks <= 0:
        raise ValueError("--pq-chunks must be > 0")

    build = {
        "data_type": "float32",
        "data": "train.fbin",
        "distance": distance,
        "dim": dim,
        "max_degree": args.max_degree,
        "l_build": args.l_build,
        "num_threads": args.build_threads,
        "build_ram_limit_gb": float(args.build_ram_gb),
        "num_pq_chunks": int(args.pq_chunks),
        # diskann-disk serializes QuantizationType as a string (e.g. "PQ_64").
        "quantization_type": f"PQ_{int(args.pq_chunks)}",
        "save_path": "",
    }

    search_phase = {
        "queries": "test.fbin",
        "groundtruth": "neighbors.ibin",
        "num_threads": int(args.search_threads),
        "beam_width": int(args.beam_width),
        "search_list": [int(x) for x in search_l],
        "recall_at": int(args.recall_k),
        "is_flat_search": bool(args.is_flat_search),
        "distance": distance,
        "vector_filters_file": None,
        "num_nodes_to_cache": None,
        "search_io_limit": None,
    }

    job = {
        "type": "disk-index",
        "content": {
            "source": {"disk-index-source": "Build", **build},
            "search_phase": search_phase,
        },
    }

    cfg = {
        "search_directories": [str(data_dir)],
        "jobs": [job],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print("Wrote:", str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
