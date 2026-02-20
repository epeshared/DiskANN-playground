#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate diskann-benchmark config: async full-precision in-memory index")
    ap.add_argument("--data-dir", required=True, help="Directory containing train.fbin/test.fbin/neighbors.ibin")
    ap.add_argument("--out", required=True, help="Output JSON path")

    ap.add_argument(
        "--distance",
        default="cosine",
        choices=["squared_l2", "inner_product", "cosine", "cosine_normalized"],
    )

    ap.add_argument("--max-degree", type=int, default=64)
    ap.add_argument("--l-build", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=1.2)
    ap.add_argument("--backedge-ratio", type=float, default=1.0)

    ap.add_argument("--build-threads", type=int, default=32)
    ap.add_argument("--search-threads", type=int, default=32)

    ap.add_argument(
        "--loop",
        "--reps",
        dest="reps",
        type=int,
        default=3,
        help="How many times to repeat the full query set search (alias: --reps)",
    )
    ap.add_argument("--search-n", type=int, default=100)
    ap.add_argument("--recall-k", type=int, default=10)
    ap.add_argument("--search-l", type=str, default="50,100,200,400", help="Comma-separated list")

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)

    train = data_dir / "train.fbin"
    test = data_dir / "test.fbin"
    gt = data_dir / "neighbors.ibin"

    for p in [train, test, gt]:
        if not p.is_file():
            raise FileNotFoundError(str(p))

    search_l_raw = [int(x) for x in args.search_l.split(",") if x.strip()]
    search_l = [x for x in search_l_raw if x >= args.search_n]
    if not search_l:
        search_l = [args.search_n]

    # cosine_normalized is only supported in PQ path; full precision uses cosine.
    distance_fp = "cosine" if args.distance == "cosine_normalized" else args.distance

    index_build = {
        "data_type": "float32",
        "data": "train.fbin",
        "distance": distance_fp,
        "start_point_strategy": "medoid",
        "max_degree": args.max_degree,
        "l_build": args.l_build,
        "alpha": args.alpha,
        "backedge_ratio": args.backedge_ratio,
        "num_threads": args.build_threads,
        "multi_insert": None,
        "save_path": None,
    }

    search_phase = {
        "search-type": "topk",
        "queries": "test.fbin",
        "groundtruth": "neighbors.ibin",
        "reps": args.reps,
        "num_threads": [args.search_threads],
        "runs": [
            {
                "search_n": args.search_n,
                "search_l": search_l,
                "recall_k": args.recall_k,
            }
        ],
    }

    job = {
        "type": "async-index-build",
        "content": {
            "source": {"index-source": "Build", **index_build},
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
