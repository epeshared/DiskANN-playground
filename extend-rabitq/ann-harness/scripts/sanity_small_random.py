#!/usr/bin/env python3

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def write_fbin(path: Path, x: np.ndarray) -> None:
    x = np.asarray(x, dtype=np.float32, order='C')
    if x.ndim != 2:
        raise ValueError('expected 2D array')
    n, d = x.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(struct.pack('<II', n, d))
        f.write(x.tobytes(order='C'))


def write_ibin(path: Path, x: np.ndarray) -> None:
    x = np.asarray(x, dtype=np.uint32, order='C')
    if x.ndim != 2:
        raise ValueError('expected 2D array')
    n, d = x.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(struct.pack('<II', n, d))
        f.write(x.tobytes(order='C'))


def cosine_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


def main() -> int:
    ap = argparse.ArgumentParser(description='Generate a tiny dataset + groundtruth to sanity-check diskann-benchmark PQ vs spherical')
    ap.add_argument('--out-dir', default='DiskANN-playground/extend-rabitq/ann-harness/sanity_data')
    ap.add_argument('--n-train', type=int, default=3000)
    ap.add_argument('--n-test', type=int, default=200)
    ap.add_argument('--dim', type=int, default=64)
    ap.add_argument('--k', type=int, default=50)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--config-out', default='DiskANN-playground/extend-rabitq/ann-harness/configs/sanity-pq-vs-spherical.json')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    rng = np.random.default_rng(args.seed)

    train = rng.standard_normal((args.n_train, args.dim), dtype=np.float32)
    test = rng.standard_normal((args.n_test, args.dim), dtype=np.float32)

    # angular ~ cosine normalized
    train_n = cosine_normalize(train)
    test_n = cosine_normalize(test)

    # brute force top-k via dot products
    sims = test_n @ train_n.T
    # largest similarity => nearest
    idx = np.argpartition(-sims, kth=args.k - 1, axis=1)[:, : args.k]
    # sort within top-k
    top_sims = np.take_along_axis(sims, idx, axis=1)
    order = np.argsort(-top_sims, axis=1)
    idx_sorted = np.take_along_axis(idx, order, axis=1)

    write_fbin(out_dir / 'train.fbin', train)
    write_fbin(out_dir / 'test.fbin', test)
    write_ibin(out_dir / 'neighbors.ibin', idx_sorted)

    # Make a minimal config directly (avoid dependency on config generator)
    cfg = {
        'search_directories': [str(out_dir)],
        'jobs': [
            {
                'type': 'async-index-build-pq',
                'content': {
                    'index_operation': {
                        'source': {
                            'index-source': 'Build',
                            'data_type': 'float32',
                            'data': 'train.fbin',
                            'distance': 'cosine',
                            'start_point_strategy': 'medoid',
                            'max_degree': 32,
                            'l_build': 64,
                            'alpha': 1.2,
                            'backedge_ratio': 1.0,
                            'num_threads': 8,
                            'multi_insert': None,
                            'save_path': None,
                        },
                        'search_phase': {
                            'search-type': 'topk',
                            'queries': 'test.fbin',
                            'groundtruth': 'neighbors.ibin',
                            'reps': 2,
                            'num_threads': [8],
                            'runs': [
                                {
                                    'search_n': args.k,
                                    'search_l': [args.k, max(args.k, 80), max(args.k, 120)],
                                    'recall_k': 10,
                                }
                            ],
                        },
                    },
                    'num_pq_chunks': 16,
                    'seed': 0xb578b71e688e65e3,
                    'max_fp_vecs_per_prune': 48,
                    'use_fp_for_search': False,
                },
            },
            {
                'type': 'async-index-build-spherical-quantization',
                'content': {
                    'index_operation': {
                        'source': {
                            'index-source': 'Build',
                            'data_type': 'float32',
                            'data': 'train.fbin',
                            'distance': 'cosine',
                            'start_point_strategy': 'medoid',
                            'max_degree': 32,
                            'l_build': 64,
                            'alpha': 1.2,
                            'backedge_ratio': 1.0,
                            'num_threads': 8,
                            'multi_insert': None,
                            'save_path': None,
                        },
                        'search_phase': {
                            'search-type': 'topk',
                            'queries': 'test.fbin',
                            'groundtruth': 'neighbors.ibin',
                            'reps': 2,
                            'num_threads': [8],
                            'runs': [
                                {
                                    'search_n': args.k,
                                    'search_l': [args.k, max(args.k, 80), max(args.k, 120)],
                                    'recall_k': 10,
                                }
                            ],
                        },
                    },
                    'seed': 0xc0ffee,
                    'transform_kind': {'padding_hadamard': 'same'},
                    'query_layouts': ['scalar_quantized', 'same_as_data', 'full_precision'],
                    'num_bits': 4,
                    'pre_scale': None,
                },
            },
        ],
    }

    config_out = Path(args.config_out)
    config_out.parent.mkdir(parents=True, exist_ok=True)
    config_out.write_text(json.dumps(cfg, indent=2) + '\n', encoding='utf-8')
    print('Wrote sanity dataset to:', out_dir)
    print('Wrote config to:', config_out)
    print('Next: run benchmark with scripts/run_benchmark.sh')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
