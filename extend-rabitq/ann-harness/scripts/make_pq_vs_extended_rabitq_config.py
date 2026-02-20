#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description='Generate diskann-benchmark config: PQ vs spherical (Extended RaBitQ)')
    ap.add_argument('--data-dir', required=True, help='Directory containing train.fbin/test.fbin/neighbors.ibin')
    ap.add_argument('--out', required=True, help='Output JSON path')

    ap.add_argument('--distance', default='cosine', choices=['squared_l2', 'inner_product', 'cosine', 'cosine_normalized'])

    ap.add_argument('--max-degree', type=int, default=64)
    ap.add_argument('--l-build', type=int, default=128)
    ap.add_argument('--alpha', type=float, default=1.2)
    ap.add_argument('--backedge-ratio', type=float, default=1.0)

    ap.add_argument('--build-threads', type=int, default=32)
    ap.add_argument('--search-threads', type=int, default=32)

    ap.add_argument(
        '--loop',
        '--reps',
        dest='reps',
        type=int,
        default=3,
        help='How many times to repeat the full query set search (alias: --reps)',
    )
    ap.add_argument('--search-n', type=int, default=100)
    ap.add_argument('--recall-k', type=int, default=10)
    ap.add_argument('--search-l', type=str, default='50,100,200,400', help='Comma-separated list')

    ap.add_argument('--pq-chunks', type=int, default=64)

    ap.add_argument('--spherical-num-bits', type=int, default=4, choices=[1, 2, 4, 8])
    ap.add_argument('--transform-kind', default='padding_hadamard', choices=['padding_hadamard', 'random_rotation', 'double_hadamard'])
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)

    train = data_dir / 'train.fbin'
    test = data_dir / 'test.fbin'
    gt = data_dir / 'neighbors.ibin'

    for p in [train, test, gt]:
        if not p.is_file():
            raise FileNotFoundError(str(p))

    search_l_raw = [int(x) for x in args.search_l.split(',') if x.strip()]
    search_l = [x for x in search_l_raw if x >= args.search_n]
    if not search_l:
        search_l = [args.search_n]

    # diskann-benchmark uses snake_case keys and tagged enums.
    # Spherical quantization does not support cosine_normalized.
    distance_pq = args.distance
    distance_spherical = 'cosine' if args.distance == 'cosine_normalized' else args.distance

    # Async graph build (in-memory)
    base_index_build = {
        'data_type': 'float32',
        'data': 'train.fbin',
        'distance': distance_pq,
        'start_point_strategy': 'medoid',
        'max_degree': args.max_degree,
        'l_build': args.l_build,
        'alpha': args.alpha,
        'backedge_ratio': args.backedge_ratio,
        'num_threads': args.build_threads,
        'multi_insert': None,
        'save_path': None,
    }

    search_phase = {
        'search-type': 'topk',
        'queries': 'test.fbin',
        'groundtruth': 'neighbors.ibin',
        'reps': args.reps,
        'num_threads': [args.search_threads],
        'runs': [
            {
                'search_n': args.search_n,
                'search_l': search_l,
                'recall_k': args.recall_k,
            }
        ],
    }

    pq_job = {
        'type': 'async-index-build-pq',
        'content': {
            'index_operation': {
                'source': {
                    'index-source': 'Build',
                    **base_index_build,
                },
                'search_phase': search_phase,
            },
            'num_pq_chunks': args.pq_chunks,
            'seed': 0xb578b71e688e65e3,
            'max_fp_vecs_per_prune': 48,
            'use_fp_for_search': False,
        },
    }

    if args.transform_kind == 'padding_hadamard':
        transform_kind = {'padding_hadamard': 'same'}
    elif args.transform_kind == 'random_rotation':
        transform_kind = {'random_rotation': {'seed': 0xc0ffee}}
    else:
        transform_kind = {'double_hadamard': 'same'}

    # query_layout compatibility depends on num_bits
    if args.spherical_num_bits == 1:
        query_layouts = ['four_bit_transposed', 'same_as_data', 'full_precision']
    else:
        query_layouts = ['scalar_quantized', 'same_as_data', 'full_precision']

    spherical_job = {
        'type': 'async-index-build-spherical-quantization',
        'content': {
            'index_operation': {
                'source': {
                    'index-source': 'Build',
                    **{**base_index_build, 'distance': distance_spherical},
                },
                'search_phase': search_phase,
            },
            'seed': 0xc0ffee,
            'transform_kind': transform_kind,
            'query_layouts': query_layouts,
            'num_bits': args.spherical_num_bits,
            'pre_scale': None,
        },
    }

    cfg = {
        'search_directories': [str(data_dir)],
        'jobs': [pq_job, spherical_job],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cfg, indent=2) + '\n', encoding='utf-8')
    print('Wrote:', str(out_path))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
