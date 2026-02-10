#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], *, cwd: Path | None = None, capture: bool = False) -> subprocess.CompletedProcess:
    print('+', ' '.join(cmd))
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        check=True,
        text=True,
        capture_output=capture,
    )


def main() -> int:
    here = Path(__file__).resolve()
    scripts_dir = here.parent
    harness_dir = scripts_dir.parent

    ap = argparse.ArgumentParser(
        description=(
            'One-command pipeline: convert ann-benchmarks HDF5 -> DiskANN bin, '
            'generate PQ vs spherical config, run diskann-benchmark, summarize results.'
        )
    )
    ap.add_argument('--hdf5', required=True, help='Path to ann-benchmarks-style .hdf5 file')
    ap.add_argument('--dataset', default=None, help='Dataset name for output folder (default: from hdf5 filename)')
    ap.add_argument(
        '--work-dir',
        default=None,
        help='Working directory to place data/configs/outputs (default: ann-harness/runs/<dataset>)',
    )

    # HDF5 keys
    ap.add_argument('--train-key', default='train')
    ap.add_argument('--test-key', default='test')
    ap.add_argument('--neighbors-key', default='neighbors')
    ap.add_argument('--chunk-rows', type=int, default=4096)

    # Config knobs (subset; extend as needed)
    ap.add_argument(
        '--distance',
        default='cosine',
        choices=['squared_l2', 'inner_product', 'cosine', 'cosine_normalized'],
        help='Distance metric for PQ; spherical will downgrade cosine_normalized -> cosine',
    )
    ap.add_argument('--max-degree', type=int, default=64)
    ap.add_argument('--l-build', type=int, default=128)
    ap.add_argument('--alpha', type=float, default=1.2)
    ap.add_argument('--backedge-ratio', type=float, default=1.0)
    ap.add_argument('--build-threads', type=int, default=32)
    ap.add_argument('--search-threads', type=int, default=32)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--search-n', type=int, default=100)
    ap.add_argument('--recall-k', type=int, default=10)
    ap.add_argument('--search-l', type=str, default='50,100,200,400')
    ap.add_argument('--pq-chunks', type=int, default=64)
    ap.add_argument('--spherical-num-bits', type=int, default=4, choices=[1, 2, 4, 8])
    ap.add_argument(
        '--transform-kind',
        default='padding_hadamard',
        choices=['padding_hadamard', 'random_rotation', 'double_hadamard'],
    )

    ap.add_argument('--skip-convert', action='store_true', help='Assume work-dir/data already exists')
    ap.add_argument('--skip-run', action='store_true', help='Only generate config (and maybe convert)')
    args = ap.parse_args()

    hdf5_path = Path(args.hdf5).resolve()
    if not hdf5_path.is_file():
        raise FileNotFoundError(str(hdf5_path))

    dataset = args.dataset or hdf5_path.stem

    work_dir = Path(args.work_dir) if args.work_dir else (harness_dir / 'runs' / dataset)
    work_dir = work_dir.resolve()

    data_dir = work_dir / 'data'
    config_dir = work_dir / 'configs'
    outputs_dir = work_dir / 'outputs'

    config_path = config_dir / 'pq-vs-spherical.json'
    output_json = outputs_dir / 'output.json'
    summary_tsv = outputs_dir / 'summary.tsv'
    details_md = outputs_dir / 'details.md'

    data_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    if not args.skip_convert:
        run(
            [
                py,
                str(scripts_dir / 'convert_hdf5_to_diskann_bin.py'),
                '--hdf5',
                str(hdf5_path),
                '--out-dir',
                str(data_dir),
                '--train-key',
                args.train_key,
                '--test-key',
                args.test_key,
                '--neighbors-key',
                args.neighbors_key,
                '--chunk-rows',
                str(int(args.chunk_rows)),
            ]
        )

    run(
        [
            py,
            str(scripts_dir / 'make_pq_vs_extended_rabitq_config.py'),
            '--data-dir',
            str(data_dir),
            '--out',
            str(config_path),
            '--distance',
            args.distance,
            '--max-degree',
            str(int(args.max_degree)),
            '--l-build',
            str(int(args.l_build)),
            '--alpha',
            str(float(args.alpha)),
            '--backedge-ratio',
            str(float(args.backedge_ratio)),
            '--build-threads',
            str(int(args.build_threads)),
            '--search-threads',
            str(int(args.search_threads)),
            '--reps',
            str(int(args.reps)),
            '--search-n',
            str(int(args.search_n)),
            '--recall-k',
            str(int(args.recall_k)),
            '--search-l',
            args.search_l,
            '--pq-chunks',
            str(int(args.pq_chunks)),
            '--spherical-num-bits',
            str(int(args.spherical_num_bits)),
            '--transform-kind',
            args.transform_kind,
        ]
    )

    if args.skip_run:
        print('Skip run; config written to:', str(config_path))
        return 0

    run([
        'bash',
        str(scripts_dir / 'run_benchmark.sh'),
        str(config_path),
        str(output_json),
    ])

    proc = run([py, str(scripts_dir / 'summarize_output.py'), '--input', str(output_json)], capture=True)
    summary_tsv.write_text(proc.stdout, encoding='utf-8')

    details = run([py, str(scripts_dir / 'explain_output.py'), '--input', str(output_json)], capture=True)
    details_md.write_text(details.stdout, encoding='utf-8')

    print('Wrote output:', str(output_json))
    print('Wrote summary:', str(summary_tsv))
    print('Wrote details:', str(details_md))
    print(proc.stdout)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
