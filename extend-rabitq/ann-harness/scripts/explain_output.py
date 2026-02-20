#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def _us_to_s(us: int | float | None) -> float | None:
    if us is None:
        return None
    try:
        return float(us) / 1_000_000.0
    except Exception:
        return None


def _mean(xs):
    if xs is None:
        return None
    if not isinstance(xs, list):
        xs = [xs]
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


def _max(xs):
    if xs is None:
        return None
    if not isinstance(xs, list):
        xs = [xs]
    xs = [x for x in xs if isinstance(x, (int, float))]
    return max(xs) if xs else None


def _extract_disk_index_rows(res: dict) -> list[dict]:
    if not isinstance(res, dict):
        return []
    search = res.get('search')
    if not isinstance(search, dict):
        build = res.get('build')
        if isinstance(build, dict):
            search = build.get('search')
    if not isinstance(search, dict):
        return []

    num_threads = search.get('num_threads')
    beam_width = search.get('beam_width')
    recall_at = search.get('recall_at')
    reps = search.get('reps')

    results_per_l = search.get('search_results_per_l')
    if not isinstance(results_per_l, list):
        return []

    rows = []
    for r in results_per_l:
        if not isinstance(r, dict):
            continue
        rows.append(
            {
                'L': r.get('search_l'),
                'threads': num_threads,
                'beam_width': beam_width,
                'recall_at': recall_at,
                'reps': reps,
                'qps_mean': _mean(r.get('qps')),
                'qps_max': _max(r.get('qps')),
                'lat_mean_us_mean': _mean(r.get('mean_latency')),
                'lat_p999_us_mean': _mean(r.get('p999_latency')),
                'recall_mean': _mean(r.get('recall')),
            }
        )
    return rows


def _fmt(x, nd=3):
    if x is None:
        return 'NA'
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _extract_topk(container: dict) -> list[dict]:
    if not isinstance(container, dict):
        return []
    if isinstance(container.get('Topk'), list):
        return container['Topk']
    search = container.get('search')
    if isinstance(search, dict) and isinstance(search.get('Topk'), list):
        return search['Topk']
    return []


def _summarize_topk_rows(topk_rows: list[dict]) -> list[dict]:
    out = []
    for r in topk_rows:
        mean_lat = r.get('mean_latencies', [])
        p99_lat = r.get('p99_latencies', [])
        recall = r.get('recall', {}) or {}
        out.append(
            {
                'L': r.get('search_l'),
                'N': r.get('search_n'),
                'tasks': r.get('num_tasks'),
                'recall_avg': recall.get('average'),
                'qps_mean': _mean(r.get('qps', [])),
                'qps_max': _max(r.get('qps', [])),
                'lat_mean_us_mean': _mean(mean_lat),
                'lat_p99_us_mean': _mean(p99_lat),
                'mean_cmps': r.get('mean_cmps'),
                'mean_hops': r.get('mean_hops'),
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description='Explain diskann-benchmark output.json (build/training + run structure)')
    ap.add_argument('--input', required=True, help='Path to output.json')
    args = ap.parse_args()

    path = Path(args.input)
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, list):
        raise ValueError('expected output.json to be a list')

    print(f'# diskann-benchmark details')
    print(f'- input: {path}')
    print(f'- jobs: {len(data)}')
    print('')

    for idx, item in enumerate(data, start=1):
        inp = item.get('input', {}) or {}
        res = item.get('results', {}) or {}
        job_type = inp.get('type', 'unknown')

        print(f'## Job {idx}: {job_type}')

        # Input config highlights
        content = inp.get('content', {}) or {}
        if job_type == 'async-index-build-pq':
            print(f"- pq_chunks: {content.get('num_pq_chunks')}")
        if job_type == 'async-index-build-spherical-quantization':
            print(f"- num_bits: {content.get('num_bits')}")
            print(f"- query_layouts: {content.get('query_layouts')}")
            print(f"- transform_kind: {content.get('transform_kind')}")

        # Build stats
        if job_type == 'async-index-build-pq':
            build_stats = (res.get('build', {}) or {}).get('build', {}) or {}
        else:
            build_stats = res.get('build', {}) or {}

        total_time = build_stats.get('total_time')
        vectors_inserted = build_stats.get('vectors_inserted')
        kind = build_stats.get('kind')
        ins = build_stats.get('insert_latencies') or {}

        print('- build:')
        print(f"  - vectors_inserted: {vectors_inserted}")
        print(f"  - kind: {kind}")
        print(f"  - total_time_us: {total_time}")
        ts = _us_to_s(total_time)
        if ts is not None:
            print(f"  - total_time_s: {ts:.3f}")
        if isinstance(ins, dict) and ins:
            print(f"  - insert_latency_us: mean={ins.get('mean')} median={ins.get('median')} p90={ins.get('p90')} p99={ins.get('p99')}")

        # Training / quantization metadata
        if job_type == 'async-index-build-pq':
            qt = res.get('quant_training_time')
            print(f"- pq_quant_training_time_us: {qt}")
            qts = _us_to_s(qt)
            if qts is not None:
                print(f"- pq_quant_training_time_s: {qts:.3f}")
        if job_type == 'async-index-build-spherical-quantization':
            tt = res.get('training_time')
            print(f"- spherical_training_time_us: {tt}")
            tts = _us_to_s(tt)
            if tts is not None:
                print(f"- spherical_training_time_s: {tts:.3f}")
            print(f"- original_dim: {res.get('original_dim')}")
            print(f"- quantized_dim: {res.get('quantized_dim')}")
            print(f"- quantized_bytes_per_vector: {res.get('quantized_bytes')}")

        # Search results (Topk)
        print('- search_results:')
        if job_type == 'disk-index':
            rows = _extract_disk_index_rows(res)
            if not rows:
                print('  - <no disk-index rows found>')
            for row in rows:
                rep_txt = ''
                if row.get('reps') is not None:
                    rep_txt = f" reps={row.get('reps')}"
                print(
                    "  - "
                    + f"L={row['L']} threads={row['threads']} bw={row['beam_width']} recall_at={row['recall_at']}" + rep_txt + " "
                    + f"recall={_fmt(row['recall_mean'], nd=4)} qps_mean={_fmt(row['qps_mean'], nd=1)} qps_max={_fmt(row['qps_max'], nd=1)} "
                    + f"lat_mean_us={_fmt(row['lat_mean_us_mean'], nd=1)} lat_p999_us={_fmt(row['lat_p999_us_mean'], nd=1)}"
                )
            print('')
            continue
        if job_type == 'async-index-build-spherical-quantization' and isinstance(res.get('runs'), list):
            runs = res.get('runs')
            print(f"  - runs: {len(runs)}")
            for ridx, run in enumerate(runs, start=1):
                if not isinstance(run, dict):
                    continue
                layout = run.get('layout')
                run_res = run.get('results') or {}
                topk = _extract_topk(run_res)
                rows = _summarize_topk_rows(topk)

                print(f"  - run_{ridx}: layout={layout}")
                for row in rows:
                    print(
                        "    - "
                        + f"L={row['L']} N={row['N']} tasks={row['tasks']} "
                        + f"recall_avg={_fmt(row['recall_avg'], nd=4)} "
                        + f"qps_mean={_fmt(row['qps_mean'], nd=1)} "
                        + f"lat_mean_us={_fmt(row['lat_mean_us_mean'], nd=1)} "
                        + f"lat_p99_us={_fmt(row['lat_p99_us_mean'], nd=1)} "
                        + f"mean_cmps={_fmt(row['mean_cmps'], nd=1)} mean_hops={_fmt(row['mean_hops'], nd=1)}"
                    )
        else:
            # PQ and other jobs
            build = res.get('build') if isinstance(res, dict) else {}
            container = build.get('search') if isinstance(build, dict) and isinstance(build.get('search'), dict) else build
            topk = _extract_topk(container)
            rows = _summarize_topk_rows(topk)
            for row in rows:
                print(
                    "  - "
                    + f"L={row['L']} N={row['N']} tasks={row['tasks']} "
                    + f"recall_avg={_fmt(row['recall_avg'], nd=4)} "
                    + f"qps_mean={_fmt(row['qps_mean'], nd=1)} "
                    + f"lat_mean_us={_fmt(row['lat_mean_us_mean'], nd=1)} "
                    + f"lat_p99_us={_fmt(row['lat_p99_us_mean'], nd=1)} "
                    + f"mean_cmps={_fmt(row['mean_cmps'], nd=1)} mean_hops={_fmt(row['mean_hops'], nd=1)}"
                )

        print('')

    print('Notes:')
    print('- Latencies are reported in microseconds (us) in output.json.')
    print('- total_time_us / training_time_us fields are assumed to be microseconds based on magnitude and benchmark output.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
