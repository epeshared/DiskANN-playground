#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path


def _as_list(xs):
    if xs is None:
        return []
    if isinstance(xs, list):
        return xs
    return [xs]


def _mean(xs):
    xs = [x for x in _as_list(xs) if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


def _max(xs):
    xs = [x for x in _as_list(xs) if isinstance(x, (int, float))]
    return max(xs) if xs else None


def _fmt(x, nd=3):
    if x is None:
        return 'NA'
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return 'NA'
        return f"{x:.{nd}f}"
    return str(x)


def _extract_topk_rows(container):
    # Handle output shapes from different job types:
    # - PQ: results.build.search.Topk
    # - Spherical: results.runs[*].results.Topk
    # - Some shapes also use results.search.Topk or results.Topk

    if not isinstance(container, dict):
        return []

    # Direct Topk
    topk = container.get('Topk')
    if isinstance(topk, list):
        topk_list = topk
    else:
        search = container.get('search')
        if isinstance(search, dict) and isinstance(search.get('Topk'), list):
            topk_list = search.get('Topk')
        else:
            return []

    rows = []
    for r in topk_list:
        recall = r.get('recall', {}) if isinstance(r, dict) else {}

        mean_latencies = r.get('mean_latencies', []) if isinstance(r, dict) else []
        p99_latencies = r.get('p99_latencies', []) if isinstance(r, dict) else []
        rows.append(
            {
                'search_l': r.get('search_l'),
                'search_n': r.get('search_n'),
                'num_tasks': r.get('num_tasks'),
                'qps_mean': _mean(r.get('qps', [])) if isinstance(r, dict) else None,
                'qps_max': _max(r.get('qps', [])) if isinstance(r, dict) else None,
                # Latencies are reported in microseconds in diskann-benchmark output.
                'lat_mean_us_mean': _mean(mean_latencies),
                'lat_mean_us_max': _max(mean_latencies),
                'lat_p99_us_mean': _mean(p99_latencies),
                'lat_p99_us_max': _max(p99_latencies),
                'recall_avg': recall.get('average') if isinstance(recall, dict) else None,
                'recall_k': recall.get('recall_k') if isinstance(recall, dict) else None,
                'recall_n': recall.get('recall_n') if isinstance(recall, dict) else None,
            }
        )
    return rows


def _extract_disk_index_rows(item: dict):
    if not isinstance(item, dict):
        return []
    res = item.get('results', {})
    if not isinstance(res, dict):
        return []

    # DiskIndexStats is { build: <opt>, search: DiskSearchStats }
    search = res.get('search')
    if not isinstance(search, dict):
        # Be tolerant of alternate nesting.
        build = res.get('build')
        if isinstance(build, dict):
            search = build.get('search')
    if not isinstance(search, dict):
        return []

    num_threads = search.get('num_threads')
    recall_at = search.get('recall_at')
    results_per_l = search.get('search_results_per_l')
    if not isinstance(results_per_l, list):
        return []

    rows = []
    for r in results_per_l:
        if not isinstance(r, dict):
            continue
        rows.append(
            {
                'search_l': r.get('search_l'),
                # disk-index doesn't have search_n in its schema; use recall_at as a proxy.
                'search_n': recall_at,
                'num_tasks': num_threads,
                'qps_mean': _mean(r.get('qps')),
                'qps_max': _max(r.get('qps')),
                # Latencies are reported in microseconds.
                'lat_mean_us_mean': _mean(r.get('mean_latency')),
                'lat_mean_us_max': _max(r.get('mean_latency')),
                # Map p999_latency onto the existing p99 columns (best available).
                'lat_p99_us_mean': _mean(r.get('p999_latency')),
                'lat_p99_us_max': _max(r.get('p999_latency')),
                'recall_avg': _mean(r.get('recall')),
                'recall_k': recall_at,
                'recall_n': recall_at,
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description='Summarize diskann-benchmark output.json')
    ap.add_argument('--input', required=True)
    args = ap.parse_args()

    path = Path(args.input)
    data = json.loads(path.read_text(encoding='utf-8'))

    if not isinstance(data, list):
        raise ValueError('expected output to be a JSON list')

    # Print compact table
    headers = [
        'job',
        'detail',
        'tasks',
        'L',
        'N',
        'recall(avg)',
        'QPS(mean)',
        'QPS(max)',
        'lat_mean_us(mean)',
        'lat_mean_us(max)',
        'lat_p99_us(mean)',
        'lat_p99_us(max)',
    ]
    print('\t'.join(headers))

    for item in data:
        inp = item.get('input', {})
        res = item.get('results', {})
        job_type = inp.get('type', 'unknown')

        if job_type == 'disk-index':
            for row in _extract_disk_index_rows(item):
                print(
                    '\t'.join(
                        [
                            str(job_type),
                            str(''),
                            str(row.get('num_tasks')),
                            str(row.get('search_l')),
                            str(row.get('search_n')),
                            _fmt(row.get('recall_avg'), nd=4),
                            _fmt(row.get('qps_mean'), nd=1),
                            _fmt(row.get('qps_max'), nd=1),
                            _fmt(row.get('lat_mean_us_mean'), nd=1),
                            _fmt(row.get('lat_mean_us_max'), nd=1),
                            _fmt(row.get('lat_p99_us_mean'), nd=1),
                            _fmt(row.get('lat_p99_us_max'), nd=1),
                        ]
                    )
                )
            continue

        detail = ''
        if job_type == 'async-index-build-pq':
            chunks = inp.get('content', {}).get('num_pq_chunks')
            detail = f"pq_chunks={chunks}"
        elif job_type == 'async-index-build-spherical-quantization':
            content = inp.get('content', {})
            nb = content.get('num_bits')
            ql = content.get('query_layouts')
            detail = f"num_bits={nb}; layouts={ql}"

        if job_type == 'async-index-build-spherical-quantization' and isinstance(res, dict) and isinstance(res.get('runs'), list):
            for run in res.get('runs'):
                if not isinstance(run, dict):
                    continue
                layout = run.get('layout')
                run_results = run.get('results', {})
                run_detail = detail
                if layout is not None:
                    run_detail = (run_detail + f"; layout={layout}") if run_detail else f"layout={layout}"

                for row in _extract_topk_rows(run_results):
                    print(
                        '\t'.join(
                            [
                                str(job_type),
                                str(run_detail),
                                str(row.get('num_tasks')),
                                str(row.get('search_l')),
                                str(row.get('search_n')),
                                _fmt(row.get('recall_avg'), nd=4),
                                _fmt(row.get('qps_mean'), nd=1),
                                _fmt(row.get('qps_max'), nd=1),
                                _fmt(row.get('lat_mean_us_mean'), nd=1),
                                _fmt(row.get('lat_mean_us_max'), nd=1),
                                _fmt(row.get('lat_p99_us_mean'), nd=1),
                                _fmt(row.get('lat_p99_us_max'), nd=1),
                            ]
                        )
                    )
        else:
            # PQ and other jobs
            container = res.get('build') if isinstance(res, dict) and isinstance(res.get('build'), dict) else res
            for row in _extract_topk_rows(container):
                print(
                    '\t'.join(
                        [
                            str(job_type),
                            str(detail),
                            str(row.get('num_tasks')),
                            str(row.get('search_l')),
                            str(row.get('search_n')),
                            _fmt(row.get('recall_avg'), nd=4),
                            _fmt(row.get('qps_mean'), nd=1),
                            _fmt(row.get('qps_max'), nd=1),
                            _fmt(row.get('lat_mean_us_mean'), nd=1),
                            _fmt(row.get('lat_mean_us_max'), nd=1),
                            _fmt(row.get('lat_p99_us_mean'), nd=1),
                            _fmt(row.get('lat_p99_us_max'), nd=1),
                        ]
                    )
                )

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
