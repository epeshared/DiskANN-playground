#!/usr/bin/env python3

import argparse
import copy
import json
import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import shutil


def run(cmd: list[str], *, cwd: Path | None = None, capture: bool = False) -> subprocess.CompletedProcess:
    print('+', ' '.join(cmd))
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        check=True,
        text=True,
        capture_output=capture,
    )


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def write_server_info(out_path: Path, *, run_id: str) -> None:
    lines: list[str] = []
    lines.append(f"run_id: {run_id}")
    try:
        proc = subprocess.run(['uname', '-a'], check=True, text=True, capture_output=True)
        lines.append('uname -a:')
        lines.append(proc.stdout.strip())
    except Exception as e:  # noqa: BLE001
        lines.append(f"uname -a: <failed: {e}>")

    lines.append('')
    lines.append('lscpu:')
    try:
        proc = subprocess.run(['lscpu'], check=True, text=True, capture_output=True)
        lines.append(proc.stdout.rstrip())
        if proc.stderr.strip():
            lines.append('')
            lines.append('lscpu stderr:')
            lines.append(proc.stderr.rstrip())
    except FileNotFoundError:
        lines.append('<lscpu not found on PATH>')
    except subprocess.CalledProcessError as e:
        lines.append(f"<lscpu failed: exit={e.returncode}>")
        if e.stdout:
            lines.append('')
            lines.append('stdout:')
            lines.append(e.stdout.rstrip())
        if e.stderr:
            lines.append('')
            lines.append('stderr:')
            lines.append(e.stderr.rstrip())

    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def stable_json_hash(obj: object) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha1(payload).hexdigest()[:10]


def parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in ('1', 'true', 't', 'yes', 'y', 'on'):
        return True
    if v in ('0', 'false', 'f', 'no', 'n', 'off'):
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got: {value!r}")


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
        help=(
            'Working directory to place data/configs/outputs. '
            'If omitted, writes to ann-harness/runs/<dataset>/<timestamp>/ (timestamp is UTC).'
        ),
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
    ap.add_argument(
        '--emon-enable',
        nargs='?',
        const=True,
        default=False,
        type=parse_bool,
        help=(
            'Enable Intel EMON collection around SEARCH ONLY (per slice). '
            'Accepts optional boolean value (e.g. --emon-enable true/false). '
            'Runs a build-only benchmark first (saving indices), then splits the search sweep so that '
            'each (threads, N, L) point is run in its own invocation; for spherical, each layout is also '
            'run separately. For each slice: starts `emon -collect-edp`, runs one-job load+search, then '
            'runs `emon -stop`. Writes outputs under <work-dir>/emon/<slice>/emon.dat.'
        ),
    )
    ap.add_argument(
        '--rebuild-index',
        nargs='?',
        const=True,
        default=False,
        type=parse_bool,
        help=(
            'Force rebuild of cached indices. Accepts optional boolean value (e.g. --rebuild-index true/false). '
            'Default is to REUSE cached indices when present: '
            'if a job\'s index artifacts already exist on disk, it will load them; otherwise it will build '
            'and save them. If enabled, existing cached index artifacts for this config '
            'will be deleted before rebuilding and saving again.'
        ),
    )
    args = ap.parse_args()

    hdf5_path = Path(args.hdf5).resolve()
    if not hdf5_path.is_file():
        raise FileNotFoundError(str(hdf5_path))

    dataset = args.dataset or hdf5_path.stem

    run_id = utc_timestamp()

    if args.work_dir:
        work_dir = Path(args.work_dir).resolve()
    else:
        dataset_root = (harness_dir / 'runs' / dataset).resolve()
        dataset_root.mkdir(parents=True, exist_ok=True)
        ts = run_id
        work_dir = dataset_root / ts
        # Avoid collisions if multiple runs start within the same second.
        if work_dir.exists():
            i = 1
            while (dataset_root / f"{ts}-{i}").exists():
                i += 1
            run_id = f"{ts}-{i}"
            work_dir = dataset_root / run_id

    data_dir = work_dir / 'data'
    config_dir = work_dir / 'configs'
    outputs_dir = work_dir / 'outputs'

    work_dir.mkdir(parents=True, exist_ok=True)
    write_server_info(work_dir / 'server-info.txt', run_id=run_id)

    config_path = config_dir / 'pq-vs-spherical.json'
    output_json = outputs_dir / 'output.json'
    output_json_build = outputs_dir / 'output.build.json'
    output_json_search = outputs_dir / 'output.search.json'
    summary_tsv = outputs_dir / 'summary.tsv'
    summary_tsv_build = outputs_dir / 'summary.build.tsv'
    summary_tsv_search = outputs_dir / 'summary.search.tsv'
    details_md = outputs_dir / 'details.md'
    details_md_build = outputs_dir / 'details.build.md'
    details_md_search = outputs_dir / 'details.search.md'

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

    emon_proc: subprocess.Popen[str] | None = None
    emon_stdout = None
    emon_dir = work_dir / 'emon'

    # Persistent index cache root so repeated runs (new timestamps) can reuse.
    # NOTE: this is intentionally outside work_dir.
    index_cache_root = (harness_dir / 'index-cache' / dataset).resolve()
    index_cache_root.mkdir(parents=True, exist_ok=True)

    def _clear_search_runs(cfg: dict) -> None:
        """Best-effort: make config effectively build-only by clearing run sweeps."""
        for job in cfg.get('jobs', []):
            content = job.get('content', {})
            op = content.get('index_operation')
            if not isinstance(op, dict):
                continue
            phase = op.get('search_phase')
            if isinstance(phase, dict) and 'runs' in phase:
                phase['runs'] = []

    def _job_save_prefix(job: dict, *, job_index: int) -> Path:
        """Compute a stable save prefix for a job under the persistent index cache."""
        job_type = job.get('type', f'job{job_index}')
        content = job.get('content', {})
        op = content.get('index_operation')
        source = None
        if isinstance(op, dict):
            source = op.get('source')
        if not isinstance(source, dict):
            # Fall back to something stable-ish.
            key_obj = {'type': job_type}
        else:
            key_obj = copy.deepcopy(source)
            key_obj.pop('save_path', None)
            key_obj.pop('load_path', None)
        key = stable_json_hash({'type': job_type, 'source': key_obj})
        return (index_cache_root / f"{job_index:02d}-{job_type}-{key}" / 'index').resolve()

    def _index_marker(save_prefix: Path) -> Path:
        return save_prefix.parent / 'BUILD_OK'

    def _index_exists(save_prefix: Path) -> bool:
        marker = _index_marker(save_prefix)
        if not marker.is_file():
            return False
        # Best-effort: ensure there is at least one artifact besides the marker.
        try:
            for p in save_prefix.parent.iterdir():
                if p.name == marker.name:
                    continue
                if p.is_file():
                    return True
        except FileNotFoundError:
            return False
        return False

    def _split_build_and_search_configs(cfg: dict) -> tuple[dict, dict, list[Path]]:
        build_cfg = copy.deepcopy(cfg)
        search_cfg = copy.deepcopy(cfg)

        _clear_search_runs(build_cfg)

        save_prefixes: list[Path] = []

        jobs = cfg.get('jobs', [])
        for i, job in enumerate(jobs):
            job_type = job.get('type', f'job{i}')
            # Put each job's index artifacts into its own stable directory (across runs)
            # and also avoid diskann-benchmark's output directory collision checks.
            save_prefix = _job_save_prefix(job, job_index=i)
            save_prefix.parent.mkdir(parents=True, exist_ok=True)
            save_prefixes.append(save_prefix)

            build_content = build_cfg['jobs'][i]['content']
            build_op = build_content.get('index_operation')
            if not isinstance(build_op, dict):
                continue

            build_source = build_op.get('source')
            if not isinstance(build_source, dict):
                continue

            # Force save_path for the build phase.
            build_source['save_path'] = str(save_prefix)

            # Configure the search phase to load from the build artifacts.
            distance = build_source.get('distance')
            data_type = build_source.get('data_type')

            search_content = search_cfg['jobs'][i]['content']
            search_op = search_content.get('index_operation')
            if not isinstance(search_op, dict):
                continue

            search_op['source'] = {
                'index-source': 'Load',
                'data_type': data_type,
                'distance': distance,
                'load_path': str(save_prefix),
            }

        return build_cfg, search_cfg, save_prefixes

    def _config_with_only_job(cfg: dict, job_index: int) -> dict:
        """Return a shallow copy of config containing only jobs[job_index]."""
        out = {k: copy.deepcopy(v) for k, v in cfg.items() if k != 'jobs'}
        out['jobs'] = [copy.deepcopy(cfg['jobs'][job_index])]
        return out

    def _split_search_phase_into_slices(search_phase: dict) -> list[tuple[str, dict]]:
        """Split a search_phase dict into independent slices.

        Each slice attempts to isolate a single measurement point by narrowing:
        - num_threads -> one value
        - runs -> one run
        - (Topk/Beta/MultiHop) search_l -> one value
        - (Range) initial_search_l -> one value

        Returns list of (slug, phase_dict).
        """
        if not isinstance(search_phase, dict):
            return [("phase", copy.deepcopy(search_phase))]

        stype = search_phase.get('search-type', 'unknown')
        num_threads = search_phase.get('num_threads')
        if isinstance(num_threads, list) and num_threads:
            thread_values = list(num_threads)
        else:
            thread_values = [None]

        runs = search_phase.get('runs')
        if not isinstance(runs, list) or not runs:
            # build-only / empty sweep
            phase = copy.deepcopy(search_phase)
            if thread_values != [None]:
                # keep as-is
                pass
            return [(f"{stype}", phase)]

        slices: list[tuple[str, dict]] = []

        def _with_threads(phase: dict, t) -> dict:
            out = copy.deepcopy(phase)
            if t is not None:
                out['num_threads'] = [t]
            return out

        if stype in ('topk', 'topk-beta-filter', 'topk-multihop-filter'):
            for t in thread_values:
                for run in runs:
                    if not isinstance(run, dict):
                        continue
                    search_n = run.get('search_n')
                    search_l_list = run.get('search_l')
                    if isinstance(search_l_list, list) and search_l_list:
                        ls = list(search_l_list)
                    else:
                        ls = [None]

                    for l in ls:
                        phase = _with_threads(search_phase, t)
                        run_one = copy.deepcopy(run)
                        if l is not None:
                            run_one['search_l'] = [l]
                        phase['runs'] = [run_one]
                        slug_parts = [stype]
                        if t is not None:
                            slug_parts.append(f"t{t}")
                        if search_n is not None:
                            slug_parts.append(f"n{search_n}")
                        if l is not None:
                            slug_parts.append(f"l{l}")
                        slices.append(('-'.join(slug_parts), phase))

        elif stype == 'range':
            for t in thread_values:
                for run in runs:
                    if not isinstance(run, dict):
                        continue
                    isl_list = run.get('initial_search_l')
                    if isinstance(isl_list, list) and isl_list:
                        isls = list(isl_list)
                    else:
                        isls = [None]

                    for isl in isls:
                        phase = _with_threads(search_phase, t)
                        run_one = copy.deepcopy(run)
                        if isl is not None:
                            run_one['initial_search_l'] = [isl]
                        phase['runs'] = [run_one]
                        slug_parts = [stype]
                        if t is not None:
                            slug_parts.append(f"t{t}")
                        if isl is not None:
                            slug_parts.append(f"isl{isl}")
                        slices.append(('-'.join(slug_parts), phase))

        else:
            # Unknown search type; at least split by thread and run index.
            for t in thread_values:
                for run_i, run in enumerate(runs):
                    phase = _with_threads(search_phase, t)
                    phase['runs'] = [copy.deepcopy(run)]
                    slug_parts = [stype]
                    if t is not None:
                        slug_parts.append(f"t{t}")
                    slug_parts.append(f"run{run_i}")
                    slices.append(('-'.join(slug_parts), phase))

        return slices

    try:
        if args.emon_enable and shutil.which('emon') is None:
            raise FileNotFoundError(
                "`emon` not found on PATH. If Intel SEP is installed, try `source /opt/intel/sep/sep_vars.sh`."
            )

        base_cfg = json.loads(config_path.read_text(encoding='utf-8'))
        build_cfg, search_cfg, save_prefixes = _split_build_and_search_configs(base_cfg)

        build_config_path = config_dir / 'pq-vs-spherical.build.json'
        search_config_path = config_dir / 'pq-vs-spherical.search.json'
        build_config_path.write_text(json.dumps(build_cfg, indent=2) + '\n', encoding='utf-8')
        search_config_path.write_text(json.dumps(search_cfg, indent=2) + '\n', encoding='utf-8')

        # Decide which indices need to be (re)built.
        jobs_to_build: list[int] = []
        for i, save_prefix in enumerate(save_prefixes):
            if args.rebuild_index:
                # Delete cached artifacts for this job/config, then rebuild.
                if save_prefix.parent.exists():
                    shutil.rmtree(save_prefix.parent, ignore_errors=True)
                save_prefix.parent.mkdir(parents=True, exist_ok=True)
                jobs_to_build.append(i)
            else:
                if not _index_exists(save_prefix):
                    jobs_to_build.append(i)

        # Build missing indices (no EMON)
        aggregated_build_results: list[dict] = []
        if jobs_to_build:
            for i in jobs_to_build:
                job_type = base_cfg.get('jobs', [])[i].get('type', f'job{i}')
                job_slug = f"{i:02d}-{job_type}"
                job_build_cfg = _config_with_only_job(build_cfg, i)
                job_build_config_path = config_dir / f"pq-vs-spherical.build.{job_slug}.json"
                job_build_config_path.write_text(
                    json.dumps(job_build_cfg, indent=2) + '\n',
                    encoding='utf-8',
                )

                job_output_json_build = outputs_dir / f"output.build.{job_slug}.json"
                run([
                    'bash',
                    str(scripts_dir / 'run_benchmark.sh'),
                    str(job_build_config_path),
                    str(job_output_json_build),
                ])

                # Mark index as built for reuse.
                marker = _index_marker(save_prefixes[i])
                marker.write_text('ok\n', encoding='utf-8')

                job_results = json.loads(job_output_json_build.read_text(encoding='utf-8'))
                if not isinstance(job_results, list):
                    raise ValueError(
                        f"Unexpected build output JSON shape for {job_slug}: expected list"
                    )
                aggregated_build_results.extend(job_results)
        else:
            print('All cached indices present; skipping build phase.')

        output_json_build.write_text(
            json.dumps(aggregated_build_results, indent=2) + '\n',
            encoding='utf-8',
        )

        if not args.emon_enable:
            # Search (no EMON) - single invocation using Load source.
            run([
                'bash',
                str(scripts_dir / 'run_benchmark.sh'),
                str(search_config_path),
                str(output_json_search),
            ])
        else:
            # Search (EMON wraps each slice; 1 slice -> 1 emon.dat)
            aggregated_search_results: list[dict] = []
            for i, job in enumerate(search_cfg.get('jobs', [])):
                job_type = job.get('type', f'job{i}')
                job_slug = f"{i:02d}-{job_type}"

                base_job_config = _config_with_only_job(search_cfg, i)
                base_content = base_job_config['jobs'][0].get('content', {})

                # Layout slicing: if spherical, run one layout per invocation.
                layouts = base_content.get('query_layouts')
                if isinstance(layouts, list) and layouts:
                    layout_values = list(layouts)
                else:
                    layout_values = [None]

                # L/N/thread slicing: split search_phase into independent slices.
                op = base_content.get('index_operation', {})
                phase = op.get('search_phase', {})
                phase_slices = _split_search_phase_into_slices(phase)

                for layout in layout_values:
                    for phase_slug, phase_one in phase_slices:
                        sliced_config = copy.deepcopy(base_job_config)
                        sliced_content = sliced_config['jobs'][0]['content']

                        if layout is not None:
                            sliced_content['query_layouts'] = [layout]
                            layout_slug = f"layout-{layout}"
                        else:
                            layout_slug = None

                        sliced_content['index_operation']['search_phase'] = phase_one

                        slug_parts = [job_slug]
                        if layout_slug is not None:
                            slug_parts.append(layout_slug)
                        slug_parts.append(phase_slug)
                        slice_slug = '.'.join(slug_parts)

                        job_config_path = config_dir / f"pq-vs-spherical.search.{slice_slug}.json"
                        job_config_path.write_text(
                            json.dumps(sliced_config, indent=2) + '\n',
                            encoding='utf-8',
                        )

                        job_output_json = outputs_dir / f"output.search.{slice_slug}.json"
                        job_emon_dir = emon_dir / slice_slug
                        job_emon_dir.mkdir(parents=True, exist_ok=True)
                        job_emon_dat = job_emon_dir / 'emon.dat'

                        emon_proc = None
                        emon_stdout = None
                        try:
                            emon_stdout = open(job_emon_dat, 'w', encoding='utf-8')
                            # Use a long -t so the collector stays active until we explicitly stop it.
                            print('+', 'emon -collect-edp -t 3600 >', str(job_emon_dat), '&')
                            emon_proc = subprocess.Popen(
                                ['emon', '-collect-edp', '-t', '3600'],
                                cwd=str(job_emon_dir),
                                stdout=emon_stdout,
                                stderr=subprocess.STDOUT,
                                text=True,
                            )

                            run([
                                'bash',
                                str(scripts_dir / 'run_benchmark.sh'),
                                str(job_config_path),
                                str(job_output_json),
                            ])
                        finally:
                            if emon_proc is not None:
                                print('+', 'emon -stop')
                                try:
                                    proc = subprocess.run(
                                        ['emon', '-stop'],
                                        check=False,
                                        text=True,
                                        capture_output=True,
                                    )
                                    if proc.returncode not in (0, 59):
                                        raise subprocess.CalledProcessError(
                                            proc.returncode, proc.args, proc.stdout, proc.stderr
                                        )
                                except Exception as e:  # noqa: BLE001
                                    print(f"WARN: failed to run `emon -stop` for {slice_slug}: {e}")

                                try:
                                    emon_proc.wait(timeout=10)
                                except Exception:
                                    try:
                                        emon_proc.terminate()
                                    except Exception:
                                        pass

                            if emon_stdout is not None:
                                try:
                                    emon_stdout.close()
                                except Exception:
                                    pass

                        # Accumulate results so we still produce a combined output.search.json.
                        job_results = json.loads(job_output_json.read_text(encoding='utf-8'))
                        if not isinstance(job_results, list):
                            raise ValueError(
                                f"Unexpected output JSON shape for {slice_slug}: expected list"
                            )
                        aggregated_search_results.extend(job_results)

            output_json_search.write_text(
                json.dumps(aggregated_search_results, indent=2) + '\n',
                encoding='utf-8',
            )
    finally:
        # EMON stop/cleanup is handled per-job inside the loop above.
        pass

    # Back-compat: output.json points at the main (search) output.
    if output_json_search.is_file():
        output_json.write_text(output_json_search.read_text(encoding='utf-8'), encoding='utf-8')

    if not args.emon_enable:
        proc = run([py, str(scripts_dir / 'summarize_output.py'), '--input', str(output_json)], capture=True)
        summary_tsv.write_text(proc.stdout, encoding='utf-8')

        details = run([py, str(scripts_dir / 'explain_output.py'), '--input', str(output_json)], capture=True)
        details_md.write_text(details.stdout, encoding='utf-8')

        print('Wrote output:', str(output_json))
        print('Wrote summary:', str(summary_tsv))
        print('Wrote details:', str(details_md))
        print(proc.stdout)
    else:
        # Build-only artifacts (useful for debugging)
        proc_build = run(
            [py, str(scripts_dir / 'summarize_output.py'), '--input', str(output_json_build)],
            capture=True,
        )
        summary_tsv_build.write_text(proc_build.stdout, encoding='utf-8')

        details_build = run(
            [py, str(scripts_dir / 'explain_output.py'), '--input', str(output_json_build)],
            capture=True,
        )
        details_md_build.write_text(details_build.stdout, encoding='utf-8')

        # Search-only artifacts (this is the EMON-covered run)
        proc_search = run(
            [py, str(scripts_dir / 'summarize_output.py'), '--input', str(output_json_search)],
            capture=True,
        )
        summary_tsv_search.write_text(proc_search.stdout, encoding='utf-8')

        details_search = run(
            [py, str(scripts_dir / 'explain_output.py'), '--input', str(output_json_search)],
            capture=True,
        )
        details_md_search.write_text(details_search.stdout, encoding='utf-8')

        # Preserve old names to point at search output by default.
        summary_tsv.write_text(proc_search.stdout, encoding='utf-8')
        details_md.write_text(details_search.stdout, encoding='utf-8')

        print('Wrote build output:', str(output_json_build))
        print('Wrote build summary:', str(summary_tsv_build))
        print('Wrote build details:', str(details_md_build))
        print('Wrote search output:', str(output_json_search))
        print('Wrote search summary:', str(summary_tsv_search))
        print('Wrote search details:', str(details_md_search))
        print(proc_search.stdout)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
