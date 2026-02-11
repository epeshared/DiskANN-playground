#!/usr/bin/env python3

import argparse
import json
import os
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print('+', ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def run_env(cmd: list[str], *, env: dict[str, str], cwd: Path | None = None) -> None:
    print('+', ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def shell_join(cmd: list[str]) -> str:
    return ' '.join(shlex.quote(x) for x in cmd)


def parse_ssh_opts(ssh_opts: str | None) -> list[str]:
    if not ssh_opts:
        return []
    toks = shlex.split(ssh_opts)
    out: list[str] = []
    for t in toks:
        if t == '~' or t.startswith('~/'):
            out.append(os.path.expanduser(t))
        else:
            out.append(t)
    return out


def ssh_target(host: str, user: str | None) -> str:
    return f"{user}@{host}" if user else host


def run_ssh(
    *,
    host: str,
    user: str | None,
    ssh_opts: list[str],
    remote_cmd: list[str],
) -> None:
    target = ssh_target(host, user)
    # Use bash -lc so remote PATH/env/profile works.
    cmd = ['ssh', *ssh_opts, target, 'bash', '-lc', shell_join(remote_cmd)]
    run(cmd)


def rsync_from_remote(
    *,
    host: str,
    user: str | None,
    ssh_opts: list[str],
    remote_dir: str,
    local_dir: Path,
) -> None:
    target = ssh_target(host, user)
    ssh_cmd = 'ssh ' + ' '.join(shlex.quote(x) for x in ssh_opts)
    # Copy contents of remote_dir into local_dir.
    run(
        [
            'rsync',
            '-az',
            '-e',
            ssh_cmd,
            f'{target}:{remote_dir.rstrip("/")}/',
            f'{str(local_dir).rstrip("/")}/',
        ]
    )


def scp_from_remote(
    *,
    host: str,
    user: str | None,
    ssh_opts: list[str],
    remote_dir: str,
    local_dir: Path,
) -> None:
    target = ssh_target(host, user)
    # Copy directory contents (including dotfiles) by using trailing '/.'
    run(
        [
            'scp',
            '-r',
            *ssh_opts,
            f'{target}:{remote_dir.rstrip("/")}/.',
            str(local_dir),
        ]
    )


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def docker_image_exists(tag: str) -> bool:
    try:
        subprocess.run(
            ['docker', 'image', 'inspect', tag],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except Exception:
        return False


BUILD_SCRIPT = r'''import argparse, json, time
from pathlib import Path

import h5py
import numpy as np

from diskann_rs_native import Index


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--hdf5', required=True)
    ap.add_argument('--train-key', default='train')
    ap.add_argument('--metric', required=True)
    ap.add_argument('--l-build', type=int, required=True)
    ap.add_argument('--max-outdegree', type=int, required=True)
    ap.add_argument('--alpha', type=float, required=True)
    ap.add_argument('--index-prefix', required=True)
    ap.add_argument('--out-json', required=True)
    args = ap.parse_args()

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.index_prefix).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.hdf5, 'r') as f:
        X = np.asarray(f[args.train_key], dtype=np.float32, order='C')

    t0 = time.perf_counter()
    index = Index(args.metric, args.l_build, args.max_outdegree, args.alpha)
    index.fit(X)
    index.save(args.index_prefix)
    t1 = time.perf_counter()

    out = {
        'metric': args.metric,
        'l_build': args.l_build,
        'max_outdegree': args.max_outdegree,
        'alpha': args.alpha,
        'n_points': int(X.shape[0]),
        'dim': int(X.shape[1]),
        'build_s': t1 - t0,
        'index_prefix': args.index_prefix,
    }

    Path(args.out_json).write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
'''

SEARCH_SCRIPT = r'''import argparse, json, time, math, statistics
from pathlib import Path

import h5py
import numpy as np

from diskann_rs_native import Index


def recall_at_k(gt: np.ndarray, pred: list[int], k: int) -> float:
    g = gt[:k]
    if k == 0:
        return 0.0
    gs = set(int(x) for x in g)
    hit = sum(1 for x in pred[:k] if int(x) in gs)
    return hit / float(k)


def p99_us(lat_us: list[float]) -> float:
    if not lat_us:
        return float('nan')
    xs = sorted(lat_us)
    idx = int(math.ceil(0.99 * len(xs))) - 1
    idx = max(0, min(idx, len(xs) - 1))
    return float(xs[idx])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--hdf5', required=True)
    ap.add_argument('--test-key', default='test')
    ap.add_argument('--neighbors-key', default='neighbors')
    ap.add_argument('--index-prefix', required=True)
    ap.add_argument('-k', type=int, required=True)
    ap.add_argument('--l-search', type=int, required=True)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--out-json', required=True)
    args = ap.parse_args()

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.hdf5, 'r') as f:
        Q = np.asarray(f[args.test_key], dtype=np.float32, order='C')
        GT = np.asarray(f[args.neighbors_key], dtype=np.int64)

    index = Index.load(args.index_prefix)

    k = int(args.k)
    l_search = int(args.l_search)
    reps = int(args.reps)
    if k <= 0:
        raise ValueError('k must be > 0')
    if l_search < k:
        raise ValueError('l_search must be >= k')
    if reps <= 0:
        raise ValueError('reps must be > 0')

    rep_stats = []
    for r in range(reps):
        lat_us = []
        t0 = time.perf_counter()
        recall_sum = 0.0
        for i in range(Q.shape[0]):
            q0 = time.perf_counter_ns()
            pred = index.search(Q[i], k, l_search)
            q1 = time.perf_counter_ns()
            lat_us.append((q1 - q0) / 1000.0)
            recall_sum += recall_at_k(GT[i], pred, k)
        t1 = time.perf_counter()

        total_s = t1 - t0
        nq = int(Q.shape[0])
        rep_stats.append(
            {
                'rep': r,
                'n_queries': nq,
                'total_s': total_s,
                'qps': (nq / total_s) if total_s > 0 else 0.0,
                'lat_mean_us': float(statistics.mean(lat_us)) if lat_us else float('nan'),
                'lat_p99_us': p99_us(lat_us),
                'recall_avg': recall_sum / float(nq) if nq > 0 else 0.0,
            }
        )

    out = {
        'k': k,
        'l_search': l_search,
        'reps': reps,
        'rep_stats': rep_stats,
        'recall_avg_mean': float(statistics.mean([x['recall_avg'] for x in rep_stats])),
        'qps_mean': float(statistics.mean([x['qps'] for x in rep_stats])),
        'qps_max': float(max([x['qps'] for x in rep_stats])),
        'lat_mean_us_mean': float(statistics.mean([x['lat_mean_us'] for x in rep_stats])),
        'lat_mean_us_max': float(max([x['lat_mean_us'] for x in rep_stats])),
        'lat_p99_us_mean': float(statistics.mean([x['lat_p99_us'] for x in rep_stats])),
        'lat_p99_us_max': float(max([x['lat_p99_us'] for x in rep_stats])),
    }

    Path(args.out_json).write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
'''


def write_summary_tsv(
    out_path: Path,
    *,
    job: str,
    detail: str,
    tasks: int,
    L: int,
    N: int,
    metrics: dict,
) -> None:
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

    def fmt(x, nd=4):
        if x is None:
            return ''
        if isinstance(x, (int, float)):
            return f"{x:.{nd}f}" if isinstance(x, float) else str(x)
        return str(x)

    row = {
        'job': job,
        'detail': detail,
        'tasks': tasks,
        'L': L,
        'N': N,
        'recall(avg)': metrics.get('recall_avg_mean'),
        'QPS(mean)': metrics.get('qps_mean'),
        'QPS(max)': metrics.get('qps_max'),
        'lat_mean_us(mean)': metrics.get('lat_mean_us_mean'),
        'lat_mean_us(max)': metrics.get('lat_mean_us_max'),
        'lat_p99_us(mean)': metrics.get('lat_p99_us_mean'),
        'lat_p99_us(max)': metrics.get('lat_p99_us_max'),
    }

    lines = ['\t'.join(headers)]
    lines.append('\t'.join([fmt(row[h], nd=4 if 'recall' in h else 1) for h in headers]))
    write_text(out_path, '\n'.join(lines) + '\n')


def main() -> int:
    ap = argparse.ArgumentParser(
        description='Run diskann_rs (ann-benchmark native extension) with explicit build/search split using Docker.'
    )
    ap.add_argument('--hdf5', required=True, help='Path to ann-benchmarks-style .hdf5')
    ap.add_argument('--dataset', default=None, help='Dataset name for run folder (default: hdf5 stem)')
    ap.add_argument('--run-id', default=None, help='Run id (default: UTC timestamp)')
    ap.add_argument(
        '--work-dir',
        default=None,
        help=(
            'Output directory. If omitted, writes to extend-rabitq/ann-harness/runs/<dataset>/<timestamp>/'
        ),
    )

    ap.add_argument('--train-key', default='train')
    ap.add_argument('--test-key', default='test')
    ap.add_argument('--neighbors-key', default='neighbors')

    ap.add_argument('--metric', default='cosine', choices=['cosine', 'l2'])
    ap.add_argument('--l-build', type=int, default=128)
    ap.add_argument('--max-outdegree', type=int, default=64)
    ap.add_argument('--alpha', type=float, default=1.2)

    ap.add_argument('-k', type=int, default=10)
    ap.add_argument('--l-search', type=int, default=100)
    ap.add_argument('--reps', type=int, default=3)

    ap.add_argument('--docker-tag', default='ann-benchmarks-diskann_rs')
    ap.add_argument(
        '--runner',
        default='auto',
        choices=['auto', 'docker', 'host', 'remote'],
        help='Execution mode. auto: docker if image exists else host. remote: run on a remote host via ssh, then sync run folder back.',
    )
    ap.add_argument('--emon-enable', action='store_true', help='Wrap SEARCH docker run with `emon -collect-edp`')

    ap.add_argument(
        '--cpu-bind',
        default=None,
        help='Optional CPU binding description to write into cpu-bind.txt (e.g. "0-16"). This script does not enforce affinity; use taskset/numactl for that.',
    )

    # Remote runner options
    ap.add_argument('--remote-host', default=None, help='Remote host for --runner remote')
    ap.add_argument('--remote-user', default=None, help='Remote SSH user (optional)')
    ap.add_argument(
        '--ssh-opts',
        default=None,
        help='Extra ssh/scp/rsync options as a single string (e.g. "-i ~/.ssh/id_rsa -o StrictHostKeyChecking=no")',
    )
    ap.add_argument(
        '--remote-workspace-root',
        default=None,
        help='Remote workspace root (the folder containing DiskANN-playground/ and ann-benchmark-epeshared/). Required for --runner remote unless --remote-script is provided.',
    )
    ap.add_argument(
        '--remote-script',
        default=None,
        help='Path to run_diskann_rs_split.py on the remote machine (default: <remote-workspace-root>/DiskANN-playground/diskann-ann-bench/run_diskann_rs_split.py)',
    )
    ap.add_argument(
        '--remote-python',
        default='python3',
        help='Python executable to use on the remote host (default: python3).',
    )
    ap.add_argument(
        '--remote-inner-runner',
        default='host',
        choices=['auto', 'docker', 'host'],
        help='Runner to use on the remote side (default: host).',
    )
    ap.add_argument(
        '--remote-work-dir',
        default=None,
        help='Remote run folder path. If omitted, uses <remote-workspace-root>/DiskANN-playground/extend-rabitq/ann-harness/runs/<dataset>/<run-id>/',
    )
    ap.add_argument(
        '--remote-hdf5',
        default=None,
        help='HDF5 path on the remote host. If omitted, uses the same path as --hdf5.',
    )
    ap.add_argument(
        '--remote-copy-hdf5',
        action='store_true',
        help='Copy the local --hdf5 file to the remote work dir before running (useful when remote cannot access the same filesystem path).',
    )
    ap.add_argument(
        '--remote-build-native',
        action='store_true',
        help='Before the remote run, run `cargo build` in the remote native crate directory to ensure diskann_rs_native is built.',
    )
    ap.add_argument(
        '--remote-sync',
        default='auto',
        choices=['auto', 'rsync', 'scp'],
        help='How to sync remote run folder back (default: auto uses rsync if available, else scp).',
    )

    args = ap.parse_args()

    hdf5_path = Path(args.hdf5).resolve()
    if not hdf5_path.is_file():
        raise FileNotFoundError(str(hdf5_path))

    dataset = args.dataset or hdf5_path.stem

    here = Path(__file__).resolve()
    playground_dir = here.parent.parent
    workspace_root = playground_dir.parent
    harness_runs_dir = playground_dir / 'extend-rabitq' / 'ann-harness' / 'runs'

    run_id = args.run_id or utc_timestamp()

    if args.work_dir:
        work_dir = Path(args.work_dir).resolve()
    else:
        work_dir = (harness_runs_dir / dataset / run_id).resolve()

    scripts_dir = work_dir / 'scripts'
    outputs_dir = work_dir / 'outputs'
    index_dir = work_dir / 'index'

    # Create local destination folder early (especially for remote sync destination).
    work_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    if args.cpu_bind:
        write_text(work_dir / 'cpu-bind.txt', str(args.cpu_bind).strip() + '\n')

    build_json = outputs_dir / 'output.build.json'
    search_json = outputs_dir / 'output.search.json'

    runner = args.runner
    if runner == 'auto':
        runner = 'docker' if docker_image_exists(args.docker_tag) else 'host'

    if runner != 'remote':
        write_text(work_dir / 'mode.txt', 'ann_bench_diskann_rs\n')
        write_text(scripts_dir / 'build.py', BUILD_SCRIPT)
        write_text(scripts_dir / 'search.py', SEARCH_SCRIPT)

    if runner == 'remote':
        if not args.remote_host:
            raise ValueError('--remote-host is required for --runner remote')

        ssh_opts = parse_ssh_opts(args.ssh_opts)

        remote_script = args.remote_script
        if not remote_script:
            if not args.remote_workspace_root:
                raise ValueError(
                    '--remote-workspace-root is required for --runner remote unless --remote-script is provided'
                )
            remote_script = (
                Path(args.remote_workspace_root)
                / 'DiskANN-playground'
                / 'diskann-ann-bench'
                / 'run_diskann_rs_split.py'
            )
            remote_script = str(remote_script)

        remote_work_dir = args.remote_work_dir
        if not remote_work_dir:
            if not args.remote_workspace_root:
                raise ValueError('--remote-work-dir or --remote-workspace-root is required for --runner remote')
            remote_work_dir = str(
                Path(args.remote_workspace_root)
                / 'DiskANN-playground'
                / 'extend-rabitq'
                / 'ann-harness'
                / 'runs'
                / dataset
                / run_id
            )

        remote_hdf5 = args.remote_hdf5 or str(hdf5_path)

        if args.remote_build_native:
            if not args.remote_workspace_root:
                raise ValueError('--remote-workspace-root is required for --remote-build-native')
            remote_native_dir = (
                Path(args.remote_workspace_root)
                / 'ann-benchmark-epeshared'
                / 'ann_benchmarks'
                / 'algorithms'
                / 'diskann_rs'
                / 'native'
            )
            run_ssh(
                host=args.remote_host,
                user=args.remote_user,
                ssh_opts=ssh_opts,
                remote_cmd=['cd', str(remote_native_dir), '&&', 'cargo', 'build'],
            )

        if args.remote_copy_hdf5:
            # Copy the dataset to the remote run folder so the remote run does not
            # depend on a shared filesystem path.
            remote_data_dir = remote_work_dir.rstrip('/') + '/data'
            run_ssh(
                host=args.remote_host,
                user=args.remote_user,
                ssh_opts=ssh_opts,
                remote_cmd=['mkdir', '-p', remote_data_dir],
            )
            target = ssh_target(args.remote_host, args.remote_user)
            run(
                [
                    'scp',
                    *ssh_opts,
                    str(hdf5_path),
                    f'{target}:{remote_data_dir}/{hdf5_path.name}',
                ]
            )
            remote_hdf5 = remote_data_dir.rstrip('/') + '/' + hdf5_path.name

        # Execute the same split runner on the remote host, then sync the whole run folder back.
        remote_cmd = [
            args.remote_python,
            remote_script,
            '--runner',
            args.remote_inner_runner,
            '--hdf5',
            remote_hdf5,
            '--dataset',
            dataset,
            '--run-id',
            run_id,
            '--work-dir',
            remote_work_dir,
            '--train-key',
            args.train_key,
            '--test-key',
            args.test_key,
            '--neighbors-key',
            args.neighbors_key,
            '--metric',
            args.metric,
            '--l-build',
            str(args.l_build),
            '--max-outdegree',
            str(args.max_outdegree),
            '--alpha',
            str(args.alpha),
            '-k',
            str(args.k),
            '--l-search',
            str(args.l_search),
            '--reps',
            str(args.reps),
            '--docker-tag',
            args.docker_tag,
        ]
        if args.emon_enable:
            remote_cmd.append('--emon-enable')
        if args.cpu_bind:
            remote_cmd.extend(['--cpu-bind', str(args.cpu_bind)])

        run_ssh(
            host=args.remote_host,
            user=args.remote_user,
            ssh_opts=ssh_opts,
            remote_cmd=remote_cmd,
        )

        sync_mode = args.remote_sync
        if sync_mode == 'auto':
            sync_mode = 'rsync' if shutil.which('rsync') else 'scp'

        if sync_mode == 'rsync':
            if not shutil.which('rsync'):
                raise FileNotFoundError('rsync not found on local machine; use --remote-sync scp')
            rsync_from_remote(
                host=args.remote_host,
                user=args.remote_user,
                ssh_opts=ssh_opts,
                remote_dir=remote_work_dir,
                local_dir=work_dir,
            )
        else:
            scp_from_remote(
                host=args.remote_host,
                user=args.remote_user,
                ssh_opts=ssh_opts,
                remote_dir=remote_work_dir,
                local_dir=work_dir,
            )

    elif runner == 'docker':
        # Mount dataset parent as /data, work_dir as /out.
        docker_base = [
            'docker',
            'run',
            '--rm',
            '--network=host',
            '-v',
            f'{hdf5_path.parent}:/data:ro',
            '-v',
            f'{work_dir}:/out',
            args.docker_tag,
        ]

        # Build stage
        run(
            docker_base
            + [
                'python',
                '/out/scripts/build.py',
                '--hdf5',
                f'/data/{hdf5_path.name}',
                '--train-key',
                args.train_key,
                '--metric',
                args.metric,
                '--l-build',
                str(args.l_build),
                '--max-outdegree',
                str(args.max_outdegree),
                '--alpha',
                str(args.alpha),
                '--index-prefix',
                f'/out/index/diskann_rs',
                '--out-json',
                '/out/outputs/output.build.json',
            ]
        )

        search_cmd = (
            docker_base
            + [
                'python',
                '/out/scripts/search.py',
                '--hdf5',
                f'/data/{hdf5_path.name}',
                '--test-key',
                args.test_key,
                '--neighbors-key',
                args.neighbors_key,
                '--index-prefix',
                f'/out/index/diskann_rs',
                '-k',
                str(args.k),
                '--l-search',
                str(args.l_search),
                '--reps',
                str(args.reps),
                '--out-json',
                '/out/outputs/output.search.json',
            ]
        )

        if args.emon_enable:
            emon_dir = work_dir / 'emon'
            emon_dir.mkdir(parents=True, exist_ok=True)
            emon_dat = emon_dir / 'emon.dat'
            run(['emon', '-collect-edp', '-f', str(emon_dat)])
            try:
                run(search_cmd)
            finally:
                run(['emon', '-stop'])
        else:
            run(search_cmd)

    elif runner == 'host':
        # Use the cargo-built PyO3 library directly (no maturin), by placing a
        # diskann_rs_native.so symlink next to libdiskann_rs_native.so and adding
        # that folder to PYTHONPATH.
        native_debug_dir = (
            workspace_root
            / 'ann-benchmark-epeshared'
            / 'ann_benchmarks'
            / 'algorithms'
            / 'diskann_rs'
            / 'native'
            / 'target'
            / 'debug'
        ).resolve()
        lib_so = native_debug_dir / 'libdiskann_rs_native.so'
        if not lib_so.is_file():
            raise FileNotFoundError(
                f"Missing {lib_so}. Build it first with: cargo build (in the native crate)."
            )
        shim_so = native_debug_dir / 'diskann_rs_native.so'
        if shim_so.exists() or shim_so.is_symlink():
            try:
                shim_so.unlink()
            except Exception:
                pass
        shim_so.symlink_to(lib_so)

        env = dict(os.environ)
        old_pp = env.get('PYTHONPATH')
        env['PYTHONPATH'] = str(native_debug_dir) + ((':' + old_pp) if old_pp else '')

        run_env(
            [
                'python',
                str(scripts_dir / 'build.py'),
                '--hdf5',
                str(hdf5_path),
                '--train-key',
                args.train_key,
                '--metric',
                args.metric,
                '--l-build',
                str(args.l_build),
                '--max-outdegree',
                str(args.max_outdegree),
                '--alpha',
                str(args.alpha),
                '--index-prefix',
                str(index_dir / 'diskann_rs'),
                '--out-json',
                str(build_json),
            ],
            env=env,
        )

        search_cmd = [
            'python',
            str(scripts_dir / 'search.py'),
            '--hdf5',
            str(hdf5_path),
            '--test-key',
            args.test_key,
            '--neighbors-key',
            args.neighbors_key,
            '--index-prefix',
            str(index_dir / 'diskann_rs'),
            '-k',
            str(args.k),
            '--l-search',
            str(args.l_search),
            '--reps',
            str(args.reps),
            '--out-json',
            str(search_json),
        ]

        if args.emon_enable:
            emon_dir = work_dir / 'emon'
            emon_dir.mkdir(parents=True, exist_ok=True)
            emon_dat = emon_dir / 'emon.dat'
            run(['emon', '-collect-edp', '-f', str(emon_dat)])
            try:
                run_env(search_cmd, env=env)
            finally:
                run(['emon', '-stop'])
        else:
            run_env(search_cmd, env=env)

    else:
        raise ValueError(f'unknown runner: {runner}')

    build = json.loads(build_json.read_text(encoding='utf-8'))
    search = json.loads(search_json.read_text(encoding='utf-8'))

    # A single-row summary.tsv compatible with the existing web UI.
    job = 'ann-bench-diskann-rs'
    detail = f"metric={args.metric}; L_build={args.l_build}; R={args.max_outdegree}; alpha={args.alpha}"
    write_summary_tsv(
        outputs_dir / 'summary.tsv',
        job=job,
        detail=detail,
        tasks=1,
        L=int(args.l_search),
        N=int(args.k),
        metrics=search,
    )

    details_lines = []
    details_lines.append('# ann-bench diskann_rs (split build/search)')
    details_lines.append('')
    details_lines.append('## Build')
    details_lines.append('')
    details_lines.append(f"- build_s: {build.get('build_s')}")
    details_lines.append(f"- n_points: {build.get('n_points')}")
    details_lines.append(f"- dim: {build.get('dim')}")
    details_lines.append(f"- index_prefix: {build.get('index_prefix')}")
    details_lines.append('')
    details_lines.append('## Search')
    details_lines.append('')
    details_lines.append(f"- k: {search.get('k')}")
    details_lines.append(f"- l_search: {search.get('l_search')}")
    details_lines.append(f"- reps: {search.get('reps')}")
    details_lines.append(f"- recall_avg_mean: {search.get('recall_avg_mean')}")
    details_lines.append(f"- qps_mean: {search.get('qps_mean')}")
    details_lines.append(f"- lat_mean_us_mean: {search.get('lat_mean_us_mean')}")
    details_lines.append(f"- lat_p99_us_mean: {search.get('lat_p99_us_mean')}")
    details_lines.append('')
    details_lines.append('## Rep Stats')
    details_lines.append('')
    for r in search.get('rep_stats', []) or []:
        details_lines.append(
            f"- rep={r.get('rep')} qps={r.get('qps'):.1f} "
            f"lat_mean_us={r.get('lat_mean_us'):.1f} lat_p99_us={r.get('lat_p99_us'):.1f} "
            f"recall_avg={r.get('recall_avg'):.4f}"
        )

    write_text(outputs_dir / 'details.md', '\n'.join(details_lines) + '\n')

    print(f"Wrote run: {work_dir}")
    print(f"- {build_json}")
    print(f"- {search_json}")
    print(f"- {outputs_dir / 'summary.tsv'}")
    print(f"- {outputs_dir / 'details.md'}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
