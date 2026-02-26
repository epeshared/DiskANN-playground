#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(shlex.quote(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _run_capture(cmd: list[str]) -> str:
    print("+", " ".join(shlex.quote(x) for x in cmd), flush=True)
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True, text=True)
    return p.stdout


def _shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def _parse_ssh_opts(ssh_opts: str | None) -> list[str]:
    if not ssh_opts:
        return []
    toks = shlex.split(ssh_opts)
    out: list[str] = []
    for t in toks:
        if t == "~" or t.startswith("~/"):
            out.append(os.path.expanduser(t))
        else:
            out.append(t)
    return out


def _ssh_target(host: str, user: str | None) -> str:
    return f"{user}@{host}" if user else host


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _find_bench_dir() -> Path:
    """Find the diskann-ann-bench directory that contains run_local.sh.

    This file lives under diskann-ann-bench/src/, but avoid assuming a fixed layout.
    """

    here = Path(__file__).resolve()
    for p in (here.parent,) + tuple(here.parents):
        if (p / "run_local.sh").is_file():
            return p
    return here.parents[1]


def _find_opt_value(argv: list[str], opt: str) -> str | None:
    for i, a in enumerate(argv):
        if a == opt and i + 1 < len(argv):
            return argv[i + 1]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run diskann-ann-bench on a remote host via ssh, then sync the run folder back locally"
    )
    ap.add_argument("--remote-host", required=True)
    ap.add_argument("--remote-user")
    ap.add_argument("--ssh-opts", help='Extra ssh/scp options, e.g. "-i ~/.ssh/id_rsa"')
    ap.add_argument(
        "--remote-workspace-root",
        required=True,
        help="Remote workspace root containing DiskANN-playground/ and ann-benchmark-epeshared/",
    )
    ap.add_argument(
        "--remote-copy-hdf5",
        action="store_true",
        help="Copy the local HDF5 to the remote host before running",
    )
    ap.add_argument(
        "--run-id",
        default=None,
        help="Run id to use (default: utc timestamp); forwarded to remote run_local.sh",
    )
    ap.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to remote run_local.sh (prefix with --)",
    )
    args = ap.parse_args()

    ssh_opts = _parse_ssh_opts(args.ssh_opts)
    target = _ssh_target(args.remote_host, args.remote_user)

    forwarded = list(args.args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    hdf5 = _find_opt_value(forwarded, "--hdf5")
    if not hdf5:
        raise SystemExit("ERROR: forwarded args must include --hdf5 /path/to/file.hdf5")

    run_id = _find_opt_value(forwarded, "--run-id")
    if not run_id:
        run_id = args.run_id or _utc_timestamp()
        forwarded += ["--run-id", run_id]

    hdf5_path = Path(hdf5).expanduser().resolve()
    if not hdf5_path.is_file():
        raise SystemExit(f"ERROR: local hdf5 not found: {hdf5_path}")

    dataset = hdf5_path.name
    if dataset.endswith(".hdf5"):
        dataset = dataset[: -len(".hdf5")]

    remote_workspace_root = Path(args.remote_workspace_root).as_posix().rstrip("/")
    remote_run_local = f"{remote_workspace_root}/DiskANN-playground/diskann-ann-bench/run_local.sh"

    remote_hdf5 = str(hdf5_path)
    if args.remote_copy_hdf5:
        remote_dir = f"/tmp/diskann-ann-bench-datasets"
        remote_hdf5 = f"{remote_dir}/{hdf5_path.name}"
        _run(["ssh", *ssh_opts, target, "bash", "-lc", f"mkdir -p {shlex.quote(remote_dir)}"])
        _run(["scp", *ssh_opts, str(hdf5_path), f"{target}:{remote_hdf5}"])
        # Rewrite the forwarded args to point at the remote copy.
        for i, a in enumerate(forwarded):
            if a == "--hdf5" and i + 1 < len(forwarded):
                forwarded[i + 1] = remote_hdf5
                break

    remote_cmd = ["bash", remote_run_local, *forwarded]
    _run(["ssh", *ssh_opts, target, "bash", "-lc", _shell_join(remote_cmd)])

    # Sync results back.
    bench_dir = _find_bench_dir()
    local_runs_dir = Path(os.environ.get("RUNS_DIR", str(bench_dir / "result"))).resolve()
    local_dataset_dir = local_runs_dir / dataset
    local_dataset_dir.mkdir(parents=True, exist_ok=True)

    remote_dataset_dir = f"{remote_workspace_root}/DiskANN-playground/diskann-ann-bench/result/{dataset}"

    # List matching run dirs on remote (wildcard kept for backwards compatibility).
    ls_cmd = [
        "ssh",
        *ssh_opts,
        target,
        "bash",
        "-lc",
        _shell_join(["ls", "-1d", f"{remote_dataset_dir}/{run_id}*" ]),
    ]
    remote_list_raw = _run_capture(ls_cmd)
    remote_dirs = [ln.strip() for ln in remote_list_raw.splitlines() if ln.strip()]

    ssh_cmd = "ssh " + " ".join(shlex.quote(x) for x in ssh_opts)
    for d in remote_dirs:
        _run(
            [
                "rsync",
                "-az",
                "-e",
                ssh_cmd,
                f"{target}:{d.rstrip('/')}/",
                f"{str(local_dataset_dir).rstrip('/')}/{Path(d).name}/",
            ]
        )

    print(f"OK: synced runs into {local_dataset_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
