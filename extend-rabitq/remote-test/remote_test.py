#!/usr/bin/env python3

import argparse
import json
import os
import re
import shlex
import subprocess
import tarfile
import tempfile
import getpass
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


try:
    import paramiko  # type: ignore
except Exception as e:  # noqa: BLE001
    paramiko = None
    _PARAMIKO_IMPORT_ERROR = e


DISKANN_GIT_URL = "https://github.com/microsoft/DiskANN.git"


@dataclass(frozen=True)
class RemoteConfig:
    host: str
    port: int
    username: str
    password: str | None
    allow_unknown_host_keys: bool


@dataclass(frozen=True)
class PathConfig:
    remote_workspace_root: str
    remote_playground_dir: str
    remote_diskann_rs_dir: str
    remote_diskann_dir: str
    dataset_name: str | None
    remote_hdf5_path: str | None
    remote_results_dir: str | None


@dataclass(frozen=True)
class OptionsConfig:
    clean_remote_playground: bool
    clean_remote_diskann_rs: bool
    clone_diskann_if_missing: bool
    upload_diskann_rs: bool


@dataclass(frozen=True)
class CpuBindConfig:
    # If set, uses taskset -c <cpus>
    taskset_cpus: str | None
    # If set, uses numactl -C <...>
    numactl_physcpubind: str | None
    # If set, uses numactl -m <...>
    numactl_membind: str | None
    # Apply binding to build/test. Default is ['test'].
    apply_to: list[str]


@dataclass(frozen=True)
class CondaConfig:
    env_name: str | None
    auto_activate: bool
    create_if_missing: bool
    create_command: str | None


@dataclass(frozen=True)
class PreRequirementsConfig:
    enable: bool
    scripts: list[str]


@dataclass(frozen=True)
class DataCopyConfig:
    enable: bool
    local_hdf5_path: str | None
    remote_hdf5_path: str | None
    if_missing_only: bool


def _run_local(cmd: list[str], *, cwd: Path | None = None, capture: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        capture_output=capture,
    )


def _git_file_list(repo_root: Path) -> list[Path]:
    """Return tracked + untracked (not ignored) files, relative to repo_root."""
    proc = _run_local(["git", "-C", str(repo_root), "ls-files", "-co", "--exclude-standard", "-z"], capture=True)
    raw = proc.stdout
    out: list[Path] = []
    for part in raw.split("\0"):
        if not part:
            continue
        out.append(Path(part))
    return out


def _make_tar_from_file_list(workspace_root: Path, arc_rel_files: Iterable[Path], out_tar_gz: Path) -> None:
    """Create a tar.gz from a list of files that are already workspace-root relative."""
    out_tar_gz.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_tar_gz, "w:gz") as tf:
        for rel in arc_rel_files:
            full = workspace_root / rel
            if not full.exists() and not full.is_symlink():
                continue
            arcname = str(rel.as_posix()).lstrip("/")
            if arcname.startswith(".."):
                continue
            tf.add(str(full), arcname=arcname, recursive=False)


def _extract_dataset_from_run_dataset_sh(run_dataset_sh: Path) -> str | None:
    """Best-effort: parse '--dataset <name>' from run_dataset.sh."""
    text = run_dataset_sh.read_text(encoding="utf-8")

    # New style: DATASET="${DATASET:-foo}"
    m = re.search(r'DATASET\s*=\s*"\$\{DATASET:-([^}]+)\}"', text)
    if m:
        return m.group(1).strip().strip("\"").strip("'")

    # Handles: --dataset foo, --dataset=foo
    m = re.search(r"--dataset(?:\s+|=)([^\s\\]+)", text)
    if not m:
        return None
    v = m.group(1).strip().strip("\"").strip("'")
    if v in ("$DATASET", "${DATASET}"):
        return None
    return v


def _extract_hdf5_from_run_dataset_sh(run_dataset_sh: Path) -> str | None:
    """Best-effort: parse default HDF5 path from run_dataset.sh."""
    text = run_dataset_sh.read_text(encoding="utf-8")

    # New style: HDF5_PATH="${HDF5_PATH:-/path/to/file.hdf5}"
    m = re.search(r'HDF5_PATH\s*=\s*"\$\{HDF5_PATH:-([^}]+)\}"', text)
    if m:
        return m.group(1).strip().strip("\"").strip("'")

    # Old style: --hdf5 /path/to/file.hdf5
    m = re.search(r"--hdf5(?:\s+|=)([^\s\\]+)", text)
    if m:
        return m.group(1).strip().strip("\"").strip("'")

    return None


def _load_config(path: Path) -> tuple[
    RemoteConfig,
    PathConfig,
    OptionsConfig,
    CpuBindConfig,
    CondaConfig,
    PreRequirementsConfig,
    DataCopyConfig,
]:
    obj = json.loads(path.read_text(encoding="utf-8"))

    remote = obj.get("remote", {})
    paths = obj.get("paths", {})
    options = obj.get("options", {})
    cpu_bind = obj.get("cpu_bind", {})
    conda = obj.get("conda", {})
    pre_req = obj.get("pre_requirements", {})
    data_copy = obj.get("data_copy", {})

    rcfg = RemoteConfig(
        host=str(remote["host"]),
        port=int(remote.get("port", 22)),
        username=str(remote["username"]),
        password=(None if remote.get("password") in (None, "") else str(remote.get("password"))),
        allow_unknown_host_keys=bool(remote.get("allow_unknown_host_keys", True)),
    )

    pcfg = PathConfig(
        remote_workspace_root=str(paths["remote_workspace_root"]),
        remote_playground_dir=str(paths["remote_playground_dir"]),
        remote_diskann_rs_dir=str(paths["remote_diskann_rs_dir"]),
        remote_diskann_dir=str(paths["remote_diskann_dir"]),
        dataset_name=paths.get("dataset_name"),
        remote_hdf5_path=(None if paths.get("remote_hdf5_path") in (None, "") else str(paths.get("remote_hdf5_path"))),
        remote_results_dir=(None if paths.get("remote_results_dir") in (None, "") else str(paths.get("remote_results_dir"))),
    )

    ocfg = OptionsConfig(
        clean_remote_playground=bool(options.get("clean_remote_playground", True)),
        clean_remote_diskann_rs=bool(options.get("clean_remote_diskann_rs", True)),
        clone_diskann_if_missing=bool(options.get("clone_diskann_if_missing", True)),
        upload_diskann_rs=bool(options.get("upload_diskann_rs", True)),
    )

    apply_to = cpu_bind.get("apply_to", ["test"])
    if not isinstance(apply_to, list):
        raise ValueError("cpu_bind.apply_to must be a list like ['test'] or ['build','test']")
    apply_to = [str(x) for x in apply_to]
    for phase in apply_to:
        if phase not in ("build", "test"):
            raise ValueError("cpu_bind.apply_to entries must be 'build' or 'test'")

    cbcfg = CpuBindConfig(
        taskset_cpus=cpu_bind.get("taskset_cpus"),
        numactl_physcpubind=cpu_bind.get("numactl_physcpubind"),
        numactl_membind=cpu_bind.get("numactl_membind"),
        apply_to=apply_to,
    )

    env_name = conda.get("env_name")
    if env_name is not None:
        env_name = str(env_name)

    create_command = None if conda.get("create_command") in (None, "") else str(conda.get("create_command"))
    if "create_if_missing" in conda:
        create_if_missing = bool(conda.get("create_if_missing"))
    else:
        # If user provided a create command, default to creating env when missing.
        create_if_missing = create_command is not None

    concfg = CondaConfig(
        env_name=env_name,
        auto_activate=bool(conda.get("auto_activate", True)),
        create_if_missing=create_if_missing,
        create_command=create_command,
    )

    scripts = pre_req.get("scripts", [])
    if scripts is None:
        scripts = []
    if not isinstance(scripts, list):
        raise ValueError("pre_requirements.scripts must be a list of script paths")
    scripts = [str(x) for x in scripts]
    prcfg = PreRequirementsConfig(
        enable=bool(pre_req.get("enable", False)),
        scripts=scripts,
    )

    dccfg = DataCopyConfig(
        enable=bool(data_copy.get("enable", False)),
        local_hdf5_path=(
            None
            if data_copy.get("local_hdf5_path") in (None, "")
            else str(data_copy.get("local_hdf5_path"))
        ),
        remote_hdf5_path=(
            None
            if data_copy.get("remote_hdf5_path") in (None, "")
            else str(data_copy.get("remote_hdf5_path"))
        ),
        if_missing_only=bool(data_copy.get("if_missing_only", True)),
    )

    return rcfg, pcfg, ocfg, cbcfg, concfg, prcfg, dccfg


def _remote_command_prefix(cpu_bind: CpuBindConfig) -> str:
    parts: list[str] = []
    if cpu_bind.numactl_physcpubind is not None or cpu_bind.numactl_membind is not None:
        numa_parts = ["numactl"]
        if cpu_bind.numactl_physcpubind is not None:
            numa_parts.append("-C")
            numa_parts.append(str(cpu_bind.numactl_physcpubind))
        if cpu_bind.numactl_membind is not None:
            numa_parts.append("-m")
            numa_parts.append(str(cpu_bind.numactl_membind))
        parts.append(" ".join(shlex.quote(x) for x in numa_parts))

    if cpu_bind.taskset_cpus is not None:
        parts.append(f"taskset -c {shlex.quote(cpu_bind.taskset_cpus)}")

    return " ".join(parts)


def _conda_base_prefix() -> str:
    # Uses conda itself to locate base, then sources conda.sh to enable `conda activate`.
    return 'CONDA_BASE="$(conda info --base)" && source "$CONDA_BASE/etc/profile.d/conda.sh"'


def _conda_activate_prefix(conda_cfg: CondaConfig) -> str:
    if not conda_cfg.env_name or not conda_cfg.auto_activate:
        return ""
    return f"{_conda_base_prefix()} && conda activate {shlex.quote(conda_cfg.env_name)}"


def _ssh_connect(cfg: RemoteConfig):
    if paramiko is None:
        raise RuntimeError(
            "paramiko is required for password SSH. "
            "Install it with: pip install paramiko\n"
            f"Import error: {_PARAMIKO_IMPORT_ERROR}"
        )

    client = paramiko.SSHClient()
    if cfg.allow_unknown_host_keys:
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    else:
        client.load_system_host_keys()

    client.connect(
        hostname=cfg.host,
        port=cfg.port,
        username=cfg.username,
        password=cfg.password or "",
        look_for_keys=False,
        allow_agent=False,
        timeout=30,
    )
    return client


def _ssh_run(client, command: str, *, cwd: str | None = None) -> None:
    # Use bash -lc so PATH/profile is consistent.
    full = command
    if cwd is not None:
        full = f"cd {shlex.quote(cwd)} && {command}"

    print("+ ssh:", full)
    stdin, stdout, stderr = client.exec_command(f"bash -lc {shlex.quote(full)}")
    _ = stdin
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    status = stdout.channel.recv_exit_status()
    if out.strip():
        print(out.rstrip())
    if status != 0:
        if err.strip():
            print(err.rstrip())
        raise RuntimeError(f"Remote command failed (exit={status}): {full}")


def _ssh_capture(client, command: str, *, cwd: str | None = None) -> str:
    full = command
    if cwd is not None:
        full = f"cd {shlex.quote(cwd)} && {command}"

    print("+ ssh:", full)
    stdin, stdout, stderr = client.exec_command(f"bash -lc {shlex.quote(full)}")
    _ = stdin
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    status = stdout.channel.recv_exit_status()
    if status != 0:
        raise RuntimeError(f"Remote command failed (exit={status}): {full}\n{err}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Remote test runner: upload DiskANN-playground (excluding .gitignored), "
            "ensure remote DiskANN clone, build, run dataset test, download results."
        )
    )
    ap.add_argument("--config", required=True, help="Path to remote-test config JSON")
    ap.add_argument(
        "--password",
        default=None,
        help="SSH password override (discouraged: may leak via shell history/process list).",
    )
    ap.add_argument(
        "--password-file",
        default=None,
        help="Path to a local file containing the SSH password (preferred for non-interactive runs).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be done; do not connect/upload/run.",
    )
    args = ap.parse_args()

    here = Path(__file__).resolve()
    playground_root = here.parent.parent
    workspace_root = playground_root.parent
    diskann_rs_root = workspace_root / "DiskANN-rs"

    config_path = Path(args.config).resolve()
    rcfg, pcfg, ocfg, cbcfg, concfg, prcfg, dccfg = _load_config(config_path)

    # Password resolution: CLI > file > config > prompt.
    if args.password is not None:
        rcfg = RemoteConfig(
            host=rcfg.host,
            port=rcfg.port,
            username=rcfg.username,
            password=str(args.password),
            allow_unknown_host_keys=rcfg.allow_unknown_host_keys,
        )
    elif args.password_file is not None:
        pw_path = Path(args.password_file).expanduser().resolve()
        pw = pw_path.read_text(encoding="utf-8").rstrip("\n")
        rcfg = RemoteConfig(
            host=rcfg.host,
            port=rcfg.port,
            username=rcfg.username,
            password=pw,
            allow_unknown_host_keys=rcfg.allow_unknown_host_keys,
        )

    if rcfg.password is None and not args.dry_run:
        pw = getpass.getpass(f"SSH password for {rcfg.username}@{rcfg.host}: ")
        # Rebind as a concrete value for connect().
        rcfg = RemoteConfig(
            host=rcfg.host,
            port=rcfg.port,
            username=rcfg.username,
            password=pw,
            allow_unknown_host_keys=rcfg.allow_unknown_host_keys,
        )

    prefix = _remote_command_prefix(cbcfg)

    def _maybe_bind(cmd: str, *, phase: str) -> str:
        if not prefix:
            return cmd
        if phase not in cbcfg.apply_to:
            return cmd
        return f"{prefix} {cmd}"

    conda_prefix = _conda_activate_prefix(concfg)

    def _wrap(cmd: str, *, phase: str | None = None) -> str:
        inner = cmd
        if phase is not None:
            inner = _maybe_bind(inner, phase=phase)
        # rustup installs cargo under ~/.cargo/bin; ensure current shell loads it if available.
        inner = 'if [ -f "$HOME/.cargo/env" ]; then . "$HOME/.cargo/env"; fi && ' + inner
        if conda_prefix:
            inner = f"{conda_prefix} && {inner}"
        return inner

    run_dataset_sh = playground_root / "extend-rabitq" / "ann-harness" / "scripts" / "run_dataset.sh"
    dataset = pcfg.dataset_name or _extract_dataset_from_run_dataset_sh(run_dataset_sh)
    local_default_hdf5 = _extract_hdf5_from_run_dataset_sh(run_dataset_sh)
    if dataset is None:
        raise RuntimeError(
            "Dataset name not provided and failed to parse --dataset from run_dataset.sh. "
            "Set paths.dataset_name in config."
        )

    effective_remote_hdf5 = pcfg.remote_hdf5_path or (dccfg.remote_hdf5_path if dccfg.enable else None)

    if ocfg.upload_diskann_rs and not diskann_rs_root.is_dir():
        raise FileNotFoundError(
            f"Expected DiskANN-rs at {diskann_rs_root}. "
            "remote-test expects workspace layout: <root>/DiskANN-playground and <root>/DiskANN-rs"
        )

    # Build a workspace-relative file list for tar: DiskANN-playground + (optional) DiskANN-rs.
    arc_files: list[Path] = []
    for rel in _git_file_list(playground_root):
        arc_files.append(Path("DiskANN-playground") / rel)
    if ocfg.upload_diskann_rs:
        for rel in _git_file_list(diskann_rs_root):
            arc_files.append(Path("DiskANN-rs") / rel)

    if args.dry_run:
        print("Local workspace root:", str(workspace_root))
        print("Will upload file count:", len(arc_files))
        print("Remote host:", rcfg.host)
        if args.password is not None:
            pw_src = "<cli>"
        elif args.password_file is not None:
            pw_src = "<file>"
        else:
            pw_src = "<from config>" if rcfg.password is not None else "<interactive>"
        print("Password:", pw_src)
        print("Remote workspace root:", pcfg.remote_workspace_root)
        print("Remote playground dir:", pcfg.remote_playground_dir)
        print("Remote DiskANN-rs dir:", pcfg.remote_diskann_rs_dir)
        print("Remote DiskANN dir:", pcfg.remote_diskann_dir)
        if prcfg.enable:
            print("Pre-requirements scripts:", ",".join(prcfg.scripts) if prcfg.scripts else "<none>")
        if concfg.env_name is not None:
            print("Conda env:", concfg.env_name)
            print("Conda auto_activate:", str(concfg.auto_activate))
            print("Conda create_if_missing:", str(concfg.create_if_missing))
        if prefix:
            print("CPU bind prefix:", prefix)
            print("CPU bind apply_to:", ",".join(cbcfg.apply_to))
        print("Dataset:", dataset)
        if effective_remote_hdf5 is not None:
            print("Remote HDF5:", effective_remote_hdf5)
        if pcfg.remote_results_dir is not None:
            print("Remote results dir:", pcfg.remote_results_dir)
        if dccfg.enable:
            print("Data copy:", "enabled")
            print("Data copy if_missing_only:", str(dccfg.if_missing_only))
            print("Data copy local_hdf5_path:", dccfg.local_hdf5_path or "<none>")
            print("Data copy remote_hdf5_path:", (dccfg.remote_hdf5_path or pcfg.remote_hdf5_path) or "<none>")
        return 0

    with tempfile.TemporaryDirectory(prefix="diskann-remote-test-") as td:
        td_path = Path(td)
        tar_path = td_path / "DiskANN-workspace.tar.gz"
        _make_tar_from_file_list(workspace_root, arc_files, tar_path)

        client = _ssh_connect(rcfg)
        try:
            sftp = client.open_sftp()

            remote_tmp = f"/tmp/diskann-playground-{os.getpid()}"
            _ssh_run(client, f"rm -rf {shlex.quote(remote_tmp)} && mkdir -p {shlex.quote(remote_tmp)}")

            remote_tar = f"{remote_tmp}/DiskANN-workspace.tar.gz"
            print(f"+ upload: {tar_path} -> {remote_tar}")
            sftp.put(str(tar_path), remote_tar)

            # Prepare remote workspace + extract.
            _ssh_run(client, f"mkdir -p {shlex.quote(pcfg.remote_workspace_root)}")
            if ocfg.clean_remote_playground:
                _ssh_run(client, f"rm -rf {shlex.quote(pcfg.remote_playground_dir)}")
            if ocfg.clean_remote_diskann_rs and ocfg.upload_diskann_rs:
                _ssh_run(client, f"rm -rf {shlex.quote(pcfg.remote_diskann_rs_dir)}")

            _ssh_run(client, f"tar -xzf {shlex.quote(remote_tar)} -C {shlex.quote(pcfg.remote_workspace_root)}")

            # Optional: move results outside the playground so cleaning playground won't delete results.
            # Implemented by symlinking ann-harness/runs -> <remote_results_dir>.
            if pcfg.remote_results_dir is not None:
                results_dir = pcfg.remote_results_dir
                ann_harness_dir = f"{pcfg.remote_playground_dir}/extend-rabitq/ann-harness"
                runs_link = f"{ann_harness_dir}/runs"
                _ssh_run(client, f"mkdir -p {shlex.quote(results_dir)}")
                _ssh_run(client, f"mkdir -p {shlex.quote(ann_harness_dir)}")
                _ssh_run(
                    client,
                    (
                        f"rm -rf {shlex.quote(runs_link)} && "
                        f"ln -s {shlex.quote(results_dir)} {shlex.quote(runs_link)}"
                    ),
                )

            # Optional: run pre-requirements scripts on remote (typically installs packages/conda/etc).
            if prcfg.enable:
                for script_rel in prcfg.scripts:
                    script_path = f"{pcfg.remote_playground_dir}/{script_rel.lstrip('/')}"
                    _ssh_run(client, f"bash {shlex.quote(script_path)}")

            # Optional: copy dataset file to remote if missing.
            if dccfg.enable:
                if dccfg.local_hdf5_path is None:
                    raise RuntimeError("data_copy.enable=true but data_copy.local_hdf5_path is not set")
                if effective_remote_hdf5 is None:
                    raise RuntimeError(
                        "data_copy.enable=true but no remote path provided. "
                        "Set paths.remote_hdf5_path or data_copy.remote_hdf5_path."
                    )

                local_hdf5 = Path(dccfg.local_hdf5_path).expanduser()
                if not local_hdf5.is_file():
                    raise FileNotFoundError(f"Local HDF5 not found: {local_hdf5}")

                remote_hdf5 = effective_remote_hdf5
                remote_exists = True
                try:
                    _ssh_run(client, f"test -f {shlex.quote(remote_hdf5)}")
                except Exception:
                    remote_exists = False

                if remote_exists and dccfg.if_missing_only:
                    print(f"INFO: Remote HDF5 exists, skip copy: {remote_hdf5}")
                else:
                    print(f"INFO: Uploading HDF5 to remote: {local_hdf5} -> {remote_hdf5}")
                    _ssh_run(client, f"mkdir -p $(dirname {shlex.quote(remote_hdf5)})")
                    tmp_remote = f"{remote_hdf5}.tmp.{os.getpid()}"
                    sftp.put(str(local_hdf5), tmp_remote)
                    _ssh_run(client, f"mv -f {shlex.quote(tmp_remote)} {shlex.quote(remote_hdf5)}")
                    _ssh_run(client, f"test -f {shlex.quote(remote_hdf5)}")

            # Optional: conda env activation + create-if-missing.
            if concfg.env_name is not None:
                _ssh_run(client, "command -v conda >/dev/null 2>&1 || (echo 'ERROR: conda not found' >&2; exit 1)")

                # Check env exists.
                env = shlex.quote(concfg.env_name)
                check_cmd = (
                    f"{_conda_base_prefix()} && "
                    f"conda env list | awk '{{print $1}}' | grep -Fxq {env}"
                )
                env_exists = True
                try:
                    _ssh_run(client, check_cmd)
                except Exception:
                    env_exists = False

                if not env_exists:
                    if concfg.create_if_missing:
                        if not concfg.create_command:
                            raise RuntimeError(
                                f"Conda env '{concfg.env_name}' missing, but conda.create_command not provided."
                            )
                        create_cmd = f"cd {shlex.quote(pcfg.remote_workspace_root)} && {_conda_base_prefix()} && {concfg.create_command}"
                        _ssh_run(client, create_cmd)
                    else:
                        raise RuntimeError(
                            (
                                f"Conda env '{concfg.env_name}' missing and conda.create_if_missing is false. "
                                "Fix: set conda.create_if_missing=true (and provide conda.create_command), "
                                "or set conda.env_name=null to disable conda integration."
                            )
                        )

                # Now that env exists, ensure conda_prefix is active for subsequent commands.
                conda_prefix = _conda_activate_prefix(concfg)

                # Ensure basic Python deps exist for ann-harness scripts.
                # convert_hdf5_to_diskann_bin.py requires numpy + h5py.
                env_name_q = shlex.quote(concfg.env_name)
                try:
                    _ssh_run(
                        client,
                        f"{_conda_base_prefix()} && conda run -n {env_name_q} python -c 'import numpy, h5py'",
                    )
                except Exception:
                    print("INFO: Installing missing Python deps in conda env:", concfg.env_name)
                    _ssh_run(
                        client,
                        f"{_conda_base_prefix()} && conda install -n {env_name_q} -y -c conda-forge numpy h5py",
                    )
                    _ssh_run(
                        client,
                        f"{_conda_base_prefix()} && conda run -n {env_name_q} python -c 'import numpy, h5py'",
                    )

            # Ensure DiskANN repo exists on remote.
            if ocfg.clone_diskann_if_missing:
                _ssh_run(
                    client,
                    (
                        f"if [ ! -d {shlex.quote(pcfg.remote_diskann_dir)} ]; then "
                        f"git clone {shlex.quote(DISKANN_GIT_URL)} {shlex.quote(pcfg.remote_diskann_dir)}; "
                        "fi"
                    ),
                )

            # Build DiskANN via the copied script.
            build_script = f"{pcfg.remote_playground_dir}/diskann-rs/build_all_targets.sh"
            if ocfg.upload_diskann_rs:
                cmd = _wrap(
                    f"DISKANN_RS_DIR={shlex.quote(pcfg.remote_diskann_rs_dir)} bash {shlex.quote(build_script)}",
                    phase="build",
                )
                _ssh_run(client, cmd)
            else:
                cmd = _wrap(f"bash {shlex.quote(build_script)}", phase="build")
                _ssh_run(client, cmd)

            # Run dataset test.
            test_dir = f"{pcfg.remote_playground_dir}/extend-rabitq/ann-harness/scripts"
            env_parts: list[str] = [f"DATASET={shlex.quote(dataset)}"]
            # Record CPU binding used for test runs (run_dataset.py will save it).
            if prefix and "test" in cbcfg.apply_to:
                env_parts.append(f"CPU_BIND_DESC={shlex.quote(prefix)}")
            if effective_remote_hdf5 is not None:
                try:
                    _ssh_run(client, f"test -f {shlex.quote(effective_remote_hdf5)}")
                    env_parts.append(f"HDF5_PATH={shlex.quote(effective_remote_hdf5)}")
                except RuntimeError:
                    # Preferred remote path missing; fall back to auto-detect if possible.
                    if local_default_hdf5 is None:
                        raise
            if not any(p.startswith("HDF5_PATH=") for p in env_parts) and local_default_hdf5 is not None:
                try:
                    _ssh_run(client, f"test -f {shlex.quote(local_default_hdf5)}")
                    env_parts.append(f"HDF5_PATH={shlex.quote(local_default_hdf5)}")
                except RuntimeError as e:
                    base = os.path.basename(local_default_hdf5)
                    search = _ssh_capture(
                        client,
                        (
                            "for d in /mnt /data /home /root; do "
                            "  if [ -d \"$d\" ]; then "
                            f"    p=$(find \"$d\" -maxdepth 6 -type f -name {shlex.quote(base)} -print -quit 2>/dev/null || true); "
                            "    if [ -n \"$p\" ]; then echo \"$p\"; break; fi; "
                            "  fi; "
                            "done"
                        ),
                    ).strip()
                    if search:
                        print(f"INFO: Using auto-detected remote HDF5 path: {search}")
                        env_parts.append(f"HDF5_PATH={shlex.quote(search)}")
                    else:
                        raise RuntimeError(
                            "Remote dataset file not found. Either put the dataset on the remote machine, "
                            "or set paths.remote_hdf5_path in config (or enable data_copy). "
                            f"Default from run_dataset.sh was: {local_default_hdf5}"
                        ) from e
            env_prefix = " ".join(env_parts)
            cmd = _wrap(f"env {env_prefix} bash run_dataset.sh", phase="test")
            _ssh_run(client, cmd, cwd=test_dir)

            # Find newest run directory for that dataset.
            if pcfg.remote_results_dir is not None:
                remote_runs_root = f"{pcfg.remote_results_dir}/{dataset}"
            else:
                remote_runs_root = f"{pcfg.remote_playground_dir}/extend-rabitq/ann-harness/runs/{dataset}"
            newest = _ssh_capture(
                client,
                f"ls -1dt {shlex.quote(remote_runs_root)}/* 2>/dev/null | head -n 1",
            ).strip()
            if not newest:
                raise RuntimeError(f"No run output found under remote runs dir: {remote_runs_root}")

            remote_out_tar = f"{remote_tmp}/runs-latest.tar.gz"
            if pcfg.remote_results_dir is not None:
                # Tar the newest timestamp dir from its parent (<remote_results_dir>/<dataset>/).
                _ssh_run(
                    client,
                    f"cd {shlex.quote(remote_runs_root)} && tar -czf {shlex.quote(remote_out_tar)} {shlex.quote(os.path.basename(newest))}",
                )
            else:
                # Tar just that run directory relative to remote playground root.
                remote_run_rel = newest.removeprefix(pcfg.remote_playground_dir + "/").lstrip("/")
                _ssh_run(
                    client,
                    f"cd {shlex.quote(pcfg.remote_playground_dir)} && tar -czf {shlex.quote(remote_out_tar)} {shlex.quote(remote_run_rel)}",
                )

            local_out_tar = td_path / "runs-latest.tar.gz"
            print(f"+ download: {remote_out_tar} -> {local_out_tar}")
            sftp.get(remote_out_tar, str(local_out_tar))

            if pcfg.remote_results_dir is not None:
                # Tar contains only the timestamp directory name.
                local_runs_dataset_dir = playground_root / "extend-rabitq" / "ann-harness" / "runs" / dataset
                local_runs_dataset_dir.mkdir(parents=True, exist_ok=True)
                with tarfile.open(local_out_tar, "r:gz") as tf:
                    tf.extractall(path=str(local_runs_dataset_dir))
                print(
                    "Downloaded and extracted remote results into:",
                    str(playground_root / "extend-rabitq" / "ann-harness" / "runs"),
                )
            else:
                # Extract into local playground root so paths match.
                with tarfile.open(local_out_tar, "r:gz") as tf:
                    tf.extractall(path=str(playground_root))
                print(
                    "Downloaded and extracted remote results into:",
                    str(playground_root / "extend-rabitq" / "ann-harness" / "runs"),
                )

        finally:
            try:
                client.close()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
