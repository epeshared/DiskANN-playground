#!/usr/bin/env bash
set -euo pipefail

# 远端执行脚本
# 1) 同步 DiskANN-rs, ann-benchmark-epeshared, DiskANN-playground (排除 .gitignore 文件)
# 2) 同步指定的 HDF5 文件
# 3) (可选) 在远端执行环境部署脚本
# 4) 在远端执行 run_local.sh
# 5) 将远端 result/ 目录同步回本地

# Usage:
#   ./run_remote.sh [--conf-dir conf]
#
# Notes:
# - This script is config-file driven (YAML default; JSON/TOML also supported).
# - Remote connection info comes from: conf/remote-conf.(yml|yaml|json|toml)
# - Job info (including local HDF5 path + remote_hdf5_dir) comes from: conf/job-conf.(yml|yaml|json|toml)
# - Only --conf-dir is accepted; any other CLI flags are rejected.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PLAYGROUND_DIR="$(realpath "$SCRIPT_DIR/..")"
WORKSPACE_ROOT="$(realpath "$DEFAULT_PLAYGROUND_DIR/..")"

CONF_DIR="$SCRIPT_DIR/conf"
REMOTE_CONF="$CONF_DIR/remote-conf.yml"
# Optional override: set this to an absolute path or a path relative to CONF_DIR.
# If empty, the script will auto-pick conf/job-conf.(yml|yaml|json|toml).
JOB_CONF=""
PASSWORD_FILE="$CONF_DIR/password"
PROXY_CONF="$CONF_DIR/proxy-conf.yml"

HOST=""
USER=""
PORT=""
REMOTE_DIR=""
CONNECT_MODE=""

HDF5=""
HDF5_NAME=""
REMOTE_HDF5_DIR=""
SETUP=0
DRY_RUN=0

DISKANN_DIR="$WORKSPACE_ROOT/DiskANN-rs"
ANN_BENCH_DIR="$WORKSPACE_ROOT/ann-benchmark-epeshared"
PLAYGROUND_DIR="$DEFAULT_PLAYGROUND_DIR"

LOG_DIR="$SCRIPT_DIR/_remote_logs"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE=""

usage() {
  cat >&2 <<EOF
Usage:
  ./run_remote.sh [--conf-dir conf]

Config:
  Default conf dir:    $CONF_DIR
  Remote config:       $REMOTE_CONF
  Job config:          $JOB_CONF
  Copy templates from: $CONF_DIR/*.example.yml
EOF
}

if [[ $# -gt 0 ]]; then
  if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    usage
    exit 0
  fi
  if [[ "$1" == "--conf-dir" && $# -eq 2 ]]; then
    CONF_DIR="$2"
  else
    echo "ERROR: this script is config-file driven; only --conf-dir is accepted." >&2
    echo "Got args: $*" >&2
    usage
    exit 2
  fi
fi

case "$CONF_DIR" in
  /*) : ;;
  *) CONF_DIR="$SCRIPT_DIR/$CONF_DIR" ;;
esac

pick_conf_file() {
  # Prefer YAML, then JSON, then TOML.
  local base="$1"
  if [[ -f "$CONF_DIR/${base}.yml" ]]; then
    printf '%s' "$CONF_DIR/${base}.yml"
  elif [[ -f "$CONF_DIR/${base}.yaml" ]]; then
    printf '%s' "$CONF_DIR/${base}.yaml"
  elif [[ -f "$CONF_DIR/${base}.json" ]]; then
    printf '%s' "$CONF_DIR/${base}.json"
  elif [[ -f "$CONF_DIR/${base}.toml" ]]; then
    printf '%s' "$CONF_DIR/${base}.toml"
  else
    # Default path (even if missing) for error messages.
    printf '%s' "$CONF_DIR/${base}.yml"
  fi
}

REMOTE_CONF="$(pick_conf_file remote-conf)"
if [[ -n "$JOB_CONF" ]]; then
  # Interpret relative paths as relative to the selected CONF_DIR.
  case "$JOB_CONF" in
    /*|~/*) : ;;
    *) JOB_CONF="$CONF_DIR/$JOB_CONF" ;;
  esac
  if [[ ! -f "$JOB_CONF" ]]; then
    echo "ERROR: specified JOB_CONF not found: $JOB_CONF" >&2
    echo "Hint: set JOB_CONF to an absolute path or a path relative to CONF_DIR=$CONF_DIR" >&2
    exit 2
  fi
else
  JOB_CONF="$(pick_conf_file job-conf)"
fi
PASSWORD_FILE="$CONF_DIR/password"
PROXY_CONF="$(pick_conf_file proxy-conf)"

echo "==> Using conf dir: $CONF_DIR" >&2
echo "==> Using remote conf: $REMOTE_CONF" >&2
echo "==> Using job conf: $JOB_CONF" >&2
echo "==> Using proxy conf: $PROXY_CONF" >&2

load_remote_conf() {
  local conf="$1"
  python3 - "$conf" <<'PY'
import json
import sys
from pathlib import Path

def load_config(path: Path):
  suffix = path.suffix.lower()
  text = path.read_text(encoding="utf-8")
  if suffix in (".yml", ".yaml"):
    import yaml  # type: ignore
    return yaml.safe_load(text)
  if suffix == ".json":
    return json.loads(text)
  if suffix == ".toml":
    import tomllib
    return tomllib.loads(text)
  # Fallback: try JSON then YAML.
  try:
    return json.loads(text)
  except Exception:
    import yaml  # type: ignore
    return yaml.safe_load(text)

p = Path(sys.argv[1]).expanduser()
if not p.is_file():
    raise SystemExit(f"remote conf not found: {p}")

obj = load_config(p)
if not isinstance(obj, dict):
  raise SystemExit(f"remote conf must be a mapping/object: {p}")

def g(key, default=""):
    v = obj.get(key, default)
    if v is None:
        return ""
    return str(v)

print(g("host"))
print(g("user"))
print(g("port", "22"))
print(g("remote_dir", "~/diskann-workspace"))
print(g("connect", "auto"))
PY
}

load_job_conf_for_remote() {
  local conf="$1"
  python3 - "$conf" <<'PY'
import json
import sys
from pathlib import Path

def load_config(path: Path):
  suffix = path.suffix.lower()
  text = path.read_text(encoding="utf-8")
  if suffix in (".yml", ".yaml"):
    import yaml  # type: ignore
    return yaml.safe_load(text)
  if suffix == ".json":
    return json.loads(text)
  if suffix == ".toml":
    import tomllib
    return tomllib.loads(text)
  try:
    return json.loads(text)
  except Exception:
    import yaml  # type: ignore
    return yaml.safe_load(text)

p = Path(sys.argv[1]).expanduser()
if not p.is_file():
  raise SystemExit(f"job conf not found: {p}")

obj = load_config(p)
if not isinstance(obj, dict):
  raise SystemExit(f"job conf must be a mapping/object: {p}")

def g(key, default=""):
  v = obj.get(key, default)
  if v is None:
    return ""
  return str(v)

def gb(key, default=False) -> str:
  v = obj.get(key, default)
  return "1" if bool(v) else "0"

print(g("hdf5"))
print(g("remote_hdf5_dir"))
print(g("stage"))
print(gb("remote_setup", False))
print(gb("remote_dry_run", False))
PY
}

if [[ -z "$HOST" || -z "$USER" || -z "$PORT" || -z "$REMOTE_DIR" ]]; then
  if [[ -f "$REMOTE_CONF" ]]; then
    mapfile -t _conf < <(load_remote_conf "$REMOTE_CONF")
    # Only fill unset values from config.
    [[ -z "$HOST" ]] && HOST="${_conf[0]:-}"
    [[ -z "$USER" ]] && USER="${_conf[1]:-}"
    [[ -z "$PORT" ]] && PORT="${_conf[2]:-}"
    [[ -z "$REMOTE_DIR" ]] && REMOTE_DIR="${_conf[3]:-}"
    [[ -z "$CONNECT_MODE" ]] && CONNECT_MODE="${_conf[4]:-}"
  fi
fi

if [[ -z "$HOST" || -z "$USER" ]]; then
  echo "ERROR: missing remote host/user." >&2
  echo "Create $REMOTE_CONF" >&2
  exit 1
fi

if [[ -z "$PORT" ]]; then
  PORT="22"
fi
if [[ -z "$REMOTE_DIR" ]]; then
  REMOTE_DIR="~/diskann-workspace"
fi
if [[ -z "$CONNECT_MODE" ]]; then
  CONNECT_MODE="auto"
fi

case "${CONNECT_MODE,,}" in
  auto|ssh|socks|socks5|sock5|socket5|http) ;;
  *)
    echo "ERROR: invalid connect mode: $CONNECT_MODE (expected: auto|ssh|socks|http)" >&2
    exit 2
    ;;
esac

mapfile -t _jobconf < <(load_job_conf_for_remote "$JOB_CONF")
HDF5="${_jobconf[0]:-}"
REMOTE_HDF5_DIR="${_jobconf[1]:-}"
STAGE="${_jobconf[2]:-}"
SETUP="${_jobconf[3]:-0}"
DRY_RUN="${_jobconf[4]:-0}"

STAGE="${STAGE,,}"

HDF5_NAME=""
REMOTE_HDF5=""
if [[ "$STAGE" != "native" ]]; then
  if [[ -z "$HDF5" ]]; then
    echo "ERROR: missing 'hdf5' in $JOB_CONF" >&2
    exit 1
  fi
  if [[ ! -f "$HDF5" ]]; then
    echo "ERROR: Local HDF5 file not found: $HDF5" >&2
    exit 1
  fi
  HDF5_NAME="$(basename "$HDF5")"
fi

# Resolve remote HDF5 destination.
# If remote_hdf5_dir is relative, interpret it as relative to REMOTE_DIR.
if [[ -z "$REMOTE_HDF5_DIR" ]]; then
  REMOTE_HDF5_DIR="$REMOTE_DIR/data"
else
  case "$REMOTE_HDF5_DIR" in
    /*|~/*) : ;;
    *) REMOTE_HDF5_DIR="$REMOTE_DIR/$REMOTE_HDF5_DIR" ;;
  esac
fi
REMOTE_HDF5_DIR="${REMOTE_HDF5_DIR%/}"
if [[ -n "$HDF5_NAME" ]]; then
  REMOTE_HDF5="$REMOTE_HDF5_DIR/$HDF5_NAME"
fi

if [[ ! -f "$PASSWORD_FILE" ]]; then
  echo "ERROR: password file not found: $PASSWORD_FILE" >&2
  echo "Create it with the SSH password (single line)." >&2
  exit 1
fi

PASS="$(python3 - "$PASSWORD_FILE" "$HOST" "$USER" <<'PY'
import json
import re
import sys
from pathlib import Path

pw_path = Path(sys.argv[1]).expanduser()
host = sys.argv[2].strip()
user = sys.argv[3].strip()
key = f"{user}@{host}"

text = pw_path.read_text(encoding="utf-8", errors="replace")

def die(msg: str) -> None:
  print(msg, file=sys.stderr)
  raise SystemExit(2)

def normalize(s: str) -> str:
  return s.strip().strip("\ufeff")

lines = []
for raw in text.splitlines():
  s = normalize(raw)
  if not s or s.startswith("#"):
    continue
  lines.append(s)

if not lines:
  die(f"password file is empty: {pw_path}")

# JSON mapping support: {"root@1.2.3.4": "pw", "user@host": "pw2"}
if pw_path.suffix.lower() == ".json":
  try:
    obj = json.loads("\n".join(lines))
  except Exception as e:
    die(f"failed to parse password json: {pw_path}: {e}")
  if not isinstance(obj, dict):
    die(f"password json must be an object map: {pw_path}")
  pw = obj.get(key) or obj.get(host)
  if pw is None:
    die(
      "no password entry for this host/user. "
      f"expected key {key!r} in {pw_path}"
    )
  print(str(pw).strip())
  raise SystemExit(0)

# Text formats supported (first match wins):
#   root@172.16.116.90 intel@123
#   root@172.16.116.90:intel@123
#   172.16.116.90 root intel@123
#   172.16.116.90 root:intel@123
patterns = [
  (re.compile(r"^(?P<k>\S+@\S+)\s+(?P<p>.+)$"), "k"),
  (re.compile(r"^(?P<k>\S+@\S+):(?P<p>.+)$"), "k"),
  (re.compile(r"^(?P<h>\S+)\s+(?P<u>\S+)\s+(?P<p>.+)$"), "hu"),
  (re.compile(r"^(?P<h>\S+)\s+(?P<u>\S+):(?P<p>.+)$"), "hu"),
]

for s in lines:
  for rx, kind in patterns:
    m = rx.match(s)
    if not m:
      continue
    if kind == "k":
      if m.group("k") == key:
        print(m.group("p").strip())
        raise SystemExit(0)
    else:
      if m.group("h") == host and m.group("u") == user:
        print(m.group("p").strip())
        raise SystemExit(0)

# Backwards compat: if the file contains exactly one non-comment line and it has
# no obvious key, treat it as the password.
if len(lines) == 1 and ("@" not in lines[0].split()[0]):
  print(lines[0].strip())
  raise SystemExit(0)

die(
  "no matching password entry found. "
  f"Add a line for {key!r} to {pw_path} (see password.example)."
)
PY
)"

PASS="${PASS%"${PASS##*[![:space:]]}"}" # rtrim
PASS="${PASS#"${PASS%%[![:space:]]*}"}" # ltrim

if [[ -z "$PASS" ]]; then
  echo "ERROR: resolved password is empty for user=$USER host=$HOST (file=$PASSWORD_FILE)" >&2
  exit 1
fi

echo "==> resolved password entry for ${USER}@${HOST} (len=${#PASS})" >&2

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${HOST}_${RUN_TS}.log"
echo "==> remote output log: $LOG_FILE" >&2

if [[ "$STAGE" != "native" && -z "$HDF5" ]]; then
  echo "ERROR: missing 'hdf5' in $JOB_CONF" >&2
  exit 1
fi

if ! command -v sshpass >/dev/null 2>&1; then
  echo "ERROR: 'sshpass' is required for password authentication."
  echo "Please install it locally (e.g., sudo apt-get install sshpass)."
  exit 1
fi

# If you run this script from within a conda env, system tools like ssh/rsync may
# accidentally pick up conda's libcrypto via LD_LIBRARY_PATH and fail with:
#   "OpenSSL version mismatch"
# Run network/file transfer tools with a sanitized env to avoid that.
clean_env=(
  env
  -u LD_LIBRARY_PATH
  -u DYLD_LIBRARY_PATH
  -u CONDA_PREFIX
  -u CONDA_DEFAULT_ENV
  -u CONDA_SHLVL
  -u PYTHONHOME
)

load_proxy_for_host() {
  local proxy_conf="$1"
  local host="$2"
  python3 - "$proxy_conf" "$host" <<'PY'
import json
import sys
from pathlib import Path

def load_config(path: Path):
  suffix = path.suffix.lower()
  text = path.read_text(encoding="utf-8")
  if suffix in (".yml", ".yaml"):
    import yaml  # type: ignore
    return yaml.safe_load(text)
  if suffix == ".json":
    return json.loads(text)
  if suffix == ".toml":
    import tomllib
    return tomllib.loads(text)
  try:
    return json.loads(text)
  except Exception:
    import yaml  # type: ignore
    return yaml.safe_load(text)

p = Path(sys.argv[1]).expanduser()
host = sys.argv[2].strip()

if not p.is_file():
    # no proxy config
    print("")
    raise SystemExit(0)

obj = load_config(p)
if not isinstance(obj, dict):
    raise SystemExit(f"invalid proxy conf (expected object): {p}")

entry = obj.get(host)
if entry is None:
    print("")
    raise SystemExit(0)

if not isinstance(entry, dict):
    raise SystemExit(f"invalid proxy entry for {host} (expected object)")

ptype = str(entry.get("type", "")).strip().lower()
ptype = {"socket5": "socks5", "sock5": "socks5", "socks": "socks5"}.get(ptype, ptype)
phost = str(entry.get("host", "")).strip()
pport = entry.get("port", "")
try:
    pport_i = int(str(pport).strip())
except Exception:
    pport_i = 0

if ptype not in ("socks5", "http"):
  raise SystemExit(f"invalid proxy type for {host}: {ptype!r} (expected socks|http)")
if not phost:
    raise SystemExit(f"invalid proxy host for {host}")
if pport_i <= 0:
    raise SystemExit(f"invalid proxy port for {host}: {pport!r}")

print(f"{ptype}\n{phost}\n{pport_i}")
PY
}

proxy_type=""
proxy_host=""
proxy_port=""

proxy_blob="$(load_proxy_for_host "$PROXY_CONF" "$HOST")"
if [[ -n "$proxy_blob" ]]; then
  proxy_type="$(printf '%s\n' "$proxy_blob" | sed -n '1p')"
  proxy_host="$(printf '%s\n' "$proxy_blob" | sed -n '2p')"
  proxy_port="$(printf '%s\n' "$proxy_blob" | sed -n '3p')"
fi

use_proxy=0
connect_lc="${CONNECT_MODE,,}"
connect_lc="${connect_lc/socks/socks5}"
connect_lc="${connect_lc/socket5/socks5}"
connect_lc="${connect_lc/sock5/socks5}"

if [[ "$connect_lc" == "ssh" ]]; then
  use_proxy=0
elif [[ "$connect_lc" == "socks5" || "$connect_lc" == "http" ]]; then
  use_proxy=1
  # Allow connect mode to override proxy type if proxy-conf omitted type.
  if [[ -z "$proxy_type" ]]; then
    proxy_type="$connect_lc"
  fi
else
  # auto
  if [[ -n "$proxy_type" ]]; then
    use_proxy=1
  fi
fi

if [[ "$use_proxy" -eq 1 ]]; then
  if [[ -z "$proxy_type" || -z "$proxy_host" || -z "$proxy_port" ]]; then
    echo "ERROR: connect=$CONNECT_MODE requires a proxy entry for $HOST in $PROXY_CONF" >&2
    exit 2
  fi
  echo "==> proxy enabled for $HOST: type=$proxy_type via $proxy_host:$proxy_port" >&2
else
  echo "==> proxy disabled for $HOST (connect=$CONNECT_MODE)" >&2
fi

SSH_OPTS=(
  -p "$PORT"
  -o StrictHostKeyChecking=no
  -o ConnectTimeout=10
  -o ServerAliveInterval=15
  -o ServerAliveCountMax=3
)

if [[ "$use_proxy" -eq 1 ]]; then
  proxy_cmd=""
  if command -v ncat >/dev/null 2>&1; then
    # ncat provides a consistent proxy interface across distros.
    proxy_cmd="ncat --proxy ${proxy_host}:${proxy_port} --proxy-type ${proxy_type} %h %p"
  elif command -v nc >/dev/null 2>&1; then
    if [[ "$proxy_type" == "socks5" ]]; then
      proxy_cmd="nc -x ${proxy_host}:${proxy_port} -X 5 %h %p"
    else
      proxy_cmd="nc -x ${proxy_host}:${proxy_port} -X connect %h %p"
    fi
  else
    echo "ERROR: proxy requires 'ncat' or 'nc' installed locally" >&2
    echo "Install: sudo apt-get install -y nmap (for ncat) or netcat-openbsd" >&2
    exit 2
  fi
  echo "==> proxy ProxyCommand: $proxy_cmd" >&2
  SSH_OPTS+=( -o "ProxyCommand=${proxy_cmd}" )
fi

run_ssh() {
  local remote_cmd="$1"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[dry-run] ssh'
    local o
    for o in "${SSH_OPTS[@]}"; do
      printf ' %q' "$o"
    done
    printf ' %q %q\n' "${USER}@${HOST}" "${remote_cmd}" >&2
    return 0
  fi
  "${clean_env[@]}" sshpass -p "$PASS" ssh "${SSH_OPTS[@]}" "$USER@$HOST" "$remote_cmd"
}

run_ssh_stream() {
  # Stream stdout+stderr to local terminal (prefixed) and append to LOG_FILE.
  local remote_cmd="$1"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    run_ssh "$remote_cmd"
    return 0
  fi
  {
    echo "==> [Local] ssh $USER@$HOST: $remote_cmd"
  } | tee -a "$LOG_FILE" >&2

  local ssh_rc=0
  local awk_rc=0
  local tee_rc=0
  local pipeline_rc=0

  # Force line-by-line flush in awk so local output updates immediately.
  set +e
  "${clean_env[@]}" sshpass -p "$PASS" ssh "${SSH_OPTS[@]}" "$USER@$HOST" "$remote_cmd" 2>&1 \
    | awk -v p="[$HOST] " '{print p $0; fflush();}' \
    | tee -a "$LOG_FILE" >&2
  pipeline_rc=$?
  local -a ps
  ps=("${PIPESTATUS[@]}")
  ssh_rc="${ps[0]:-$pipeline_rc}"
  awk_rc="${ps[1]:-0}"
  tee_rc="${ps[2]:-0}"
  set -e

  if [[ "$ssh_rc" -ne 0 ]]; then
    {
      echo "ERROR: ssh command failed (rc=$ssh_rc)"
      echo "ERROR: remote cmd: $remote_cmd"
    } | tee -a "$LOG_FILE" >&2
    return "$ssh_rc"
  fi
  if [[ "$awk_rc" -ne 0 || "$tee_rc" -ne 0 ]]; then
    {
      echo "ERROR: logging pipeline failed (awk_rc=$awk_rc tee_rc=$tee_rc)"
      echo "ERROR: remote cmd: $remote_cmd"
    } | tee -a "$LOG_FILE" >&2
    return 1
  fi
}

run_ssh_stream_tty() {
  # Same as run_ssh_stream, but forces pseudo-TTY allocation on the remote side.
  # This helps tools that change/buffer output when stdout is not a TTY.
  local remote_cmd="$1"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[dry-run] ssh -tt'
    local o
    for o in "${SSH_OPTS[@]}"; do
      printf ' %q' "$o"
    done
    printf ' %q %q\n' "${USER}@${HOST}" "${remote_cmd}" >&2
    return 0
  fi

  {
    echo "==> [Local] ssh -tt $USER@$HOST: $remote_cmd"
  } | tee -a "$LOG_FILE" >&2

  local ssh_rc=0
  local awk_rc=0
  local tee_rc=0
  local pipeline_rc=0

  set +e
  "${clean_env[@]}" sshpass -p "$PASS" ssh -tt "${SSH_OPTS[@]}" "$USER@$HOST" "$remote_cmd" 2>&1 \
    | awk -v p="[$HOST] " '{print p $0; fflush();}' \
    | tee -a "$LOG_FILE" >&2
  pipeline_rc=$?
  local -a ps
  ps=("${PIPESTATUS[@]}")
  ssh_rc="${ps[0]:-$pipeline_rc}"
  awk_rc="${ps[1]:-0}"
  tee_rc="${ps[2]:-0}"
  set -e

  if [[ "$ssh_rc" -ne 0 ]]; then
    {
      echo "ERROR: ssh -tt command failed (rc=$ssh_rc)"
      echo "ERROR: remote cmd: $remote_cmd"
    } | tee -a "$LOG_FILE" >&2
    return "$ssh_rc"
  fi
  if [[ "$awk_rc" -ne 0 || "$tee_rc" -ne 0 ]]; then
    {
      echo "ERROR: logging pipeline failed (awk_rc=$awk_rc tee_rc=$tee_rc)"
      echo "ERROR: remote cmd: $remote_cmd"
    } | tee -a "$LOG_FILE" >&2
    return 1
  fi
}

rsync_ssh_rsh() {
  # Build an rsync "-e" command string.
  # IMPORTANT: rsync parses this string itself (not a shell), so backslash-escaping
  # can be passed through literally. Prefer quote-grouping for args with spaces.
  rsync_quote_arg() {
    local s="$1"
    if [[ "$s" == *$'\n'* ]]; then
      echo "ERROR: rsync ssh option contains newline" >&2
      exit 2
    fi
    if [[ "$s" == *[[:space:]]* ]]; then
      # Double-quote grouping; escape any embedded double quotes defensively.
      printf '"%s"' "${s//\"/\\\"}"
    else
      printf '%s' "$s"
    fi
  }

  printf 'ssh'
  local o
  for o in "${SSH_OPTS[@]}"; do
    printf ' %s' "$(rsync_quote_arg "$o")"
  done
}

run_rsync_dir() {
  local src="$1"
  local dest="$2"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    local rsh
    rsh="$(rsync_ssh_rsh)"
    echo "[dry-run] rsync -e: $rsh" >&2
    echo "[dry-run] rsync dir: $src/ -> $USER@$HOST:$dest/ (respects .gitignore, excludes .git)" >&2
    return 0
  fi
  local rsh
  rsh="$(rsync_ssh_rsh)"
  "${clean_env[@]}" sshpass -p "$PASS" rsync -avz \
    --info=progress2 \
    --partial \
    --timeout=60 \
    --filter=':- .gitignore' --exclude='.git' \
    -e "$rsh" \
    "$src/" "$USER@$HOST:$dest/"
}

# ann-benchmark-epeshared contains vendored Rust sources (native/vendor/) which
# include their own .gitignore files. If we use rsync's ":- .gitignore" merge,
# rsync may accidentally exclude files Cargo expects (breaking checksum checks).
# So for this tree, prefer explicit excludes over recursive .gitignore honoring.
run_rsync_ann_bench_dir() {
  local src="$1"
  local dest="$2"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    local rsh
    rsh="$(rsync_ssh_rsh)"
    echo "[dry-run] rsync -e: $rsh" >&2
    echo "[dry-run] rsync dir: $src/ -> $USER@$HOST:$dest/ (explicit excludes; includes vendor/)" >&2
    return 0
  fi
  local rsh
  rsh="$(rsync_ssh_rsh)"
  "${clean_env[@]}" sshpass -p "$PASS" rsync -avz \
    --info=progress2 \
    --partial \
    --timeout=60 \
    --exclude='.git' \
    --exclude='data/' \
    --exclude='results/' \
    --exclude='venv/' \
    --exclude='.idea/' \
    --exclude='**/__pycache__/' \
    --exclude='**/*.pyc' \
    -e "$rsh" \
    "$src/" "$USER@$HOST:$dest/"
}

run_rsync_file() {
  local src_file="$1"
  local dest_file="$2"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    local rsh
    rsh="$(rsync_ssh_rsh)"
    echo "[dry-run] rsync -e: $rsh" >&2
    echo "[dry-run] rsync file: $src_file -> $USER@$HOST:$dest_file" >&2
    return 0
  fi
  local rsh
  rsh="$(rsync_ssh_rsh)"
  # For large binary files (e.g., .hdf5), compression (-z) often hurts. Use resume-friendly options.
  "${clean_env[@]}" sshpass -p "$PASS" rsync -av \
    --info=progress2 \
    --partial \
    --append-verify \
    --timeout=60 \
    -e "$rsh" \
    "$src_file" "$USER@$HOST:$dest_file"
}

make_remote_job_conf() {
  local src_conf="$1"
  local dst_conf="$2"
  local remote_hdf5="$3"
  python3 - "$src_conf" "$dst_conf" "$remote_hdf5" <<'PY'
import json
import sys
from pathlib import Path

def load_config(path: Path):
  suffix = path.suffix.lower()
  text = path.read_text(encoding="utf-8")
  if suffix in (".yml", ".yaml"):
    import yaml  # type: ignore
    return yaml.safe_load(text)
  if suffix == ".json":
    return json.loads(text)
  if suffix == ".toml":
    import tomllib
    return tomllib.loads(text)
  try:
    return json.loads(text)
  except Exception:
    import yaml  # type: ignore
    return yaml.safe_load(text)

src = Path(sys.argv[1]).expanduser().resolve()
dst = Path(sys.argv[2]).expanduser().resolve()
remote_hdf5 = str(sys.argv[3])

obj = load_config(src)
if not isinstance(obj, dict):
  raise SystemExit(f"job conf must be a mapping/object: {src}")

obj["hdf5"] = remote_hdf5

dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(str(dst))
PY
}

echo "==> Creating remote directories..."
if [[ "$STAGE" == "native" ]]; then
  run_ssh_stream "mkdir -p $REMOTE_DIR $REMOTE_DIR/DiskANN-playground/diskann-ann-bench/conf"
else
  run_ssh_stream "mkdir -p $REMOTE_DIR $REMOTE_HDF5_DIR $REMOTE_DIR/DiskANN-playground/diskann-ann-bench/conf"
fi

sync_dir() {
  local src="$1"
  local dest="$2"
  echo "==> Syncing $src to $USER@$HOST:$dest"
  # --filter=':- .gitignore' 排除 .gitignore 中列出的文件
  # --exclude='.git' 排除 git 历史记录
  run_rsync_dir "$src" "$dest"
}

# 1) 同步 DiskANN-rs 和 ann-benchmark-epeshared
sync_dir "$DISKANN_DIR" "$REMOTE_DIR/DiskANN-rs"
echo "==> Syncing $ANN_BENCH_DIR to $USER@$HOST:$REMOTE_DIR/ann-benchmark-epeshared"
run_rsync_ann_bench_dir "$ANN_BENCH_DIR" "$REMOTE_DIR/ann-benchmark-epeshared"

# 2) 同步 DiskANN-playground (包含 diskann-ann-bench)
sync_dir "$PLAYGROUND_DIR" "$REMOTE_DIR/DiskANN-playground"

# 3) 同步 HDF5 文件
if [[ "$STAGE" == "native" ]]; then
  echo "==> Skipping HDF5 sync (stage=native)" >&2
else
  echo "==> Syncing HDF5 file $HDF5 to $USER@$HOST:$REMOTE_HDF5"
  run_rsync_file "$HDF5" "$REMOTE_HDF5"
fi

# 3.5) Generate+sync remote job conf (not included by dir rsync due to .gitignore)
# Use a unique remote filename per run to avoid rsync "same size+mtime" short-circuiting
# and to avoid cross-run overwrites when multiple jobs target the same host.
JOB_CONF_REMOTE_LOCAL="$LOG_DIR/job-conf.remote.json"
JOB_CONF_REMOTE_ON_REMOTE="/tmp/diskann-ann-bench/job-conf.remote.${RUN_TS}.json"
echo "==> Generating remote job conf: $JOB_CONF_REMOTE_LOCAL (hdf5=$REMOTE_HDF5)" >&2
make_remote_job_conf "$JOB_CONF" "$JOB_CONF_REMOTE_LOCAL" "$REMOTE_HDF5" >/dev/null
echo "==> Ensuring remote temp dir exists: /tmp/diskann-ann-bench" >&2
run_ssh_stream "mkdir -p /tmp/diskann-ann-bench"
echo "==> Syncing remote job conf to $USER@$HOST:$JOB_CONF_REMOTE_ON_REMOTE" >&2
run_rsync_file "$JOB_CONF_REMOTE_LOCAL" "$JOB_CONF_REMOTE_ON_REMOTE"

echo "==> Verifying remote job conf (head): $JOB_CONF_REMOTE_ON_REMOTE" >&2
run_ssh_stream_tty "echo '==> [Remote] job-conf.remote.json (head)'; head -n 60 $JOB_CONF_REMOTE_ON_REMOTE"

# 6) 远端环境部署
if [[ "$SETUP" -eq 1 ]]; then
  echo "==> Running remote setup script..."
  if [[ "$STAGE" == "build" || "$STAGE" == "native" ]]; then
    run_ssh_stream_tty "bash $REMOTE_DIR/DiskANN-playground/diskann-ann-bench/setup_remote.sh $REMOTE_DIR --mode build"
  else
    run_ssh_stream_tty "bash $REMOTE_DIR/DiskANN-playground/diskann-ann-bench/setup_remote.sh $REMOTE_DIR"
  fi
fi

echo "==> Ensuring DiskANN-rs is vendored under ann-benchmark third_party/..."
run_ssh_stream_tty "if [[ -f $REMOTE_DIR/ann-benchmark-epeshared/ann_benchmarks/algorithms/diskann_rs/third_party/DiskANN-rs/diskann/Cargo.toml ]]; then
  echo '[Remote] third_party/DiskANN-rs already present.'
else
  if ! command -v rsync >/dev/null 2>&1; then
    echo 'ERROR: rsync not found on remote; run with --setup first.' >&2
    exit 2
  fi
  bash $REMOTE_DIR/ann-benchmark-epeshared/ann_benchmarks/algorithms/diskann_rs/sync_diskann_rs.sh
fi"

# 4) 在远端执行 run_local.sh
echo "==> Executing run_local.sh on remote (conf=$JOB_CONF_REMOTE_ON_REMOTE)..."
if [[ "$STAGE" == "build" || "$STAGE" == "native" ]]; then
  run_ssh_stream_tty "source \$HOME/.cargo/env || true; export PYTHONUNBUFFERED=1; cd $REMOTE_DIR/DiskANN-playground/diskann-ann-bench; bash run_local.sh --conf $JOB_CONF_REMOTE_ON_REMOTE"
else
  run_ssh_stream_tty "source \$HOME/.cargo/env || true; source $REMOTE_DIR/.venv/bin/activate 2>/dev/null || true; export PYTHONUNBUFFERED=1; cd $REMOTE_DIR/DiskANN-playground/diskann-ann-bench; bash run_local.sh --conf $JOB_CONF_REMOTE_ON_REMOTE"
fi

# 5) 将远端结果同步回本地
echo "==> Fetching results back to local..."
mkdir -p "$SCRIPT_DIR/result"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] rsync results: $USER@$HOST:$REMOTE_DIR/DiskANN-playground/diskann-ann-bench/result/ -> $SCRIPT_DIR/result/" >&2
else
  rsync_rsh="$(rsync_ssh_rsh)"
  "${clean_env[@]}" sshpass -p "$PASS" rsync -avz \
    --info=progress2 \
    --partial \
    --timeout=60 \
    -e "$rsync_rsh" \
    "$USER@$HOST:$REMOTE_DIR/DiskANN-playground/diskann-ann-bench/result/" "$SCRIPT_DIR/result/"
fi

echo "==> All done! Results are synced to $SCRIPT_DIR/result/"
