#!/usr/bin/env bash
set -euo pipefail

# 远端执行脚本
# 1) 同步 DiskANN-rs, ann-benchmark-epeshared, DiskANN-playground (排除 .gitignore 文件)
# 2) 同步指定的 HDF5 文件
# 3) (可选) 在远端执行环境部署脚本
# 4) 在远端执行 run_local.sh
# 5) 将远端 result/ 目录同步回本地

# Usage:
#   ./run_remote.sh --hdf5 /path/to/dataset.hdf5 [--setup] [--dry-run] [--pq-mode disk|memory | --pq-name <algo-name>] [<run_local.sh args...>]
#
# Notes:
# - Any unrecognized args are forwarded to run_local.sh on the remote.
# - --pq-mode/--pq-name are convenience options that translate to run_local.sh's --name-pq.
# - --pq-mode and --pq-name are mutually exclusive.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PLAYGROUND_DIR="$(realpath "$SCRIPT_DIR/..")"
WORKSPACE_ROOT="$(realpath "$DEFAULT_PLAYGROUND_DIR/..")"

CONF_JSON="$SCRIPT_DIR/remote-conf.json"
PASSWORD_FILE="$SCRIPT_DIR/password"
PROXY_CONF_JSON="$SCRIPT_DIR/proxy-conf.json"

HOST=""
USER=""
PORT=""
REMOTE_DIR=""
CONNECT_MODE=""

HDF5=""
HDF5_DIR=""
HDF5_NAME=""
SETUP=0
DRY_RUN=0

DISKANN_DIR="$WORKSPACE_ROOT/DiskANN-rs"
ANN_BENCH_DIR="$WORKSPACE_ROOT/ann-benchmark-epeshared"
PLAYGROUND_DIR="$DEFAULT_PLAYGROUND_DIR"

RUN_ARGS=()

# Convenience options for selecting which PQ algorithm name to run on the remote.
# These are translated to run_local.sh's existing --name-pq flag.
PQ_MODE=""
PQ_NAME=""

LOG_DIR="$SCRIPT_DIR/_remote_logs"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --conf) CONF_JSON="$2"; shift 2 ;;
    --password-file) PASSWORD_FILE="$2"; shift 2 ;;
    --proxy-conf) PROXY_CONF_JSON="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --user) USER="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
    --connect) CONNECT_MODE="$2"; shift 2 ;;
    --hdf5) HDF5="$2"; shift 2 ;;
    --hdf5-dir) HDF5_DIR="$2"; shift 2 ;;
    --hdf5-name) HDF5_NAME="$2"; shift 2 ;;
    --diskann-dir) DISKANN_DIR="$2"; shift 2 ;;
    --ann-bench-dir) ANN_BENCH_DIR="$2"; shift 2 ;;
    --playground-dir) PLAYGROUND_DIR="$2"; shift 2 ;;
    --setup) SETUP=1; shift 1 ;;
    --dry-run) DRY_RUN=1; shift 1 ;;
    --pq-mode) PQ_MODE="$2"; shift 2 ;;
    --pq-name) PQ_NAME="$2"; shift 2 ;;
    *) RUN_ARGS+=("$1"); shift 1 ;;
  esac
done

if [[ -n "$PQ_MODE" && -n "$PQ_NAME" ]]; then
  echo "ERROR: --pq-mode and --pq-name are mutually exclusive" >&2
  exit 2
fi

if [[ -n "$PQ_MODE" ]]; then
  case "${PQ_MODE,,}" in
    disk)
      PQ_NAME="diskann-rs-pq-disk"
      ;;
    mem|memory)
      PQ_NAME="diskann-rs-pq-memory"
      ;;
    *)
      echo "ERROR: invalid --pq-mode: $PQ_MODE (expected: disk|memory)" >&2
      exit 2
      ;;
  esac
fi

has_run_arg() {
  local needle="$1"
  local a
  for a in "${RUN_ARGS[@]}"; do
    if [[ "$a" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

if [[ -n "$PQ_NAME" ]]; then
  # Backward compatibility: keep passing --name-pq for compare-mode flows.
  RUN_ARGS+=("--name-pq" "$PQ_NAME")

  # When user selects --pq-mode/--pq-name and doesn't provide explicit run-mode flags,
  # default to PQ-only workflow for clearer behavior.
  if ! has_run_arg "--compare" \
    && ! has_run_arg "--no-compare" \
    && ! has_run_arg "--algo" \
    && ! has_run_arg "--name" \
    && ! has_run_arg "--run-group" \
    && ! has_run_arg "--run-group-pq" \
    && ! has_run_arg "--run-group-spherical"; then
    RUN_ARGS+=("--no-compare" "--algo" "pq" "--name" "$PQ_NAME")
  fi
fi

load_conf() {
  local conf="$1"
  python3 - "$conf" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1]).expanduser()
if not p.is_file():
    raise SystemExit(f"remote conf not found: {p}")

obj = json.loads(p.read_text(encoding="utf-8"))

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

if [[ -z "$HOST" || -z "$USER" || -z "$PORT" || -z "$REMOTE_DIR" ]]; then
  if [[ -f "$CONF_JSON" ]]; then
    mapfile -t _conf < <(load_conf "$CONF_JSON")
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
  echo "Provide --host/--user or create $CONF_JSON" >&2
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

if [[ -n "$HDF5" && -n "$HDF5_DIR" ]]; then
  echo "ERROR: --hdf5 and --hdf5-dir are mutually exclusive" >&2
  exit 1
fi

if [[ -n "$HDF5" ]]; then
  if [[ ! -f "$HDF5" ]]; then
    echo "ERROR: Local HDF5 file not found: $HDF5" >&2
    exit 1
  fi
  HDF5_NAME="$(basename "$HDF5")"
else
  if [[ -z "$HDF5_DIR" ]]; then
    echo "ERROR: must provide --hdf5 <file> OR --hdf5-dir <dir>" >&2
    exit 1
  fi
  if [[ ! -d "$HDF5_DIR" ]]; then
    echo "ERROR: Local HDF5 dir not found: $HDF5_DIR" >&2
    exit 1
  fi
  if [[ -z "$HDF5_NAME" ]]; then
    # Auto-pick if exactly one .hdf5 exists.
    mapfile -t _hdf5s < <(find "$HDF5_DIR" -maxdepth 1 -type f -name '*.hdf5' -printf '%f\n' | sort)
    if [[ ${#_hdf5s[@]} -ne 1 ]]; then
      echo "ERROR: --hdf5-name is required when --hdf5-dir contains multiple .hdf5 files" >&2
      echo "Found: ${#_hdf5s[@]} files" >&2
      exit 1
    fi
    HDF5_NAME="${_hdf5s[0]}"
  fi
  if [[ ! -f "$HDF5_DIR/$HDF5_NAME" ]]; then
    echo "ERROR: HDF5 file not found in dir: $HDF5_DIR/$HDF5_NAME" >&2
    exit 1
  fi
  HDF5="$HDF5_DIR/$HDF5_NAME"
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

if [[ -z "$HDF5" ]]; then
  echo "ERROR: missing --hdf5 (or --hdf5-dir/--hdf5-name)" >&2
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

p = Path(sys.argv[1]).expanduser()
host = sys.argv[2].strip()

if not p.is_file():
    # no proxy config
    print("")
    raise SystemExit(0)

obj = json.loads(p.read_text(encoding="utf-8"))
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

proxy_blob="$(load_proxy_for_host "$PROXY_CONF_JSON" "$HOST")"
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
    echo "ERROR: connect=$CONNECT_MODE requires a proxy entry for $HOST in $PROXY_CONF_JSON" >&2
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
  # Force line-by-line flush in awk so local output updates immediately.
  "${clean_env[@]}" sshpass -p "$PASS" ssh "${SSH_OPTS[@]}" "$USER@$HOST" "$remote_cmd" 2>&1 \
    | awk -v p="[$HOST] " '{print p $0; fflush();}' \
    | tee -a "$LOG_FILE" >&2
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
  "${clean_env[@]}" sshpass -p "$PASS" ssh -tt "${SSH_OPTS[@]}" "$USER@$HOST" "$remote_cmd" 2>&1 \
    | awk -v p="[$HOST] " '{print p $0; fflush();}' \
    | tee -a "$LOG_FILE" >&2
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

echo "==> Creating remote directories..."
run_ssh_stream "mkdir -p $REMOTE_DIR/data"

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
sync_dir "$ANN_BENCH_DIR" "$REMOTE_DIR/ann-benchmark-epeshared"

# 2) 同步 DiskANN-playground (包含 diskann-ann-bench)
sync_dir "$PLAYGROUND_DIR" "$REMOTE_DIR/DiskANN-playground"

# 3) 同步 HDF5 文件
REMOTE_HDF5="$REMOTE_DIR/data/$HDF5_NAME"
echo "==> Syncing HDF5 file $HDF5 to $USER@$HOST:$REMOTE_HDF5"
run_rsync_file "$HDF5" "$REMOTE_HDF5"

# 6) 远端环境部署
if [[ "$SETUP" -eq 1 ]]; then
  echo "==> Running remote setup script..."
  run_ssh_stream_tty "bash $REMOTE_DIR/DiskANN-playground/diskann-ann-bench/setup_remote.sh $REMOTE_DIR"
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
echo "==> Executing run_local.sh on remote..."
# 安全地格式化传递给 run_local.sh 的参数，防止空格被截断
FORMATTED_ARGS=$(printf "%q " "${RUN_ARGS[@]}")

run_ssh_stream_tty "source \$HOME/.cargo/env || true; source $REMOTE_DIR/.venv/bin/activate 2>/dev/null || true; export PYTHONUNBUFFERED=1; cd $REMOTE_DIR/DiskANN-playground/diskann-ann-bench; bash run_local.sh --hdf5 ../../data/$HDF5_NAME $FORMATTED_ARGS"

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
