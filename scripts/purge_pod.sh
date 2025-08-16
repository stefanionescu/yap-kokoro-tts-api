#!/bin/bash
set -euo pipefail

# Default knobs
PORT=${PORT:-8000}
DO_CLEAN_FILES=false
DO_AGGRESSIVE=false
DO_GPU_RESET=false   # resetting GPU can kill other CUDA users; off by default
DO_KILL_JUPYTER=false
DO_KILL_SESSIONS=false

usage() {
  cat <<EOF
Usage: $0 [--port N] [--clean-files] [--aggressive] [--gpu-reset] [--kill-jupyter] [--kill-sessions]
 - Default: stop only the voice server (uvicorn + vLLM children) using server.pgid or :PORT detection.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --clean-files) DO_CLEAN_FILES=true; shift ;;
    --aggressive) DO_AGGRESSIVE=true; DO_CLEAN_FILES=true; shift ;;
    --gpu-reset) DO_GPU_RESET=true; shift ;;
    --kill-jupyter) DO_KILL_JUPYTER=true; shift ;;
    --kill-sessions) DO_KILL_SESSIONS=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

WORK=/workspace
SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
REPO_DIR=$(dirname "$SCRIPT_DIR")

echo "[purge] Targeting only the TTS server on :$PORT"

# 1) Prefer killing by recorded PGID (cleanest)
PGID_FILE="$REPO_DIR/server.pgid"
PID_FILE="$REPO_DIR/server.pid"

kill_group() {
  local pg="$1"
  if [[ -n "$pg" ]]; then
    echo "[purge] Killing process group -$pg ..."
    kill -TERM "-- -$pg" 2>/dev/null || true
    sleep 1
    kill -KILL "-- -$pg" 2>/dev/null || true
  fi
}

if [[ -s "$PGID_FILE" ]]; then
  PGID=$(tr -d ' ' < "$PGID_FILE" || true)
  kill_group "$PGID"
  rm -f "$PGID_FILE" "$PID_FILE"
else
  # 2) Fallback: no PGID recorded; do a safe, targeted pkill (no console impact)
  echo "[purge] No recorded PGID; stopping uvicorn via pkill"
  pkill -f uvicorn || true
  pkill -f "python .*main.py" || true
  # vLLM-specific processes removed in Kokoro setup
fi

# 3) DO NOT broad-pkill here. Keep the console alive.
if $DO_KILL_JUPYTER; then
  echo "[purge] Killing Jupyter as requested (--kill-jupyter)"
  pkill -f '^jupyter-(lab|notebook)' || true
fi
if $DO_KILL_SESSIONS; then
  echo "[purge] Killing tmux/screen as requested (--kill-sessions)"
  tmux kill-server 2>/dev/null || true
  screen -ls | awk '/Detached|Attached/ {print $1}' | xargs -r -n1 screen -S 2>/dev/null || true
fi

# 4) Optional cleanup (safe set)
if $DO_CLEAN_FILES; then
  echo "[purge] Cleaning in-repo runtime dirs (safe set)..."
  rm -rf \
    "$REPO_DIR/venv" \
    "$REPO_DIR/model" \
    "$REPO_DIR/snac_model" \
    "$REPO_DIR/cache" \
    "$REPO_DIR/logs" \
    "$REPO_DIR/server.log" \
    "$REPO_DIR/server.pid" \
    "$REPO_DIR/server.pgid" \
    "$REPO_DIR/warmup_audio" \
    "$WORK/.cache/huggingface" \
    "$HOME/.cache/huggingface" \
    "$HOME/.nv/ComputeCache" || true
  # Do NOT wipe /tmp by default; it kills the web console.
fi

# 5) Aggressive cleanup (explicit opt-in; may disrupt console)
if $DO_AGGRESSIVE; then
  echo "[purge] Aggressive purge enabled -- this may disrupt Jupyter/console!"
  pip cache purge || true
  apt-get clean || true
  rm -rf /var/lib/apt/lists/* || true
  # torch extensions cache only (safe)
  rm -rf "${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions}" || true
  # still avoid 'rm -rf /tmp/*' unless you absolutely must:
  # rm -rf /tmp/*   # <- DANGEROUS, commented by design
fi

# 6) Optional GPU reset (kills any remaining CUDA users)
if $DO_GPU_RESET && command -v nvidia-smi >/dev/null 2>&1; then
  echo "[purge] Attempting GPU reset..."
  nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r -n1 kill -9 2>/dev/null || true
  nvidia-smi -pm 0 || true
  nvidia-smi --gpu-reset -i 0 || true
  nvidia-smi -pm 1 || true
fi

echo "[purge] Done. Disk usage:"
df -h || true
