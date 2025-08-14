#!/bin/bash
set -euo pipefail

WORK=/workspace
SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
REPO_DIR=$(dirname "$SCRIPT_DIR")

echo "[purge] Stopping GPU-related server processes (safe mode)..."

# Stop background server by PID if present
PID_FILE="$REPO_DIR/server.pid"
if [ -f "$PID_FILE" ]; then
  if [ -s "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE" 2>/dev/null || true)
    if [ -n "${PID:-}" ] && kill -0 "$PID" 2>/dev/null; then
      kill "$PID" || true
      sleep 1
      kill -9 "$PID" || true
      echo "[purge] Killed server PID $PID"
    fi
  fi
  rm -f "$PID_FILE"
fi

# Broad kill of common processes (do NOT kill Jupyter/console by default)
pkill -f uvicorn || true
pkill -f "python .*main.py" || true
pkill -f "python -m vllm" || true
pkill -f vllm || true
pkill -f ray || true
pkill -f gunicorn || true
pkill -f start.sh || true
pkill -f start_bg.sh || true

# Optionally kill tmux/screen sessions (only if requested later)

# (paths already set above)

# Flags (defaults = GPU-only safe purge)
CLEAN_FILES=false       # remove venv/model/cache/logs
DELETE_REPO=false       # remove repo dir
KILL_JUPYTER=false      # also stop Jupyter (web console)
KILL_SESSIONS=false     # kill tmux/screen sessions
AGGRESSIVE=false        # also remove global site-packages and apt/pip caches

for arg in "$@"; do
  case "$arg" in
    --clean-files) CLEAN_FILES=true ;;
    --delete-repo) DELETE_REPO=true ;;
    --kill-jupyter) KILL_JUPYTER=true ;;
    --kill-sessions) KILL_SESSIONS=true ;;
    --aggressive) AGGRESSIVE=true; CLEAN_FILES=true ;;
    --all) CLEAN_FILES=true; DELETE_REPO=true; KILL_JUPYTER=true; KILL_SESSIONS=true; AGGRESSIVE=true ;;
  esac
done

if $KILL_JUPYTER; then
  echo "[purge] Killing Jupyter as requested (--kill-jupyter)"
  pkill -f jupyter || true
fi

if $KILL_SESSIONS; then
  echo "[purge] Killing tmux/screen sessions as requested (--kill-sessions)"
  tmux kill-server || true
  screen -ls | awk '/Detached|Attached/ {print $1}' | xargs -r -n1 screen -S || true
fi

if $CLEAN_FILES; then
  echo "[purge] Removing in-repo runtime dirs (venv, model, snac_model, cache, logs) and caches..."
  rm -rf \
    "$REPO_DIR/venv" \
    "$REPO_DIR/model" \
    "$REPO_DIR/snac_model" \
    "$REPO_DIR/cache" \
    "$REPO_DIR/logs" \
    "$REPO_DIR/server.log" \
    "$REPO_DIR/server.pid" \
    "$REPO_DIR/warmup_audio" \
    "$WORK/.cache/huggingface" \
    "$WORK/hf" \
    "$HOME/.cache/huggingface" \
    "$HOME/.cache/pip" \
    "$HOME/.nv/ComputeCache" \
    /tmp/* || true
else
  echo "[purge] Skipping file cleanup (use --clean-files to remove venv/model/cache/logs)"
fi

if $DELETE_REPO; then
  echo "[purge] Deleting repo dir: $REPO_DIR"
  rm -rf "$REPO_DIR"
else
  echo "[purge] Keeping repo dir: $REPO_DIR (pass --delete-repo to remove)"
fi

if $AGGRESSIVE; then
  echo "[purge] Clearing pip/apt caches..."
  pip cache purge || true
  apt-get clean || true
  rm -rf /var/lib/apt/lists/* || true
fi

# Aggressive: remove globally installed heavy python deps (outside venv)
if $AGGRESSIVE; then
  echo "[purge] Removing global Python GPU/ML packages (torch, vllm, etc.)..."
  GLOBAL_SITE=/usr/local/lib/python3.11/dist-packages
  rm -rf \
    "$GLOBAL_SITE"/torch* \
    "$GLOBAL_SITE"/torchvision* \
    "$GLOBAL_SITE"/triton* \
    "$GLOBAL_SITE"/xformers* \
    "$GLOBAL_SITE"/vllm* \
    "$GLOBAL_SITE"/transformers* \
    "$GLOBAL_SITE"/deepspeed* \
    "$GLOBAL_SITE"/ray* \
    "$GLOBAL_SITE"/sentencepiece* \
    "$GLOBAL_SITE"/nvidia* \
    "$GLOBAL_SITE"/numpy* || true
fi

# Remove potential global entry points
rm -f /usr/local/bin/uvicorn /usr/local/bin/ray /usr/local/bin/transformers-cli || true

# Final cache sweep
rm -rf ~/.cache/pip /root/.cache/pip /tmp/* || true

echo "[purge] Attempting to kill any remaining GPU processes..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
  PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr '\n' ' ')
  if [ -n "$PIDS" ]; then
    echo "[purge] Killing GPU PIDs: $PIDS"
    for p in $PIDS; do kill -9 "$p" 2>/dev/null || true; done
  else
    echo "[purge] No GPU compute processes reported by nvidia-smi"
  fi

  echo "[purge] Resetting GPU (if idle)..."
  nvidia-smi -pm 0 || true
  nvidia-smi --gpu-reset -i 0 || true
  nvidia-smi -pm 1 || true
  nvidia-smi || true
else
  echo "[purge] nvidia-smi not found; skipping GPU reset"
fi

echo "[purge] Disk usage after purge:"
df -h || true

echo "[purge] Done."


