#!/bin/bash
set -euo pipefail

echo "[purge] Stopping running servers (if any)..."
pkill -f uvicorn || true

WORK=/workspace
SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
REPO_DIR=$(dirname "$SCRIPT_DIR")

# Args: --delete-repo to also remove the cloned repo directory
DELETE_REPO=false
if [[ ${1:-} == "--delete-repo" ]]; then
  DELETE_REPO=true
fi

echo "[purge] Removing runtime dir and caches..."
rm -rf \
  "$WORK/orpheus-tts" \
  "$WORK/.cache/huggingface" \
  "$WORK/hf" \
  "$HOME/.cache/huggingface" \
  "$HOME/.cache/pip" \
  "$HOME/.nv/ComputeCache" \
  /tmp/* || true

if $DELETE_REPO; then
  echo "[purge] Deleting repo dir: $REPO_DIR"
  rm -rf "$REPO_DIR"
else
  echo "[purge] Keeping repo dir: $REPO_DIR (pass --delete-repo to remove)"
fi

echo "[purge] Clearing pip/apt caches..."
pip cache purge || true
apt-get clean || true
rm -rf /var/lib/apt/lists/* || true

# Aggressive: remove globally installed heavy python deps (outside venv)
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

# Remove potential global entry points
rm -f /usr/local/bin/uvicorn /usr/local/bin/ray /usr/local/bin/transformers-cli || true

# Final cache sweep
rm -rf ~/.cache/pip /root/.cache/pip /tmp/* || true

echo "[purge] Disk usage after purge:"
df -h || true

echo "[purge] Done."


