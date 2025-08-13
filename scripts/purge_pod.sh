#!/bin/bash
set -euo pipefail

echo "[purge] Stopping running servers (if any)..."
pkill -f uvicorn || true

WORK=/workspace

echo "[purge] Removing project dirs and caches..."
rm -rf \
  "$WORK/orpheus-tts" \
  "$WORK/yap-voice-model-deployment" \
  "$WORK/.cache/huggingface" \
  "$WORK/hf" \
  "$HOME/.cache/huggingface" \
  "$HOME/.cache/pip" \
  "$HOME/.nv/ComputeCache" \
  /tmp/* || true

echo "[purge] Clearing pip/apt caches..."
pip cache purge || true
apt-get clean || true
rm -rf /var/lib/apt/lists/* || true

echo "[purge] Disk usage after purge:"
df -h || true

echo "[purge] Done."


