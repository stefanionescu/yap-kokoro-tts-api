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


