#!/bin/bash
set -euo pipefail

REPO_DIR=/workspace/orpheus-tts

if [ -z "${HF_TOKEN:-}" ]; then
  echo "[install] Warning: HF_TOKEN is not set. Set it to access gated models."
fi

echo "[install] Creating repo dir at $REPO_DIR"
rm -rf "$REPO_DIR"
mkdir -p "$REPO_DIR"
cd "$REPO_DIR"

echo "[install] Copying current workspace repo contents if present (fallback)..."
# If the code is already present where this script lives, copy everything there
SRC_DIR=$(dirname "$(dirname "$(readlink -f "$0")")")
rsync -a --exclude venv --exclude cache --exclude logs "$SRC_DIR/" "$REPO_DIR/"

echo "[install] Creating venv and installing dependencies..."
python3 -m venv venv
source venv/bin/activate
pip install -U pip "setuptools<70" wheel
pip install --no-cache-dir -r requirements.txt

echo "[install] Running setup.sh to create .env and defaults..."
chmod +x setup.sh || true
./setup.sh

echo "[install] Done. Use ./start.sh to launch, or run the run_all.sh helper."


