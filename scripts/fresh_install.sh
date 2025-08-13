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

echo "[install] Materializing project into $REPO_DIR..."
# Prefer git clone (most robust), fallback to tar copy
SRC_DIR=$(dirname "$(dirname "$(readlink -f "$0")")")
if [ -d "$SRC_DIR/.git" ] && command -v git >/dev/null 2>&1; then
  set +e
  REMOTE_URL=$(git -C "$SRC_DIR" config --get remote.origin.url)
  BRANCH=$(git -C "$SRC_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null)
  set -e
  if [ -n "$REMOTE_URL" ]; then
    echo "[install] Cloning $REMOTE_URL (branch: ${BRANCH:-default}) -> $REPO_DIR"
    if [ -n "$BRANCH" ] && [ "$BRANCH" != "HEAD" ]; then
      git clone --depth 1 -b "$BRANCH" "$REMOTE_URL" "$REPO_DIR"
    else
      git clone --depth 1 "$REMOTE_URL" "$REPO_DIR"
    fi
  else
    echo "[install] No remote found; copying via tar"
    (cd "$SRC_DIR" && tar --exclude='./venv' --exclude='./cache' --exclude='./logs' -cf - .) | (cd "$REPO_DIR" && tar -xf -)
  fi
else
  echo "[install] Copying via tar (no git detected)"
  (cd "$SRC_DIR" && tar --exclude='./venv' --exclude='./cache' --exclude='./logs' -cf - .) | (cd "$REPO_DIR" && tar -xf -)
fi

echo "[install] Creating venv and installing dependencies..."
python3 -m venv venv
source venv/bin/activate
pip install -U pip "setuptools<70" wheel
pip install --no-cache-dir -r requirements.txt

echo "[install] Running setup.sh to create .env and defaults..."
chmod +x setup.sh || true
./setup.sh

echo "[install] Done. Use ./start.sh to launch, or run the run_all.sh helper."


