#!/bin/bash
set -euo pipefail

# Full cycle: purge -> fresh install -> start -> warmup

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
ROOT_DIR=$(dirname "$SCRIPT_DIR")

echo "[run_all] Purging pod..."
bash "$SCRIPT_DIR/purge_pod.sh"

echo "[run_all] Fresh install..."
bash "$SCRIPT_DIR/fresh_install.sh"

echo "[run_all] Starting server..."
cd /workspace/orpheus-tts
chmod +x start.sh warmup.py || true
./start.sh &
SERVER_PID=$!

echo "[run_all] Waiting for server to come up..."
sleep 25

echo "[run_all] Warmup both voices..."
python warmup.py --save || true

echo "[run_all] Server PID: $SERVER_PID"
echo "[run_all] Logs: tail -f /workspace/orpheus-tts/server.log (if using nohup)"
echo "[run_all] Done."


