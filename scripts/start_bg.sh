#!/bin/bash
set -euo pipefail

# Always execute from repo root
SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")
cd "$ROOT_DIR"

# Ensure executable
chmod +x "$SCRIPT_DIR/start.sh" || true

# Start detached with nohup, piping logs to server.log and PID to server.pid
nohup bash -c '"$0"/start.sh' "$SCRIPT_DIR" > server.log 2>&1 & echo $! > server.pid

echo "[start_bg] Server started with PID $(cat server.pid). Logs: $ROOT_DIR/server.log"
echo "[start_bg] Tail logs: ./scripts/logs.sh"


