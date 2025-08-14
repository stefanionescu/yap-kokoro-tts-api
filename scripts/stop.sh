#!/bin/bash
set -euo pipefail

cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

if [ -f server.pid ]; then
  PID=$(cat server.pid)
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
    echo "Stopped server PID $PID"
  fi
  rm -f server.pid
else
  echo "server.pid not found; attempting to kill uvicorn..."
  pkill -f uvicorn || true
fi


