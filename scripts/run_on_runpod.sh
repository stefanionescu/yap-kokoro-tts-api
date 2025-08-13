#!/bin/bash
set -e

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")
cd "$ROOT_DIR"

echo "==== Setting up environment ===="
bash "$SCRIPT_DIR/setup.sh"

echo "==== Starting API server ===="
nohup bash "$SCRIPT_DIR/start.sh" > server.log 2>&1 &
SERVER_PID=$!
echo "API server started with PID: $SERVER_PID"

echo "Waiting for the server to initialize (30 seconds)..."
sleep 30

echo "==== Running model warmup ===="
source venv/bin/activate
python warmup.py --save || true

echo "==== Deployment complete ===="
echo "API server is running on http://0.0.0.0:8000"
echo "Server logs are in server.log"
echo "To stop the server, run: kill $SERVER_PID"
echo
echo "Example API call:"
echo "curl -X POST http://localhost:8000/v1/audio/speech/stream \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"input\":\"Hello world\", \"voice\":\"female\"}' \\"
echo "  --output test.pcm"

echo
echo "==== Recent server logs ===="
sleep 2
tail -n 10 server.log || true


