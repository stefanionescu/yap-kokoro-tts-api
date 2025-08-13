#!/bin/bash
# Script to deploy and run the Orpheus TTS service on RunPod

# Step 1: Setup environment and install dependencies
echo "==== Setting up environment ===="
./setup.sh

# Step 2: Start the API server in the background
echo "==== Starting API server ===="
nohup ./start.sh > server.log 2>&1 &
SERVER_PID=$!
echo "API server started with PID: $SERVER_PID"

# Step 3: Wait for the server to initialize
echo "Waiting for the server to initialize (30 seconds)..."
sleep 30

# Step 4: Run the warmup script to optimize performance
echo "==== Running model warmup ===="
python warmup.py --save

# Step 5: Display status information
echo "==== Deployment complete ===="
echo "API server is running on http://0.0.0.0:8000"
echo "Server logs are in server.log"
echo "To stop the server, run: kill $SERVER_PID"
echo
echo "Example API call:"
echo "curl -X POST http://localhost:8000/v1/audio/speech/stream \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"input\":\"Hello world\", \"voice\":\"tara\"}' \\"
echo "  --output test.pcm"

# Display server log tail
echo
echo "==== Recent server logs ===="
sleep 2
tail -n 10 server.log
