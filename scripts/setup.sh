#!/bin/bash
set -e

# Always execute from repo root
SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")
cd "$ROOT_DIR"

echo "Setting up Kokoro TTS deployment environment..."

echo "Updating system packages..."
if command -v apt-get >/dev/null 2>&1; then
  apt-get update
  apt-get install -y libsndfile1 ffmpeg libopenmpi-dev python3-venv nano htop espeak-ng
else
  echo "apt-get not found (likely macOS). Please ensure espeak-ng is installed if needed."
fi

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

mkdir -p logs cache

echo "Setting up environment variables (WS-only, Pipecat-ready)..."
echo ""

cat > .env << EOL
# Model configuration
MODEL_NAME=hexgrad/Kokoro-82M
LOG_LEVEL=INFO

# Kokoro voice mapping
DEFAULT_VOICE_FEMALE=af_aoede
DEFAULT_VOICE_MALE=am_michael
LANG_CODE=a
KOKORO_SPEED=1.0
STREAM_CHUNK_SECONDS=0.02
FIRST_SEGMENT_MAX_WORDS=2
FIRST_SEGMENT_BOUNDARIES=.,?!;:
FIRST_SEGMENT_REQUIRE_BOUNDARY=1
MAX_UTTERANCE_WORDS=150

# Stream priming to defeat proxy buffering (0=off, 1=on)
PRIME_STREAM=1
PRIME_BYTES=512
 
# Optional CUDA/throughput knobs
KOKORO_DEVICE=cuda:0
KOKORO_GPU_MEMORY_FRACTION=0.95
MAX_CONCURRENT_JOBS=8
QUEUE_MAXSIZE=128
SCHED_QUANTUM_BYTES=4096
PRIORITY_QUANTUM_BYTES=1024
QUEUE_WAIT_SLA_MS=800

# WebSocket send controls (small buffers for fast TTFB)
WS_FIRST_CHUNK_IMMEDIATE=1
WS_BUFFER_BYTES=960
WS_FLUSH_EVERY=1
WS_SEND_TIMEOUT=3.0
WS_LONG_SEND_LOG_MS=250
EOL

echo "Setup complete! You can now run scripts/start.sh to launch the API server."


