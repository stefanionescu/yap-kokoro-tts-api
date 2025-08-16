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

echo "Setting up environment variables..."
echo ""
echo "Quantization disabled (not applicable to Kokoro)"

cat > .env << EOL
# Model configuration
MODEL_NAME=hexgrad/Kokoro-82M
QUANTIZATION=none
LOG_LEVEL=INFO

GPU_MEMORY_UTILIZATION=0.9

# Kokoro voice mapping
DEFAULT_VOICE_FEMALE=aoede
DEFAULT_VOICE_MALE=michael
LANG_CODE=a
KOKORO_SPEED=1.0
KOKORO_SPLIT_PATTERN="\\n+"
STREAM_CHUNK_SECONDS=0.5

# Context parameters (not used by Kokoro, kept for API compat logging)
NUM_CTX=8192
NUM_PREDICT=49152

# HuggingFace cache directory (for model downloads)
HF_HOME=

# Hugging Face auth token for gated models (set before running start.sh)
# Get one from https://huggingface.co/settings/tokens and paste below or
# export HF_TOKEN in the shell environment.
HF_TOKEN=

# Stream priming to defeat proxy buffering (0=off, 1=on)
PRIME_STREAM=0
 
# Optional CUDA knobs
# KOKORO_DEVICE=cuda
# KOKORO_GPU_MEMORY_FRACTION=0.90
EOL

echo "Setup complete! You can now run scripts/start.sh to launch the API server."


