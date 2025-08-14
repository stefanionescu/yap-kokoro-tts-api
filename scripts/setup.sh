#!/bin/bash
set -e

# Always execute from repo root
SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")
cd "$ROOT_DIR"

echo "Setting up Orpheus TTS deployment environment..."

echo "Updating system packages..."
apt-get update
apt-get install -y libsndfile1 ffmpeg libopenmpi-dev python3-venv nano htop

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install lightweight deepspeed needed by vLLM DSFP importer (no CUDA kernels)
pip uninstall -y deepspeed || true
DS_BUILD_OPS=0 DS_BUILD_AIO=0 DS_BUILD_SPARSE_ATTN=0 \
  pip install deepspeed==0.14.4 --no-build-isolation --no-cache-dir
python - <<'PY'
import deepspeed
print('[setup] deepspeed', deepspeed.__version__)
PY

mkdir -p logs cache

echo "Setting up environment variables..."
echo ""
QUANT_METHOD="deepspeedfp"
echo "Quantization method fixed to DeepSpeed FP6/FP8 (deepspeedfp)"

cat > .env << EOL
# Model configuration
MODEL_NAME=canopylabs/orpheus-3b-0.1-ft
QUANTIZATION=$QUANT_METHOD
LOG_LEVEL=INFO

# VLLM parameters
GPU_MEMORY_UTILIZATION=0.9
# For compatibility with main.py which reads TRT_* envs
TRT_MAX_INPUT_LEN=8192
TRT_MAX_SEQ_LEN=8192
MAX_NUM_BATCHED_TOKENS=8192
MAX_NUM_SEQS=4
ENABLE_CHUNKED_PREFILL=True
VLLM_DISABLE_MULTIMODAL=1

# Voice specific parameters (latency-friendly defaults)
TEMPERATURE_TARA=0.5
TEMPERATURE_ZAC=0.3
TOP_P=0.95
REP_PENALTY_TARA=1.15
REP_PENALTY_ZAC=1.12

# Context parameters (for long-form text)
NUM_CTX=8192
NUM_PREDICT=49152

# Sampling parameters - matching Ollama reference implementation
MAX_TOKENS=49152
STOP_TOKEN_IDS=128258
N_EXTRA_AFTER_EOT=8192

# HuggingFace cache directory (for model downloads)
HF_HOME=

# Hugging Face auth token for gated models (set before running start.sh)
# Get one from https://huggingface.co/settings/tokens and paste below or
# export HF_TOKEN in the shell environment.
HF_TOKEN=

# Keep GPU kernels hot between requests
KEEP_GPU_HOT_INTERVAL=20
KEEP_GPU_HOT_VOICE=male
KEEP_GPU_HOT_PROMPT=.

# Stream priming to defeat proxy buffering (0=off, 1=on)
PRIME_STREAM=0
EOL

echo "Setup complete! You can now run scripts/start.sh to launch the API server."


