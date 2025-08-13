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

mkdir -p logs cache

echo "Setting up environment variables..."
echo ""
echo "Select quantization method:"
echo "1) DeepSpeed FP6/FP8 (deepspeedfp)"
echo "2) AWQ (6-bit)"
echo "3) GPTQ"
echo "4) SqueezeLLM"
echo "5) Marlin"
read -p "Enter choice [1-5] (default: 1): " quant_choice
case $quant_choice in
    2) QUANT_METHOD="awq"; echo "Selected AWQ";;
    3) QUANT_METHOD="gptq"; echo "Selected GPTQ";;
    4) QUANT_METHOD="squeezellm"; echo "Selected SqueezeLLM";;
    5) QUANT_METHOD="marlin"; echo "Selected Marlin";;
    *) QUANT_METHOD="deepspeedfp"; echo "Selected DeepSpeed FP6/FP8 (default)";;
esac
echo ""

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

# Voice specific parameters
TEMPERATURE_TARA=0.8
TEMPERATURE_ZAC=0.4
TOP_P=0.8
REP_PENALTY_TARA=1.9
REP_PENALTY_ZAC=1.85

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
EOL

echo "Setup complete! You can now run scripts/start.sh to launch the API server."


