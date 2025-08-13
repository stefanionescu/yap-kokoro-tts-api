#!/bin/bash
set -e

# Setup script for Orpheus TTS on RunPod with vLLM and 6-bit quantization
echo "Setting up Orpheus TTS deployment environment..."

# Update and install system packages
echo "Updating system packages..."
apt-get update
apt-get install -y libsndfile1 ffmpeg libopenmpi-dev python3-venv nano htop

# Create and activate Python virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs
mkdir -p cache

# Set environment variables
echo "Setting up environment variables..."

# Choose quantization method
echo ""
echo "Select quantization method:"
echo "1) DeepSpeed FP6/FP8 (recommended for L40S GPU, higher performance)"
echo "2) AWQ (alternative 6-bit quantization)"
read -p "Enter choice [1-2] (default: 1): " quant_choice
case $quant_choice in
    2)
        QUANT_METHOD="awq"
        echo "Selected AWQ quantization"
        ;;
    *)
        QUANT_METHOD="deepspeedfp"
        echo "Selected DeepSpeed FP6/FP8 quantization (default)"
        ;;
esac
echo ""

cat > .env << EOL
# Model configuration
MODEL_NAME=canopylabs/orpheus-3b-0.1-ft
QUANTIZATION=$QUANT_METHOD
LOG_LEVEL=INFO

# VLLM parameters
GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN=2048
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

# Sampling parameters
MAX_TOKENS=2000
STOP_TOKEN_IDS=128258

# HuggingFace cache directory (for model downloads)
HF_HOME=$(pwd)/cache
EOL

echo "Setup complete! You can now run start.sh to launch the API server."