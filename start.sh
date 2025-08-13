#!/bin/bash
set -e

# Load environment variables
source .env

# Activate virtual environment if not already activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if model is downloaded
echo "Checking for model files..."
if [ ! -d "./cache/models--${MODEL_NAME/\//__}" ]; then
    echo "Model not found in cache, downloading may take some time..."
    echo "Note: You can stop the download after the tokenizer and safetensors files are downloaded"
    echo "      and set TRANSFORMERS_OFFLINE=1 to use local files only."
fi

# Export additional environment variables for offline mode if needed
if [ -n "$TRANSFORMERS_OFFLINE" ]; then
    export TRANSFORMERS_OFFLINE=1
    echo "Running in offline mode (TRANSFORMERS_OFFLINE=1)"
fi

# Configure server port (default: 8000 for API)
PORT=${PORT:-8000}
HOST=${HOST:-"0.0.0.0"}

echo "Starting Orpheus TTS API server on $HOST:$PORT..."
if [ "$QUANTIZATION" = "deepspeedfp" ]; then
    echo "Model: $MODEL_NAME with DeepSpeed FP6/FP8 quantization (optimized for L40S)"
else
    echo "Model: $MODEL_NAME with $QUANTIZATION quantization"
fi
echo "Log level: $LOG_LEVEL"
echo "Voice settings:"
echo " - tara (female): temperature=$TEMPERATURE_TARA, top_p=$TOP_P, repetition_penalty=$REP_PENALTY_TARA"
echo " - zac (male): temperature=$TEMPERATURE_ZAC, top_p=$TOP_P, repetition_penalty=$REP_PENALTY_ZAC"
echo "Context window: $NUM_CTX, Max prediction: $NUM_PREDICT"

# Export HF token if provided in .env
if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN
fi

# Start the server
uvicorn main:app --host $HOST --port $PORT --log-level ${LOG_LEVEL,,} --workers 1