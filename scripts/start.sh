#!/bin/bash
set -e

# Always execute from repo root
SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")
cd "$ROOT_DIR"

# Load environment variables
source .env

# Activate virtual environment if not already activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo "Checking for model files..."
if [ ! -d "./cache/models--${MODEL_NAME/\//__}" ]; then
    echo "Model not found in cache, downloading may take some time..."
    echo "Note: You can stop the download after the tokenizer and safetensors files are downloaded"
    echo "      and set TRANSFORMERS_OFFLINE=1 to use local files only."
fi

if [ -n "$TRANSFORMERS_OFFLINE" ]; then
    export TRANSFORMERS_OFFLINE=1
    echo "Running in offline mode (TRANSFORMERS_OFFLINE=1)"
fi

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

# Force eager/no-compile for DSFP path unless overridden, and set kv-cache dtype
export TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE:-1}
export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}
export KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
echo "Torch compile disabled: TORCH_COMPILE_DISABLE=$TORCH_COMPILE_DISABLE, TORCHDYNAMO_DISABLE=$TORCHDYNAMO_DISABLE"
echo "vLLM KV cache dtype: $KV_CACHE_DTYPE | vLLM logging: $VLLM_LOGGING_LEVEL"

# Export HF tokens for gated models
if [ -n "$HUGGING_FACE_HUB_TOKEN" ]; then
    export HUGGING_FACE_HUB_TOKEN
elif [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

# Ensure HF_HOME is set to repo cache
if [ -z "${HF_HOME:-}" ]; then
    export HF_HOME="$ROOT_DIR/cache"
fi

# Prepare local model snapshot (downloads only inference files and removes rope_scaling)
SNAC_DIR="$ROOT_DIR/snac_model" SNAC_MODEL_PATH="$ROOT_DIR/snac_model" bash "$SCRIPT_DIR/prepare_model.sh"

# Launch uvicorn in a dedicated process group so we can kill the whole tree by PGID
mkdir -p logs
setsid bash -lc "uvicorn main:app --host $HOST --port $PORT --log-level ${LOG_LEVEL,,} --workers 1" \
  > logs/app.out 2>&1 < /dev/null &

SVR_PID=$!
SVR_PGID=$(ps -o pgid= -p "$SVR_PID" | tr -d ' ')
echo "$SVR_PID"  > server.pid
echo "$SVR_PGID" > server.pgid
echo "[start] server pid=$SVR_PID pgid=$SVR_PGID (logs at logs/app.out)"


