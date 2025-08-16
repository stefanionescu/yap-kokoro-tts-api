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

echo "Starting Kokoro TTS API server on $HOST:$PORT..."
echo "Model: $MODEL_NAME (quantization: $QUANTIZATION)"
echo "Log level: $LOG_LEVEL"
echo "Kokoro voices: female=${DEFAULT_VOICE_FEMALE:-aoede}, male=${DEFAULT_VOICE_MALE:-michael}"
echo "Speed: ${KOKORO_SPEED:-1.0} | Split: ${KOKORO_SPLIT_PATTERN:-\\n+} | Chunk: ${STREAM_CHUNK_SECONDS:-0.5}s"

# Force priming and smaller initial chunks for proxy friendliness unless overridden
export PRIME_STREAM=${PRIME_STREAM:-1}
export STREAM_CHUNK_SECONDS=${STREAM_CHUNK_SECONDS:-0.1}
echo "Priming: $PRIME_STREAM | Stream chunk seconds: $STREAM_CHUNK_SECONDS"

# Optional CUDA reservation/capping
if [ -n "${KOKORO_DEVICE:-}" ]; then
  echo "Forcing Kokoro device: $KOKORO_DEVICE"
fi
if [ -n "${KOKORO_GPU_MEMORY_FRACTION:-}" ]; then
  echo "Per-process GPU memory fraction: $KOKORO_GPU_MEMORY_FRACTION"
fi

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

# No heavy model prep needed for Kokoro; it will download via pip/hub as needed

if command -v setsid >/dev/null 2>&1; then
  # Launch uvicorn in a dedicated process group so we can kill the whole tree by PGID (Linux)
  setsid bash -lc "uvicorn main:app --host $HOST --port $PORT --log-level ${LOG_LEVEL,,} --workers 1 --http httptools --loop uvloop --timeout-keep-alive 120" \
    > server.log 2>&1 < /dev/null &
  SVR_PID=$!
  SVR_PGID=$(ps -o pgid= -p "$SVR_PID" | tr -d ' ')
  echo "$SVR_PID"  > server.pid
  echo "$SVR_PGID" > server.pgid
  echo "[start] server pid=$SVR_PID pgid=$SVR_PGID (logs at server.log)"
else
  # macOS fallback without setsid
  nohup uvicorn main:app --host "$HOST" --port "$PORT" --log-level "${LOG_LEVEL,,}" --workers 1 --http httptools --loop uvloop --timeout-keep-alive 120 \
    > server.log 2>&1 &
  SVR_PID=$!
  echo "$SVR_PID" > server.pid
  echo "[start] server pid=$SVR_PID (logs at server.log)"
fi


