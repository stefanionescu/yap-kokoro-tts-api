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
echo "Model: $MODEL_NAME"
echo "Log level: $LOG_LEVEL"
echo "Kokoro voices: female=${DEFAULT_VOICE_FEMALE:-heart}, male=${DEFAULT_VOICE_MALE:-michael}"
echo "Speed: ${KOKORO_SPEED:-1.0} | Split: ${KOKORO_SPLIT_PATTERN:-\\n+} | Chunk: ${STREAM_CHUNK_SECONDS:-0.02}s"

# Disable priming by default and use small chunks for fast real audio
export PRIME_STREAM=${PRIME_STREAM:-0}
export STREAM_CHUNK_SECONDS=${STREAM_CHUNK_SECONDS:-0.02}
echo "Priming: $PRIME_STREAM | Stream chunk seconds: $STREAM_CHUNK_SECONDS"

# TTFB tuning defaults (overridable)
export FIRST_SEGMENT_MAX_WORDS=${FIRST_SEGMENT_MAX_WORDS:-2}
export FIRST_SEGMENT_BOUNDARIES=${FIRST_SEGMENT_BOUNDARIES:-",?!;:"}
export FIRST_SEGMENT_REQUIRE_BOUNDARY=${FIRST_SEGMENT_REQUIRE_BOUNDARY:-1}
# Conservative defaults to avoid model/library deadlocks under load
export MAX_CONCURRENT_JOBS=${MAX_CONCURRENT_JOBS:-4}
export QUEUE_MAXSIZE=${QUEUE_MAXSIZE:-128}
export PRIME_BYTES=${PRIME_BYTES:-512}
echo "First segment max words: $FIRST_SEGMENT_MAX_WORDS | boundaries: $FIRST_SEGMENT_BOUNDARIES | require_boundary: $FIRST_SEGMENT_REQUIRE_BOUNDARY"
echo "Concurrency: $MAX_CONCURRENT_JOBS | Queue size: $QUEUE_MAXSIZE"

# Optional CUDA reservation/capping
if [ -n "${KOKORO_DEVICE:-}" ]; then
  echo "Forcing Kokoro device: $KOKORO_DEVICE"
fi
if [ -n "${KOKORO_GPU_MEMORY_FRACTION:-}" ]; then
  echo "TTS max GPU memory fraction: $KOKORO_GPU_MEMORY_FRACTION"
fi

# CUDA allocator tuning to reduce fragmentation and improve long uptimes
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.9

# Thread limits to avoid CPU thrash on small hosts (Kokoro is GPU-bound)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# No heavy model prep needed for Kokoro; it will download via pip/hub as needed

if command -v setsid >/dev/null 2>&1; then
  # Launch uvicorn in a dedicated process group so we can kill the whole tree by PGID (Linux)
  setsid bash -lc "uvicorn src.main:app --host $HOST --port $PORT --log-level ${LOG_LEVEL,,} --workers 1 --http httptools --loop uvloop --ws websockets --timeout-keep-alive 120" \
    > server.log 2>&1 < /dev/null &
  SVR_PID=$!
  SVR_PGID=$(ps -o pgid= -p "$SVR_PID" | tr -d ' ')
  echo "$SVR_PID"  > server.pid
  echo "$SVR_PGID" > server.pgid
  echo "[start] server pid=$SVR_PID pgid=$SVR_PGID (logs at server.log)"
else
  # macOS fallback without setsid
  nohup uvicorn src.main:app --host "$HOST" --port "$PORT" --log-level "${LOG_LEVEL,,}" --workers 1 --http httptools --loop uvloop --ws websockets --timeout-keep-alive 120 \
    > server.log 2>&1 &
  SVR_PID=$!
  echo "$SVR_PID" > server.pid
  echo "[start] server pid=$SVR_PID (logs at server.log)"
fi


