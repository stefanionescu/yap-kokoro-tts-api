#!/bin/bash
set -euo pipefail

# One-shot AWQ 6-bit quantization pipeline: fetch -> quantize -> validate -> push
# Usage:
#   export HF_TOKEN=hf_xxx; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
#   export BASE_REPO=canopylabs/orpheus-3b-0.1-ft          # optional (defaults as in 01_fetch)
#   export REPO_ID=your-org/orpheus-3b-awq-6bit            # required for push step
#   bash run_all_awq.sh

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

if [[ -z "${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}" ]]; then
  echo "[run_all_awq] ERROR: Set HUGGING_FACE_HUB_TOKEN or HF_TOKEN before running." >&2
  exit 1
fi

if [[ -z "${REPO_ID:-}" ]]; then
  echo "[run_all_awq] ERROR: Set REPO_ID to the target HF repo id (e.g., your-org/orpheus-3b-awq-6bit)." >&2
  exit 1
fi

# Create dedicated venv for quantization to avoid conflicts with runtime venv
if [[ ! -d venv ]]; then
  python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

echo "[run_all_awq] Step 1: Fetch base FP model..."
python 01_fetch.py

echo "[run_all_awq] Step 2: Quantize to AWQ 6-bit..."
python 02_awq_quantize.py

echo "[run_all_awq] Step 3: Validate quantized model..."
python 03_validate.py || echo "[run_all_awq] Validation printed sample decode; continuing."

echo "[run_all_awq] Step 4: Push to Hugging Face: $REPO_ID"
python 04_push_hf.py

echo "[run_all_awq] Done. Set in deployment .env:"
echo "  MODEL_NAME=$REPO_ID"
echo "  QUANTIZATION=awq"


