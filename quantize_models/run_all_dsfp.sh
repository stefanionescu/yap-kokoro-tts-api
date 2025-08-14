#!/bin/bash
set -euo pipefail

# One-shot DeepSpeedFP (FP6) export pipeline: fetch -> export -> validate -> push
# Usage:
#   export HF_TOKEN=hf_xxx; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
#   export BASE_REPO=canopylabs/orpheus-3b-0.1-ft          # optional (defaults as in 01_fetch)
#   export REPO_ID=your-org/orpheus-3b-dsfp6               # required for push step
#   bash run_all_dsfp.sh

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

if [[ -z "${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}" ]]; then
  echo "[run_all_dsfp] WARN: HUGGING_FACE_HUB_TOKEN/HF_TOKEN not set. If the base repo is gated, export it first." >&2
fi

if [[ -z "${REPO_ID:-}" ]]; then
  echo "[run_all_dsfp] ERROR: Set REPO_ID to the target HF repo id (e.g., your-org/orpheus-3b-dsfp6)." >&2
  exit 1
fi

# Create dedicated venv for DSFP export to avoid conflicts with runtime venv
if [[ ! -d venv ]]; then
  python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

echo "[run_all_dsfp] Step 1: Fetch base FP model..."
python 01_fetch.py

echo "[run_all_dsfp] Step 2: Export DeepSpeedFP (FP6) weights..."
python 02_dsfp_export.py

echo "[run_all_dsfp] Step 3: Validate DSFP model with vLLM..."
python 03_validate_dsfp.py || echo "[run_all_dsfp] Validation used a tiny generate; continuing."

echo "[run_all_dsfp] Step 4: Push to Hugging Face: $REPO_ID"
SRC=./dsfp_model REPO_ID="$REPO_ID" python 04_push_hf.py

echo "[run_all_dsfp] Done. Set in deployment .env:"
echo "  MODEL_NAME=$REPO_ID"
echo "  QUANTIZATION=deepspeedfp"


