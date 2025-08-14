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

# Try loading tokens from a repo-level .env if present
if [[ -f "$SCRIPT_DIR/../.env" ]]; then
  # shellcheck disable=SC2046
  export $(grep -E '^(HF_TOKEN|HUGGING_FACE_HUB_TOKEN)=' "$SCRIPT_DIR/../.env" | xargs -d '\n') || true
fi

# Normalize token env
if [[ -n "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

if [[ -z "${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}" ]]; then
  echo "[run_all_dsfp] WARN: HUGGING_FACE_HUB_TOKEN/HF_TOKEN not set. If the base repo is gated, export it first." >&2
fi

if [[ -z "${REPO_ID:-}" ]]; then
  echo "[run_all_dsfp] REPO_ID not set; attempting to derive from token..." >&2
  REPO_ID=$(python - <<'PY'
import os
from huggingface_hub import HfApi
tok = os.environ.get('HUGGING_FACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
if not tok:
    print('')
else:
    try:
        u = HfApi(token=tok).whoami()
        # default repo name
        print(f"{u.get('name')}/orpheus-3b-dsfp6")
    except Exception:
        print('')
PY
  )
  if [[ -z "$REPO_ID" ]]; then
    echo "[run_all_dsfp] ERROR: Could not determine REPO_ID. Export REPO_ID or ensure token is valid." >&2
    exit 1
  else
    export REPO_ID
    echo "[run_all_dsfp] Using REPO_ID=$REPO_ID"
  fi
fi

# Create dedicated venv for DSFP export to avoid conflicts with runtime venv
if [[ -d venv ]]; then
  echo "[run_all_dsfp] Reusing existing venv"
else
  python3 -m venv venv
fi
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements.txt

# Ensure torch is present; install a CUDA build if not found
if ! python - <<'PY'
import importlib.util
exit(0 if importlib.util.find_spec('torch') else 1)
PY
then
  echo "[run_all_dsfp] torch not found; attempting install..." >&2
  if [[ -n "${TORCH_SPEC:-}" ]]; then
    if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
      pip install --no-cache-dir --extra-index-url "$TORCH_INDEX_URL" "$TORCH_SPEC"
    else
      pip install --no-cache-dir "$TORCH_SPEC"
    fi
  else
    # Try common CUDA wheels, fallback to CPU
    python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 || \
    python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 || \
    python -m pip install --no-cache-dir torch
  fi
fi

echo "[run_all_dsfp] Step 1: Fetch base FP model..."
python 01_fetch.py

echo "[run_all_dsfp] Step 2: Export DeepSpeedFP (FP6) weights..."
python 02_dsfp_export.py

if [[ -z "${SKIP_VALIDATE:-}" ]]; then
  echo "[run_all_dsfp] Step 3: Validate DSFP model with vLLM..."
  python 03_validate_dsfp.py || echo "[run_all_dsfp] Validation used a tiny generate; continuing."
else
  echo "[run_all_dsfp] Skipping validation (SKIP_VALIDATE set)."
fi

echo "[run_all_dsfp] Step 4: Push to Hugging Face: $REPO_ID"
SRC=./dsfp_model REPO_ID="$REPO_ID" python 04_push_hf.py

echo "[run_all_dsfp] Done. Set in deployment .env:"
echo "  MODEL_NAME=$REPO_ID"
echo "  QUANTIZATION=deepspeedfp"


