#!/bin/bash
set -euo pipefail

# Always execute from repo root
SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")
cd "$ROOT_DIR"

# Read env
source .env || true

MODEL_PATH_DEFAULT="/workspace/orpheus-tts/model"
HF_REPO=${MODEL_NAME:-canopylabs/orpheus-3b-0.1-ft}

# Temporarily force online mode for downloads
PREV_HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-}
PREV_TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-}
unset HF_HUB_OFFLINE
export TRANSFORMERS_OFFLINE=0

echo "[prepare_model] Preparing local snapshot for $HF_REPO"

# Ensure venv is active (needed for huggingface_hub)
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  source venv/bin/activate
fi

if [[ ! -d "$HF_REPO" ]]; then
python - <<'PY'
import os
from huggingface_hub import snapshot_download

repo=os.getenv("MODEL_NAME","canopylabs/orpheus-3b-0.1-ft")
token=os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
local_dir=os.getenv("MODEL_PATH_DEFAULT","/workspace/orpheus-tts/model")

allow=[
  "*.safetensors",
  "config.json",
  "generation_config.json",
  "tokenizer.json",
  "tokenizer_config.json",
  "special_tokens_map.json",
  "*.model","*.spm","*.tiktoken*",
]
ignore=[
  "optimizer*.bin",
  "pytorch_model_fsdp.bin",
  "training_args*",
  "trainer_state*",
  "rng_state_*.pth",
  "scheduler.pt",
  "*.bin",
]

snapshot_download(repo_id=repo, token=token, local_dir=local_dir,
                  local_dir_use_symlinks=False,
                  allow_patterns=allow, ignore_patterns=ignore)
print("[prepare_model] Downloaded only inference files to", local_dir)
PY
else
  echo "[prepare_model] Local model directory detected: $HF_REPO"
fi

# Patch/normalize rope_scaling by removing it (not needed for 8k)
python - <<'PY'
import os, json, glob
local_dir=os.getenv("MODEL_PATH_DEFAULT","/workspace/orpheus-tts/model")
changed=False
for p in glob.glob(local_dir+"/**/config.json", recursive=True):
    try:
        with open(p,"r",encoding="utf-8") as f: cfg=json.load(f)
        c=False
        if "rope_scaling" in cfg: cfg.pop("rope_scaling"); c=True
        if isinstance(cfg.get("text_config"),dict) and "rope_scaling" in cfg["text_config"]:
            cfg["text_config"].pop("rope_scaling"); c=True
        if c:
            with open(p,"w",encoding="utf-8") as f: json.dump(cfg,f)
            print("[prepare_model] Removed rope_scaling in", p)
            changed=True
    except Exception as e:
        print("[prepare_model] Skip", p, e)
print("[prepare_model] Patch complete; changed:", changed)
PY

# Also prepare SNAC locally to avoid online fetches (always attempt; idempotent)
python - <<'PY'
import os
from huggingface_hub import snapshot_download
token=os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
snac_dir="/workspace/orpheus-tts/snac_model"
snapshot_download(repo_id="hubertsiuzdak/snac_24khz", token=token,
                  local_dir=snac_dir, local_dir_use_symlinks=False)
print("[prepare_model] SNAC downloaded to", snac_dir)
PY

# Persist SNAC path
if ! grep -q '^SNAC_MODEL_PATH=' .env 2>/dev/null; then
  echo "SNAC_MODEL_PATH=/workspace/orpheus-tts/snac_model" >> .env
fi

# Persist env to use local model and offline mode
if ! grep -q '^MODEL_NAME=' .env 2>/dev/null || [[ "${MODEL_NAME:-}" != "$MODEL_PATH_DEFAULT" ]]; then
  sed -i '/^MODEL_NAME=/d' .env || true
  echo "MODEL_NAME=$MODEL_PATH_DEFAULT" >> .env
fi

# Restore offline mode and persist
export TRANSFORMERS_OFFLINE=1
sed -i '/^TRANSFORMERS_OFFLINE=/d' .env || true
echo "TRANSFORMERS_OFFLINE=1" >> .env

# Restore previous HF_HUB_OFFLINE if it existed
if [[ -n "$PREV_HF_HUB_OFFLINE" ]]; then export HF_HUB_OFFLINE=$PREV_HF_HUB_OFFLINE; else unset HF_HUB_OFFLINE; fi
if [[ -n "$PREV_TRANSFORMERS_OFFLINE" ]]; then export TRANSFORMERS_OFFLINE=$PREV_TRANSFORMERS_OFFLINE; fi

echo "[prepare_model] MODEL_NAME set to $MODEL_PATH_DEFAULT and offline mode enabled."


