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

# If MODEL_NAME is already a local directory, nothing to do
if [[ -d "$HF_REPO" ]]; then
  echo "[prepare_model] Local model directory detected: $HF_REPO"
  exit 0
fi

echo "[prepare_model] Preparing local snapshot for $HF_REPO"

# Ensure venv is active (needed for huggingface_hub)
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  source venv/bin/activate
fi

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

# Persist env to use local model and offline mode
if ! grep -q '^MODEL_NAME=' .env 2>/dev/null || [[ "${MODEL_NAME:-}" != "$MODEL_PATH_DEFAULT" ]]; then
  sed -i '/^MODEL_NAME=/d' .env || true
  echo "MODEL_NAME=$MODEL_PATH_DEFAULT" >> .env
fi

if ! grep -q '^TRANSFORMERS_OFFLINE=' .env 2>/dev/null; then
  echo "TRANSFORMERS_OFFLINE=1" >> .env
fi

echo "[prepare_model] MODEL_NAME set to $MODEL_PATH_DEFAULT and offline mode enabled."


