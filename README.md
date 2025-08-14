# Orpheus TTS Deployment

FastAPI-based service for [Canopy Labs' Orpheus 3B](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) using vLLM with 6-bit (DeepSpeed FP6/FP8) quantization and SNAC decoding.

## Features

- 🚀 Optimized performance with DeepSpeed FP6/FP8
- 🔊 Low-latency PCM streaming (TTFB logged)
- 🗣️ Voice selector via API: `female` or `male` (internally mapped to Tara/Zac)
- 🔌 REST and WebSocket endpoints
- 📝 Centralized logging

## Prerequisites

- Hugging Face token with access to `canopylabs/orpheus-3b-0.1-ft` (set `HF_TOKEN`)

## Quick Start

1) Setup
```bash
bash scripts/setup.sh
```

2) Start (foreground)
```bash
bash scripts/start.sh
```

3) Start (background, survives closing web console)
```bash
bash scripts/start_bg.sh          # writes server.pid / server.pgid, logs → server.log
bash scripts/tail_bg_logs.sh      # view logs
bash scripts/stop.sh              # stop background server
```

4) Test HTTP streaming
```bash
curl -X POST http://localhost:8000/v1/audio/speech/stream \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello, this is a test of the Orpheus text-to-speech system.", "voice":"female"}' \
  --output test.pcm

# male voice
curl -X POST http://localhost:8000/v1/audio/speech/stream \
  -H "Content-Type: application/json" \
  -d '{"input":"Testing male voice.", "voice":"male"}' \
  --output test_male.pcm
```

5) Warmup (optional)
```bash
source venv/bin/activate
python warmup.py --host localhost --port 8000 --save   # saves warmup_audio/*.pcm
```

## API Docs

Visit:
```
http://localhost:8000/docs
```

## Voice Parameters

- female: temperature=0.8, top_p=0.8, repetition_penalty=1.9
- male:   temperature=0.4, top_p=0.8, repetition_penalty=1.85

## Long-form Settings

- num_ctx: 8192
- num_predict: 49152
- N_EXTRA_AFTER_EOT: 8192

## Deployment (RunPod)

Quick one-shot:
```bash
cd /workspace
git clone https://github.com/yourusername/yap-voice-model-deployment.git
cd yap-voice-model-deployment
export HF_TOKEN=hf_xxx; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
bash scripts/fresh_install.sh
bash scripts/start_bg.sh
```

Manual:
```bash
bash scripts/setup.sh
bash scripts/start_bg.sh
bash scripts/tail_bg_logs.sh
source venv/bin/activate && python warmup.py --save
```

Stopping and cleaning (keeps web console alive):
```bash
bash scripts/stop.sh                 # stop background server
bash scripts/purge_pod.sh            # stop server group only (safe)
bash scripts/purge_pod.sh --clean-files   # also remove venv/model/cache/logs
```

Token setup:
```bash
export HF_TOKEN=hf_xxx
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
```

## Configuration (.env)

- MODEL_NAME, QUANTIZATION (`deepspeedfp`)
- GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN (default 8192)
- NUM_CTX, NUM_PREDICT, MAX_TOKENS
- HF_HOME (cache dir), HF_TOKEN (required for gated model)

Runtime envs (sensible defaults set by scripts/start.sh):
- TORCH_COMPILE_DISABLE=1, TORCHDYNAMO_DISABLE=1 (disable Torch compile for DSFP)
- KV_CACHE_DTYPE=fp8 (smaller KV cache); set to `auto` if you prefer default
- VLLM_LOGGING_LEVEL=INFO (use DEBUG for detailed logs)

Notes:
- DSFP-6 is enabled by `quantization=deepspeedfp` and a `model/quant_config.json` with `{ "bits": 6, "group_size": 512 }` (generated automatically on first start)
- Voices exposed by API: `female`, `male`

## Components

- `main.py` – FastAPI app and endpoints
- `src/vllm.py` – vLLM integration (token gen + auth + quantization)
- `src/decoder.py` – SNAC decoding to PCM
- `src/logger.py` – centralized logging

## Requirements

See `requirements.txt` (vLLM 0.10.0, torch 2.7.1).

## Remote client

Use the included client to call the API from your local machine or another host:
```bash
python client.py --host <RUNPOD_PUBLIC_IP> --port 8000 \
  --text "Hello there" --voice female --out hello.pcm
```

## Acknowledgements

- [Canopy Labs](https://canopylabs.ai/) – Orpheus TTS
- [vLLM](https://github.com/vllm-project/vllm)
- [SNAC](https://github.com/hubert-siuzdak/snac/)