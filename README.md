# Kokoro TTS Deployment

FastAPI-based service for Kokoro-82M using the official `kokoro` Python package. No quantization or separate decoder required.

## Features

- 🔊 Low-latency PCM16 streaming at 24 kHz (TTFB logged)
- 🗣️ Voice selector via API: `female` or `male` (mapped to Kokoro voices Aoede/Michael)
- 🔌 REST and WebSocket endpoints
- 📝 Centralized logging

## Prerequisites

- Python 3.10+
- Optional: espeak-ng (for English OOD fallback and some languages)

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
  -d '{"input":"Hello, this is a test of the Kokoro text-to-speech system.", "voice":"female"}' \
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

## Voices

- API voices exposed: `female` → Kokoro voice `aoede` (American), `male` → `michael` (American)
  - Override via `.env`: `DEFAULT_VOICE_FEMALE`, `DEFAULT_VOICE_MALE`

## Audio

- PCM16 mono at 24000 Hz is streamed by default (saved by curl as `.pcm`)
- OPUS (Ogg/Opus) available by setting `{"format":"opus"}` if `ffmpeg` is installed

## Deployment (RunPod)

Quick one-shot:
```bash
cd /workspace
git clone https://github.com/yourusername/yap-voice-model-deployment.git
cd yap-voice-model-deployment
bash scripts/setup.sh
bash scripts/start_bg.sh
```

Manual:
```bash
bash scripts/setup.sh
bash scripts/start_bg.sh
bash scripts/tail_bg_logs.sh
source venv/bin/activate && python test/warmup.py --save
```

Stopping and cleaning (keeps web console alive):
```bash
bash scripts/stop.sh                 # stop background server
bash scripts/purge_pod.sh            # stop server group only (safe)
bash scripts/purge_pod.sh --clean-files   # also remove venv/model/cache/logs
```

## Configuration (.env)

- MODEL_NAME (default `hexgrad/Kokoro-82M`), QUANTIZATION=`none`
- DEFAULT_VOICE_FEMALE=`aoede`, DEFAULT_VOICE_MALE=`michael`, LANG_CODE=`a`
- KOKORO_SPEED, KOKORO_SPLIT_PATTERN, STREAM_CHUNK_SECONDS (default 0.25)
- FIRST_SEGMENT_MAX_WORDS (default 10), FIRST_SEGMENT_BOUNDARIES (first-chunk cut; for sub-200ms TTFB)
- HF_HOME (cache dir)

Runtime envs:
- None required beyond defaults. `LANG_CODE`, `DEFAULT_VOICE_*`, and `STREAM_CHUNK_SECONDS` can be customized in `.env`.

Notes:
- The API preserves the same endpoints and payload shape as before, but sampling params are ignored by Kokoro.

## Components

- `main.py` – FastAPI app and endpoints
- `src/engine.py` – Kokoro integration, fast-TTFB segmentation, PCM/OPUS streaming
- `src/logger.py` – centralized logging

## Requirements

See `requirements.txt` (kokoro, misaki[en], torch).

## Remote client

Use the included client to call the API from your local machine or another host:
```bash
python client.py --host <RUNPOD_PUBLIC_IP> --port 8000 \
  --text "Hello there" --voice female --out hello.pcm
```

## References

- Kokoro GitHub: [hexgrad/kokoro](https://github.com/hexgrad/kokoro)
- Kokoro-82M on Hugging Face: [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)