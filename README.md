## Kokoro TTS Deployment (OpenAI Realtime Compatible)

FastAPI service around Kokoro‑82M with a persistent WebSocket using an OpenAI Realtime‑compatible event schema for low‑latency sentence‑level streaming and barge‑in. Output is 24 kHz mono PCM16 (base64 in WS events).

All parameters were optimized for an L40S.

### Run
```bash
bash scripts/setup.sh
bash scripts/start.sh
bash scripts/tail_bg_logs.sh
# optional warmup (on the pod)
source venv/bin/activate && python test/warmup.py --save
```

Manual launch (alternative):
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --log-level info
```

### Authentication (production)
- The WebSocket endpoint requires an API key if `API_KEY` is set in `.env`.
- `scripts/setup.sh` creates `.env` with a default `API_KEY=dev-key`.
- For production, set a strong key:
```bash
sed -i 's/^API_KEY=.*/API_KEY=your-strong-key/' .env
bash scripts/stop.sh || true
bash scripts/start.sh
```
- Clients append the key as a query parameter, e.g. `wss://host/v1/audio/speech/stream/ws?api_key=your-strong-key`.
- Our tools (`test/client.py`, `test/bench.py`, `test/warmup.py`) read `API_KEY` from `.env` automatically and include it.

### Protocol (WebSocket only; OpenAI Realtime‑compatible)
- Route: `/v1/audio/speech/stream/ws`
- Messages:
  - Client → Server
    - `{"type":"session.update", "session": {"voice":"female", "audio": {"format":"pcm", "sample_rate":24000}}}`
    - For each sentence: `{"type":"response.create", "response_id":"uuid", "input":"Okay, let's go.", "voice":"female", "speed":1.0}`
    - Incremental input (optional): `{"type":"input.append", "text":"partial..."}` then `{"type":"input.commit"}` to flush
    - Barge‑in: `{"type":"response.cancel", "response":"uuid"}`
    - Immediate barge for incremental mode: `{"type":"barge"}` (stops playback and clears pending input)
    - End session: `{"type":"session.end"}`
  - Server → Client
    - `{"type":"session.updated"}` and `{"type":"response.created", "response":"uuid"}`
    - Streaming audio: `{"type":"response.output_audio.delta", "response":"uuid", "delta":"<base64>"}`
    - Completion: `{"type":"response.completed", "response":"uuid"}` or `{"type":"response.canceled", "response":"uuid"}`

Notes:
- One connection stays open for many sentences. Concurrency is handled inside the engine via a fair round‑robin scheduler.
- Cancel is immediate: queued or in‑flight audio for that `request_id` is dropped.
- Authentication: append `?api_key=YOUR_KEY` to the WS URL. Set `API_KEY` in `.env`.

### Pipecat integration
- Use a custom WebSocket TTS service or adapt Pipecat’s OpenAI TTS service if you prefer sentence‑per‑HTTP (slightly higher latency).
- Configure PCM output at 24 kHz mono. Sentence aggregation should remain enabled in Pipecat.

Minimal client example is provided in `test/client.py` and a load benchmark in `test/bench.py` using the new protocol.

By default, the client and tools read `RUNPOD_TCP_HOST` and `RUNPOD_TCP_PORT` from `.env`. Override on the CLI if needed.

### Example WS session
```json
{ "type": "session.update", "session": {"voice": "female", "audio": {"format":"pcm", "sample_rate": 24000}} }
{ "type": "response.create", "response_id": "uuid-1", "input": "Okay, let's go." }
{ "type": "response.created", "response": "uuid-1" }
{ "type": "response.output_audio.delta", "response": "uuid-1", "delta": "<base64>" }
{ "type": "response.completed", "response": "uuid-1" }
{ "type": "session.end" }
```

### Realtime incremental input (optional)
```json
{ "type": "session.update", "session": {"voice": "female", "audio": {"format":"pcm", "sample_rate": 24000}} }
{ "type": "input.append", "text": "Hello there" }
{ "type": "input.append", "text": ". This is" }
{ "type": "input.append", "text": " a demo." }
{ "type": "input.commit" }
```
Notes:
- The server buffers small `input.append` segments and auto‑flushes after a short idle (`FLUSH_IDLE_MS`, default 160 ms), or immediately on `input.commit`.
- Use `{"type":"barge"}` to stop current playback and clear any pending input.

### Examples: send full text vs sentence‑by‑sentence

All tools support both modes where applicable.

#### test/client.py
```bash
# Full text in one request
python test/client.py --voice female --text "Hello there. This is a demo." --out hello.wav --mode single

# Sentence‑by‑sentence (each sentence as its own request)
python test/client.py --voice female --text "Hello there. This is a demo." --out hello.wav --mode sentences
```

#### test/bench.py
```bash
# Full text per request
python test/bench.py --n 40 --concurrency 3 --mode single

# Sentence‑by‑sentence per request (averages TTFB across sentences)
python test/bench.py --n 40 --concurrency 3 --mode sentences
```

#### test/warmup.py
```bash
# Full text per request
python test/warmup.py --mode single

# Sentence‑by‑sentence per request
python test/warmup.py --mode sentences
```

#### test/tpm.py
```bash
# Full text per transaction (TX counted when the single request completes)
python test/tpm.py --mode single

# Sentence‑by‑sentence per transaction (TX counted when all sentences complete)
python test/tpm.py --mode sentences --duration 120
```

### Voices
- Only two logical voices are supported: `female` and `male`.
- The actual Kokoro voice IDs come from environment variables:
  - `DEFAULT_VOICE_FEMALE` (default: `af_heart`)
  - `DEFAULT_VOICE_MALE` (default: `am_michael`)
  
Custom voice names are not accepted directly by the API. Map your desired Kokoro voice IDs to `DEFAULT_VOICE_FEMALE`/`DEFAULT_VOICE_MALE` in `.env`.

### Benchmark & find optimal concurrency
```bash
# On the pod
source venv/bin/activate
# Try several conc values to find the sweet spot
python test/bench.py --n 60 --concurrency 1
python test/bench.py --n 60 --concurrency 2
python test/bench.py --n 60 --concurrency 3
python test/bench.py --n 60 --concurrency 4
```
- Pick the highest concurrency where p95 TTFB is acceptable. Set it via `MAX_CONCURRENT_JOBS` (see below) and scale replicas for more QPS.

### Config knobs (what they do)
- **Auth & logging**
  - `API_KEY`: If set, the WS endpoint requires `?api_key=...`. Tools auto‑append from `.env`.
  - `LOG_LEVEL`: Uvicorn/app log level (`DEBUG`, `INFO`, ...).

- **Model & voices**
  - `MODEL_NAME`: Hugging Face model id to load (e.g., `hexgrad/Kokoro-82M`).
  - `DEFAULT_VOICE_FEMALE`, `DEFAULT_VOICE_MALE`: Kokoro voice ids (or custom voice names) used when the client passes `voice: "female"|"male"`.
  - `LANG_CODE`: Model language variant; `a` is American English for Kokoro.
  - `KOKORO_SPEED`: Default speech speed (0.5–2.0). Per‑request `speed` overrides it.
  - `KOKORO_SPLIT_PATTERN`: Regex for sentence segmentation inside Kokoro (affects natural boundaries).

- **Streaming/TTFB behavior**
  - `FIRST_SEGMENT_MAX_WORDS`: Emit a tiny first piece (≤ N words) to minimize TTFB.
  - `FIRST_SEGMENT_BOUNDARIES`: Punctuation considered a natural cut for the first piece.
  - `FIRST_SEGMENT_REQUIRE_BOUNDARY`: If `1`, only cut on punctuation; otherwise fall back to a word cut.
  - `STREAM_CHUNK_SECONDS`: Engine chunk size in seconds (PCM per chunk). `0.02` → 20 ms (@24 kHz ≈ 480 samples).
  - `MAX_UTTERANCE_WORDS`: Reject a `response.create` if text exceeds this word count (returns `response.error` with `code: "too_long"`).
  - `PRIME_STREAM`: If `1`, send `PRIME_BYTES` all‑zero bytes once per speak to defeat proxy buffering (does not affect metrics).
  - `PRIME_BYTES`: Size of the primer chunk when `PRIME_STREAM=1`.

- **WebSocket send**
  - `WS_FIRST_CHUNK_IMMEDIATE`: Flush immediately once some audio is ready to cut TTFB.
  - `WS_BUFFER_BYTES`: Byte threshold to flush buffered PCM (typ. around the size of one engine chunk).
  - `WS_FLUSH_EVERY`: Flush after N chunks regardless of buffer bytes.
  - `WS_SEND_TIMEOUT`: Max seconds for a single send operation.
  - `WS_LONG_SEND_LOG_MS`: Log a warning when a send operation exceeds this duration.

- **Concurrency & admission control (API level)**
  - `MAX_CONCURRENT_JOBS`: Max concurrent synth streams actively running (engine scheduler `active_limit`).
  - `MAX_QUEUED_REQUESTS`: Max number of requests allowed to wait behind the running ones. Beyond this, the server replies `{code:"busy"}`.
  - `QUEUE_WAIT_SLA_MS`: Estimated start‑wait SLA. If the predicted wait to start streaming exceeds this, the request is rejected with `{code:"busy"}`.
  - `QUEUE_MAXSIZE`: Internal engine queue capacity (safety cap for `_TTSJob` objects). Not the public backlog; you usually won’t hit it.

- **Scheduler fairness (engine)**
  - `SCHED_QUANTUM_BYTES`: Bytes budget per scheduler turn for normal jobs (≈ how much PCM each stream emits before yielding).
  - `PRIORITY_QUANTUM_BYTES`: Bytes budget for priority work (keeps first pieces snappy under light load).

- **GPU device & memory**
  - `KOKORO_DEVICE`: Target device (`cuda:0`, `cpu`, ...). If CUDA is available, the engine binds to that device.
  - `KOKORO_GPU_MEMORY_FRACTION`: Soft cap for per‑process GPU memory (0.0–1.0).

- **Encoding (optional Ogg/Opus when requested)**
  - `OPUS_BITRATE`: Bitrate passed to ffmpeg when output format is `opus`.
  - `OPUS_APPLICATION`: `audio` (default) or `voip` passed to the Opus encoder.

### Health & status
- `/healthz` (200 OK), `/readyz` (engine + device)

### Authentication
- Set `API_KEY` in `.env` (created by `scripts/setup.sh`). Replace the default `dev-key` in production.
- Clients append the key to the WS URL: `/v1/audio/speech/stream/ws?api_key=YOUR_KEY`. Our client/bench/warmup add it automatically from `.env`.

### Metrics & reporting
- Each request writes JSON to `logs/metrics.log`: `ts`, `request_id`, `ttfb_ms`, `wall_s`, `audio_s`, `rtf`, `xrt`, `kbps`, `canceled`.
- Print rolling summaries:
```bash
source venv/bin/activate && python src/metrics.py
# Custom windows
python src/metrics.py --periods "30m,1h,6h,24h,3d"
```

### Purge / reset the pod
Use `scripts/purge_pod.sh` to stop the server and optionally clean runtime files.

Common operations:
```bash
# Stop only the TTS server on the default port (safe; keeps files intact)
bash scripts/purge_pod.sh --port 8000

# Stop the server and clean in-repo runtime files (includes logs/metrics.log)
bash scripts/purge_pod.sh --port 8000 --clean-files

# Aggressive cleanup (implies --clean-files and adds extra cache purges)
bash scripts/purge_pod.sh --port 8000 --aggressive

# Optional extras (combine as needed)
bash scripts/purge_pod.sh --port 8000 \
  --drop-caches --clear-shm --clear-tmp --gpu-reset --kill-jupyter --kill-sessions

# Remove system packages installed by setup.sh (ffmpeg/espeak/etc.) — irreversible in a live image
bash scripts/purge_pod.sh --purge-system

# Prune container cache (if a container engine is available in the environment)
bash scripts/purge_pod.sh --docker-prune

# Self-delete the pod via RunPod API (requires RUNPOD_API_KEY and RUNPOD_POD_ID)
bash scripts/purge_pod.sh --self-remove
```

Notes:
- `--clean-files` removes `venv/`, `cache/`, `logs/` (including `logs/metrics.log`), model caches, warmup audio, and process files. Use with care.
- `--aggressive` implies `--clean-files` and additionally purges pip/apt caches and Torch extension caches.
- We avoid wiping `/tmp` by default; pass `--clear-tmp` if you truly need it (may disrupt consoles).

### Notes
- Voices come from Kokoro‑82M: see the official list (`af_heart`, `am_michael`, etc.).
- Default API voices: `female→af_heart`, `male→am_michael`.
- Audio: PCM16 mono @ 24 kHz by default; Ogg/Opus with `{"format":"opus"}` if ffmpeg installed.

### References
- Kokoro GitHub: https://github.com/hexgrad/kokoro
- Kokoro‑82M model card: https://huggingface.co/hexgrad/Kokoro-82M