## Kokoro TTS Deployment (Pipecat‑ready)

FastAPI service around Kokoro‑82M with a persistent WebSocket protocol designed for low‑latency sentence‑level streaming and barge‑in. Output is 24 kHz mono PCM16.

### Run
```bash
bash scripts/setup.sh
bash scripts/start.sh
bash scripts/tail_bg_logs.sh
# optional warmup (on the pod)
source venv/bin/activate && python test/warmup.py --save
```

### Protocol (WebSocket only)
- Route: `/v1/audio/speech/stream/ws`
- Messages:
  - Client → Server
    - `{"type":"start", "voice":"female", "format":"pcm", "sample_rate":24000, "timestamps":false}`
    - For each sentence: `{"type":"speak", "request_id":"uuid", "text":"Okay, let's go.", "voice":"female", "speed":1.0}`
    - Barge‑in: `{"type":"cancel", "request_id":"uuid"}`
    - End session: `{"type":"stop"}`
  - Server → Client
    - `{"type":"started"}` then `{"type":"started_speak", "request_id":"uuid"}`
    - Binary frames with 20–40 ms PCM16 chunks
    - Periodic `{"type":"meta", "request_id":"uuid", ...}`
    - Completion: `{"type":"done", "request_id":"uuid", "duration_s":0.62}` or `{"type":"canceled", ...}`

Notes:
- One connection stays open for many sentences. Concurrency is handled inside the engine via a fair round‑robin scheduler.
- Cancel is immediate: queued or in‑flight audio for that `request_id` is dropped.

### Pipecat integration
- Use a custom WebSocket TTS service or adapt PipeCat’s OpenAI TTS service if you prefer sentence‑per‑HTTP (slightly higher latency).
- Configure PCM output at 24 kHz mono. Sentence aggregation should remain enabled in Pipecat.

Minimal client example is provided in `test/client.py` and a load benchmark in `test/bench.py` using the new protocol.

### Example WS session
```json
{ "type": "start", "voice": "female", "format": "pcm", "sample_rate": 24000 }
{ "type": "speak", "request_id": "uuid-1", "text": "Okay, let's go." }
<binary PCM> ...
{ "type": "done", "request_id": "uuid-1" }
{ "type": "stop" }
```

### Custom voices (on‑pod only)
- Blends are persisted in `custom_voices/custom_voices.json`.
- Create/update a blend (recipe is `+`‑separated base voices, repeats weight the mix):
```bash
python voices/custom_voices.py add --name my_blend --recipe "af_aoede+af_nicole" --validate
# 60/40 weighting (10 parts): 6× aoede, 4× nicole
python voices/custom_voices.py add --name aoede60_nicole40 \
  --recipe "af_aoede+af_aoede+af_aoede+af_aoede+af_aoede+af_aoede+af_nicole+af_nicole+af_nicole+af_nicole" --validate
```
- Use in API calls by setting `voice:"my_blend"`.
- List/remove:
```bash
python voices/custom_voices.py list
python voices/custom_voices.py remove --name my_blend
```
Restart the server after changes:
```bash
bash scripts/stop.sh || true && bash scripts/start.sh
```

### Benchmark & find optimal concurrency
```bash
# On the pod
source venv/bin/activate
# Try several conc values to find the sweet spot
python test/bench.py --proto ws --n 60 --concurrency 1
python test/bench.py --proto ws --n 60 --concurrency 2
python test/bench.py --proto ws --n 60 --concurrency 4
python test/bench.py --proto ws --n 60 --concurrency 8
python test/bench.py --proto ws --n 60 --concurrency 12
```
- Pick the highest concurrency where p95 TTFB is acceptable. Set it via `MAX_CONCURRENT_JOBS` (see below) and scale replicas for more QPS.

### Config knobs (sane defaults set by setup.sh)
- TTFB/streaming
  - `FIRST_SEGMENT_MAX_WORDS` (default 3)
  - `FIRST_SEGMENT_BOUNDARIES` (default `.,?!;:`)
  - `STREAM_CHUNK_SECONDS` (default 0.02)
  - `PRIME_STREAM=1`, `PRIME_BYTES=512`
- Concurrency/backpressure
  - `MAX_CONCURRENT_JOBS` (default 12)
  - `QUEUE_MAXSIZE` (default 128) – excess requests block in queue
- GPU
  - `KOKORO_DEVICE` (e.g., `cuda:0`)
  - `KOKORO_GPU_MEMORY_FRACTION` (e.g., `0.95`) – soft cap; use MIG for hard isolation

### Health & status
- `/healthz` (200 OK), `/readyz` (engine + device)

### Notes
- Voices come from Kokoro‑82M: see the official list (`af_aoede`, `am_michael`, etc.).
- Default API voices: `female→af_aoede`, `male→am_michael`.
- Audio: PCM16 mono @ 24 kHz by default; Ogg/Opus with `{"format":"opus"}` if ffmpeg installed.

### References
- Kokoro GitHub: https://github.com/hexgrad/kokoro
- Kokoro‑82M model card: https://huggingface.co/hexgrad/Kokoro-82M