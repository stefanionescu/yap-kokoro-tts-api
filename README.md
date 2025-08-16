## Kokoro TTS Deployment

FastAPI service around Kokoro‑82M (24 kHz PCM streaming). Minimal, production‑oriented.

### One‑shot run
```bash
bash scripts/setup.sh
bash scripts/start.sh
bash scripts/tail_bg_logs.sh
# optional warmup (on the pod)
source venv/bin/activate && python test/warmup.py --save
```

### Call the API (HTTP streaming)
```bash
curl -X POST http://localhost:8000/v1/audio/speech/stream \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello from Kokoro","voice":"female","format":"pcm"}' \
  --output out.pcm
```

### WebSocket (binary PCM)
- Route: `/v1/audio/speech/stream/ws`
- Send JSON `{continue:true, segment_id:"seg-1", input:"...", voice:"female", format:"pcm"}` then read binary audio frames until `{type:"end"}`.

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
# Try several conc values to find the sweet spot (WS recommended)
python test/bench.py --proto ws --n 60 --concurrency 1
python test/bench.py --proto ws --n 60 --concurrency 2
python test/bench.py --proto ws --n 60 --concurrency 4
python test/bench.py --proto ws --n 60 --concurrency 8
python test/bench.py --proto ws --n 60 --concurrency 12
```
- Pick the highest concurrency where p95 TTFB is acceptable. Set it via `MAX_CONCURRENT_JOBS` (see below) and scale replicas for more QPS.

### Config knobs (sane defaults set by start.sh)
- TTFB/streaming
  - `FIRST_SEGMENT_MAX_WORDS` (default 6)
  - `FIRST_SEGMENT_BOUNDARIES` (default `.,?!;:`)
  - `STREAM_CHUNK_SECONDS` (default 0.1)
  - `PRIME_STREAM=1`, `PRIME_BYTES=512`
- Concurrency/backpressure
  - `MAX_CONCURRENT_JOBS` (default 12)
  - `QUEUE_MAXSIZE` (default 256) – excess requests block in queue
- GPU
  - `KOKORO_DEVICE` (e.g., `cuda:0`)
  - `KOKORO_GPU_MEMORY_FRACTION` (e.g., `0.95`) – soft cap; use MIG for hard isolation

### Health & status
- `/healthz` (200 OK), `/readyz` (engine + device), `/api/status` (device, CUDA, GPU mem, ffmpeg)

### Notes
- Voices come from Kokoro‑82M: see the official list (`af_aoede`, `am_michael`, etc.).
- Default API voices: `female→af_aoede`, `male→am_michael`.
- Audio: PCM16 mono @ 24 kHz by default; Ogg/Opus with `{"format":"opus"}` if ffmpeg installed.

### References
- Kokoro GitHub: https://github.com/hexgrad/kokoro
- Kokoro‑82M model card: https://huggingface.co/hexgrad/Kokoro-82M