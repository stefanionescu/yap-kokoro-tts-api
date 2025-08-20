# Kokoro TTS Deployment (OpenAI Realtime Compatible)

FastAPI service around Kokoro‑82M with a persistent WebSocket using an OpenAI Realtime‑compatible event schema for low‑latency sentence‑level streaming and barge‑in. Output is 24 kHz mono PCM16 (base64 in WS events).

## 🚀 Quick Start

```bash
# Setup and start the service
bash scripts/setup.sh
bash scripts/start.sh
bash scripts/tail_bg_logs.sh

# Optional warmup (on the pod)
source venv/bin/activate && python test/warmup.py --save
```

**Manual launch (alternative):**
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --log-level info
```

## 🎯 Client Usage Examples

The `test/client.py` script provides an easy way to generate speech from text. Here are common usage patterns:

### Basic Usage

**Generate WAV file (recommended for quality):**
```bash
# Simple text to speech
python test/client.py --text "Hello, this is a test." --voice female --out hello.wav

# Using short sample text
python test/client.py --short-reply --voice male --out response.wav
```

**Different output formats:**
```bash
# High-quality WAV (default)
python test/client.py --text "Hello world" --format wav --out audio.wav

# Compressed Ogg/Opus (smaller file size)
python test/client.py --text "Hello world" --format ogg --out audio.ogg

# MP3 format
python test/client.py --text "Hello world" --format mp3 --out audio.mp3

# Raw PCM (no ffmpeg required)
python test/client.py --text "Hello world" --format pcm --out audio.pcm
```

### Voice and Speed Control

```bash
# Female voice at normal speed
python test/client.py --voice female --text "Welcome to our service"

# Male voice with faster speech
python test/client.py --voice male --speed 1.5 --text "This is faster speech"

# Slower, more deliberate speech
python test/client.py --voice female --speed 0.8 --text "This is slower speech"
```

### Processing Modes

```bash
# Single request mode (default) - entire text in one go
python test/client.py --mode single --text "First sentence. Second sentence. Third sentence."

# Sentence-by-sentence mode - better TTFB, shows per-sentence timing
python test/client.py --mode sentences --text "First sentence. Second sentence. Third sentence."
```

### Remote Server Usage

```bash
# Connect to remote server (replace with your host)
python test/client.py --host your-server.com --port 8000 --text "Hello from remote"

# RunPod with auto-TLS detection
python test/client.py --host "your-pod-id-xyz.proxy.runpod.net" --text "Hello from RunPod"

# Explicit TLS
python test/client.py --host your-server.com --tls --text "Hello over TLS"
```

### Long-form Content

```bash
# Default long text (good for testing)
python test/client.py --voice female --out long_demo.wav

# Custom long text
python test/client.py --text "$(cat your_long_text.txt)" --out custom_long.wav --mode sentences
```

### Practical Examples

```bash
# Quick voice memo
python test/client.py --voice male --speed 1.2 --text "Remember to buy groceries after work" --out reminder.wav

# Podcast-style narration
python test/client.py --voice female --speed 1.0 --format mp3 --text "Welcome to today's episode..." --out intro.mp3

# Fast preview/testing
python test/client.py --short-reply --speed 1.5 --format ogg --out test.ogg
```

**Note**: The client automatically reads `API_KEY`, `RUNPOD_TCP_HOST`, and `RUNPOD_TCP_PORT` from `.env` file when available.

## 🔐 Authentication

- The WebSocket endpoint requires an API key if `API_KEY` is set in `.env`
- `scripts/setup.sh` creates `.env` with a default `API_KEY=dev-key`
- **For production, set a strong key:**
```bash
sed -i 's/^API_KEY=.*/API_KEY=your-strong-key/' .env
bash scripts/stop.sh || true
bash scripts/start.sh
```
- Clients append the key as a query parameter: `wss://host/v1/audio/speech/stream/ws?api_key=your-strong-key`
- Our tools (`test/client.py`, `test/bench.py`, `test/warmup.py`) read `API_KEY` from `.env` automatically

## 🎮 Available Voices

- Only two logical voices are supported: **`female`** and **`male`**
- The actual Kokoro voice IDs come from environment variables:
  - `DEFAULT_VOICE_FEMALE` (default: `af_heart`)
  - `DEFAULT_VOICE_MALE` (default: `am_michael`)
  
**Custom voice mapping**: To use different Kokoro voice IDs, set them in `.env`:
```bash
DEFAULT_VOICE_FEMALE=af_sarah
DEFAULT_VOICE_MALE=am_adam
```

## 🧪 Testing & Benchmarking Tools

### Warmup Tool
```bash
# Warm up the model (single mode)
python test/warmup.py --mode single

# Warm up with sentence-by-sentence processing
python test/warmup.py --mode sentences
```

### Benchmark Tool
```bash
# Basic benchmark
python test/bench.py --n 40 --concurrency 3 --mode single

# Sentence-by-sentence benchmark (shows average TTFB across sentences)  
python test/bench.py --n 40 --concurrency 3 --mode sentences

# Find optimal concurrency for your setup
python test/bench.py --n 60 --concurrency 1
python test/bench.py --n 60 --concurrency 2
python test/bench.py --n 60 --concurrency 3
python test/bench.py --n 60 --concurrency 4
```

### Transactions Per Minute (TPM) Testing
```bash
# Single mode TPM test
python test/tpm.py --mode single

# Sentence mode with custom duration
python test/tpm.py --mode sentences --duration 120
```

**Tip**: Pick the highest concurrency where p95 TTFB is acceptable. Set it via `MAX_CONCURRENT_JOBS` and scale replicas for more QPS.

## 🔌 WebSocket Protocol (OpenAI Realtime Compatible)

**Endpoint**: `/v1/audio/speech/stream/ws`

### Basic Message Flow
```json
// 1. Initialize session
{ "type": "session.update", "session": {"voice": "female", "audio": {"format":"pcm", "sample_rate": 24000}} }

// 2. Create response
{ "type": "response.create", "response_id": "uuid-1", "input": "Okay, let's go.", "voice": "female", "speed": 1.0 }

// 3. Receive streaming audio (server sends multiple)
{ "type": "response.output_audio.delta", "response": "uuid-1", "delta": "<base64>" }

// 4. Completion
{ "type": "response.completed", "response": "uuid-1" }

// 5. End session
{ "type": "session.end" }
```

### Advanced Features

**Incremental input** (optional):
```json
{ "type": "input.append", "text": "Hello there" }
{ "type": "input.append", "text": ". This is" }
{ "type": "input.append", "text": " a demo." }
{ "type": "input.commit" }
```

**Barge-in capabilities**:
```json
// Cancel specific response
{ "type": "response.cancel", "response": "uuid" }

// Immediate stop and clear (incremental mode)
{ "type": "barge" }
```

### Protocol Notes
- One connection stays open for many sentences
- Concurrency handled via fair round‑robin scheduler  
- Cancel is immediate: queued or in‑flight audio dropped instantly
- Authentication via `?api_key=YOUR_KEY` query parameter

### Pipecat Integration
- Use a custom WebSocket TTS service or adapt Pipecat's OpenAI TTS service
- Configure PCM output at 24 kHz mono
- Keep sentence aggregation enabled in Pipecat

## ⚙️ Configuration

### Core Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | `dev-key` | Authentication key for WebSocket endpoint |
| `LOG_LEVEL` | `INFO` | Uvicorn/app log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `MODEL_NAME` | `hexgrad/Kokoro-82M` | Hugging Face model ID to load |
| `LANG_CODE` | `a` | Language variant (American English for Kokoro) |

### Voice Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_VOICE_FEMALE` | `af_heart` | Kokoro voice ID for "female" requests |
| `DEFAULT_VOICE_MALE` | `am_michael` | Kokoro voice ID for "male" requests |
| `KOKORO_SPEED` | `1.0` | Default speech speed (0.5–2.0) |
| `KOKORO_SPLIT_PATTERN` | `\n+` | Regex for sentence segmentation |

### Performance & Latency  
| Variable | Default | Description |
|----------|---------|-------------|
| `FIRST_SEGMENT_MAX_WORDS` | `2` | Words in first chunk (for fast TTFB) |
| `FIRST_SEGMENT_BOUNDARIES` | `,?!;:` | Punctuation for natural cuts |
| `FIRST_SEGMENT_REQUIRE_BOUNDARY` | `1` | Require punctuation boundary (vs word cut) |
| `STREAM_CHUNK_SECONDS` | `0.04` | Engine chunk size (40ms @ 24kHz) |
| `MAX_UTTERANCE_WORDS` | `150` | Max words per request |

### Concurrency Control
| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CONCURRENT_JOBS` | `4` | Max concurrent synthesis streams |
| `MAX_QUEUED_REQUESTS` | `4` | Max requests waiting to start |
| `QUEUE_WAIT_SLA_MS` | `1000` | Reject if predicted wait > this |
| `SCHED_QUANTUM_BYTES` | `16384` | Bytes per scheduler turn (fairness) |
| `PRIORITY_QUANTUM_BYTES` | `2048` | Bytes per turn (priority jobs) |

### WebSocket Streaming
| Variable | Default | Description |
|----------|---------|-------------|
| `WS_BUFFER_BYTES` | `16384` | Buffer threshold before flush |
| `WS_FLUSH_EVERY` | `16` | Flush after N chunks regardless |
| `WS_SEND_TIMEOUT` | `3.0` | Max seconds per send operation |
| `PRIME_STREAM` | `1` | Send zero-byte primer (anti-buffering) |
| `PRIME_BYTES` | `512` | Size of primer chunk |

### Hardware & GPU
| Variable | Default | Description |
|----------|---------|-------------|
| `KOKORO_DEVICE` | `cuda` (if available) | Target device (`cuda:0`, `cpu`, etc.) |
| `KOKORO_GPU_MEMORY_FRACTION` | unset | GPU memory limit (0.0-1.0) |

### Audio Encoding
| Variable | Default | Description |  
|----------|---------|-------------|
| `OPUS_BITRATE` | `48k` | Opus encoding bitrate |
| `OPUS_APPLICATION` | `audio` | Opus application type (`audio`/`voip`) |

## 📊 Monitoring & Health

### Health Endpoints
- **`/healthz`** - Basic health check (200 OK)
- **`/readyz`** - Readiness check (engine + device status)

### Metrics & Reporting
Each request logs detailed metrics to `logs/metrics.log` in JSON format:
- `ts`, `request_id`, `ttfb_ms`, `wall_s`, `audio_s`, `rtf`, `xrt`, `kbps`, `canceled`

**View rolling summaries:**
```bash
# Default time windows
source venv/bin/activate && python -m src.metrics

# Custom time windows
python -m src.metrics --periods "30m,1h,6h,24h,3d"
```

## 🧹 Maintenance & Troubleshooting

### Pod Management
Use `scripts/purge_pod.sh` to stop the server and optionally clean runtime files:

```bash
# Stop TTS server (safe - keeps files intact)
bash scripts/purge_pod.sh --port 8000

# Stop server and clean runtime files (logs, cache, etc.)
bash scripts/purge_pod.sh --port 8000 --clean-files

# Aggressive cleanup (clears additional caches)
bash scripts/purge_pod.sh --port 8000 --aggressive

# System cleanup with extras
bash scripts/purge_pod.sh --port 8000 --drop-caches --clear-shm --gpu-reset

# Remove system packages (⚠️ irreversible)
bash scripts/purge_pod.sh --purge-system

# Self-delete pod (requires RUNPOD_API_KEY and RUNPOD_POD_ID)
bash scripts/purge_pod.sh --self-remove
```

**⚠️ Important Notes:**
- `--clean-files` removes `venv/`, `cache/`, `logs/`, model caches, warmup audio
- `--aggressive` adds pip/apt caches and Torch extension cache cleanup  
- Use `--clear-tmp` carefully as it may disrupt active console sessions

### Common Issues
- **Import errors**: Ensure dependencies are installed via `bash scripts/setup.sh`
- **GPU memory issues**: Set `KOKORO_GPU_MEMORY_FRACTION` (e.g., `0.8`) in `.env`
- **High latency**: Tune `FIRST_SEGMENT_MAX_WORDS` and `STREAM_CHUNK_SECONDS`
- **Connection issues**: Check `API_KEY` and firewall settings
- **Audio quality**: Use WAV format for best quality, Ogg/Opus for smaller files

## 📈 Performance Results

Performance data from L40S GPU RunPod testing:

### Warmup Performance (Long Text)
| Voice | TTFB | Total Time | Audio Length | RTF | xRT | Throughput |
|-------|------|------------|--------------|-----|-----|------------|
| **Female** | 95.36ms | 0.58s | 69.15s | 0.0084 | 118.75x | 5,566 KB/s |
| **Male** | 65.21ms | 0.57s | 75.90s | 0.0075 | 133.84x | 6,274 KB/s |

### Benchmark Results (20 requests, concurrency=4)
| Metric | Value |
|--------|-------|
| **TTFB** | avg=199ms, p50=252ms, p95=255ms |
| **Wall Time** | avg=2.21s |
| **Audio Length** | avg=71.85s |
| **xRT** | avg=32.48x |
| **Throughput** | avg=1,523 KB/s |
| **Rejected** | 0 |
| **Total Time** | 11.09s |

### Single Client Performance
- **TTFB**: 116ms (long text, no sentence splitting)

## 💡 Key Features Summary

✅ **OpenAI Realtime Compatible** - Drop-in WebSocket replacement  
✅ **Ultra-low Latency** - TTFB ~65-116ms with smart chunking  
✅ **High Throughput** - 32x+ real-time generation speed  
✅ **Streaming Audio** - PCM16 @ 24kHz + Opus compression support  
✅ **Fair Scheduling** - Round-robin concurrency with barge-in  
✅ **Production Ready** - Authentication, metrics, health checks  
✅ **Easy Integration** - Works with Pipecat and custom clients  

## 📚 References

- **Kokoro TTS**: [GitHub Repository](https://github.com/hexgrad/kokoro)
- **Kokoro-82M Model**: [Hugging Face Model Card](https://huggingface.co/hexgrad/Kokoro-82M)
- **OpenAI Realtime API**: [Documentation](https://platform.openai.com/docs/guides/realtime)