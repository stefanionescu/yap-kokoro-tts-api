"""
Global constants and default knobs for the Kokoro TTS service.

Each constant documents what it controls. Environment variables can override
most defaults at runtime; these values are used as sane fallbacks.
"""

# Audio / sampling
# Mono PCM16 at 24 kHz; used everywhere to convert samples ↔ seconds and to set ffmpeg rate
SAMPLE_RATE: int = 24000

# WebSocket send behavior (fallback defaults if env vars not set)
# Size threshold (bytes) before flushing buffered audio frames
WS_DEFAULT_BUFFER_BYTES: int = 16384
# Flush after this many chunks regardless of bytes
WS_DEFAULT_FLUSH_EVERY: int = 16
# Per-send timeout in seconds for ws.send_bytes
WS_DEFAULT_SEND_TIMEOUT_S: float = 3.0
# Log a warning if a ws.send_bytes takes longer than this (ms)
WS_DEFAULT_LONG_SEND_LOG_MS: float = 250.0

# Engine chunking and scheduler fairness
# Per-chunk synthesis size in seconds (smaller = lower latency, more overhead)
STREAM_DEFAULT_CHUNK_SECONDS: float = 0.04
# Bytes per scheduler turn for normal jobs
SCHED_DEFAULT_QUANTUM_BYTES: int = 16384
# Bytes per scheduler turn for priority jobs (first segments)
PRIORITY_DEFAULT_QUANTUM_BYTES: int = 2048

# Concurrency / admission control (API level; engine reads env overrides)
# Max concurrent synthesis streams (active_limit)
MAX_DEFAULT_CONCURRENT_JOBS: int = 4
# Internal engine queue capacity safety cap (TTS jobs)
DEFAULT_QUEUE_MAXSIZE: int = 128
# SLA: reject if predicted wait to start exceeds this (ms)
DEFAULT_QUEUE_WAIT_SLA_MS: int = 1000
# Allow up to this many queued requests beyond active_limit
DEFAULT_MAX_QUEUED_REQUESTS: int = 4

# EWMA of job wall-time for admission prediction
# Initial estimate of one job wall time in ms; updated after each request
DEFAULT_EWMA_WALL_MS: float = 2500.0
# Smoothing factor for EWMA update (0..1). Higher = adapt faster.
EWMA_ALPHA: float = 0.2

# API request validation / defaults
# Minimum and maximum allowed per-request speed multipliers
SPEED_MIN: float = 0.5
SPEED_MAX: float = 2.0
# Max words allowed per speak before rejecting (protects enormous utterances)
MAX_UTTERANCE_WORDS_DEFAULT: int = 150
# Optional primer: send N zero bytes to defeat proxy buffering
PRIME_STREAM_DEFAULT: int = 1
PRIME_BYTES_DEFAULT: int = 512

# Idle/scheduling sleeps (seconds) to avoid tight loops
WORKER_IDLE_SLEEP_S: float = 0.001
PROCESSOR_LOOP_SLEEP_S: float = 0.005
JOB_QUEUE_GET_TIMEOUT_S: float = 0.05

# Opus encoding defaults (only used when format="opus")
OPUS_DEFAULT_BITRATE: str = "48k"
OPUS_DEFAULT_APPLICATION: str = "audio"

# Metrics
# JSONL sink for request metrics
from pathlib import Path
METRICS_LOG_PATH: Path = Path("logs/metrics.log")


