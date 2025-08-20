"""Core Kokoro engine integrating all components."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import asyncio
import logging
import contextlib
from typing import AsyncGenerator, Optional
from pathlib import Path

import torch
from kokoro import KPipeline

from .audio.utils import float_to_pcm16_bytes, iter_pcm16_chunks
from .scheduling.jobs import TTSJob, RequestManager
from .scheduling.admission import AdmissionControl
from .scheduling.scheduler import WorkerScheduler
from .synthesis.pipeline import SynthesisPipeline
from .monitoring.metrics import MetricsCollector, SystemStatus

from constants import (
    SAMPLE_RATE,
    STREAM_DEFAULT_CHUNK_SECONDS,
    SCHED_DEFAULT_QUANTUM_BYTES,
    DEFAULT_QUEUE_WAIT_SLA_MS,
    DEFAULT_MAX_QUEUED_REQUESTS,
    DEFAULT_EWMA_WALL_MS,
    DEFAULT_QUEUE_MAXSIZE,
    MAX_DEFAULT_CONCURRENT_JOBS,
)

logger = logging.getLogger(__name__)


class KokoroEngine:
    """
    Lightweight engine for Kokoro using the official KPipeline.

    - Exposes available_voices = ["female", "male"]
    - Maps 'female' → 'heart' and 'male' → 'michael' by default (env-overridable)
    - Streams raw PCM16 bytes at 24 kHz
    - Single async worker with a job queue (one process per GPU)
    """

    def __init__(self, lang_code: Optional[str] = None) -> None:
        self.available_voices = ["female", "male"]
        self._voice_mapping = {
            "female": os.getenv("DEFAULT_VOICE_FEMALE", "af_heart"),
            "male": os.getenv("DEFAULT_VOICE_MALE", "am_michael"),
        }

        # Custom voices are no longer supported; only "female" and "male" via voice mapping

        lang = lang_code or os.getenv("LANG_CODE", "a")  # 'a' = American English
        # Device selection and optional memory cap BEFORE model init
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = os.getenv("KOKORO_DEVICE", default_device)
        cuda_index = 0
        if self.device.startswith("cuda") and ":" in self.device:
            with contextlib.suppress(Exception):
                cuda_index = int(self.device.split(":", 1)[1])
        try:
            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.set_device(cuda_index)
                fraction_env = os.getenv("KOKORO_GPU_MEMORY_FRACTION")
                if fraction_env:
                    frac = max(0.0, min(1.0, float(fraction_env)))
                    torch.cuda.set_per_process_memory_fraction(frac, device=cuda_index)
                    logger.info("Set CUDA per-process memory fraction to %s", frac)
        except Exception as e:
            logger.debug("Could not set CUDA device/memory fraction: %s", e)

        logger.info("Initializing Kokoro KPipeline | lang_code=%s device=%s", lang, self.device)
        # Try to pass device to KPipeline if supported; fall back gracefully
        try:
            self.pipeline = KPipeline(lang_code=lang, device=self.device)
        except TypeError:
            self.pipeline = KPipeline(lang_code=lang)

        # Streaming/chunking behavior
        self.speed = float(os.getenv("KOKORO_SPEED", "1.0"))
        self.split_pattern = os.getenv("KOKORO_SPLIT_PATTERN", r"\n+")
        # Stream chunking in samples
        self.stream_chunk_samples = int(
            float(os.getenv("STREAM_CHUNK_SECONDS", str(STREAM_DEFAULT_CHUNK_SECONDS))) * SAMPLE_RATE
        )
        # Fast-TTFB first-segment control
        self.first_segment_max_words = int(os.getenv("FIRST_SEGMENT_MAX_WORDS", "2"))
        self.first_segment_boundary_chars = os.getenv("FIRST_SEGMENT_BOUNDARIES", ",?!;:")

        logger.info(
            "Kokoro ready | voices: female=%s male=%s | speed=%s split=%s chunk=%d samples",
            self._voice_mapping["female"],
            self._voice_mapping["male"],
            self.speed,
            self.split_pattern,
            self.stream_chunk_samples,
        )

        # Preload voice packs once to remove first-hit latency
        try:
            for v in {self._voice_mapping["female"], self._voice_mapping["male"]}:
                try:
                    self.pipeline.load_voice(v)
                except Exception:
                    pass
        except Exception:
            pass

        # Initialize components
        self._setup_components()

        # Set available voices (female/male only)
        self._refresh_available_voices()

    def _setup_components(self) -> None:
        """Initialize all engine components."""
        # Async job queues 
        queue_size = int(os.getenv("QUEUE_MAXSIZE", str(DEFAULT_QUEUE_MAXSIZE)))
        self._pri_queue: asyncio.Queue[TTSJob] = asyncio.Queue(maxsize=queue_size)
        self._job_queue: asyncio.Queue[TTSJob] = asyncio.Queue(maxsize=queue_size)
        
        # Concurrency and admission control
        self.max_concurrent = int(os.getenv("MAX_CONCURRENT_JOBS", str(MAX_DEFAULT_CONCURRENT_JOBS)))
        queue_wait_sla_ms = int(os.getenv("QUEUE_WAIT_SLA_MS", str(DEFAULT_QUEUE_WAIT_SLA_MS)))
        max_queued_requests = int(os.getenv("MAX_QUEUED_REQUESTS", str(DEFAULT_MAX_QUEUED_REQUESTS)))
        initial_ewma_wall_ms = float(os.getenv("AVG_JOB_WALL_MS", str(DEFAULT_EWMA_WALL_MS)))
        
        # Initialize all components
        self.request_manager = RequestManager()
        self.admission_control = AdmissionControl(
            self.max_concurrent, queue_wait_sla_ms, max_queued_requests, initial_ewma_wall_ms
        )
        
        self.synthesis_pipeline = SynthesisPipeline(
            self.pipeline,
            self.speed,
            self.split_pattern,
            self.stream_chunk_samples,
            self.first_segment_max_words,
            self.first_segment_boundary_chars
        )
        
        # Round-robin scheduling parameters
        self.quantum_bytes = int(float(os.getenv("SCHED_QUANTUM_BYTES", str(SCHED_DEFAULT_QUANTUM_BYTES))))
        
        self.scheduler = WorkerScheduler(
            self.synthesis_pipeline,
            self.admission_control,
            self._pri_queue,
            self._job_queue,
            self.quantum_bytes,
            self.request_manager
        )
        
        self.metrics_collector = MetricsCollector()
        
        self.system_status = SystemStatus(
            self.device,
            self._voice_mapping,
            self.speed,
            self.split_pattern,
            self.stream_chunk_samples
        )

    # Custom voices are not supported; available voices are fixed
    def _refresh_available_voices(self) -> None:
        self.available_voices = ["female", "male"]

    def start_worker(self) -> None:
        """Run exactly ONE scheduler loop; it interleaves up to active_limit streams."""
        self.scheduler.start_worker()

    async def stop_worker(self) -> None:
        """Stop all worker tasks."""
        await self.scheduler.stop_worker()

    def can_accept(self) -> bool:
        """Non-blocking admission check: token available?"""
        return self.admission_control.can_accept()

    def try_accept_request(self) -> bool:
        """Non-blocking: admit only if a running slot token is available."""
        return self.admission_control.try_accept_request()

    async def try_accept_request_async(self) -> bool:
        """Immediate token acquire; return False if none (no queue)."""
        return await self.admission_control.try_accept_request_async()

    def release_accept_slot(self) -> None:
        """Release an admission slot."""
        self.admission_control.release_accept_slot()

    def record_job_wall_ms(self, wall_ms: float) -> None:
        """Update EWMA for job wall time used in SLA estimation."""
        self.admission_control.record_job_wall_ms(wall_ms)

    def try_reserve_slot(self) -> bool:
        """Atomically reserve a concurrency slot if available."""
        return self.admission_control.try_reserve_slot()

    def release_slot(self) -> None:
        """Release a previously reserved concurrency slot."""
        self.admission_control.release_slot()

    def register_request(self, request_id: Optional[str]) -> None:
        """Register a request for cancellation tracking."""
        self.request_manager.register_request(request_id)

    def cancel_request(self, request_id: Optional[str]) -> bool:
        """Mark a request as canceled."""
        return self.request_manager.cancel_request(request_id)

    def clear_request(self, request_id: Optional[str]) -> None:
        """Clear a request from the registry."""
        self.request_manager.clear_request(request_id)

    def validate_voice(self, voice: str) -> None:
        """Validate that a voice is available."""
        if voice not in self.available_voices:
            raise ValueError(
                f"Voice {voice} is not available. Valid options are: {', '.join(self.available_voices)}"
            )

    def _iter_pcm16_chunks(self, audio: object) -> iter:
        """Yield PCM16 bytes from various possible audio types."""
        return iter_pcm16_chunks(audio, self.stream_chunk_samples)

    def _segment_for_fast_ttfb(self, text: str) -> list[str]:
        """Split input so the first piece is tiny to minimize TTFB."""
        return self.synthesis_pipeline.segment_for_fast_ttfb(text)

    async def generate_speech_async(
        self, prompt: str, voice: str | None = None, output_format: str = "pcm", speed: Optional[float] = None, request_id: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Enqueue a synthesis job and stream encoded bytes (default PCM16 @ 24 kHz).
        """
        # Lazy-start worker if not running
        self.start_worker()
        # Mark in-flight start
        self.admission_control._inflight_count += 1

        out_q: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=32)
        text = (prompt or "").strip()
        pieces = self._segment_for_fast_ttfb(text)
        # Register request for cancellation tracking
        self.register_request(request_id)
        # IMPORTANT: enqueue as a single job to avoid RR interleaving across segments
        if not voice:
            raise ValueError("Voice must be provided (should be resolved by caller)")
        await self._pri_queue.put(TTSJob(
            pieces=pieces,
            voice=voice,
            output_format=output_format,
            out_queue=out_q,
            end_stream=True,
            priority=True,
            speed=speed,
            request_id=request_id,
        ))

        try:
            while True:
                chunk = await out_q.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            # Clear cancellation entry at end
            self.clear_request(request_id)
            # Mark in-flight end
            self.admission_control._inflight_count = max(0, self.admission_control._inflight_count - 1)

    def get_status(self) -> dict:
        """Return runtime status for diagnostics."""
        return self.system_status.get_status()

    def log_request_metrics(self, metrics: dict) -> None:
        """Append request metrics to a log file and update memory cache."""
        self.metrics_collector.log_request_metrics(metrics)
