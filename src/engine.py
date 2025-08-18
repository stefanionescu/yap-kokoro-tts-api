import asyncio
from asyncio import QueueEmpty
import logging
import os
import contextlib
import subprocess
import threading
import shutil
from typing import AsyncGenerator, Iterable, Optional, List
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
from kokoro import KPipeline
import torch

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000  # Kokoro outputs 24 kHz

def _float_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    if audio is None or audio.size == 0:
        return b""
    # Ensure float32 in [-1, 1]
    audio = np.clip(audio.astype(np.float32), -1.0, 1.0)
    pcm = (audio * 32767.0).round().astype(np.int16)
    return pcm.tobytes()

class KokoroEngine:
    """
    Lightweight engine for Kokoro using the official KPipeline.

    - Exposes available_voices = ["female", "male"]
    - Maps 'female' → 'aoede' and 'male' → 'michael' by default (env-overridable)
    - Streams raw PCM16 bytes at 24 kHz (or Ogg Opus if requested and ffmpeg is available)
    - Single async worker with a job queue (one process per GPU)
    """

    def __init__(self, lang_code: Optional[str] = None) -> None:
        self.available_voices = ["female", "male"]
        self._voice_mapping = {
            "female": os.getenv("DEFAULT_VOICE_FEMALE", "af_aoede"),
            "male": os.getenv("DEFAULT_VOICE_MALE", "am_michael"),
        }

        # Custom voice recipes (e.g., "my_blend": "af_aoede+am_michael").
        # ALWAYS save/load from the fixed directory "custom_voices".
        self.custom_dir = Path("custom_voices").resolve()
        self.custom_dir.mkdir(parents=True, exist_ok=True)
        self.custom_json_path = self.custom_dir / "custom_voices.json"
        self._custom_voices: dict[str, str] = {}
        self._load_custom_voices()

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
            float(os.getenv("STREAM_CHUNK_SECONDS", "0.04")) * SAMPLE_RATE
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

        # (memory cap already applied above before model init)

        # Async job queues and worker pool
        queue_size = int(os.getenv("QUEUE_MAXSIZE", "128"))
        self._pri_queue: asyncio.Queue["_TTSJob"] = asyncio.Queue(maxsize=queue_size)
        self._job_queue: asyncio.Queue["_TTSJob"] = asyncio.Queue(maxsize=queue_size)
        self.max_concurrent = int(os.getenv("MAX_CONCURRENT_JOBS", "4"))
        self._worker_tasks: list[asyncio.Task] = []
        # Number of in-flight synthesis streams (process-wide)
        self._inflight_count: int = 0
        # Total accepted requests (running + queued/reserved at API)
        self._accepted_slots: int = 0
        # EWMA of observed job wall time (ms) used for SLA estimation
        self._ewma_wall_ms: float = float(os.getenv("AVG_JOB_WALL_MS", "2500"))
        
        # Round-robin scheduling parameters
        self.quantum_bytes = int(float(os.getenv("SCHED_QUANTUM_BYTES", "16384")))
        self.active_limit = self.max_concurrent
        
        # Admission control
        self.queue_wait_sla_ms = int(os.getenv("QUEUE_WAIT_SLA_MS", "1000"))
        self.max_queued_requests = int(os.getenv("MAX_QUEUED_REQUESTS", "4"))

        # Cancellation registry per request_id
        self._cancel_flags: dict[str, bool] = {}

        # Extend available voices with custom names
        self._refresh_available_voices()

    def _load_custom_voices(self) -> None:
        try:
            if self.custom_json_path.exists():
                self._custom_voices = json.loads(self.custom_json_path.read_text(encoding="utf-8")) or {}
            else:
                self._custom_voices = {}
        except Exception:
            self._custom_voices = {}

    def _save_custom_voices(self) -> None:
        try:
            self.custom_json_path.write_text(json.dumps(self._custom_voices, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def _refresh_available_voices(self) -> None:
        names = ["female", "male"] + sorted(list(self._custom_voices.keys()))
        self.available_voices = names

    def add_custom_voice(self, name: str, recipe: str) -> None:
        if not name or not recipe:
            raise ValueError("name and recipe are required")
        self._custom_voices[name] = recipe
        self._save_custom_voices()
        self._refresh_available_voices()

    def remove_custom_voice(self, name: str) -> None:
        if name in self._custom_voices:
            self._custom_voices.pop(name, None)
            self._save_custom_voices()
            self._refresh_available_voices()

    def list_custom_voices(self) -> dict[str, str]:
        return dict(self._custom_voices)

    def start_worker(self) -> None:
        # Run exactly ONE scheduler loop; it interleaves up to self.active_limit streams.
        alive = [t for t in self._worker_tasks if not t.done()]
        if alive:
            self._worker_tasks = alive
            return
        t = asyncio.create_task(self._worker_loop())
        self._worker_tasks = [t]
        logger.info("Kokoro RR worker started (active_limit=%d, quantum=%d bytes)",
                    self.active_limit, self.quantum_bytes)

    async def stop_worker(self) -> None:
        try:
            for t in self._worker_tasks:
                if not t.done():
                    t.cancel()
            for t in self._worker_tasks:
                with contextlib.suppress(Exception):
                    await t
        finally:
            self._worker_tasks = []

    async def _worker_loop(self) -> None:
        from collections import deque
        logger.info("Kokoro RR scheduler loop running")
        active = deque()  # items: (job, agen) where agen is async generator of PCM bytes

        async def start_from_job(job: "_TTSJob"):
            # If canceled before start, short-circuit by yielding nothing
            if job.request_id and self._cancel_flags.get(job.request_id, False):
                async def _empty():
                    if False:
                        yield b""  # pragma: no cover
                    return
                return (job, _empty())
            agen = self._synthesize_stream_pieces(job.pieces, job.voice, job.output_format, job.speed, job.request_id)
            return (job, agen)

        while True:
            # Fill active set up to the effective number of inflight streams to avoid blocking under partial load
            effective_limit = min(self.active_limit, max(1, self._inflight_count))
            while len(active) < effective_limit:
                try:
                    # Priority first
                    job = self._pri_queue.get_nowait()
                    self._pri_queue.task_done()
                except QueueEmpty:
                    try:
                        job = self._job_queue.get_nowait()
                        self._job_queue.task_done()
                    except QueueEmpty:
                        break
                active.append(await start_from_job(job))

            if not active:
                # nothing to do; small sleep so we don't spin
                await asyncio.sleep(0.001)
                continue

            # Round-robin one quantum
            job, agen = active.popleft()
            sent = 0
            prio_q = int(float(os.getenv("PRIORITY_QUANTUM_BYTES", "2048")))
            budget = prio_q if getattr(job, "priority", False) else self.quantum_bytes
            try:
                while sent < budget:
                    chunk = await anext(agen)
                    if not chunk:
                        continue
                    await job.out_queue.put(chunk)
                    sent += len(chunk)
            except StopAsyncIteration:
                # done
                if job.end_stream:
                    await job.out_queue.put(None)
            except Exception as e:
                logger.exception("Worker error: %s", e)
                if job.end_stream:
                    await job.out_queue.put(None)
            else:
                # Not finished; rotate back
                active.append((job, agen))

    async def _synthesize_stream_pieces(self, pieces: List[str], voice: str, output_format: str, speed: Optional[float] = None, request_id: Optional[str] = None) -> AsyncGenerator[bytes, None]:
        selected_voice = voice or "female"
        # Allow direct Kokoro IDs like af_aoede/am_michael without forcing fallback
        if (
            selected_voice not in self.available_voices
            and not str(selected_voice).startswith(("af_", "am_"))
        ):
            selected_voice = "female"

        # Map API voice to Kokoro voice string (built-in or custom recipe)
        if selected_voice in self._custom_voices:
            kokoro_voice = self._custom_voices[selected_voice]
        else:
            kokoro_voice = self._voice_mapping.get(selected_voice, selected_voice)
        # guard against legacy unprefixed IDs in env
        if kokoro_voice == "aoede":
            kokoro_voice = "af_aoede"
        if kokoro_voice == "michael":
            kokoro_voice = "am_michael"

        def pipeline_for(piece: str):
            # Use per-request speed if provided, otherwise engine default
            eff_speed = float(speed if speed is not None else self.speed)
            # Try common Kokoro voice IDs if mapping fails silently
            try:
                return self.pipeline(
                    piece,
                    voice=kokoro_voice,
                    speed=eff_speed,
                    split_pattern=self.split_pattern,
                )
            except Exception as e:
                alt_voice = {"female": "af_aoede", "male": "am_michael"}.get(selected_voice, kokoro_voice)
                logger.warning("primary voice '%s' failed: %s; retrying with '%s'", kokoro_voice, e, alt_voice)
                return self.pipeline(
                    piece,
                    voice=alt_voice,
                    speed=eff_speed,
                    split_pattern=self.split_pattern,
                )

        if output_format == "pcm":
            for piece in pieces:
                # Kokoro v1 yields (gs, ps, audio) tuples; audio is np.ndarray float32
                for _, _, audio in pipeline_for(piece):
                    if request_id and self._cancel_flags.get(request_id, False):
                        return
                    if audio is None:
                        continue
                    audio_arr = np.asarray(audio, dtype=np.float32).flatten()
                    pcm = _float_to_pcm16_bytes(audio_arr)
                    if not pcm:
                        continue
                    if self.stream_chunk_samples > 0:
                        bpc = self.stream_chunk_samples * 2  # 2 bytes/sample
                        for i in range(0, len(pcm), bpc):
                            chunk = pcm[i : i + bpc]
                            if chunk:
                                yield chunk
                    else:
                        yield pcm
            return

        if output_format == "opus" and shutil.which("ffmpeg"):
            def pcm_iter():
                for piece in pieces:
                    result = pipeline_for(piece)
                    if isinstance(result, dict) or isinstance(result, np.ndarray) or isinstance(result, (bytes, bytearray)):
                        iterator = [result]
                    elif isinstance(result, (list, tuple)):
                        iterator = [result]
                    else:
                        iterator = result
                    for out in iterator:
                        audio_src = None
                        if isinstance(out, dict):
                            audio_src = out.get("audio") or out.get("wav") or out.get("audio_np")
                        elif isinstance(out, (list, tuple)):
                            audio_src = out[-1] if out else None
                        elif isinstance(out, (np.ndarray, bytes, bytearray)):
                            audio_src = out
                        if audio_src is None:
                            continue
                        for pcm_bytes in self._iter_pcm16_chunks(audio_src):
                            if pcm_bytes:
                                yield pcm_bytes

            for opus_bytes in self._opus_encode_via_ffmpeg(pcm_iter()):
                yield opus_bytes
            return

        logger.warning("Requested format '%s' not available; falling back to PCM", output_format)
        for piece in pieces:
            for _, _, audio_np in pipeline_for(piece):
                if request_id and self._cancel_flags.get(request_id, False):
                    return
                for pcm_bytes in self._iter_pcm16_chunks(audio_np):
                    yield pcm_bytes


    def _opus_encode_via_ffmpeg(self, pcm_iter: Iterable[bytes]) -> Iterable[bytes]:
        """Encode PCM16 mono 24kHz to Ogg Opus using ffmpeg subprocess."""
        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "-",
            "-c:a", "libopus",
            "-b:a", os.getenv("OPUS_BITRATE", "48k"),
            "-application", os.getenv("OPUS_APPLICATION", "audio"),
            "-f", "ogg",
            "-",
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        def writer():
            try:
                assert proc.stdin is not None
                for chunk in pcm_iter:
                    proc.stdin.write(chunk)
                proc.stdin.close()
            except Exception:
                pass

        t = threading.Thread(target=writer, daemon=True)
        t.start()

        try:
            assert proc.stdout is not None
            while True:
                buf = proc.stdout.read(4096)
                if not buf:
                    break
                yield buf
        finally:
            with contextlib.suppress(Exception):
                proc.kill()

    def can_accept(self) -> bool:
        """Check if we can accept a new request within SLA (conservative estimate)."""
        # Allow queueing up to max_queued_requests beyond active_limit
        q = self._pri_queue.qsize() + self._job_queue.qsize()
        # Conservative predicted wait: add ~half a job duration for residual before the next slot frees
        est_wait_ms = ((q / max(1, self.active_limit)) + 0.5) * max(1.0, self._ewma_wall_ms)
        within_sla = est_wait_ms < self.queue_wait_sla_ms
        within_queue_cap = q < max(0, self.max_queued_requests)
        return within_sla and within_queue_cap and (not self._job_queue.full())

    def try_accept_request(self) -> bool:
        """Reserve a slot (running or queued) if capacity and SLA permit.

        Allows up to (active_limit + max_queued_requests) total accepted requests
        and rejects if estimated start wait would exceed SLA.
        """
        # Capacity cap across running + queued
        max_total = max(0, self.active_limit) + max(0, self.max_queued_requests)
        if self._accepted_slots >= max_total:
            return False
        # Always admit up to active_limit immediately (no queue wait)
        if self._accepted_slots < self.active_limit:
            self._accepted_slots += 1
            return True
        # Beyond active_limit → queued
        queued_est = (self._accepted_slots - self.active_limit) + self._pri_queue.qsize() + self._job_queue.qsize()
        # Conservative wait estimate: N_per_slot + residual 0.5 job
        est_wait_ms = ((queued_est / max(1, self.active_limit)) + 0.5) * max(1.0, self._ewma_wall_ms)
        if est_wait_ms >= self.queue_wait_sla_ms:
            return False
        self._accepted_slots += 1
        return True

    def release_accept_slot(self) -> None:
        self._accepted_slots = max(0, self._accepted_slots - 1)

    def record_job_wall_ms(self, wall_ms: float) -> None:
        """Update EWMA for job wall time used in SLA estimation."""
        try:
            alpha = 0.2
            self._ewma_wall_ms = (1 - alpha) * self._ewma_wall_ms + alpha * max(1.0, float(wall_ms))
        except Exception:
            pass

    def try_reserve_slot(self) -> bool:
        """Atomically reserve a concurrency slot if available.

        Returns True on success. Caller must later call release_slot().
        """
        if self._inflight_count >= self.active_limit:
            return False
        self._inflight_count += 1
        return True

    def release_slot(self) -> None:
        """Release a previously reserved concurrency slot."""
        self._inflight_count = max(0, self._inflight_count - 1)

    def register_request(self, request_id: Optional[str]) -> None:
        if not request_id:
            return
        # initialize cancel flag as False
        self._cancel_flags[request_id] = False

    def cancel_request(self, request_id: Optional[str]) -> bool:
        if not request_id:
            return False
        if request_id in self._cancel_flags:
            self._cancel_flags[request_id] = True
            return True
        # If not registered yet, mark as canceled so future start short-circuits
        self._cancel_flags[request_id] = True
        return True

    def clear_request(self, request_id: Optional[str]) -> None:
        if not request_id:
            return
        with contextlib.suppress(Exception):
            self._cancel_flags.pop(request_id, None)

    def validate_voice(self, voice: str) -> None:
        if voice not in self.available_voices:
            raise ValueError(
                f"Voice {voice} is not available. Valid options are: {', '.join(self.available_voices)}"
            )

    def _iter_pcm16_chunks(self, audio: object) -> Iterable[bytes]:
        """Yield PCM16 bytes from various possible audio types.

        Accepts numpy arrays (float), torch tensors, or already-encoded bytes.
        """
        if audio is None:
            return
        # Fast path: already bytes/bytearray (assumed PCM16 mono 24k)
        if isinstance(audio, (bytes, bytearray)):
            if self.stream_chunk_samples <= 0:
                yield bytes(audio)
                return
            bytes_per_chunk = self.stream_chunk_samples * 2  # 2 bytes per sample
            for i in range(0, len(audio), bytes_per_chunk):
                yield bytes(audio[i : i + bytes_per_chunk])
            return

        # Convert arrays / tensors to np.float32
        try:
            if isinstance(audio, np.ndarray):
                arr = audio
            elif 'torch' in str(type(audio)):
                try:
                    arr = audio.detach().to('cpu').numpy()
                except Exception:
                    return
            else:
                arr = np.asarray(audio)
        except Exception:
            return

        if arr.size == 0:
            return
        arr = np.asarray(arr).astype(np.float32).flatten()
        if self.stream_chunk_samples <= 0 or arr.size <= self.stream_chunk_samples:
            yield _float_to_pcm16_bytes(arr)
            return
        for i in range(0, arr.size, self.stream_chunk_samples):
            segment = arr[i : i + self.stream_chunk_samples]
            yield _float_to_pcm16_bytes(segment)

    def _segment_for_fast_ttfb(self, text: str) -> list[str]:
        """Split input so the first piece is tiny (<= N words) to minimize TTFB.
        Tries to cut on natural boundaries near the limit; falls back to word cut.
        """
        t = (text or "").strip()
        if not t:
            return [""]
        words = t.split()
        n = self.first_segment_max_words
        if len(words) <= n:
            return [t]
        # find a boundary within first ~n+4 words
        boundary_chars = set(self.first_segment_boundary_chars)
        cut_idx = 0
        acc_words = []
        require_boundary = os.getenv("FIRST_SEGMENT_REQUIRE_BOUNDARY", "1") == "1"
        for i, w in enumerate(words):
            acc_words.append(w)
            if i + 1 >= n and any(ch in boundary_chars for ch in w):
                cut_idx = i + 1
                break
            # Only allow no-boundary fallback when not requiring boundary
            if not require_boundary and (i + 1 >= n + 4):
                cut_idx = i + 1
                break
        # If no natural boundary found and require-boundary is enabled, do NOT split
        if cut_idx == 0:
            if require_boundary:
                return [t]
            cut_idx = n
        first = " ".join(words[:cut_idx]).strip()
        rest = " ".join(words[cut_idx:]).strip()
        return [p for p in [first, rest] if p]

    async def generate_speech_async(
        self, prompt: str, voice: str | None = None, output_format: str = "pcm", speed: Optional[float] = None, request_id: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Enqueue a synthesis job and stream encoded bytes (default PCM16 @ 24 kHz).
        """
        # Lazy-start worker if not running
        self.start_worker()
        # Mark in-flight start
        self._inflight_count += 1

        out_q: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=32)
        text = (prompt or "").strip()
        pieces = self._segment_for_fast_ttfb(text)
        # Register request for cancellation tracking
        self.register_request(request_id)
        # IMPORTANT: enqueue as a single job to avoid RR interleaving across segments
        await self._pri_queue.put(_TTSJob(
            pieces=pieces,
            voice=voice or "female",
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
            self._inflight_count = max(0, self._inflight_count - 1)

    def get_status(self) -> dict:
        """Return runtime status for diagnostics."""
        gpu_name = None
        cuda_available = False
        device_index = None
        free_mem = None
        total_mem = None
        try:
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                device_index = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(device_index)
                try:
                    free_mem, total_mem = torch.cuda.mem_get_info(device_index)
                except Exception:
                    free_mem = total_mem = None
        except Exception:
            pass
        ffmpeg = shutil.which("ffmpeg") is not None
        return {
            "device": getattr(self, "device", "cpu"),
            "cuda_available": cuda_available,
            "gpu_name": gpu_name,
            "device_index": device_index,
            "gpu_mem_free_bytes": free_mem,
            "gpu_mem_total_bytes": total_mem,
            "ffmpeg_available": ffmpeg,
            "voices": self._voice_mapping,
            "speed": self.speed,
            "split_pattern": self.split_pattern,
            "stream_chunk_seconds": self.stream_chunk_samples / float(SAMPLE_RATE),
        }

    # Simple rolling metrics store (in-memory + file append)
    _metrics_log_path = Path("logs/metrics.log")
    _last_metrics: dict[str, float] = {}

    def log_request_metrics(self, metrics: dict) -> None:
        """Append request metrics to a log file and update memory cache.

        Expected keys: request_id, ttfb_ms, wall_s, audio_s, rtf, xrt, kbps, canceled(bool)
        """
        ts = time.time()
        try:
            self._last_metrics = {
                "ts": ts,
                "ttfb_ms": float(metrics.get("ttfb_ms", 0.0)),
                "wall_s": float(metrics.get("wall_s", 0.0)),
                "audio_s": float(metrics.get("audio_s", 0.0)),
                "rtf": float(metrics.get("rtf", 0.0)),
                "xrt": float(metrics.get("xrt", 0.0)),
                "kbps": float(metrics.get("kbps", 0.0)),
                "canceled": bool(metrics.get("canceled", False)),
            }
            rec = {
                "ts": ts,
                "request_id": metrics.get("request_id"),
                **self._last_metrics,
            }
            self._metrics_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._metrics_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass


@dataclass
class _TTSJob:
    pieces: List[str]
    voice: str
    output_format: str
    out_queue: asyncio.Queue
    end_stream: bool
    priority: bool
    speed: Optional[float] = None
    request_id: Optional[str] = None


