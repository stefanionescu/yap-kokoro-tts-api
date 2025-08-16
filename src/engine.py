import asyncio
import logging
import os
import contextlib
import subprocess
import threading
import shutil
from typing import AsyncGenerator, Iterable, Optional
from dataclasses import dataclass

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
            "female": os.getenv("DEFAULT_VOICE_FEMALE", "aoede"),
            "male": os.getenv("DEFAULT_VOICE_MALE", "michael"),
        }

        lang = lang_code or os.getenv("LANG_CODE", "a")  # 'a' = American English
        # Device selection
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = os.getenv("KOKORO_DEVICE", default_device)
        logger.info("Initializing Kokoro KPipeline | lang_code=%s device=%s", lang, self.device)
        # Try to pass device to KPipeline if supported; fall back gracefully
        try:
            self.pipeline = KPipeline(lang_code=lang, device=self.device)
        except TypeError:
            self.pipeline = KPipeline(lang_code=lang)

        # Streaming/chunking behavior
        self.speed = float(os.getenv("KOKORO_SPEED", "1.0"))
        self.split_pattern = os.getenv("KOKORO_SPLIT_PATTERN", r"\n+")
        # Stream chunking in samples (defaults to 0.5s)
        self.stream_chunk_samples = int(
            float(os.getenv("STREAM_CHUNK_SECONDS", "0.25")) * SAMPLE_RATE
        )
        # Fast-TTFB first-segment control
        self.first_segment_max_words = int(os.getenv("FIRST_SEGMENT_MAX_WORDS", "10"))
        self.first_segment_boundary_chars = os.getenv("FIRST_SEGMENT_BOUNDARIES", ".?!,;:-—")

        logger.info(
            "Kokoro ready | voices: female=%s male=%s | speed=%s split=%s chunk=%d samples",
            self._voice_mapping["female"],
            self._voice_mapping["male"],
            self.speed,
            self.split_pattern,
            self.stream_chunk_samples,
        )

        # Optional: cap GPU memory usage per process (0.0–1.0)
        try:
            fraction_env = os.getenv("KOKORO_GPU_MEMORY_FRACTION")
            if fraction_env and self.device.startswith("cuda"):
                frac = max(0.0, min(1.0, float(fraction_env)))
                torch.cuda.set_per_process_memory_fraction(frac, device=torch.device(self.device))
                logger.info("Set CUDA per-process memory fraction to %s", frac)
        except Exception as e:
            logger.debug("Could not set CUDA memory fraction: %s", e)

        # Async job queue and single-worker loop (one process per GPU)
        self._job_queue: asyncio.Queue["_TTSJob"] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

    def start_worker(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())
            logger.info("Kokoro worker started")

    async def stop_worker(self) -> None:
        try:
            if self._worker_task and not self._worker_task.done():
                self._worker_task.cancel()
                with contextlib.suppress(Exception):
                    await self._worker_task
        finally:
            self._worker_task = None

    async def _worker_loop(self) -> None:
        logger.info("Kokoro worker loop running")
        while True:
            job = await self._job_queue.get()
            try:
                async for chunk in self._synthesize_stream(job.text, job.voice, job.output_format):
                    await job.out_queue.put(chunk)
            except Exception as e:
                logger.exception("Worker error: %s", e)
                # signal error by closing stream
            finally:
                await job.out_queue.put(None)  # sentinel
                self._job_queue.task_done()

    async def _synthesize_stream(self, text: str, voice: str, output_format: str) -> AsyncGenerator[bytes, None]:
        selected_voice = voice or "female"
        if selected_voice not in self.available_voices:
            selected_voice = "female"

        kokoro_voice = self._voice_mapping[selected_voice]
        text = text or ""
        if not text.strip():
            return

        logger.info(
            "Kokoro generating | voice=%s (%s) text_len=%d format=%s",
            selected_voice,
            kokoro_voice,
            len(text),
            output_format,
        )

        # Pre-segment to force a tiny first segment for sub-200ms TTFB
        pieces = self._segment_for_fast_ttfb(text)
        def pipeline_for(piece: str):
            # Allow per-call speed/mapping override via env; kept simple here
            return self.pipeline(
                piece,
                voice=kokoro_voice,
                speed=self.speed,
                split_pattern=self.split_pattern,
            )

        if output_format == "pcm":
            for idx, piece in enumerate(pieces):
                for item in pipeline_for(piece):
                    audio_np = None
                    if isinstance(item, dict):
                        audio_np = item.get("audio") or item.get("wav") or item.get("audio_np")
                    elif isinstance(item, (list, tuple)):
                        # Assume last element is audio array
                        audio_np = item[-1] if item else None
                    elif isinstance(item, np.ndarray):
                        audio_np = item
                    if audio_np is None:
                        continue
                    for pcm_bytes in self._iter_pcm16_chunks(audio_np):
                        if pcm_bytes:
                            yield pcm_bytes
            return

        if output_format == "opus" and shutil.which("ffmpeg"):
            def pcm_iter():
                for piece in pieces:
                    for item in pipeline_for(piece):
                        audio_np = None
                        if isinstance(item, dict):
                            audio_np = item.get("audio") or item.get("wav") or item.get("audio_np")
                        elif isinstance(item, (list, tuple)):
                            audio_np = item[-1] if item else None
                        elif isinstance(item, np.ndarray):
                            audio_np = item
                        if audio_np is None:
                            continue
                        for pcm_bytes in self._iter_pcm16_chunks(audio_np):
                            if pcm_bytes:
                                yield pcm_bytes

            for opus_bytes in self._opus_encode_via_ffmpeg(pcm_iter()):
                yield opus_bytes
            return

        logger.warning("Requested format '%s' not available; falling back to PCM", output_format)
        for piece in pieces:
            for _, _, audio_np in pipeline_for(piece):
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

    def validate_voice(self, voice: str) -> None:
        if voice not in self.available_voices:
            raise ValueError(
                f"Voice {voice} is not available. Valid options are: {', '.join(self.available_voices)}"
            )

    def _iter_pcm16_chunks(self, audio: np.ndarray) -> Iterable[bytes]:
        if audio is None:
            return
        # Accept numpy arrays or torch tensors; ensure CPU numpy float32
        try:
            if isinstance(audio, np.ndarray):
                arr = audio
            elif 'torch' in str(type(audio)):
                try:
                    # Handle torch.Tensor without importing torch here
                    arr = audio.detach().to('cpu').numpy()
                except Exception:
                    return
            else:
                arr = np.asarray(audio)
        except Exception:
            return

        if arr.size == 0:
            return
        # Ensure 1D float array
        audio = np.asarray(arr).astype(np.float32).flatten()
        if self.stream_chunk_samples <= 0 or audio.size <= self.stream_chunk_samples:
            yield _float_to_pcm16_bytes(audio)
            return
        # Chunk into fixed-size segments for streaming
        for i in range(0, audio.size, self.stream_chunk_samples):
            segment = audio[i : i + self.stream_chunk_samples]
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
        for i, w in enumerate(words):
            acc_words.append(w)
            if i + 1 >= n and any(ch in boundary_chars for ch in w):
                cut_idx = i + 1
                break
            if i + 1 >= n + 4:
                cut_idx = i + 1
                break
        if cut_idx == 0:
            cut_idx = n
        first = " ".join(words[:cut_idx]).strip()
        rest = " ".join(words[cut_idx:]).strip()
        return [p for p in [first, rest] if p]

    async def generate_speech_async(
        self, prompt: str, voice: str | None = None, output_format: str = "pcm"
    ) -> AsyncGenerator[bytes, None]:
        """
        Enqueue a synthesis job and stream encoded bytes (default PCM16 @ 24 kHz).
        """
        # Lazy-start worker if not running
        self.start_worker()

        out_q: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=32)
        job = _TTSJob(text=prompt or "", voice=voice or "female", output_format=output_format, out_queue=out_q)
        await self._job_queue.put(job)

        while True:
            chunk = await out_q.get()
            if chunk is None:
                break
            yield chunk

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


@dataclass
class _TTSJob:
    text: str
    voice: str
    output_format: str
    out_queue: asyncio.Queue


