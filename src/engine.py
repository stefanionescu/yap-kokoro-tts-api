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
            "female": os.getenv("DEFAULT_VOICE_FEMALE", "af_aoede"),
            "male": os.getenv("DEFAULT_VOICE_MALE", "am_michael"),
        }

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

        # (memory cap already applied above before model init)

        # Async job queue and single-worker loop (one process per GPU)
        queue_size = int(os.getenv("QUEUE_MAXSIZE", "8"))
        self._job_queue: asyncio.Queue["_TTSJob"] = asyncio.Queue(maxsize=queue_size)
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
        # guard against legacy unprefixed IDs in env
        if kokoro_voice == "aoede":
            kokoro_voice = "af_aoede"
        if kokoro_voice == "michael":
            kokoro_voice = "am_michael"
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
            # Try common Kokoro voice IDs if mapping fails silently
            try:
                return self.pipeline(
                    piece,
                    voice=kokoro_voice,
                    speed=self.speed,
                    split_pattern=self.split_pattern,
                )
            except Exception as e:
                alt_voice = {"female": "af_aoede", "male": "am_michael"}.get(selected_voice, kokoro_voice)
                logger.warning("primary voice '%s' failed: %s; retrying with '%s'", kokoro_voice, e, alt_voice)
                return self.pipeline(
                    piece,
                    voice=alt_voice,
                    speed=self.speed,
                    split_pattern=self.split_pattern,
                )

        if output_format == "pcm":
            for piece in pieces:
                # Kokoro v1 yields (gs, ps, audio) tuples; audio is np.ndarray float32
                for _, _, audio in pipeline_for(piece):
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


