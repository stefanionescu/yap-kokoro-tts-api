"""Audio streaming logic for TTS WebSocket connections."""

import os
import time
import base64
import asyncio
import contextlib
import logging
from typing import Set, Optional

from fastapi import WebSocket

from constants import (
    SAMPLE_RATE,
    WS_DEFAULT_BUFFER_BYTES,
    WS_DEFAULT_FLUSH_EVERY,
    WS_DEFAULT_SEND_TIMEOUT_S,
    WS_DEFAULT_LONG_SEND_LOG_MS,
    SPEED_MIN,
    SPEED_MAX,
    PRIME_STREAM_DEFAULT,
    PRIME_BYTES_DEFAULT,
)
from metrics import log_request_metrics
from .voice import VoiceResolver

logger = logging.getLogger(__name__)


class TTSStreamer:
    """Handles TTS audio streaming for WebSocket connections."""
    
    def __init__(
        self,
        websocket: WebSocket,
        engine,
        voice_resolver: VoiceResolver,
        send_lock: asyncio.Lock,
        canceled: Set[str]
    ):
        self.websocket = websocket
        self.engine = engine
        self.voice_resolver = voice_resolver
        self.send_lock = send_lock
        self.canceled = canceled
        self.active_request_id: Optional[str] = None
    
    async def send_json_safe(self, obj: dict) -> None:
        """Send JSON to WebSocket with error handling."""
        try:
            async with self.send_lock:
                await self.websocket.send_json(obj)
        except Exception:
            pass
    
    async def stream_one(self, job: dict) -> None:
        """Stream TTS audio for a single job."""
        req_id: str = job.get("request_id")
        text: str = job.get("text", "")
        voice: str = job.get("voice") or self.voice_resolver.get_default_voice()
        speed = job.get("speed")
        test_mode = job.get("test_mode")
        response_id: str | None = job.get("response_id") if isinstance(job.get("response_id"), str) else None
        
        if not text:
            await self.send_json_safe({"type": "response.error", "response": response_id or req_id, "code": "empty_text"})
            return
        if req_id in self.canceled:
            await self.send_json_safe({"type": "response.canceled", "response": response_id or req_id})
            return

        # Resolve voice before starting synthesis
        try:
            resolved_voice = self.voice_resolver.resolve_voice(voice or self.voice_resolver.get_default_voice())
        except ValueError as e:
            await self.send_json_safe({"type": "response.error", "response": response_id or req_id, "code": "invalid_voice", "message": str(e)})
            return

        self.active_request_id = req_id
        if not job.get("suppress_created"):
            await self.send_json_safe({
                "type": "response.created",
                "response": response_id or req_id,
            })

        # Optional primer (as base64 audio delta)
        if os.getenv("PRIME_STREAM", str(PRIME_STREAM_DEFAULT)) == "1":
            with contextlib.suppress(Exception):
                prime_len = int(os.getenv("PRIME_BYTES", str(PRIME_BYTES_DEFAULT)))
                delta_b64 = base64.b64encode(b"\0" * prime_len).decode("ascii")
                await self.send_json_safe({
                    "type": "response.output_audio.delta",
                    "response": response_id or req_id,
                    "delta": delta_b64,
                })

        safe_speed = max(SPEED_MIN, min(SPEED_MAX, float(speed))) if speed is not None else None
        start_time = time.perf_counter()
        total_samples = 0
        n_ws_chunks = 0
        first_audio_at: float | None = None
        first_chunk = True
        buf = bytearray()
        chunks_since_flush = 0

        # WS send controls (align with setup.sh defaults)
        BUF_TARGET = int(os.getenv("WS_BUFFER_BYTES", str(WS_DEFAULT_BUFFER_BYTES)))
        FLUSH_EVERY = int(os.getenv("WS_FLUSH_EVERY", str(WS_DEFAULT_FLUSH_EVERY)))
        SEND_TIMEOUT = float(os.getenv("WS_SEND_TIMEOUT", str(WS_DEFAULT_SEND_TIMEOUT_S)))
        LONG_SEND_LOG_MS = float(os.getenv("WS_LONG_SEND_LOG_MS", str(WS_DEFAULT_LONG_SEND_LOG_MS)))

        async def _flush() -> None:
            nonlocal buf, chunks_since_flush, total_samples, first_chunk, first_audio_at, n_ws_chunks
            if not buf:
                return
            send_t0 = time.perf_counter()
            try:
                b64 = base64.b64encode(bytes(buf)).decode("ascii")
                event = {
                    "type": "response.output_audio.delta",
                    "response": response_id or req_id,
                    "delta": b64,
                }
                async with self.send_lock:
                    await asyncio.wait_for(self.websocket.send_json(event), timeout=SEND_TIMEOUT)
            finally:
                send_ms = (time.perf_counter() - send_t0) * 1000.0
                if send_ms > LONG_SEND_LOG_MS:
                    logger.warning("WebSocket: slow send: %.1f ms (buffer=%d bytes)", send_ms, len(buf))

            if first_chunk:
                ttfb = time.perf_counter() - start_time
                logger.info("WebSocket: TTFB %.1f ms for %s", ttfb * 1000.0, req_id)
                first_audio_at = time.perf_counter()
                first_chunk = False

            total_samples += len(buf) // 2
            buf.clear()
            chunks_since_flush = 0
            n_ws_chunks += 1

        try:
            agen = self.engine.generate_speech_async(
                prompt=text,
                voice=resolved_voice,
                output_format="pcm",
                speed=safe_speed,
                request_id=req_id,
            )

            async for chunk in agen:
                if req_id in self.canceled:
                    break
                buf.extend(chunk)
                chunks_since_flush += 1
                if first_chunk and os.getenv("WS_FIRST_CHUNK_IMMEDIATE", "1") == "1" and len(buf) > 0:
                    await _flush()
                    continue
                if len(buf) >= BUF_TARGET or chunks_since_flush >= FLUSH_EVERY:
                    await _flush()
            if buf and req_id not in self.canceled:
                await _flush()

            if req_id in self.canceled:
                if not job.get("suppress_completed"):
                    await self.send_json_safe({"type": "response.canceled", "response": response_id or req_id})
            else:
                # Compute simple per-request metrics and log via engine
                t_end = time.perf_counter()
                wall_s = (t_end - start_time)
                ttfb_ms = ((first_audio_at or t_end) - start_time) * 1000.0
                audio_s = total_samples / float(SAMPLE_RATE)
                rtf = wall_s / audio_s if audio_s > 0 else float("inf")
                xrt = audio_s / wall_s if wall_s > 0 else 0.0
                # PCM16 mono: 2 bytes per sample → convert samples→KB for throughput
                kbps = ((total_samples * 2) / 1024.0) / wall_s if wall_s > 0 else 0.0
                if not job.get("suppress_completed"):
                    log_request_metrics({
                        "request_id": req_id,
                        "ttfb_ms": ttfb_ms,
                        "wall_s": wall_s,
                        "audio_s": audio_s,
                        "rtf": rtf,
                        "xrt": xrt,
                        "kbps": kbps,
                        "canceled": False,
                        "test_mode": test_mode,
                    })
                # Feed wall time into engine EWMA (ms) to improve admission estimation under load
                with contextlib.suppress(Exception):
                    self.engine.record_job_wall_ms(wall_s * 1000.0)
                if not job.get("suppress_completed"):
                    await self.send_json_safe({
                        "type": "response.completed",
                        "response": response_id or req_id,
                        "duration_s": total_samples / float(SAMPLE_RATE),
                        "total_samples": total_samples,
                    })
        except Exception as e:
            logger.exception("WebSocket: error in stream_one(%s): %s", req_id, e)
            await self.send_json_safe({"type": "response.error", "response": response_id or req_id, "code": "stream_error", "message": str(e)})
        finally:
            self.active_request_id = None
            # Release accept slot whether completed or canceled/error
            with contextlib.suppress(Exception):
                self.engine.release_accept_slot()
    
    def get_active_request_id(self) -> Optional[str]:
        """Get the currently active request ID."""
        return self.active_request_id
