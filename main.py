from src.logger import setup_logger
setup_logger()

import time
import os
from src.engine import KokoroEngine
from src.metrics import log_request_metrics
import torch
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import contextlib
import logging
import asyncio
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from starlette.websockets import WebSocketState

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

load_dotenv()

logger = logging.getLogger(__name__)

engine: KokoroEngine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize TTS engine on startup; WS-only service (no HTTP APIs)."""
    global engine
    try:
        torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
    except Exception:
        pass

    engine = KokoroEngine(lang_code=os.getenv("LANG_CODE", "a"))
    engine.start_worker()
    logger.info("Kokoro model initialized; WS-only mode")
    yield
    logger.info("Shutting down TTS engine")

app = FastAPI(lifespan=lifespan)


@app.websocket("/v1/audio/speech/stream/ws")
async def tts_stream_ws(websocket: WebSocket):
    # Simple API key guard via query param ?api_key=
    required_key = os.getenv("API_KEY")
    query = dict(websocket.query_params)
    provided = query.get("api_key")
    if required_key and provided != required_key:
        await websocket.close(code=1008)  # policy violation
        return
    await websocket.accept()
    logger.info("WebSocket connection open")

    # Per-connection defaults and state
    default_voice = os.getenv("DEFAULT_VOICE_FEMALE", "af_aoede")
    out_format = "pcm"  # wire is raw PCM16 frames
    want_timestamps = False
    send_lock = asyncio.Lock()
    job_queue: asyncio.Queue[dict] = asyncio.Queue()
    canceled: set[str] = set()
    active_request_id: str | None = None
    closing = False
    max_utterance_words = int(os.getenv("MAX_UTTERANCE_WORDS", "150"))
    # Heartbeat disabled

    async def send_json_safe(obj: dict) -> None:
        try:
            async with send_lock:
                await websocket.send_json(obj)
        except Exception:
            pass

    async def stream_one(job: dict) -> None:
        nonlocal active_request_id
        req_id: str = job.get("request_id")
        text: str = job.get("text", "")
        voice: str = job.get("voice") or default_voice
        speed = job.get("speed")
        if not text:
            await send_json_safe({"type": "error", "code": "empty_text", "request_id": req_id})
            return
        if req_id in canceled:
            await send_json_safe({"type": "canceled", "request_id": req_id})
            return

        active_request_id = req_id
        await send_json_safe({"type": "started_speak", "request_id": req_id})

        # Optional primer
        if os.getenv("PRIME_STREAM", "0") == "1":
            with contextlib.suppress(Exception):
                async with send_lock:
                    await websocket.send_bytes(b"\0" * int(os.getenv("PRIME_BYTES", "512")))

        safe_speed = max(0.5, min(2.0, float(speed))) if speed is not None else None
        start_time = time.perf_counter()
        total_samples = 0
        first_audio_at: float | None = None
        segment_index = 0
        first_chunk = True
        buf = bytearray()
        chunks_since_flush = 0

        # WS send controls (align with setup.sh defaults)
        BUF_TARGET = int(os.getenv("WS_BUFFER_BYTES", "960"))
        FLUSH_EVERY = int(os.getenv("WS_FLUSH_EVERY", "1"))
        SEND_TIMEOUT = float(os.getenv("WS_SEND_TIMEOUT", "3.0"))
        LONG_SEND_LOG_MS = float(os.getenv("WS_LONG_SEND_LOG_MS", "250.0"))

        async def _flush() -> None:
            nonlocal buf, chunks_since_flush, segment_index, total_samples, first_chunk
            if not buf:
                return
            send_t0 = time.perf_counter()
            try:
                async with send_lock:
                    await asyncio.wait_for(websocket.send_bytes(bytes(buf)), timeout=SEND_TIMEOUT)
            except asyncio.TimeoutError:
                logger.error("WebSocket: send_bytes timeout after %.2fs (buffer=%d bytes)", SEND_TIMEOUT, len(buf))
                raise
            finally:
                send_ms = (time.perf_counter() - send_t0) * 1000.0
                if send_ms > LONG_SEND_LOG_MS:
                    logger.warning("WebSocket: slow send_bytes: %.1f ms (buffer=%d bytes)", send_ms, len(buf))

            if first_chunk:
                ttfb = time.perf_counter() - start_time
                logger.info("WebSocket: TTFB %.1f ms for %s", ttfb * 1000.0, req_id)
                first_audio_at = time.perf_counter()
                first_chunk = False

            total_samples += len(buf) // 2
            buf.clear()
            chunks_since_flush = 0

            if (segment_index % 10) == 0:
                await send_json_safe({
                    "type": "meta",
                    "request_id": req_id,
                    "segment": segment_index,
                    "total_samples": total_samples,
                })
            segment_index += 1

        try:
            agen = engine.generate_speech_async(
                prompt=text,
                voice=voice,
                output_format=out_format,
                speed=safe_speed,
                request_id=req_id,
            )

            async for chunk in agen:
                if req_id in canceled:
                    break
                buf.extend(chunk)
                chunks_since_flush += 1
                if first_chunk and os.getenv("WS_FIRST_CHUNK_IMMEDIATE", "1") == "1" and len(buf) > 0:
                    await _flush()
                    continue
                if len(buf) >= BUF_TARGET or chunks_since_flush >= FLUSH_EVERY:
                    await _flush()
            if buf and req_id not in canceled:
                await _flush()

            if req_id in canceled:
                await send_json_safe({"type": "canceled", "request_id": req_id})
            else:
                # Compute simple per-request metrics and log via engine
                t_end = time.perf_counter()
                wall_s = (t_end - start_time)
                ttfb_ms = ((first_audio_at or t_end) - start_time) * 1000.0
                audio_s = total_samples / 24000.0
                rtf = wall_s / audio_s if audio_s > 0 else float("inf")
                xrt = audio_s / wall_s if wall_s > 0 else 0.0
                kbps = (total_samples / 1024.0) / wall_s if wall_s > 0 else 0.0
                log_request_metrics({
                    "request_id": req_id,
                    "ttfb_ms": ttfb_ms,
                    "wall_s": wall_s,
                    "audio_s": audio_s,
                    "rtf": rtf,
                    "xrt": xrt,
                    "kbps": kbps,
                    "canceled": False,
                })
                await send_json_safe({
                    "type": "done",
                    "request_id": req_id,
                    "duration_s": total_samples / 24000.0,
                })
        except Exception as e:
            logger.exception("WebSocket: error in stream_one(%s): %s", req_id, e)
            await send_json_safe({"type": "error", "request_id": req_id, "code": "stream_error", "message": str(e)})
        finally:
            active_request_id = None

    # Receiver loop: handles incoming control frames concurrently
    async def receiver_loop() -> None:
        nonlocal default_voice, out_format, want_timestamps, closing
        while True:
            raw = await websocket.receive()
            if raw.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect()
            if "text" in raw and raw["text"]:
                try:
                    data = json.loads(raw["text"])  # type: ignore[name-defined]
                except Exception:
                    await send_json_safe({"type": "error", "code": "bad_json"})
                    continue
            elif "bytes" in raw and raw["bytes"]:
                await send_json_safe({"type": "error", "code": "unexpected_binary"})
                continue
            else:
                continue

            msg_type = data.get("type")
            if msg_type == "start":
                default_voice = data.get("voice") or default_voice
                out_format = "pcm"
                want_timestamps = bool(data.get("timestamps", False))
                await send_json_safe({"type": "started", "voice": default_voice, "format": out_format, "timestamps": want_timestamps})
            elif msg_type == "speak":
                req_id = data.get("request_id") or f"req-{int(time.time()*1000)}"
                # Validate text
                text_val = data.get("text")
                if not isinstance(text_val, str):
                    await send_json_safe({"type": "error", "code": "bad_text", "request_id": req_id})
                    continue
                text_val = text_val.strip()
                if not text_val:
                    await send_json_safe({"type": "error", "code": "empty_text", "request_id": req_id})
                    continue
                # Word cap
                try:
                    word_count = len(text_val.split())
                except Exception:
                    word_count = 0
                if max_utterance_words > 0 and word_count > max_utterance_words:
                    await send_json_safe({
                        "type": "error",
                        "code": "too_long",
                        "request_id": req_id,
                        "max_words": max_utterance_words,
                        "got_words": word_count,
                    })
                    continue
                # Speed bound and sanitize
                spd_raw = data.get("speed")
                spd_val = None
                if spd_raw is not None:
                    try:
                        spd_val = float(spd_raw)
                    except Exception:
                        spd_val = None
                    else:
                        spd_val = max(0.5, min(2.0, spd_val))

                if not engine.can_accept():
                    await send_json_safe({"type": "queue", "request_id": req_id})
                await job_queue.put({
                    "request_id": req_id,
                    "text": text_val,
                    "voice": data.get("voice"),
                    "speed": spd_val,
                })
            elif msg_type == "cancel":
                req_id = data.get("request_id")
                if req_id:
                    canceled.add(req_id)
                    engine.cancel_request(req_id)
            elif msg_type == "stop":
                closing = True
                break
            else:
                await send_json_safe({"type": "error", "code": "unknown_type", "got": msg_type})

    try:
        recv_task = asyncio.create_task(receiver_loop())
        # Processor loop: serialize speaks per connection, preserve order
        while True:
            if closing and job_queue.empty():
                break
            if active_request_id is None:
                try:
                    job = await asyncio.wait_for(job_queue.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue
                if job.get("request_id") in canceled:
                    await send_json_safe({"type": "canceled", "request_id": job.get("request_id")})
                    job_queue.task_done()
                    continue
                await stream_one(job)
                job_queue.task_done()
            else:
                await asyncio.sleep(0.005)

        # Graceful shutdown: wait for receiver to finish or cancel it
        with contextlib.suppress(Exception):
            recv_task.cancel()
            await recv_task

    except WebSocketDisconnect:
        logger.info("Client disconnected from websocket.")
    except Exception as e:
        logger.error(f"Unexpected error in websocket endpoint: {e}")
    finally:
        logger.info("Closing websocket connection.")
        with contextlib.suppress(Exception):
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close()


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/readyz")
async def readyz():
    try:
        st = engine.get_status() if engine else None
        if not st:
            raise RuntimeError("engine not initialized")
        return {"ok": True, **st}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))