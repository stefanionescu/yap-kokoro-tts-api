from logger import setup_logger
setup_logger()

import time
import uuid as _uuid
import os
from engine import KokoroEngine
from utils import split_sentences as _split_sentences
from constants import (
    SAMPLE_RATE,
    WS_DEFAULT_BUFFER_BYTES,
    WS_DEFAULT_FLUSH_EVERY,
    WS_DEFAULT_SEND_TIMEOUT_S,
    WS_DEFAULT_LONG_SEND_LOG_MS,
    SPEED_MIN,
    SPEED_MAX,
    MAX_UTTERANCE_WORDS_DEFAULT,
    PRIME_STREAM_DEFAULT,
    PRIME_BYTES_DEFAULT,
    JOB_QUEUE_GET_TIMEOUT_S,
    PROCESSOR_LOOP_SLEEP_S,
)
from metrics import log_request_metrics
import torch
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import contextlib
import logging
import asyncio
import json
import base64

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
    # Voice mapping for API
    voice_mapping = {
        "female": os.getenv("DEFAULT_VOICE_FEMALE", "af_heart"),
        "male": os.getenv("DEFAULT_VOICE_MALE", "am_michael"),
    }
    default_voice = "female"

    def resolve_voice(requested: str) -> str:
        """Resolve API voice name to Kokoro voice id. Only 'female' and 'male' are allowed."""
        if not requested:
            return voice_mapping["female"]
        req_lower = requested.lower()
        if req_lower in ("female", "f"):
            return voice_mapping["female"]
        if req_lower in ("male", "m"):
            return voice_mapping["male"]
        raise ValueError(f"Voice '{requested}' not found (allowed: female, male)")
    out_format = "pcm"  # wire is raw PCM16 frames
    send_lock = asyncio.Lock()
    job_queue: asyncio.Queue[dict] = asyncio.Queue()
    canceled: set[str] = set()
    active_request_id: str | None = None
    closing = False
    max_utterance_words = int(os.getenv("MAX_UTTERANCE_WORDS", str(MAX_UTTERANCE_WORDS_DEFAULT)))

    # no per-connection state beyond defaults required by the protocol

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
        response_id: str | None = job.get("response_id") if isinstance(job.get("response_id"), str) else None
        if not text:
            await send_json_safe({"type": "response.error", "response": response_id or req_id, "code": "empty_text"})
            return
        if req_id in canceled:
            await send_json_safe({"type": "response.canceled", "response": response_id or req_id})
            return

        # Resolve voice before starting synthesis
        try:
            resolved_voice = resolve_voice(voice or default_voice)
        except ValueError as e:
            await send_json_safe({"type": "response.error", "response": response_id or req_id, "code": "invalid_voice", "message": str(e)})
            return

        active_request_id = req_id
        if not job.get("suppress_created"):
            await send_json_safe({
                "type": "response.created",
                "response": response_id or req_id,
            })

        # Optional primer (as base64 audio delta)
        if os.getenv("PRIME_STREAM", str(PRIME_STREAM_DEFAULT)) == "1":
            with contextlib.suppress(Exception):
                prime_len = int(os.getenv("PRIME_BYTES", str(PRIME_BYTES_DEFAULT)))
                delta_b64 = base64.b64encode(b"\0" * prime_len).decode("ascii")
                await send_json_safe({
                    "type": "response.output_audio.delta",
                    "response": response_id or req_id,
                    "delta": delta_b64,
                })

        safe_speed = max(SPEED_MIN, min(SPEED_MAX, float(speed))) if speed is not None else None
        start_time = time.perf_counter()
        total_samples = 0
        n_ws_chunks = 0
        # Heuristic: classify whether the request is a single sentence or a block
        try:
            n_sentences = len(_split_sentences(text))
        except Exception:
            n_sentences = 1
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
                async with send_lock:
                    await asyncio.wait_for(websocket.send_json(event), timeout=SEND_TIMEOUT)
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

            # no meta events in OpenAI mode

        try:
            agen = engine.generate_speech_async(
                prompt=text,
                voice=resolved_voice,
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
                if not job.get("suppress_completed"):
                    await send_json_safe({"type": "response.canceled", "response": response_id or req_id})
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
                        # new metadata for per-sentence vs block classification and chunking
                        "n_sentences": int(n_sentences) if isinstance(n_sentences, int) else 1,
                        "n_ws_chunks": int(n_ws_chunks),
                        "total_samples": int(total_samples),
                    })
                # Feed wall time into engine EWMA (ms) to improve admission estimation under load
                with contextlib.suppress(Exception):
                    engine.record_job_wall_ms(wall_s * 1000.0)
                if not job.get("suppress_completed"):
                    await send_json_safe({
                        "type": "response.completed",
                        "response": response_id or req_id,
                        "duration_s": total_samples / float(SAMPLE_RATE),
                        "total_samples": total_samples,
                    })
        except Exception as e:
            logger.exception("WebSocket: error in stream_one(%s): %s", req_id, e)
            await send_json_safe({"type": "response.error", "response": response_id or req_id, "code": "stream_error", "message": str(e)})
        finally:
            active_request_id = None
            # Release accept slot whether completed or canceled/error
            with contextlib.suppress(Exception):
                engine.release_accept_slot()

    # Receiver loop: handles incoming control frames concurrently
    # Incremental input state (for Pipecat-style streaming)
    pending_text: str = ""
    utterance_id: str | None = None
    last_append_t: float = 0.0
    flush_task: asyncio.Task | None = None
    FLUSH_IDLE_MS = int(os.getenv("FLUSH_IDLE_MS", "160"))

    async def _maybe_flush(force: bool = False) -> None:
        nonlocal pending_text, utterance_id, last_append_t
        buf = pending_text.strip()
        if not buf:
            return
        if not force:
            idle_ms = (time.perf_counter() - last_append_t) * 1000.0
            if idle_ms < FLUSH_IDLE_MS:
                return
        # consume buffer as a single unit
        pending_text = ""
        # Admission control
        ok = False
        try:
            ok = await engine.try_accept_request_async()  # type: ignore[attr-defined]
        except AttributeError:
            ok = engine.try_accept_request()
        if not ok:
            await send_json_safe({"type": "response.error", "response": utterance_id or "", "code": "busy"})
            return
        first_chunk_of_utt = utterance_id is None
        if utterance_id is None:
            utterance_id = f"utt-{int(time.time()*1000)}"
        await job_queue.put({
            "request_id": f"{utterance_id}-{_uuid.uuid4().hex[:6]}",
            "response_id": utterance_id,
            "text": buf,
            "voice": default_voice,
            "speed": None,
            # Only send response.created on the first flush of this utterance
            "suppress_created": (not first_chunk_of_utt),
            # Only send response.completed on commit/force; keep streaming otherwise
            "suppress_completed": (not force),
        })

    async def _schedule_debounce_flush() -> None:
        nonlocal flush_task
        if flush_task and not flush_task.done():
            with contextlib.suppress(Exception):
                flush_task.cancel()
        async def _runner():
            try:
                await asyncio.sleep(FLUSH_IDLE_MS / 1000.0)
                await _maybe_flush(force=True)
            except Exception:
                pass
        flush_task = asyncio.create_task(_runner())

    async def receiver_loop() -> None:
        nonlocal default_voice, out_format, closing
        while True:
            raw = await websocket.receive()
            if raw.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect()
            if "text" in raw and raw["text"]:
                try:
                    data = json.loads(raw["text"])  # type: ignore[name-defined]
                except Exception:
                    await send_json_safe({"type": "response.error", "code": "bad_json"})
                    continue
            elif "bytes" in raw and raw["bytes"]:
                await send_json_safe({"type": "response.error", "code": "unexpected_binary"})
                continue
            else:
                continue

            msg_type = data.get("type")
            # OpenAI-only protocol
            if True:
                if msg_type == "session.update":
                    sess = data.get("session", {}) if isinstance(data.get("session"), dict) else {}
                    v_top = data.get("voice")
                    v_sess = sess.get("voice") if isinstance(sess, dict) else None
                    a_top = data.get("audio") or {}
                    a_sess = sess.get("audio") if isinstance(sess, dict) else {}
                    voice_val = (v_top or v_sess or default_voice) or "female"
                    sr_val = (a_top.get("sample_rate") if isinstance(a_top, dict) else None) or (a_sess.get("sample_rate") if isinstance(a_sess, dict) else None) or SAMPLE_RATE
                    try:
                        resolve_voice(voice_val)
                        default_voice = voice_val
                    except ValueError:
                        await send_json_safe({"type": "response.error", "code": "invalid_voice", "message": str(voice_val)})
                        continue
                    out_format = "pcm"
                    await send_json_safe({
                        "type": "session.updated",
                        "session": {
                            "voice": default_voice,
                            "audio": {"format": out_format, "sample_rate": int(sr_val) if isinstance(sr_val, (int, float)) else SAMPLE_RATE},
                        },
                    })
                elif msg_type == "input.append":
                    chunk = data.get("text")
                    if isinstance(chunk, str) and chunk:
                        pending_text += chunk
                        last_append_t = time.perf_counter()
                        await _maybe_flush(force=False)
                        await _schedule_debounce_flush()
                elif msg_type == "input.commit":
                    await _maybe_flush(force=True)
                    # Reset utterance id after commit completes
                    utterance_id = None
                elif msg_type == "barge":
                    # Cancel current playback and clear buffer
                    if active_request_id:
                        canceled.add(active_request_id)
                        engine.cancel_request(str(active_request_id))
                    pending_text = ""
                    utterance_id = None
                elif msg_type == "response.create":
                    req_id = data.get("response_id") or data.get("id") or f"req-{int(time.time()*1000)}"
                    input_text = data.get("input") or data.get("text") or data.get("instructions")
                    if not isinstance(input_text, str):
                        await send_json_safe({"type": "response.error", "response": req_id, "code": "bad_text"})
                        continue
                    input_text = input_text.strip()
                    if not input_text:
                        await send_json_safe({"type": "response.error", "response": req_id, "code": "empty_text"})
                        continue
                    audio_cfg = data.get("audio") or {}
                    voice_override = data.get("voice") or (audio_cfg.get("voice") if isinstance(audio_cfg, dict) else None)
                    speed_raw = data.get("speed") or (audio_cfg.get("speed") if isinstance(audio_cfg, dict) else None)
                    spd_val = None
                    if speed_raw is not None:
                        try:
                            spd_val = float(speed_raw)
                        except Exception:
                            spd_val = None
                        else:
                            spd_val = max(0.5, min(2.0, spd_val))
                    # Word cap
                    try:
                        word_count = len(input_text.split())
                    except Exception:
                        word_count = 0
                    if max_utterance_words > 0 and word_count > max_utterance_words:
                        await send_json_safe({
                            "type": "response.error",
                            "response": req_id,
                            "code": "too_long",
                            "max_words": max_utterance_words,
                            "got_words": word_count,
                        })
                        continue
                    ok = False
                    try:
                        ok = await engine.try_accept_request_async()  # type: ignore[attr-defined]
                    except AttributeError:
                        ok = engine.try_accept_request()
                    if not ok:
                        await send_json_safe({"type": "response.error", "response": req_id, "code": "busy"})
                        continue
                    await job_queue.put({
                        "request_id": req_id,
                        "response_id": req_id,
                        "text": input_text,
                        "voice": voice_override or default_voice,
                        "speed": spd_val,
                    })
                elif msg_type == "response.cancel":
                    resp = data.get("response") or data.get("response_id")
                    if resp:
                        canceled.add(resp)
                        engine.cancel_request(str(resp))
                elif msg_type == "session.end":
                    closing = True
                    break
                else:
                    await send_json_safe({"type": "response.error", "code": "unknown_type", "got": msg_type})

    try:
        recv_task = asyncio.create_task(receiver_loop())
        # Processor loop: serialize speaks per connection, preserve order
        while True:
            if closing and job_queue.empty():
                break
            if active_request_id is None:
                try:
                    job = await asyncio.wait_for(job_queue.get(), timeout=JOB_QUEUE_GET_TIMEOUT_S)
                except asyncio.TimeoutError:
                    continue
                if job.get("request_id") in canceled:
                    await send_json_safe({"type": "response.canceled", "response": job.get("request_id")})
                    job_queue.task_done()
                    continue
                await stream_one(job)
                job_queue.task_done()
            else:
                await asyncio.sleep(PROCESSOR_LOOP_SLEEP_S)

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