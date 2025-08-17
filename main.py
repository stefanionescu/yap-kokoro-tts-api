from src.logger import setup_logger
setup_logger()

import time
import os
from src.engine import KokoroEngine
import torch
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import contextlib
import logging
import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from fastapi.responses import StreamingResponse
from fastapi import Response, HTTPException
from fastapi import BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

load_dotenv()

logger = logging.getLogger(__name__)


class TTSRequest(BaseModel):
    input: str = "Hey there, looks like you forgot to provide a prompt!"
    voice: str = "female"
    format: str = Field("pcm", description="Output format: pcm (default), opus (experimental)")
    # Optional per-request speed override for OpenAI parity
    speed: Optional[float] = Field(None, description="Override speaking speed (e.g., 0.8–1.4)")

class TTSStreamRequest(BaseModel):
    input: str
    voice: str = "female"
    format: str = Field("pcm", description="Output format: pcm (default), opus (experimental)")
    continue_: bool = Field(True, alias="continue")
    segment_id: str

class VoiceDetail(BaseModel):
    name: str
    description: str
    language: str
    gender: str
    accent: str
    preview_url: Optional[str] = None

class VoicesResponse(BaseModel):
    voices: List[VoiceDetail]
    default: str
    count: int
    
engine: KokoroEngine = None
_keep_hot_task: asyncio.Task | None = None
VOICE_DETAILS: List[VoiceDetail] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes the TTS engine on application startup."""
    global engine, VOICE_DETAILS
    
    # Torch perf knobs (small but free throughput/latency wins)
    try:
        torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
    except Exception:
        pass
    
    # Initialize the engine
    engine = KokoroEngine(lang_code=os.getenv("LANG_CODE", "a"))
    engine.start_worker()
    
    # Log Kokoro settings instead
    logger.info("Kokoro model initialized")
    
    # Define gender for the available voices
    voice_genders = {
        "female": "female",
        "male": "male"
    }

    # Dynamically generate voice details from the loaded engine
    VOICE_DETAILS = [
        VoiceDetail(
            name=voice,
            description=f"A natural-sounding {voice_genders.get(voice, 'unknown')} voice.",
            language="en",
            gender=voice_genders.get(voice, "unknown"),
            accent="american"
        ) for voice in engine.available_voices
    ]
    
    logger.info(f"TTS engine initialized with {len(VOICE_DETAILS)} voices: {', '.join([v.name for v in VOICE_DETAILS])}")

    # No keep-hot task needed for Kokoro
    yield
    logger.info("Shutting down TTS engine")

app = FastAPI(lifespan=lifespan)


@app.post('/v1/audio/speech/stream')
async def tts_stream(data: TTSRequest):
    """
    Generates audio speech from text in a streaming fashion.
    This endpoint is optimized for low latency (Time to First Byte).
    """
    # Admission control
    if not engine.can_accept():
        raise HTTPException(status_code=429, detail="busy", headers={"Retry-After": "1"})
    
    start_time = time.perf_counter()
    voice = data.voice
    
    out_format = data.format.lower() if data.format else "pcm"
    num_ctx = int(os.getenv("NUM_CTX", "8192"))
    num_predict = int(os.getenv("NUM_PREDICT", "49152"))
    
    logger.info(f"TTS request: voice={voice}, text_length={len(data.input)}")

    async def generate_audio_stream():
        first_chunk = True
        total_bytes = 0
        try:
            # Optional priming to defeat overly aggressive proxy buffering
            if os.getenv("PRIME_STREAM", "0") == "1":
                prime_bytes = int(os.getenv("PRIME_BYTES", "512"))
                yield b"\0" * prime_bytes

            # Per-request speed override if provided
            if getattr(data, "speed", None) is not None:
                os.environ["KOKORO_SPEED"] = str(max(0.5, min(2.0, float(data.speed))))

            audio_generator = engine.generate_speech_async(
                prompt=data.input,
                voice=voice,
                output_format=out_format,
            )

            async for chunk in audio_generator:
                if first_chunk:
                    ttfb = time.perf_counter() - start_time
                    logger.info(f"Time to first audio chunk (TTFB): {ttfb*1000:.2f} ms")
                    first_chunk = False
                total_bytes += len(chunk)
                yield chunk
        except Exception as e:
            logger.exception(f"Error during audio generation: {str(e)}")
        finally:
            if total_bytes > 0:
                secs = total_bytes / (24000 * 2)
                logger.info(f"HTTP stream completed: {total_bytes} bytes (~{secs:.2f}s)")

    media_type = 'audio/ogg' if out_format == 'opus' else 'audio/L16; rate=24000; channels=1'
    return StreamingResponse(
        generate_audio_stream(),
        media_type=media_type,
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-store, no-transform",
            "Content-Type": media_type,
            "Connection": "keep-alive",
        },
    )


@app.post('/v1/audio/speech')
async def tts_sync(data: TTSRequest):
    """OpenAI-style sync endpoint that returns the full audio body."""
    # Admission control
    if not engine.can_accept():
        raise HTTPException(status_code=429, detail="busy", headers={"Retry-After": "1"})
    
    voice = data.voice
    out_format = data.format.lower() if data.format else "pcm"
    try:
        chunks = []
        async for chunk in engine.generate_speech_async(
            prompt=data.input,
            voice=voice,
            output_format=out_format,
        ):
            chunks.append(chunk)
        audio_bytes = b"".join(chunks)
        mt = 'audio/ogg' if out_format == 'opus' else 'audio/pcm'
        return Response(content=audio_bytes, media_type=mt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/v1/audio/speech/stream/ws")
async def tts_stream_ws(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection open")
    
    # Admission control for WS
    if not engine.can_accept():
        await websocket.send_json({"type": "error", "code": "busy"})
        await websocket.close(code=1013)  # Try again later
        return
    
    try:
        while True:
            data = await websocket.receive_json()

            if not data.get("continue", True):
                logger.info("End of stream message received, closing connection.")
                break

            if not (input_text := data.get("input", "").strip()):
                logger.info("Empty or whitespace-only input received, skipping audio generation.")
                continue

            voice = data.get("voice", "female")
            segment_id = data.get("segment_id", "no_segment_id")
            speed = data.get("speed", 1.4)  # Default to server speed
            
            out_format = str(data.get("format", "pcm")).lower()

            start_time = time.perf_counter()
            total_samples = 0
            segment_index = 0
            # Send start; break cleanly if client is already gone
            try:
                await websocket.send_json({"type": "start", "segment_id": segment_id})
            except Exception:
                break

            # WS primer (optional)
            if os.getenv("PRIME_STREAM", "0") == "1":
                with contextlib.suppress(Exception):
                    await websocket.send_bytes(b"\0" * int(os.getenv("PRIME_BYTES", "512")))

            if input_text:
                logger.info(f"WebSocket: Generating audio for input: '{input_text[:50]}...' (length: {len(input_text)}) speed={speed}")
                
                # Set per-request speed override
                if speed is not None:
                    os.environ["KOKORO_SPEED"] = str(max(0.5, min(2.0, float(speed))))
                
                audio_generator = engine.generate_speech_async(
                    prompt=input_text,
                    voice=voice,
                    output_format=out_format,
                )

                # WS send controls
                BUF_TARGET = int(os.getenv("WS_BUFFER_BYTES", "8192"))       # ~0.17s @ 24kHz mono PCM16
                FLUSH_EVERY = int(os.getenv("WS_FLUSH_EVERY", "5"))          # or flush every N micro-chunks
                SEND_TIMEOUT = float(os.getenv("WS_SEND_TIMEOUT", "3.0"))    # hard cap per send
                LONG_SEND_LOG_MS = float(os.getenv("WS_LONG_SEND_LOG_MS", "250.0"))

                buf = bytearray()
                chunks_since_flush = 0
                first_chunk = True
                segment_index = 0
                total_samples = 0

                async def _flush():
                    nonlocal buf, chunks_since_flush, segment_index, total_samples, first_chunk
                    if not buf:
                        return
                    send_t0 = time.perf_counter()
                    try:
                        await asyncio.wait_for(websocket.send_bytes(bytes(buf)), timeout=SEND_TIMEOUT)
                    except asyncio.TimeoutError:
                        logger.error("WebSocket: send_bytes timeout after %.2fs (buffer=%d bytes)", SEND_TIMEOUT, len(buf))
                        raise
                    finally:
                        send_ms = (time.perf_counter() - send_t0) * 1000.0
                        if send_ms > LONG_SEND_LOG_MS:
                            logger.warning("WebSocket: slow send_bytes: %.1f ms (buffer=%d bytes)", send_ms, len(buf))

                    # First-chunk TTFB measured on first successful send
                    if first_chunk:
                        ttfb = time.perf_counter() - start_time
                        logger.info("WebSocket: Time to first audio chunk (TTFB): %.2f ms", ttfb * 1000.0)
                        first_chunk = False

                    total_samples += len(buf) // 2  # 2 bytes/sample (PCM16 mono)
                    buf.clear()
                    chunks_since_flush = 0

                    # Lightweight progress message (every ~10 flushes by default)
                    if (segment_index % 10) == 0:
                        with contextlib.suppress(Exception):
                            await websocket.send_json({
                                "type": "meta",
                                "segment": segment_index,
                                "total_samples": total_samples,
                            })
                    segment_index += 1

                try:
                    async for chunk in audio_generator:
                        buf.extend(chunk)
                        chunks_since_flush += 1

                        # Flush on size or cadence
                        if len(buf) >= BUF_TARGET or chunks_since_flush >= FLUSH_EVERY:
                            await _flush()

                    # Final flush
                    if buf:
                        await _flush()

                except Exception as e:
                    logger.exception("WebSocket: Error during audio generation/sending: %s", str(e))
                    break
            else:
                logger.info("WebSocket: Empty or whitespace-only input received, skipping audio generation.")

            # Send end marker; ignore errors if the client already closed
            with contextlib.suppress(Exception):
                await websocket.send_json({
                    "type": "end",
                    "segment_id": segment_id,
                    "total_samples": total_samples,
                    "duration_seconds": total_samples / 24000.0,
                })

            if not data.get("continue", True):
                break

    except WebSocketDisconnect:
        logger.info("Client disconnected from websocket.")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the websocket endpoint: {e}")
    finally:
        logger.info("Closing websocket connection.")
        with contextlib.suppress(Exception):
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close()

@app.get("/api/voices", response_model=VoicesResponse)
async def get_voices():
    """Get available voices with detailed information."""
    default_voice = engine.available_voices[0] if engine and engine.available_voices else "female"
    return {
        "voices": VOICE_DETAILS,
        "default": default_voice,
        "count": len(VOICE_DETAILS)
    }

@app.get("/api/status")
async def get_status():
    try:
        return engine.get_status() if engine else {"device": "unknown"}
    except Exception:
        return {"device": "error"}

@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.get("/readyz")
async def readyz():
    try:
        st = engine.get_status() if engine else None
        if not st:
            raise RuntimeError("engine not initialized")
        if st.get("device") is None:
            raise RuntimeError("device unknown")
        if st.get("ffmpeg_available") is False:
            # Not fatal, but report in readiness payload
            pass
        return {"ok": True, **st}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))