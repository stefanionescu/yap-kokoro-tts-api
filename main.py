from src.logger import setup_logger
setup_logger()

import time
import os
from src.engine import KokoroEngine
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import contextlib
import logging
import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from fastapi.responses import StreamingResponse
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
    
    # Get configuration from environment variables
    model_name = os.getenv("MODEL_NAME", "hexgrad/Kokoro-82M")
    quantization = os.getenv("QUANTIZATION", "none")
    
    # Retained for API compatibility; Kokoro ignores sampling params
    temperature_tara = float(os.getenv("TEMPERATURE_TARA", "0.8"))
    temperature_zac = float(os.getenv("TEMPERATURE_ZAC", "0.4"))
    top_p = float(os.getenv("TOP_P", "0.8"))
    rep_penalty_tara = float(os.getenv("REP_PENALTY_TARA", "1.9"))
    rep_penalty_zac = float(os.getenv("REP_PENALTY_ZAC", "1.85"))
    
    # Context parameters
    num_ctx = int(os.getenv("NUM_CTX", "8192"))
    num_predict = int(os.getenv("NUM_PREDICT", "49152"))
    
    logger.info(f"Initializing TTS engine with model={model_name}, quantization={quantization}")
    
    # Initialize the engine
    engine = KokoroEngine(lang_code=os.getenv("LANG_CODE", "a"))
    engine.start_worker()
    
    # Log the configuration
    logger.info(f"Voice settings - female: temp={temperature_tara}, top_p={top_p}, rep_penalty={rep_penalty_tara}")
    logger.info(f"Voice settings - male: temp={temperature_zac}, top_p={top_p}, rep_penalty={rep_penalty_zac}")
    logger.info(f"Context settings: num_ctx={num_ctx}, num_predict={num_predict}")
    
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

    async def _keep_gpu_hot_loop():
        """Periodically run a tiny generation to keep GPU kernels warm."""
        interval = float(os.getenv("KEEP_GPU_HOT_INTERVAL", "25"))
        voice_hint = os.getenv("KEEP_GPU_HOT_VOICE", "male")
        prompt_hint = os.getenv("KEEP_GPU_HOT_PROMPT", ".")
        while True:
            try:
                gen = engine.generate_speech_async(
                    prompt=prompt_hint,
                    voice=voice_hint,
                    temperature=0.0,
                    top_p=1.0,
                    repetition_penalty=1.0,
                    max_tokens=1,
                )
                async for _ in gen:
                    break
            except Exception as e:
                logger.debug(f"keep_gpu_hot error: {e}")
            await asyncio.sleep(interval)

    # Start keep-hot task
    global _keep_hot_task
    _keep_hot_task = asyncio.create_task(_keep_gpu_hot_loop())
    yield
    
    # Clean up the model and other resources if needed
    logger.info("Shutting down TTS engine")
    try:
        if _keep_hot_task is not None:
            _keep_hot_task.cancel()
            with contextlib.suppress(Exception):
                await _keep_hot_task
    except Exception:
        pass

app = FastAPI(lifespan=lifespan)


@app.post('/v1/audio/speech/stream')
async def tts_stream(data: TTSRequest):
    """
    Generates audio speech from text in a streaming fashion.
    This endpoint is optimized for low latency (Time to First Byte).
    """
    start_time = time.perf_counter()
    voice = data.voice
    
    out_format = data.format.lower() if data.format else "pcm"
    num_ctx = int(os.getenv("NUM_CTX", "8192"))
    num_predict = int(os.getenv("NUM_PREDICT", "49152"))
    
    logger.info(f"TTS request: voice={voice}, text_length={len(data.input)}")

    async def generate_audio_stream():
        first_chunk = True
        try:
            # Optional priming to defeat overly aggressive proxy buffering
            if os.getenv("PRIME_STREAM", "0") == "1":
                yield b"\0" * 128

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
                yield chunk
        except Exception as e:
            logger.exception(f"Error during audio generation: {str(e)}")

    return StreamingResponse(
        generate_audio_stream(),
        media_type='audio/pcm',
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-store",
            "Content-Type": "audio/pcm",
        },
    )


@app.websocket("/v1/audio/speech/stream/ws")
async def tts_stream_ws(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection open")
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
            
            out_format = str(data.get("format", "pcm")).lower()

            start_time = time.perf_counter()
            try:
                await websocket.send_json({"type": "start", "segment_id": segment_id})

                if input_text:
                    logger.info(f"WebSocket: Generating audio for input: '{input_text[:50]}...' (length: {len(input_text)})")
                    audio_generator = engine.generate_speech_async(
                        prompt=input_text,
                        voice=voice,
                        output_format=out_format,
                    )

                    first_chunk = True
                    async for chunk in audio_generator:
                        if first_chunk:
                            ttfb = time.perf_counter() - start_time
                            logger.info(f"WebSocket: Time to first audio chunk (TTFB): {ttfb*1000:.2f} ms")
                            first_chunk = False
                        await websocket.send_bytes(chunk)
                else:
                    logger.info("WebSocket: Empty or whitespace-only input received, skipping audio generation.")
                
                await websocket.send_json({"type": "end", "segment_id": segment_id})

                if not data.get("continue", True):
                    await websocket.send_json({"done": True})
                    break

            except Exception as e:
                logger.exception(f"WebSocket: Error during audio generation: {str(e)}")
                await websocket.send_json({"error": str(e), "done": True})
                break

    except WebSocketDisconnect:
        logger.info("Client disconnected from websocket.")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the websocket endpoint: {e}")
    finally:
        logger.info("Closing websocket connection.")
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