from src.logger import setup_logger
setup_logger()

import time
import os
from src.vllm import OrpheusModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import logging
import json
import base64

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import warnings
import asyncio

warnings.simplefilter(action='ignore', category=FutureWarning)

load_dotenv()

logger = logging.getLogger(__name__)


class TTSRequest(BaseModel):
    input: str = "Hey there, looks like you forgot to provide a prompt!"
    voice: str = "tara"


class TTSStreamRequest(BaseModel):
    input: str
    voice: str = "tara"
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
    
engine: OrpheusModel = None
VOICE_DETAILS: List[VoiceDetail] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes the TTS engine on application startup."""
    global engine, VOICE_DETAILS
    
    # Get configuration from environment variables
    model_name = os.getenv("MODEL_NAME", "canopylabs/orpheus-3b-0.1-ft")
    max_model_len = int(os.getenv("TRT_MAX_INPUT_LEN", "1024"))
    max_seq_len = int(os.getenv("TRT_MAX_SEQ_LEN", "2048"))
    gpu_utilization = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))
    quantization = os.getenv("QUANTIZATION", "awq")  # Using AWQ for 6-bit quantization
    
    # Voice-specific parameters
    # These will be used as defaults but can be overridden in API calls
    temperature_tara = float(os.getenv("TEMPERATURE_TARA", "0.8"))
    temperature_zac = float(os.getenv("TEMPERATURE_ZAC", "0.4"))
    top_p = float(os.getenv("TOP_P", "0.8"))
    rep_penalty_tara = float(os.getenv("REP_PENALTY_TARA", "1.9"))
    rep_penalty_zac = float(os.getenv("REP_PENALTY_ZAC", "1.85"))
    
    # Context parameters
    num_ctx = int(os.getenv("NUM_CTX", "8192"))
    num_predict = int(os.getenv("NUM_PREDICT", "49152"))
    
    logger.info(f"Initializing OrpheusModel with model={model_name}, quantization={quantization}")
    
    # Initialize the model
    engine = OrpheusModel(
        model_name=model_name,
        max_model_len=max_seq_len,
        gpu_memory_utilization=gpu_utilization,
        quantization=quantization
    )
    
    # Log the configuration
    logger.info(f"Voice settings - tara: temp={temperature_tara}, top_p={top_p}, rep_penalty={rep_penalty_tara}")
    logger.info(f"Voice settings - zac: temp={temperature_zac}, top_p={top_p}, rep_penalty={rep_penalty_zac}")
    logger.info(f"Context settings: num_ctx={num_ctx}, num_predict={num_predict}")
    
    # Define gender for the available voices
    voice_genders = {
        "tara": "female",
        "zac": "male"
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
    yield
    
    # Clean up the model and other resources if needed
    logger.info("Shutting down TTS engine")

app = FastAPI(lifespan=lifespan)


@app.post('/v1/audio/speech/stream')
async def tts_stream(data: TTSRequest):
    """
    Generates audio speech from text in a streaming fashion.
    This endpoint is optimized for low latency (Time to First Byte).
    """
    start_time = time.perf_counter()
    voice = data.voice
    
    # Apply voice-specific settings based on the selected voice
    temperature = None  # Will use voice-specific default from the model
    top_p = float(os.getenv("TOP_P", "0.8"))
    repetition_penalty = None  # Will use voice-specific default from the model
    num_ctx = int(os.getenv("NUM_CTX", "8192"))
    num_predict = int(os.getenv("NUM_PREDICT", "49152"))
    
    logger.info(f"TTS request: voice={voice}, text_length={len(data.input)}")

    async def generate_audio_stream():
        first_chunk = True
        try:
            audio_generator = engine.generate_speech_async(
                prompt=data.input,
                voice=voice,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_ctx=num_ctx,
                num_predict=num_predict
            )

            async for chunk in audio_generator:
                if first_chunk:
                    ttfb = time.perf_counter() - start_time
                    logger.info(f"Time to first audio chunk (TTFB): {ttfb*1000:.2f} ms")
                    first_chunk = False
                yield chunk
        except Exception as e:
            logger.exception(f"Error during audio generation: {str(e)}")

    return StreamingResponse(generate_audio_stream(), media_type='audio/pcm')


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

            voice = data.get("voice", "tara")
            segment_id = data.get("segment_id", "no_segment_id")
            
            # Apply voice-specific settings
            temperature = None  # Will use voice-specific default from the model
            top_p = float(os.getenv("TOP_P", "0.8"))
            repetition_penalty = None  # Will use voice-specific default from the model
            num_ctx = int(os.getenv("NUM_CTX", "8192"))
            num_predict = int(os.getenv("NUM_PREDICT", "49152"))

            start_time = time.perf_counter()
            try:
                await websocket.send_json({"type": "start", "segment_id": segment_id})

                if input_text:
                    logger.info(f"WebSocket: Generating audio for input: '{input_text[:50]}...' (length: {len(input_text)})")
                    audio_generator = engine.generate_speech_async(
                        prompt=input_text,
                        voice=voice,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        num_ctx=num_ctx,
                        num_predict=num_predict
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
    default_voice = engine.available_voices[0] if engine and engine.available_voices else "tara"
    return {
        "voices": VOICE_DETAILS,
        "default": default_voice,
        "count": len(VOICE_DETAILS)
    }