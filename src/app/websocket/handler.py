"""Main WebSocket handler that orchestrates all TTS streaming components."""

import asyncio
import contextlib
import logging
from typing import Callable, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from constants import JOB_QUEUE_GET_TIMEOUT_S, PROCESSOR_LOOP_SLEEP_S
from .auth import authenticate_websocket
from .voice import VoiceResolver
from .streaming import TTSStreamer
from .receiver import MessageReceiver

logger = logging.getLogger(__name__)


def create_websocket_endpoint(app: FastAPI, engine_getter: Callable):
    """Create and register the WebSocket endpoint with the FastAPI app."""
    
    @app.websocket("/v1/audio/speech/stream/ws")
    async def tts_stream_ws(websocket: WebSocket):
        """Main WebSocket endpoint for TTS streaming."""
        # Authentication
        if not await authenticate_websocket(websocket):
            return
        
        await websocket.accept()
        logger.info("WebSocket connection open")

        # Get engine instance
        engine = engine_getter()
        if not engine:
            await websocket.close(code=1011, reason="Service unavailable")
            return

        # Initialize components
        voice_resolver = VoiceResolver()
        send_lock = asyncio.Lock()
        job_queue: asyncio.Queue[dict] = asyncio.Queue()
        canceled: Set[str] = set()

        # Create send_json_safe function for components
        async def send_json_safe(obj: dict) -> None:
            try:
                async with send_lock:
                    await websocket.send_json(obj)
            except Exception:
                pass

        # Initialize streaming and message handling components
        streamer = TTSStreamer(websocket, engine, voice_resolver, send_lock, canceled)
        receiver = MessageReceiver(websocket, engine, voice_resolver, job_queue, canceled, send_json_safe)
        
        # Connect components
        receiver.set_active_request_tracker(streamer.get_active_request_id)

        try:
            # Start receiver task
            recv_task = asyncio.create_task(receiver.receiver_loop())
            
            # Main processor loop: serialize speaks per connection, preserve order
            while True:
                if receiver.is_closing() and job_queue.empty():
                    break
                
                if streamer.get_active_request_id() is None:
                    try:
                        job = await asyncio.wait_for(job_queue.get(), timeout=JOB_QUEUE_GET_TIMEOUT_S)
                    except asyncio.TimeoutError:
                        continue
                    
                    if job.get("request_id") in canceled:
                        await send_json_safe({"type": "response.canceled", "response": job.get("request_id")})
                        job_queue.task_done()
                        continue
                    
                    await streamer.stream_one(job)
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
