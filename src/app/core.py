"""Core FastAPI app creation and configuration."""

import os
import logging
import warnings
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI
from dotenv import load_dotenv

from engine import KokoroEngine
from .websocket.handler import create_websocket_endpoint
from .health.endpoints import create_health_endpoints

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Global engine instance
engine: Optional[KokoroEngine] = None


def get_engine() -> Optional[KokoroEngine]:
    """Get the global engine instance."""
    return engine


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


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(lifespan=lifespan)
    
    # Register WebSocket endpoint
    create_websocket_endpoint(app, get_engine)
    
    # Register health check endpoints
    create_health_endpoints(app, get_engine)
    
    return app
