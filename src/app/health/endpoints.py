"""Health check endpoints for the TTS service."""

from fastapi import HTTPException


def create_health_endpoints(app, engine_getter):
    """Create health check endpoints and register them with the FastAPI app."""
    
    @app.get("/healthz")
    async def healthz():
        """Basic health check endpoint."""
        return {"ok": True}

    @app.get("/readyz")
    async def readyz():
        """Readiness check endpoint that verifies engine status."""
        try:
            engine = engine_getter()
            st = engine.get_status() if engine else None
            if not st:
                raise RuntimeError("engine not initialized")
            return {"ok": True, **st}
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))
