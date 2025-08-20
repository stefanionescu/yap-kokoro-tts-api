"""WebSocket authentication handling."""

import os
from fastapi import WebSocket


async def authenticate_websocket(websocket: WebSocket) -> bool:
    """
    Authenticate WebSocket connection via API key in query parameters.
    
    Returns True if authentication succeeds or no API key is required.
    Returns False if authentication fails (caller should handle closing connection).
    """
    # Simple API key guard via query param ?api_key=
    required_key = os.getenv("API_KEY")
    if not required_key:
        # No API key required
        return True
    
    query = dict(websocket.query_params)
    provided = query.get("api_key")
    
    if provided != required_key:
        # Authentication failed
        await websocket.close(code=1008)  # policy violation
        return False
    
    # Authentication succeeded
    return True
