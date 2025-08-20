"""Job management and dataclass definitions."""

import asyncio
import contextlib
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TTSJob:
    """Represents a text-to-speech synthesis job."""
    pieces: List[str]
    voice: str
    output_format: str
    out_queue: asyncio.Queue
    end_stream: bool
    priority: bool
    speed: Optional[float] = None
    request_id: Optional[str] = None


class RequestManager:
    """Manages request registration, cancellation, and cleanup."""
    
    def __init__(self):
        # Cancellation registry per request_id
        self._cancel_flags: dict[str, bool] = {}
    
    def register_request(self, request_id: Optional[str]) -> None:
        """Register a request for cancellation tracking."""
        if not request_id:
            return
        # initialize cancel flag as False
        self._cancel_flags[request_id] = False
    
    def cancel_request(self, request_id: Optional[str]) -> bool:
        """Mark a request as canceled."""
        if not request_id:
            return False
        if request_id in self._cancel_flags:
            self._cancel_flags[request_id] = True
            return True
        # If not registered yet, mark as canceled so future start short-circuits
        self._cancel_flags[request_id] = True
        return True
    
    def clear_request(self, request_id: Optional[str]) -> None:
        """Clear a request from the registry."""
        if not request_id:
            return
        with contextlib.suppress(Exception):
            self._cancel_flags.pop(request_id, None)
    
    def is_canceled(self, request_id: Optional[str]) -> bool:
        """Check if a request is canceled."""
        if not request_id:
            return False
        return self._cancel_flags.get(request_id, False)
