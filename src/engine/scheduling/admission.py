"""Admission control and concurrency management."""

import asyncio
import contextlib
import os
from ...constants import EWMA_ALPHA


class AdmissionControl:
    """Manages admission control and concurrency slots."""
    
    def __init__(self, max_concurrent: int, queue_wait_sla_ms: int, 
                 max_queued_requests: int, initial_ewma_wall_ms: float):
        self.max_concurrent = max_concurrent
        self.queue_wait_sla_ms = queue_wait_sla_ms
        self.max_queued_requests = max_queued_requests
        
        # Number of in-flight synthesis streams (process-wide)
        self._inflight_count: int = 0
        # Total accepted requests (running + queued/reserved at API)
        self._accepted_slots: int = 0
        # EWMA of observed job wall time (ms) used for SLA estimation
        self._ewma_wall_ms: float = initial_ewma_wall_ms
        # Admission lock to make reservations atomic across connections
        self._admission_lock: asyncio.Lock = asyncio.Lock()
        
        # Set active_limit same as max_concurrent for simplicity
        self.active_limit = self.max_concurrent
        
        # Token semaphore for active slots only (no queue). We attempt immediate acquire.
        self._admission_semaphore: asyncio.Semaphore = asyncio.Semaphore(self.active_limit)
    
    def can_accept(self) -> bool:
        """Non-blocking admission check: token available?"""
        return getattr(self._admission_semaphore, "_value", 0) > 0
    
    def try_accept_request(self) -> bool:
        """Non-blocking: admit only if a running slot token is available."""
        return self.can_accept()
    
    async def try_accept_request_async(self) -> bool:
        """Immediate token acquire; return False if none (no queue)."""
        # Non-blocking: check token count first to avoid any waiting
        if getattr(self._admission_semaphore, "_value", 0) <= 0:  # type: ignore[attr-defined]
            return False
        await self._admission_semaphore.acquire()
        return True
    
    def release_accept_slot(self) -> None:
        """Release an admission slot."""
        with contextlib.suppress(Exception):
            self._admission_semaphore.release()
        self._accepted_slots = max(0, self._accepted_slots - 1)
    
    def record_job_wall_ms(self, wall_ms: float) -> None:
        """Update EWMA for job wall time used in SLA estimation."""
        try:
            alpha = EWMA_ALPHA
            self._ewma_wall_ms = (1 - alpha) * self._ewma_wall_ms + alpha * max(1.0, float(wall_ms))
        except Exception:
            pass
    
    def try_reserve_slot(self) -> bool:
        """Atomically reserve a concurrency slot if available.

        Returns True on success. Caller must later call release_slot().
        """
        if self._inflight_count >= self.active_limit:
            return False
        self._inflight_count += 1
        return True
    
    def release_slot(self) -> None:
        """Release a previously reserved concurrency slot."""
        self._inflight_count = max(0, self._inflight_count - 1)
    
    @property
    def inflight_count(self) -> int:
        """Get current inflight count."""
        return self._inflight_count
