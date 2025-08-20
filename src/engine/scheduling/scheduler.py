"""Worker scheduling and round-robin job execution."""

import os
import asyncio
import contextlib
import logging
from asyncio import QueueEmpty
from collections import deque
from typing import AsyncGenerator, List, Optional, TYPE_CHECKING

from .jobs import TTSJob
from .admission import AdmissionControl
from ...constants import (
    PRIORITY_DEFAULT_QUANTUM_BYTES,
    WORKER_IDLE_SLEEP_S,
)

if TYPE_CHECKING:
    from ..synthesis.pipeline import SynthesisPipeline

logger = logging.getLogger(__name__)


class WorkerScheduler:
    """Manages worker tasks and round-robin job scheduling."""
    
    def __init__(
        self, 
        synthesis_pipeline: "SynthesisPipeline",
        admission_control: AdmissionControl,
        pri_queue: asyncio.Queue[TTSJob],
        job_queue: asyncio.Queue[TTSJob],
        quantum_bytes: int,
        request_manager
    ):
        self.synthesis_pipeline = synthesis_pipeline
        self.admission_control = admission_control
        self._pri_queue = pri_queue
        self._job_queue = job_queue
        self.quantum_bytes = quantum_bytes
        self.request_manager = request_manager
        
        # Worker task management
        self._worker_tasks: list[asyncio.Task] = []
    
    def start_worker(self) -> None:
        """Run exactly ONE scheduler loop; it interleaves up to active_limit streams."""
        alive = [t for t in self._worker_tasks if not t.done()]
        if alive:
            self._worker_tasks = alive
            return
        t = asyncio.create_task(self._worker_loop())
        self._worker_tasks = [t]
        logger.info("Kokoro RR worker started (active_limit=%d, quantum=%d bytes)",
                    self.admission_control.active_limit, self.quantum_bytes)
    
    async def stop_worker(self) -> None:
        """Stop all worker tasks."""
        try:
            for t in self._worker_tasks:
                if not t.done():
                    t.cancel()
            for t in self._worker_tasks:
                with contextlib.suppress(Exception):
                    await t
        finally:
            self._worker_tasks = []
    
    async def _worker_loop(self) -> None:
        """Main round-robin scheduler loop."""
        logger.info("Kokoro RR scheduler loop running")
        active = deque()  # items: (job, agen) where agen is async generator of PCM bytes

        async def start_from_job(job: TTSJob):
            # If canceled before start, short-circuit by yielding nothing
            if job.request_id and self.request_manager.is_canceled(job.request_id):
                async def _empty():
                    if False:
                        yield b""  # pragma: no cover
                    return
                return (job, _empty())
            agen = self.synthesis_pipeline.synthesize_stream_pieces(
                job.pieces, job.voice, job.output_format, job.speed, job.request_id, self.request_manager
            )
            return (job, agen)

        while True:
            # Fill active set up to the effective number of inflight streams to avoid blocking under partial load
            effective_limit = min(
                self.admission_control.active_limit, 
                max(1, self.admission_control.inflight_count)
            )
            while len(active) < effective_limit:
                try:
                    # Priority first
                    job = self._pri_queue.get_nowait()
                    self._pri_queue.task_done()
                except QueueEmpty:
                    try:
                        job = self._job_queue.get_nowait()
                        self._job_queue.task_done()
                    except QueueEmpty:
                        break
                active.append(await start_from_job(job))

            if not active:
                # nothing to do; small sleep so we don't spin
                await asyncio.sleep(WORKER_IDLE_SLEEP_S)
                continue

            # Round-robin one quantum
            job, agen = active.popleft()
            sent = 0
            prio_q = int(float(os.getenv("PRIORITY_QUANTUM_BYTES", str(PRIORITY_DEFAULT_QUANTUM_BYTES))))
            budget = prio_q if getattr(job, "priority", False) else self.quantum_bytes
            try:
                while sent < budget:
                    chunk = await anext(agen)
                    if not chunk:
                        continue
                    await job.out_queue.put(chunk)
                    sent += len(chunk)
            except StopAsyncIteration:
                # done
                if job.end_stream:
                    await job.out_queue.put(None)
            except Exception as e:
                logger.exception("Worker error: %s", e)
                if job.end_stream:
                    await job.out_queue.put(None)
            else:
                # Not finished; rotate back
                active.append((job, agen))
