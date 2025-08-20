"""WebSocket message receiving and processing logic."""

import os
import time
import json
import asyncio
import contextlib
import logging
import uuid as _uuid
from typing import Set, Optional

from fastapi import WebSocket, WebSocketDisconnect

from constants import SAMPLE_RATE, MAX_UTTERANCE_WORDS_DEFAULT
from .voice import VoiceResolver

logger = logging.getLogger(__name__)


class MessageReceiver:
    """Handles incoming WebSocket messages and protocol logic."""
    
    def __init__(
        self,
        websocket: WebSocket,
        engine,
        voice_resolver: VoiceResolver,
        job_queue: asyncio.Queue,
        canceled: Set[str],
        send_json_safe_func
    ):
        self.websocket = websocket
        self.engine = engine
        self.voice_resolver = voice_resolver
        self.job_queue = job_queue
        self.canceled = canceled
        self.send_json_safe = send_json_safe_func
        
        # Connection state
        self.closing = False
        self.out_format = "pcm"
        self.max_utterance_words = int(os.getenv("MAX_UTTERANCE_WORDS", str(MAX_UTTERANCE_WORDS_DEFAULT)))
        
        # Incremental input state (for Pipecat-style streaming)
        self.pending_text: str = ""
        self.utterance_id: str | None = None
        self.last_append_t: float = 0.0
        self.flush_task: asyncio.Task | None = None
        self.FLUSH_IDLE_MS = int(os.getenv("FLUSH_IDLE_MS", "160"))
    
    def set_active_request_tracker(self, get_active_request_id_func):
        """Set function to get the currently active request ID."""
        self.get_active_request_id = get_active_request_id_func
    
    async def _maybe_flush(self, force: bool = False) -> None:
        """Handle incremental input flushing logic."""
        buf = self.pending_text.strip()
        if not buf:
            return
        if not force:
            idle_ms = (time.perf_counter() - self.last_append_t) * 1000.0
            if idle_ms < self.FLUSH_IDLE_MS:
                return
        # consume buffer as a single unit
        self.pending_text = ""
        # Admission control
        ok = False
        try:
            ok = await self.engine.try_accept_request_async()  # type: ignore[attr-defined]
        except AttributeError:
            ok = self.engine.try_accept_request()
        if not ok:
            await self.send_json_safe({"type": "response.error", "response": self.utterance_id or "", "code": "busy"})
            return
        first_chunk_of_utt = self.utterance_id is None
        if self.utterance_id is None:
            self.utterance_id = f"utt-{int(time.time()*1000)}"
        await self.job_queue.put({
            "request_id": f"{self.utterance_id}-{_uuid.uuid4().hex[:6]}",
            "response_id": self.utterance_id,
            "text": buf,
            "voice": self.voice_resolver.get_default_voice(),
            "speed": None,
            # Only send response.created on the first flush of this utterance
            "suppress_created": (not first_chunk_of_utt),
            # Only send response.completed on commit/force; keep streaming otherwise
            "suppress_completed": (not force),
        })

    async def _schedule_debounce_flush(self) -> None:
        """Schedule debounced flush for incremental input."""
        if self.flush_task and not self.flush_task.done():
            with contextlib.suppress(Exception):
                self.flush_task.cancel()
        async def _runner():
            try:
                await asyncio.sleep(self.FLUSH_IDLE_MS / 1000.0)
                await self._maybe_flush(force=True)
            except Exception:
                pass
        self.flush_task = asyncio.create_task(_runner())
    
    async def receiver_loop(self) -> None:
        """Main message receiving loop."""
        while True:
            raw = await self.websocket.receive()
            if raw.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect()
            if "text" in raw and raw["text"]:
                try:
                    data = json.loads(raw["text"])  # type: ignore[name-defined]
                except Exception:
                    await self.send_json_safe({"type": "response.error", "code": "bad_json"})
                    continue
            elif "bytes" in raw and raw["bytes"]:
                await self.send_json_safe({"type": "response.error", "code": "unexpected_binary"})
                continue
            else:
                continue

            msg_type = data.get("type")
            # OpenAI-only protocol
            if True:
                if msg_type == "session.update":
                    await self._handle_session_update(data)
                elif msg_type == "input.append":
                    await self._handle_input_append(data)
                elif msg_type == "input.commit":
                    await self._handle_input_commit(data)
                elif msg_type == "barge":
                    await self._handle_barge()
                elif msg_type == "response.create":
                    await self._handle_response_create(data)
                elif msg_type == "response.cancel":
                    await self._handle_response_cancel(data)
                elif msg_type == "session.end":
                    self.closing = True
                    break
                else:
                    await self.send_json_safe({"type": "response.error", "code": "unknown_type", "got": msg_type})
    
    async def _handle_session_update(self, data: dict) -> None:
        """Handle session.update message."""
        sess = data.get("session", {}) if isinstance(data.get("session"), dict) else {}
        v_top = data.get("voice")
        v_sess = sess.get("voice") if isinstance(sess, dict) else None
        a_top = data.get("audio") or {}
        a_sess = sess.get("audio") if isinstance(sess, dict) else {}
        voice_val = (v_top or v_sess or self.voice_resolver.get_default_voice()) or "female"
        sr_val = (a_top.get("sample_rate") if isinstance(a_top, dict) else None) or (a_sess.get("sample_rate") if isinstance(a_sess, dict) else None) or SAMPLE_RATE
        try:
            self.voice_resolver.resolve_voice(voice_val)
            self.voice_resolver.set_default_voice(voice_val)
        except ValueError:
            await self.send_json_safe({"type": "response.error", "code": "invalid_voice", "message": str(voice_val)})
            return
        self.out_format = "pcm"
        await self.send_json_safe({
            "type": "session.updated",
            "session": {
                "voice": self.voice_resolver.get_default_voice(),
                "audio": {"format": self.out_format, "sample_rate": int(sr_val) if isinstance(sr_val, (int, float)) else SAMPLE_RATE},
            },
        })
    
    async def _handle_input_append(self, data: dict) -> None:
        """Handle input.append message."""
        chunk = data.get("text")
        if isinstance(chunk, str) and chunk:
            self.pending_text += chunk
            self.last_append_t = time.perf_counter()
            await self._maybe_flush(force=False)
            await self._schedule_debounce_flush()
    
    async def _handle_input_commit(self, data: dict) -> None:
        """Handle input.commit message."""
        await self._maybe_flush(force=True)
        # Reset utterance id after commit completes
        self.utterance_id = None
    
    async def _handle_barge(self) -> None:
        """Handle barge message."""
        # Cancel current playback and clear buffer
        active_request_id = self.get_active_request_id() if hasattr(self, 'get_active_request_id') else None
        if active_request_id:
            self.canceled.add(active_request_id)
            self.engine.cancel_request(str(active_request_id))
        self.pending_text = ""
        self.utterance_id = None
    
    async def _handle_response_create(self, data: dict) -> None:
        """Handle response.create message."""
        req_id = data.get("response_id") or data.get("id") or f"req-{int(time.time()*1000)}"
        input_text = data.get("input") or data.get("text") or data.get("instructions")
        if not isinstance(input_text, str):
            await self.send_json_safe({"type": "response.error", "response": req_id, "code": "bad_text"})
            return
        input_text = input_text.strip()
        if not input_text:
            await self.send_json_safe({"type": "response.error", "response": req_id, "code": "empty_text"})
            return
        audio_cfg = data.get("audio") or {}
        voice_override = data.get("voice") or (audio_cfg.get("voice") if isinstance(audio_cfg, dict) else None)
        speed_raw = data.get("speed") or (audio_cfg.get("speed") if isinstance(audio_cfg, dict) else None)
        test_mode = data.get("test_mode")  # Track client test mode
        spd_val = None
        if speed_raw is not None:
            try:
                spd_val = float(speed_raw)
            except Exception:
                spd_val = None
            else:
                spd_val = max(0.5, min(2.0, spd_val))
        # Word cap
        try:
            word_count = len(input_text.split())
        except Exception:
            word_count = 0
        if self.max_utterance_words > 0 and word_count > self.max_utterance_words:
            await self.send_json_safe({
                "type": "response.error",
                "response": req_id,
                "code": "too_long",
                "max_words": self.max_utterance_words,
                "got_words": word_count,
            })
            return
        ok = False
        try:
            ok = await self.engine.try_accept_request_async()  # type: ignore[attr-defined]
        except AttributeError:
            ok = self.engine.try_accept_request()
        if not ok:
            await self.send_json_safe({"type": "response.error", "response": req_id, "code": "busy"})
            return
        await self.job_queue.put({
            "request_id": req_id,
            "response_id": req_id,
            "text": input_text,
            "voice": voice_override or self.voice_resolver.get_default_voice(),
            "speed": spd_val,
            "test_mode": test_mode,
        })
    
    async def _handle_response_cancel(self, data: dict) -> None:
        """Handle response.cancel message."""
        resp = data.get("response") or data.get("response_id")
        if resp:
            self.canceled.add(resp)
            self.engine.cancel_request(str(resp))
    
    def is_closing(self) -> bool:
        """Check if the connection is closing."""
        return self.closing
