"""Synthesis pipeline for text processing and audio generation."""

import os
import logging
from typing import AsyncGenerator, List, Optional
import numpy as np

from ..audio.utils import float_to_pcm16_bytes, iter_pcm16_chunks

logger = logging.getLogger(__name__)


class SynthesisPipeline:
    """Handles text segmentation and audio synthesis pipeline."""
    
    def __init__(
        self, 
        kokoro_pipeline,
        speed: float,
        split_pattern: str,
        stream_chunk_samples: int,
        first_segment_max_words: int,
        first_segment_boundary_chars: str
    ):
        self.kokoro_pipeline = kokoro_pipeline
        self.speed = speed
        self.split_pattern = split_pattern
        self.stream_chunk_samples = stream_chunk_samples
        self.first_segment_max_words = first_segment_max_words
        self.first_segment_boundary_chars = first_segment_boundary_chars
    
    def segment_for_fast_ttfb(self, text: str) -> list[str]:
        """Split input so the first piece is tiny (<= N words) to minimize TTFB.
        Tries to cut on natural boundaries near the limit; falls back to word cut.
        """
        t = (text or "").strip()
        if not t:
            return [""]
        words = t.split()
        n = self.first_segment_max_words
        if len(words) <= n:
            return [t]
        # find a boundary within first ~n+4 words
        boundary_chars = set(self.first_segment_boundary_chars)
        cut_idx = 0
        acc_words = []
        require_boundary = os.getenv("FIRST_SEGMENT_REQUIRE_BOUNDARY", "1") == "1"
        for i, w in enumerate(words):
            acc_words.append(w)
            if i + 1 >= n and any(ch in boundary_chars for ch in w):
                cut_idx = i + 1
                break
            # Only allow no-boundary fallback when not requiring boundary
            if not require_boundary and (i + 1 >= n + 4):
                cut_idx = i + 1
                break
        # If no natural boundary found and require-boundary is enabled, do NOT split
        if cut_idx == 0:
            if require_boundary:
                return [t]
            cut_idx = n
        first = " ".join(words[:cut_idx]).strip()
        rest = " ".join(words[cut_idx:]).strip()
        return [p for p in [first, rest] if p]
    
    async def synthesize_stream_pieces(
        self, 
        pieces: List[str], 
        voice: str, 
        output_format: str, 
        speed: Optional[float] = None, 
        request_id: Optional[str] = None,
        request_manager=None
    ) -> AsyncGenerator[bytes, None]:
        """Synthesize audio from text pieces and stream as bytes."""
        # Voice is already resolved by main.py; use as-is (should be a valid Kokoro voice id or custom recipe)
        if not voice:
            raise ValueError("Voice must be provided (main.py should have resolved it)")
        kokoro_voice = voice

        def pipeline_for(piece: str):
            # Use per-request speed if provided, otherwise engine default
            eff_speed = float(speed if speed is not None else self.speed)
            # Try common Kokoro voice IDs if mapping fails silently
            try:
                return self.kokoro_pipeline(
                    piece,
                    voice=kokoro_voice,
                    speed=eff_speed,
                    split_pattern=self.split_pattern,
                )
            except Exception as e:
                # No fallback - voice should have been validated by main.py
                logger.error("Voice '%s' failed and no fallback available: %s", kokoro_voice, e)
                raise

        if output_format == "pcm":
            for piece in pieces:
                # Kokoro v1 yields (gs, ps, audio) tuples; audio is np.ndarray float32
                for _, _, audio in pipeline_for(piece):
                    if request_id and request_manager and request_manager.is_canceled(request_id):
                        return
                    if audio is None:
                        continue
                    audio_arr = np.asarray(audio, dtype=np.float32).flatten()
                    pcm = float_to_pcm16_bytes(audio_arr)
                    if not pcm:
                        continue
                    if self.stream_chunk_samples > 0:
                        bpc = self.stream_chunk_samples * 2  # 2 bytes/sample
                        for i in range(0, len(pcm), bpc):
                            chunk = pcm[i : i + bpc]
                            if chunk:
                                yield chunk
                    else:
                        yield pcm
            return

        # Only PCM format supported
        if output_format != "pcm":
            logger.warning("Only PCM format is supported; ignoring requested format '%s'", output_format)
        
        for piece in pieces:
            for _, _, audio_np in pipeline_for(piece):
                if request_id and request_manager and request_manager.is_canceled(request_id):
                    return
                for pcm_bytes in iter_pcm16_chunks(audio_np, self.stream_chunk_samples):
                    yield pcm_bytes
