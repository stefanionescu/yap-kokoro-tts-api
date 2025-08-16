import asyncio
import logging
import os
from typing import AsyncGenerator, Iterable

import numpy as np
from kokoro import KPipeline


logger = logging.getLogger(__name__)


SAMPLE_RATE = 24000  # Kokoro outputs 24 kHz


def _float_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    if audio is None or audio.size == 0:
        return b""
    # Ensure float32 in [-1, 1]
    audio = np.clip(audio.astype(np.float32), -1.0, 1.0)
    pcm = (audio * 32767.0).round().astype(np.int16)
    return pcm.tobytes()


class OrpheusModel:
    """
    Wrapper class keeping the same name/API but backed by Kokoro's KPipeline.

    - Exposes available_voices = ["female", "male"] for API compatibility
    - Maps 'female' → 'aoede' and 'male' → 'michael' by default (env-overridable)
    - Streams raw PCM16 bytes at 24 kHz
    """

    def __init__(
        self,
        model_name: str | None = None,
        tokenizer: str | None = None,
        max_model_len: int | None = None,
        gpu_memory_utilization: float = 0.9,
        max_num_batched_tokens: int = 8192,
        max_num_seqs: int = 4,
        enable_chunked_prefill: bool = True,
        quantization: str | None = None,
    ) -> None:
        # Public API compatibility fields (unused by Kokoro)
        self.model_name = model_name or "hexgrad/Kokoro-82M"
        self.quantization = quantization or "none"

        # API voice options
        self.available_voices = ["female", "male"]
        self._voice_mapping = {
            "female": os.getenv("DEFAULT_VOICE_FEMALE", "aoede"),
            "male": os.getenv("DEFAULT_VOICE_MALE", "michael"),
        }

        lang_code = os.getenv("LANG_CODE", "a")  # 'a' = American English
        logger.info(
            "Initializing Kokoro KPipeline | model=%s lang_code=%s", self.model_name, lang_code
        )
        # KPipeline downloads/loads weights under the hood when first used
        self.pipeline = KPipeline(lang_code=lang_code)

        # Streaming/chunking behavior
        self.speed = float(os.getenv("KOKORO_SPEED", "1.0"))
        self.split_pattern = os.getenv("KOKORO_SPLIT_PATTERN", r"\n+")
        # Stream chunking in samples (defaults to 0.5s)
        self.stream_chunk_samples = int(
            float(os.getenv("STREAM_CHUNK_SECONDS", "0.5")) * SAMPLE_RATE
        )

        logger.info(
            "Kokoro ready | voices: female=%s male=%s | speed=%s split=%s chunk=%d samples",
            self._voice_mapping["female"],
            self._voice_mapping["male"],
            self.speed,
            self.split_pattern,
            self.stream_chunk_samples,
        )

    def validate_voice(self, voice: str) -> None:
        if voice not in self.available_voices:
            raise ValueError(
                f"Voice {voice} is not available. Valid options are: {', '.join(self.available_voices)}"
            )

    def _iter_pcm16_chunks(self, audio: np.ndarray) -> Iterable[bytes]:
        if audio is None or audio.size == 0:
            return
        # Ensure 1D float array
        audio = np.asarray(audio).astype(np.float32).flatten()
        if self.stream_chunk_samples <= 0 or audio.size <= self.stream_chunk_samples:
            yield _float_to_pcm16_bytes(audio)
            return
        # Chunk into fixed-size segments for streaming
        for i in range(0, audio.size, self.stream_chunk_samples):
            segment = audio[i : i + self.stream_chunk_samples]
            yield _float_to_pcm16_bytes(segment)

    async def generate_speech_async(
        self, prompt: str, voice: str | None = None, **kwargs
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate speech audio asynchronously as PCM16 bytes (24 kHz).

        Parameters present in kwargs like temperature/top_p/repetition_penalty are
        accepted for API compatibility but ignored by Kokoro.
        """
        selected_voice = voice or "female"
        if selected_voice not in self.available_voices:
            selected_voice = "female"

        kokoro_voice = self._voice_mapping[selected_voice]
        text = prompt or ""
        if not text.strip():
            return

        logger.info(
            "Kokoro generating | voice=%s (%s) text_len=%d",
            selected_voice,
            kokoro_voice,
            len(text),
        )

        # Kokoro yields (graphemes, phonemes, audio_np) per segment
        generator = self.pipeline(
            text,
            voice=kokoro_voice,
            speed=self.speed,
            split_pattern=self.split_pattern,
        )

        # Stream out PCM bytes per segment, chunked
        for _, _, audio_np in generator:
            for pcm_bytes in self._iter_pcm16_chunks(audio_np):
                yield pcm_bytes