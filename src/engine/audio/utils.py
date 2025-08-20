"""Audio utility functions for PCM processing."""

from typing import Iterable
import numpy as np
from ...constants import SAMPLE_RATE


def float_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 audio array to PCM16 bytes."""
    if audio is None or audio.size == 0:
        return b""
    # Ensure float32 in [-1, 1]
    audio = np.clip(audio.astype(np.float32), -1.0, 1.0)
    pcm = (audio * 32767.0).round().astype(np.int16)
    return pcm.tobytes()


def iter_pcm16_chunks(
    audio: object, stream_chunk_samples: int
) -> Iterable[bytes]:
    """Yield PCM16 bytes from various possible audio types.

    Accepts numpy arrays (float), torch tensors, or already-encoded bytes.
    """
    if audio is None:
        return
    # Fast path: already bytes/bytearray (assumed PCM16 mono 24k)
    if isinstance(audio, (bytes, bytearray)):
        if stream_chunk_samples <= 0:
            yield bytes(audio)
            return
        bytes_per_chunk = stream_chunk_samples * 2  # 2 bytes per sample
        for i in range(0, len(audio), bytes_per_chunk):
            yield bytes(audio[i : i + bytes_per_chunk])
        return

    # Convert arrays / tensors to np.float32
    try:
        if isinstance(audio, np.ndarray):
            arr = audio
        elif 'torch' in str(type(audio)):
            try:
                arr = audio.detach().to('cpu').numpy()
            except Exception:
                return
        else:
            arr = np.asarray(audio)
    except Exception:
        return

    if arr.size == 0:
        return
    arr = np.asarray(arr).astype(np.float32).flatten()
    if stream_chunk_samples <= 0 or arr.size <= stream_chunk_samples:
        yield float_to_pcm16_bytes(arr)
        return
    for i in range(0, arr.size, stream_chunk_samples):
        segment = arr[i : i + stream_chunk_samples]
        yield float_to_pcm16_bytes(segment)
