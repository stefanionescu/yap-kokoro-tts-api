"""Audio encoding utilities for Opus compression."""

import os
import contextlib
import subprocess
import threading
from typing import Iterable

from constants import (
    SAMPLE_RATE,
    OPUS_DEFAULT_BITRATE,
    OPUS_DEFAULT_APPLICATION,
)


def opus_encode_via_ffmpeg(pcm_iter: Iterable[bytes]) -> Iterable[bytes]:
    """Encode PCM16 mono 24kHz to Ogg Opus using ffmpeg subprocess."""
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "-",
        "-c:a", "libopus",
        "-b:a", os.getenv("OPUS_BITRATE", OPUS_DEFAULT_BITRATE),
        "-application", os.getenv("OPUS_APPLICATION", OPUS_DEFAULT_APPLICATION),
        "-f", "ogg",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def writer():
        try:
            assert proc.stdin is not None
            for chunk in pcm_iter:
                proc.stdin.write(chunk)
            proc.stdin.close()
        except Exception:
            pass

    t = threading.Thread(target=writer, daemon=True)
    t.start()

    try:
        assert proc.stdout is not None
        while True:
            buf = proc.stdout.read(4096)
            if not buf:
                break
            yield buf
    finally:
        with contextlib.suppress(Exception):
            proc.kill()
