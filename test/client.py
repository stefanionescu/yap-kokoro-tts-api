#!/usr/bin/env python3
"""
Kokoro TTS WebSocket client for streaming audio and saving to a listenable format.

Examples:
  # WAV output (recommended)
  python client.py --host <RUNPOD_PUBLIC_IP> --port 8000 \
    --text "Hello there" --voice female --out hello.wav --format wav

  # Ogg/Opus output
  python client.py --host <RUNPOD_PUBLIC_IP> --port 8000 \
    --text "Hello there" --voice female --out hello.ogg --format ogg

Requires ffmpeg for wav/ogg/mp3. If ffmpeg is unavailable and --format pcm is used,
raw 24kHz mono PCM16 is written.
"""
import argparse
import asyncio
import json
import os
import sys
import time
import uuid
import shutil
import subprocess
from urllib.parse import urlsplit

import websockets

SAMPLE_RATE = 24000


def build_ffmpeg_cmd(output_path: str, output_format: str, speed: float = 1.4) -> list[str]:
    """Return ffmpeg command to encode from stdin s16le 24k mono to desired format."""
    base = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "-",
    ]
    
    # Apply tempo correction to compensate for server-side speed up
    # If server generates at 1.4x speed, we slow down by 1/1.4 to get normal playback
    tempo_filter = f"atempo={1.0/speed:.3f}" if speed != 1.0 else None
    
    if output_format == "wav":
        if tempo_filter:
            return base + ["-af", tempo_filter, "-c:a", "pcm_s16le", "-f", "wav", output_path]
        else:
            return base + ["-c:a", "pcm_s16le", "-f", "wav", output_path]
    if output_format in {"ogg", "opus"}:
        bitrate = os.getenv("OPUS_BITRATE", "48k")
        application = os.getenv("OPUS_APPLICATION", "audio")
        if tempo_filter:
            return base + ["-af", tempo_filter, "-c:a", "libopus", "-b:a", bitrate, "-application", application, "-f", "ogg", output_path]
        else:
            return base + ["-c:a", "libopus", "-b:a", bitrate, "-application", application, "-f", "ogg", output_path]
    if output_format == "mp3":
        if tempo_filter:
            return base + ["-af", tempo_filter, "-c:a", "libmp3lame", "-b:a", os.getenv("MP3_BITRATE", "192k"), "-f", "mp3", output_path]
        else:
            return base + ["-c:a", "libmp3lame", "-b:a", os.getenv("MP3_BITRATE", "192k"), "-f", "mp3", output_path]
    raise ValueError(f"Unsupported format for ffmpeg: {output_format}")


def _sanitize_host_and_scheme(host: str) -> tuple[str, bool]:
    """Strip scheme and trailing slashes. Return (host[:port?], tls_from_scheme)."""
    h = (host or "").strip()
    tls_from_scheme = False
    if "://" in h:
        parts = urlsplit(h)
        # netloc contains host[:port]; if missing, parts.path may carry it
        h = parts.netloc or parts.path
        tls_from_scheme = parts.scheme in ("https", "wss")
    h = h.strip().strip("/")
    return h, tls_from_scheme


def _is_runpod_proxy_host(host: str) -> bool:
    h = host.lower()
    return ("proxy.runpod.net" in h) or h.endswith("runpod.net")


async def stream_ws_and_save(host: str, port: int, voice: str, text: str, out_path: str, out_format: str, use_tls: bool = False, speed: float = 1.4) -> int:
    """Connect to WS, stream PCM, encode to chosen format via ffmpeg or write raw PCM."""
    norm_host, tls_from_scheme = _sanitize_host_and_scheme(host)
    force_tls = use_tls or tls_from_scheme or _is_runpod_proxy_host(norm_host)
    if _is_runpod_proxy_host(norm_host):
        # RunPod proxy terminates TLS and forwards to container; no explicit port in public URL
        ws_url = f"wss://{norm_host}/v1/audio/speech/stream/ws"
    else:
        scheme = "wss" if force_tls else "ws"
        # Only append port if it's not already present
        netloc = norm_host if ":" in norm_host else f"{norm_host}:{port}"
        ws_url = f"{scheme}://{netloc}/v1/audio/speech/stream/ws"
    segment_id = f"seg-{uuid.uuid4().hex[:8]}"

    proc: subprocess.Popen | None = None
    file_handle = None
    if out_format == "pcm":
        file_handle = open(out_path, "wb")
    else:
        if not shutil.which("ffmpeg"):
            print("ffmpeg not found; install it or use --format pcm", file=sys.stderr)
            return 2
        proc = subprocess.Popen(build_ffmpeg_cmd(out_path, out_format, speed), stdin=subprocess.PIPE)

    t0 = time.time()
    first = True
    total = 0
    
    print(f"Connecting to {ws_url} with speed={speed:.1f}x (tempo correction: {1.0/speed:.3f}x)")

    # Avoid extra_headers; some builds mis-handle them
    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.send(json.dumps({
            "continue": True,
            "segment_id": segment_id,
            "input": text,
            "voice": voice,
            "format": "pcm",
            "speed": speed,
        }))

        while True:
            msg = await ws.recv()
            if isinstance(msg, (bytes, bytearray)):
                if first:
                    print(f"TTFB: {1000*(time.time()-t0):.0f}ms")
                    first = False
                total += len(msg)
                if file_handle is not None:
                    file_handle.write(msg)
                elif proc and proc.stdin:
                    proc.stdin.write(msg)
            else:
                try:
                    data = json.loads(msg)
                except Exception:
                    continue
                if isinstance(data, dict) and data.get("type") == "end":
                    break

    if proc and proc.stdin:
        try:
            proc.stdin.close()
            proc.wait(timeout=10)
        except Exception:
            pass
    if file_handle is not None:
        file_handle.close()

    print(f"Saved ~{total} bytes of PCM to {out_path}")
    return 0 if total > 0 else 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WebSocket Kokoro TTS client → save audio via ffmpeg")
    parser.add_argument("--host", default="7v9iogacp102xj-8000.proxy.runpod.net", help="API host (RunPod proxy host or hostname[:port])")
    parser.add_argument("--port", type=int, default=8000, help="API port (default: 8000)")
    parser.add_argument("--voice", choices=["female", "male"], default="female", help="Voice to use")
    parser.add_argument("--text", default="I would love to suck that juicy dick!", help="Input text to synthesize")
    parser.add_argument("--out", default="hello.wav", help="Output file path (wav/ogg/mp3/pcm)")
    parser.add_argument("--format", choices=["wav", "ogg", "opus", "mp3", "pcm"], default="wav", help="Output format")
    parser.add_argument("--speed", type=float, default=1.4, help="Speech speed multiplier (0.5-2.0, default: 1.4 to match server)")
    parser.add_argument("--tls", action="store_true", help="Use wss:// (TLS)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        # Determine TLS pref: explicit flag, scheme in host, or RunPod proxy host
        norm_host, tls_from_scheme = _sanitize_host_and_scheme(args.host)
        tls_pref = args.tls or tls_from_scheme or _is_runpod_proxy_host(norm_host)
        rc = asyncio.run(stream_ws_and_save(norm_host, args.port, args.voice, args.text, args.out, args.format, tls_pref, args.speed))
        if rc != 0:
            sys.exit(rc)
    except KeyboardInterrupt:
        print("Interrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()