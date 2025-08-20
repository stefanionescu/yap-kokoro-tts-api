#!/usr/bin/env python3
"""
Kokoro TTS WebSocket client for streaming audio and saving to a listenable format.

Examples:
  # WAV output (recommended)
  python test/client.py --host <RUNPOD_PUBLIC_IP> \
    --text "Hello there" --voice female --out hello.wav --format wav

  # Ogg/Opus output with custom speed
  python test/client.py --host <RUNPOD_PUBLIC_IP> \
    --text "Hello there" --voice female --out hello.ogg --format ogg --speed 1.2

  # Fast generation
  python test/client.py --host <RUNPOD_PUBLIC_IP> \
    --text "Hello there" --speed 2.0

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
from dotenv import load_dotenv
from text import split_sentences

import websockets

SAMPLE_RATE = 24000  # Kokoro outputs 24kHz PCM16 mono

# Load .env so defaults can come from RUNPOD_TCP_HOST/RUNPOD_TCP_PORT
load_dotenv(override=True)

# Default long text (matches test/bench.py)
DEFAULT_TEXT = (
    "The origin of humankind is a fascinating topic with deep roots in our "
    "evolutionary history. Our species, Homo sapiens, emerged in Africa "
    "around 300,000 years ago. Fossil evidence suggests we evolved from "
    "earlier hominid species like Homo erectus and Australopithecus afarensis "
    "through a process of natural selection and adaptation to changing "
    "environments. Genetic studies have traced our ancestry back to a small "
    "population that migrated out of Africa around 70,000 years ago, "
    "eventually populating the entire globe. Over time, different human "
    "populations diverged and developed unique traits due to geographic "
    "isolation and varying environmental pressures. The story of human origins "
    "is still being pieced together through ongoing archaeological discoveries "
    "and advancements in genetic research. However, the prevailing scientific "
    "consensus points to a single African origin for all modern humans, with "
    "our species' remarkable journey spanning hundreds of thousands of years "
    "and continents."
)

SHORT_TEXT = (
    "I'm not sure what's funnier, the fact that you're asking a 22-year-old woman to tell you something "
    "funny or that you think I can actually make you laugh. You're probably one of those guys who thinks "
    "laughing out loud is a valid form of humor, right? Or maybe you're into weirder stuff like fart jokes? "
    "Anyway, each to their own I guess."
)

def _is_primer_chunk(b: bytes) -> bool:
    try:
        prime_bytes = int(os.getenv("PRIME_BYTES", "512"))
    except Exception:
        prime_bytes = 512
    return len(b) <= prime_bytes and not any(b)

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


def _split_sentences(text: str) -> list[str]:
    return split_sentences(text)

async def stream_ws_and_save(host: str, port: int, voice: str, text: str, out_path: str, out_format: str, use_tls: bool = False, speed: float = 1.4, mode: str = "single") -> int:
    """OpenAI Realtime WS: session.update → response.create → response.output_audio.delta (b64) → response.completed.

    Decodes base64 PCM deltas and saves via ffmpeg (wav/ogg/mp3) or raw PCM.
    """
    norm_host, tls_from_scheme = _sanitize_host_and_scheme(host)
    force_tls = use_tls or tls_from_scheme or _is_runpod_proxy_host(norm_host)
    api_key = os.getenv("API_KEY", "")
    qs = f"?api_key={api_key}" if api_key else ""
    if _is_runpod_proxy_host(norm_host):
        # RunPod proxy terminates TLS and forwards to container; no explicit port in public URL
        ws_url = f"wss://{norm_host}/v1/audio/speech/stream/ws{qs}"
    else:
        scheme = "wss" if force_tls else "ws"
        # Only append port if it's not already present
        netloc = norm_host if ":" in norm_host else f"{norm_host}:{port}"
        ws_url = f"{scheme}://{netloc}/v1/audio/speech/stream/ws{qs}"
    request_id = f"req-{uuid.uuid4().hex[:8]}"

    proc: subprocess.Popen | None = None
    file_handle = None
    if out_format == "pcm":
        file_handle = open(out_path, "wb")
    else:
        if not shutil.which("ffmpeg"):
            print("ffmpeg not found; install it or use --format pcm", file=sys.stderr)
            return 2
        proc = subprocess.Popen(build_ffmpeg_cmd(out_path, out_format, speed), stdin=subprocess.PIPE)

    ttfb_values: list[float] = []
    total = 0
    print(f"Connecting to {ws_url} with speed={speed:.1f}x (tempo correction: {1.0/speed:.3f}x)")

    async with websockets.connect(ws_url, max_size=None) as ws:
        # OpenAI session.update
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "voice": voice,
                "audio": {"format": "pcm", "sample_rate": SAMPLE_RATE},
            },
        }))
        # Wait for session.updated
        while True:
            m = await ws.recv()
            try:
                d = json.loads(m)
            except Exception:
                continue
            if isinstance(d, dict) and d.get("type") == "session.updated":
                break

        if mode == "single":
            t0 = time.time()
            first = True
            # Create response
            await ws.send(json.dumps({
                "type": "response.create",
                "response_id": request_id,
                "input": text,
                "voice": voice,
                "speed": speed,
            }))
            while True:
                msg = await ws.recv()
                try:
                    data = json.loads(msg)
                except Exception:
                    continue
                if not isinstance(data, dict):
                    continue
                msg_type = data.get("type")
                if msg_type == "response.output_audio.delta" and data.get("response") == request_id:
                    b64 = data.get("delta") or ""
                    try:
                        import base64 as _b64
                        chunk = _b64.b64decode(b64) if isinstance(b64, str) else b""
                    except Exception:
                        chunk = b""
                    if first and _is_primer_chunk(chunk):
                        continue
                    if first:
                        ttfb_values.append(1000*(time.time()-t0))
                        print(f"TTFB: {ttfb_values[-1]:.0f}ms")
                        first = False
                    total += len(chunk)
                    if file_handle is not None:
                        file_handle.write(chunk)
                    elif proc and proc.stdin:
                        proc.stdin.write(chunk)
                    continue
                if msg_type in ("response.completed", "response.canceled") and data.get("response") == request_id:
                    break
        else:
            # sentence-by-sentence
            sentences = _split_sentences(text)
            for sent in sentences:
                rid = f"req-{uuid.uuid4().hex[:8]}"
                t0 = time.time()
                first = True
                await ws.send(json.dumps({
                    "type": "response.create",
                    "response_id": rid,
                    "input": sent,
                    "voice": voice,
                    "speed": speed,
                }))
                while True:
                    msg = await ws.recv()
                    try:
                        data = json.loads(msg)
                    except Exception:
                        continue
                    if not isinstance(data, dict):
                        continue
                    msg_type = data.get("type")
                    if msg_type == "response.output_audio.delta" and data.get("response") == rid:
                        b64 = data.get("delta") or ""
                        try:
                            import base64 as _b64
                            chunk = _b64.b64decode(b64) if isinstance(b64, str) else b""
                        except Exception:
                            chunk = b""
                        if first and _is_primer_chunk(chunk):
                            continue
                        if first:
                            ttfb_values.append(1000*(time.time()-t0))
                            first = False
                        total += len(chunk)
                        if file_handle is not None:
                            file_handle.write(chunk)
                        elif proc and proc.stdin:
                            proc.stdin.write(chunk)
                        continue
                    if msg_type in ("response.completed", "response.canceled") and data.get("response") == rid:
                        break

    if proc and proc.stdin:
        try:
            proc.stdin.close()
            proc.wait(timeout=10)
        except Exception:
            pass
    if file_handle is not None:
        file_handle.close()

    if mode == "sentences" and ttfb_values:
        avg_ttfb = sum(ttfb_values)/len(ttfb_values)
        print(f"Avg TTFB: {avg_ttfb:.0f}ms")
    print(f"Saved ~{total} bytes of PCM to {out_path}")
    return 0 if total > 0 else 3

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WebSocket Kokoro TTS client → save audio via ffmpeg")
    parser.add_argument(
        "--host",
        default=os.getenv("RUNPOD_TCP_HOST", "localhost"),
        help="API host (defaults to RUNPOD_TCP_HOST or localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("RUNPOD_TCP_PORT", "8000")),
        help="API port (defaults to RUNPOD_TCP_PORT or 8000)",
    )
    parser.add_argument("--voice", choices=["female", "male"], default="female", help="Voice to use")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Input text to synthesize")
    parser.add_argument("--short-reply", action="store_true", help="Use a much shorter sample text")
    parser.add_argument("--out", default="hello.wav", help="Output file path (wav/ogg/mp3/pcm)")
    parser.add_argument("--format", choices=["wav", "ogg", "opus", "mp3", "pcm"], default="wav", help="Output format")
    parser.add_argument("--mode", choices=["single", "sentences"], default="single", help="Send full text at once or sentence-by-sentence")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier (0.5-2.0, default: 1.0)")
    parser.add_argument("--tls", action="store_true", help="Use wss:// (TLS)")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    try:
        # Determine TLS pref: explicit flag, scheme in host, or RunPod proxy host
        norm_host, tls_from_scheme = _sanitize_host_and_scheme(args.host)
        tls_pref = args.tls or tls_from_scheme or _is_runpod_proxy_host(norm_host)
        text_to_use = SHORT_TEXT if args.short_reply else args.text
        rc = asyncio.run(stream_ws_and_save(norm_host, args.port, args.voice, text_to_use, args.out, args.format, tls_pref, args.speed, args.mode))
        if rc != 0:
            sys.exit(rc)
    except KeyboardInterrupt:
        print("Interrupted")
        sys.exit(130)

if __name__ == "__main__":
    main()