#!/usr/bin/env python3
"""
Warmup script for Kokoro TTS API over WebSocket only.
Sends a couple of requests to warm up the model for optimal performance.
"""
import asyncio
import os
import websockets
import json
import uuid
import time
import os
import logging
import argparse
from dotenv import load_dotenv
from utils import split_sentences

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000  # Hz, Kokoro 24kHz mono, 16-bit PCM

# Load .env for RUNPOD_TCP_HOST/RUNPOD_TCP_PORT defaults
load_dotenv(override=True)

def _compute_metrics(total_bytes: int, t0: float, t_first: float, t_end: float):
    ttfb_ms = (t_first - t0) * 1000.0
    wall_s = t_end - t0
    audio_s = total_bytes / (SAMPLE_RATE * 2)  # 2 bytes per 16-bit sample, mono
    rtf = wall_s / audio_s if audio_s > 0 else float('inf')
    x_rt = audio_s / wall_s if wall_s > 0 else 0.0
    kbps = (total_bytes / 1024.0) / wall_s if wall_s > 0 else 0.0
    return {
        "ttfb_ms": ttfb_ms,
        "wall_s": wall_s,
        "audio_s": audio_s,
        "rtf": rtf,
        "x_realtime": x_rt,
        "kb_per_s": kbps,
    }

def _format_metrics(tag: str, m: dict):
    return (
        f"[{tag}] TTFB={m['ttfb_ms']:.2f} ms | time={m['wall_s']:.4f}s | "
        f"audio={m['audio_s']:.4f}s | RTF={m['rtf']:.4f} | xRT={m['x_realtime']:.4f} | "
        f"throughput={m['kb_per_s']:.3f} KB/s"
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

def _split_sentences(text: str) -> list[str]:
    return split_sentences(text)

async def _ws_measure_async(base_url: str, text: str, voice: str, save_audio: bool, speed: float = 1.0, mode: str = "single"):
    api_key = os.getenv("API_KEY", "")
    qs = f"?api_key={api_key}" if api_key else ""
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://") + "/v1/audio/speech/stream/ws" + qs
    request_id = f"req-{uuid.uuid4().hex[:8]}"
    t_first = None
    total = 0
    t0 = None
    audio_buf = bytearray() if save_audio else None

    async with websockets.connect(ws_url, max_size=None, compression=None, ping_interval=20, ping_timeout=20) as ws:
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "voice": voice,
                "audio": {"format": "pcm", "sample_rate": SAMPLE_RATE},
            },
        }))
        # Await session.updated
        while True:
            m = await ws.recv()
            try:
                d = json.loads(m)
            except Exception:
                continue
            if isinstance(d, dict) and d.get("type") == "session.updated":
                break

        if mode == "single":
            await ws.send(json.dumps({
                "type": "response.create",
                "response_id": request_id,
                "input": text,
                "voice": voice,
                "speed": speed,
                "test_mode": "single",
            }))
            t0 = time.time()
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=10)
                except asyncio.TimeoutError:
                    raise RuntimeError("WS timeout waiting for first message/chunk")
                try:
                    data = json.loads(msg)
                except Exception:
                    continue
                if not isinstance(data, dict):
                    continue
                if data.get("type") == "response.output_audio.delta" and data.get("response") == request_id:
                    b64 = data.get("delta") or ""
                    try:
                        import base64 as _b64
                        chunk = _b64.b64decode(b64) if isinstance(b64, str) else b""
                    except Exception:
                        chunk = b""
                    if t_first is None and _is_primer_chunk(chunk):
                        continue
                    if t_first is None:
                        t_first = time.time()
                    total += len(chunk)
                    if audio_buf is not None:
                        audio_buf.extend(chunk)
                    continue
                if data.get("type") in ("response.completed", "response.canceled") and data.get("response") == request_id:
                    break
        else:
            # sentences mode: average TTFB across sentences; aggregate bytes
            sentences = _split_sentences(text)
            ttfb_vals = []
            for sent in sentences:
                rid = f"req-{uuid.uuid4().hex[:8]}"
                t0 = time.time()
                sent_first = None
                await ws.send(json.dumps({
                    "type": "response.create",
                    "response_id": rid,
                    "input": sent,
                    "voice": voice,
                    "speed": speed,
                    "test_mode": "sentences",
                }))
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=10)
                    except asyncio.TimeoutError:
                        raise RuntimeError("WS timeout waiting for a sentence chunk")
                    try:
                        data = json.loads(msg)
                    except Exception:
                        continue
                    if not isinstance(data, dict):
                        continue
                    if data.get("type") == "response.output_audio.delta" and data.get("response") == rid:
                        b64 = data.get("delta") or ""
                        try:
                            import base64 as _b64
                            chunk = _b64.b64decode(b64) if isinstance(b64, str) else b""
                        except Exception:
                            chunk = b""
                        if sent_first is None and _is_primer_chunk(chunk):
                            continue
                        if sent_first is None:
                            sent_first = time.time()
                            ttfb_vals.append((sent_first - t0) * 1000.0)
                        total += len(chunk)
                        if audio_buf is not None:
                            audio_buf.extend(chunk)
                        continue
                    if data.get("type") in ("response.completed", "response.canceled") and data.get("response") == rid:
                        break
        t_end = time.time()
    metrics = _compute_metrics(total, t0, t_first or t_end, t_end)
    if save_audio and total > 0 and audio_buf is not None:
        os.makedirs("warmup_audio", exist_ok=True)
        out = f"warmup_audio/warmup_ws_{voice}.pcm"
        with open(out, "wb") as f:
            f.write(audio_buf)
    return metrics

async def _ws_ready_check(base_url: str, voice: str) -> bool:
    """Perform a WS handshake (session.update→session.updated) to validate readiness."""
    api_key = os.getenv("API_KEY", "")
    qs = f"?api_key={api_key}" if api_key else ""
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://") + "/v1/audio/speech/stream/ws" + qs
    try:
        async with websockets.connect(ws_url, max_size=None, compression=None, ping_interval=20, ping_timeout=20) as ws:
            await ws.send(json.dumps({"type": "session.update", "session": {"voice": voice, "audio": {"format": "pcm", "sample_rate": SAMPLE_RATE}}}))
            # wait for session.updated
            while True:
                m = await ws.recv()
                try:
                    d = json.loads(m)
                except Exception:
                    continue
                if isinstance(d, dict) and d.get("type") == "session.updated":
                    break
        return True
    except Exception as e:
        logger.error(f"WS readiness check failed: {e}")
        return False

def warmup_api(host="localhost", port=8000, save_audio=False, short_reply: bool = False, speed: float = 1.0, mode: str = "single"):
    """Send warmup requests to the API (WS-only)."""
    base_url = f"http://{host}:{port}"
    
    # WS readiness
    logger.info("Checking API readiness via WebSocket...")
    ok = asyncio.run(_ws_ready_check(base_url, "female"))
    if not ok:
        logger.info("Make sure the API server is running (./start.sh)")
        return False

    # Warmup requests for each voice: WebSocket only
    test_text = (
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
    if short_reply:
        test_text = SHORT_TEXT

    for voice in ["female", "male"]:
        logger.info(f"[WS] {voice}: starting…")
        try:
            ws_m = asyncio.run(_ws_measure_async(base_url, test_text, voice, save_audio, speed, mode))
            logger.info(_format_metrics(f"WS   {voice}", ws_m))
        except Exception as e:
            logger.error(f"[WS] Error for voice '{voice}': {e}")
    
    logger.info("Model warmup completed!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Warm up the Kokoro TTS API")
    parser.add_argument("--host", default=os.getenv("RUNPOD_TCP_HOST", "localhost"), help="API host (default: RUNPOD_TCP_HOST or localhost)")
    parser.add_argument("--port", type=int, default=int(os.getenv("RUNPOD_TCP_PORT", "8000")), help="API port (default: RUNPOD_TCP_PORT or 8000)")
    parser.add_argument("--save", action="store_true", help="Save generated audio files")
    parser.add_argument("--short-reply", action="store_true", help="Use a much shorter sample text")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier (0.5-2.0, default: 1.0)")
    parser.add_argument("--mode", choices=["single","sentences"], default="single", help="Send full text in one request or sentence-by-sentence")
    
    args = parser.parse_args()
    warmup_api(args.host, args.port, args.save, args.short_reply, args.speed, args.mode)
