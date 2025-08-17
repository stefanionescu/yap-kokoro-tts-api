#!/usr/bin/env python3
"""
Warmup script for Kokoro TTS API.
Sends a couple of requests to warm up the model for optimal performance.
"""
import requests
import asyncio
import websockets
import json
import uuid
import time
import os
import logging
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000  # Hz, Kokoro 24kHz mono, 16-bit PCM

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
        f"[{tag}] TTFB={m['ttfb_ms']:.0f} ms | time={m['wall_s']:.2f}s | "
        f"audio={m['audio_s']:.2f}s | RTF={m['rtf']:.2f} | xRT={m['x_realtime']:.2f} | "
        f"throughput={m['kb_per_s']:.1f} KB/s"
    )

def _http_measure(base_url: str, text: str, voice: str, save_audio: bool):
    url = f"{base_url}/v1/audio/speech/stream"
    payload = {
        "input": text,
        "voice": voice,
        "format": "pcm",
    }
    headers = {"Accept-Encoding": "identity", "Cache-Control": "no-store", "Content-Type": "application/json"}
    t0 = time.time()
    total = 0
    t_first = None
    audio_buf = bytearray() if save_audio else None
    # Use context manager so the connection is always closed
    with requests.post(url, json=payload, stream=True, headers=headers, timeout=(5, 60)) as r:
        if r.status_code != 200:
            # Try to extract error body for diagnostics
            body = None
            try:
                body = r.text[:500]
            except Exception:
                body = None
            raise RuntimeError(f"HTTP {r.status_code} for {voice} | body={body}")

        try:
            for chunk in r.iter_content(chunk_size=1):
                if not chunk:
                    continue
                if t_first is None:
                    t_first = time.time()
                total += len(chunk)
                if audio_buf is not None:
                    audio_buf.extend(chunk)
        except requests.exceptions.ChunkedEncodingError as e:
            # Treat truncated chunked stream as end-of-stream; compute metrics on bytes received
            pass

    t_end = time.time()
    metrics = _compute_metrics(total, t0, t_first or t_end, t_end)
    if save_audio and total > 0 and audio_buf is not None:
        os.makedirs("warmup_audio", exist_ok=True)
        out = f"warmup_audio/warmup_http_{voice}.pcm"
        with open(out, "wb") as f:
            f.write(audio_buf)
    return metrics

async def _ws_measure_async(base_url: str, text: str, voice: str, save_audio: bool):
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://") + "/v1/audio/speech/stream/ws"
    seg_id = f"seg-{uuid.uuid4().hex[:8]}"
    t_first = None
    total = 0
    t0 = time.time()
    audio_buf = bytearray() if save_audio else None
    
    async with websockets.connect(ws_url, max_size=None, compression=None, ping_interval=20, ping_timeout=20) as ws:
        await ws.send(json.dumps({
            "continue": True,
            "segment_id": seg_id,
            "input": text,
            "voice": voice,
            "format": "pcm"
        }))

        # Receive until we get an end message
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=10)
            except asyncio.TimeoutError:
                raise RuntimeError("WS timeout waiting for first message/chunk")
                
            if isinstance(msg, (bytes, bytearray)):
                if t_first is None:
                    t_first = time.time()
                total += len(msg)
                if audio_buf is not None:
                    audio_buf.extend(msg)
            else:
                try:
                    data = json.loads(msg)
                    if data.get("type") == "start":
                        logger.info("[WS] got start")
                    elif data.get("type") == "meta":
                        logger.info(f"[WS] meta: seg={data.get('segment')} samples={data.get('total_samples')}")
                    elif data.get("type") == "end":
                        break
                except Exception:
                    continue
        t_end = time.time()
    metrics = _compute_metrics(total, t0, t_first or t_end, t_end)
    if save_audio and total > 0 and audio_buf is not None:
        os.makedirs("warmup_audio", exist_ok=True)
        out = f"warmup_audio/warmup_ws_{voice}.pcm"
        with open(out, "wb") as f:
            f.write(audio_buf)
    return metrics

def warmup_api(host="localhost", port=8000, save_audio=False):
    """Send warmup requests to the API"""
    base_url = f"http://{host}:{port}"
    
    # First check if API is up
    try:
        logger.info(f"Checking if API is available at {base_url}...")
        response = requests.get(f"{base_url}/api/voices", timeout=5)
        if response.status_code != 200:
            logger.error(f"API returned status code {response.status_code}")
            return False
        voices = response.json()
        logger.info(f"API is up. Available voices: {[v['name'] for v in voices['voices']]}")
    except requests.RequestException as e:
        logger.error(f"API is not available: {str(e)}")
        logger.info("Make sure the API server is running (./start.sh)")
        return False

    # Warmup requests for each voice: HTTP then WS
    test_text = "This is a warmup request to optimize the text-to-speech model performance."

    for voice in ["female", "male"]:
        logger.info(f"[HTTP] {voice}: starting…")
        try:
            http_m = _http_measure(base_url, test_text, voice, save_audio)
            logger.info(_format_metrics(f"HTTP {voice}", http_m))
        except Exception as e:
            logger.exception(f"[HTTP] Error for voice '{voice}': {type(e).__name__}: {e}")

        logger.info(f"[WS] {voice}: starting…")
        try:
            ws_m = asyncio.run(_ws_measure_async(base_url, test_text, voice, save_audio))
            logger.info(_format_metrics(f"WS   {voice}", ws_m))
        except Exception as e:
            logger.error(f"[WS] Error for voice '{voice}': {e}")
    
    logger.info("Model warmup completed!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Warm up the Kokoro TTS API")
    parser.add_argument("--host", default="localhost", help="API host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="API port (default: 8000)")
    parser.add_argument("--save", action="store_true", help="Save generated audio files")
    
    args = parser.parse_args()
    warmup_api(args.host, args.port, args.save)
