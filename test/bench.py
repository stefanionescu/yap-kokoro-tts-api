#!/usr/bin/env python3
"""
Simple load/latency benchmark for Kokoro TTS WebSocket API.

Measures TTFB, wall time, audio seconds, RTF, xRT and throughput for a given
number of requests, split evenly across voices (female/male), using a SINGLE
reused WebSocket connection for all requests.

Examples (run on pod):
  python test/bench.py --n 40 --concurrency 12
  python test/bench.py --n 100 --concurrency 8 --host your-host

Host/port default to localhost:8000; override with --host/--port.
Concurrency parameter now controls request rate (lower = slower rate).
"""
import argparse
import asyncio
import json
import os
import time
import uuid
import statistics as stats
from typing import Dict, List, Tuple

import numpy as np

import websockets


SAMPLE_RATE = 24000

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


def _metrics(total_bytes: int, t0: float, t_first: float, t_end: float) -> Dict[str, float]:
    ttfb_ms = (t_first - t0) * 1000.0
    wall_s = t_end - t0
    audio_s = total_bytes / (SAMPLE_RATE * 2.0)
    rtf = wall_s / audio_s if audio_s > 0 else float("inf")
    xrt = audio_s / wall_s if wall_s > 0 else 0.0
    kbps = (total_bytes / 1024.0) / wall_s if wall_s > 0 else 0.0
    return {
        "ttfb_ms": ttfb_ms,
        "wall_s": wall_s,
        "audio_s": audio_s,
        "rtf": rtf,
        "xrt": xrt,
        "kbps": kbps,
        "bytes": float(total_bytes),
    }

async def _ws_one_on_connection(ws, text: str, voice: str) -> Dict[str, float]:
    """Send one request through an existing WebSocket connection."""
    seg_id = f"seg-{uuid.uuid4().hex[:8]}"
    t0 = None
    total = 0
    t_first = None
    
    await ws.send(json.dumps({
        "continue": True,
        "segment_id": seg_id,
        "input": text,
        "voice": voice,
        "format": "pcm",
    }))
    t0 = time.time()
    
    while True:
        msg = await ws.recv()
        if isinstance(msg, (bytes, bytearray)):
            if t_first is None:
                t_first = time.time()
            total += len(msg)
        else:
            try:
                data = json.loads(msg)
            except Exception:
                continue
            if isinstance(data, dict) and data.get("type") == "end":
                break
    
    t_end = time.time()
    return _metrics(total, t0, t_first or t_end, t_end)


def summarize(title: str, results: List[Dict[str, float]]) -> None:
    if not results:
        print(f"{title}: no results")
        return
    ttfb = [r["ttfb_ms"] for r in results]
    wall = [r["wall_s"] for r in results]
    audio = [r["audio_s"] for r in results]
    xrt = [r["xrt"] for r in results]
    kbps = [r["kbps"] for r in results]
    n = len(results)
    def p(v: List[float], q: float) -> float:
        k = max(0, min(len(v)-1, int(round(q*(len(v)-1)))))
        return sorted(v)[k]
    print(f"\n== {title} ==")
    print(f"n={n}")
    print(f"TTFB ms  | avg={stats.mean(ttfb):.0f}  p50={stats.median(ttfb):.0f}  p95={p(ttfb,0.95):.0f}")
    print(f"Wall s   | avg={stats.mean(wall):.2f}")
    print(f"Audio s  | avg={stats.mean(audio):.2f}")
    print(f"xRT      | avg={stats.mean(xrt):.2f}")
    print(f"Throughput KB/s | avg={stats.mean(kbps):.0f}")


async def bench_ws(base_url: str, text: str, total_reqs: int, concurrency: int) -> List[Dict[str, float]]:
    """Run benchmark using a single reused WebSocket connection."""
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://") + "/v1/audio/speech/stream/ws"
    results: List[Dict[str, float]] = []
    errors: List[str] = []

    try:
        async with websockets.connect(ws_url, max_size=None, compression=None) as ws:
            print(f"Connected to {ws_url} - reusing connection for all {total_reqs} requests")
            
            for i in range(total_reqs):
                voice = "female" if (i % 2 == 0) else "male"
                try:
                    r = await _ws_one_on_connection(ws, text, voice)
                    results.append(r)
                    if (i + 1) % 10 == 0:
                        print(f"  Completed {i + 1}/{total_reqs} requests...")
                except Exception as e:
                    errors.append(f"Request {i}: {str(e)}")
                
                # Small delay between requests if concurrency < total_reqs
                # This simulates controlled request rate
                if concurrency < total_reqs and i < total_reqs - 1:
                    delay = 0.1  # 100ms between requests
                    await asyncio.sleep(delay)
    
    except Exception as e:
        errors.append(f"Connection error: {str(e)}")
    
    if errors:
        print(f"WS errors: {len(errors)} (showing first 3): {errors[:3]}")
    return results

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--proto", choices=["ws"], default="ws")
    ap.add_argument("--n", type=int, default=40, help="total requests")
    ap.add_argument("--concurrency", type=int, default=12, help="request rate control (lower = slower rate)")
    ap.add_argument("--text", default=DEFAULT_TEXT)
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    print(f"Benchmark → WebSocket (single reused connection) | n={args.n} | rate_control={args.concurrency} | host={args.host}:{args.port}")

    t0 = time.time()
    ws_res = asyncio.run(bench_ws(base_url, args.text, args.n, args.concurrency))
    summarize("WebSocket", ws_res)
    print(f"WebSocket elapsed: {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()


