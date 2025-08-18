#!/usr/bin/env python3
"""
Simple load/latency benchmark for Kokoro TTS WebSocket API.

Measures TTFB, wall time, audio seconds, RTF, xRT and throughput for a given
number of requests, split evenly across voices (female/male), with configurable
concurrency using WebSocket protocol only.

Examples (run on pod):
  python test/bench.py --n 40 --concurrency 12
  python test/bench.py --n 100 --concurrency 8 --host your-host

Host/port default to localhost:8000; override with --host/--port.
"""
import argparse
import asyncio
import json
import time
import uuid
import statistics as stats
from typing import Dict, List, Tuple

import websockets
import os
from dotenv import load_dotenv

SAMPLE_RATE = 24000

# Load .env so defaults can come from RUNPOD_TCP_HOST/RUNPOD_TCP_PORT
load_dotenv(override=True)
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


async def _ws_worker(base_url: str, text: str, voice_cycle: List[str], requests_count: int, worker_id: int) -> List[Dict[str, float]]:
    """Open one WS and send multiple speak() calls sequentially to simulate Pipecat."""
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://") + "/v1/audio/speech/stream/ws"
    results: List[Dict[str, float]] = []
    async with websockets.connect(ws_url, max_size=None, compression=None) as ws:
        await ws.send(json.dumps({
            "type": "start",
            "voice": voice_cycle[0],
            "format": "pcm",
            "sample_rate": SAMPLE_RATE,
        }))
        # Wait for started
        while True:
            m = await ws.recv()
            if not isinstance(m, (bytes, bytearray)):
                try:
                    d = json.loads(m)
                except Exception:
                    continue
                if isinstance(d, dict) and d.get("type") == "started":
                    break

        for i in range(requests_count):
            voice = voice_cycle[i % len(voice_cycle)]
            request_id = f"req-{uuid.uuid4().hex[:8]}"
            t_send = time.time()
            t0 = None
            t_first = None
            total = 0
            # emit speak
            await ws.send(json.dumps({
                "type": "speak",
                "request_id": request_id,
                "text": text,
                "voice": voice,
            }))
            t0 = time.time()
            while True:
                msg = await ws.recv()
                if isinstance(msg, (bytes, bytearray)):
                    if t_first is None:
                        t_first = time.time()
                        print(f"    Worker {worker_id}: First chunk after {(t_first - t0) * 1000:.0f}ms")
                    total += len(msg)
                    continue
                try:
                    data = json.loads(msg)
                except Exception:
                    continue
                if isinstance(data, dict) and data.get("type") in ("done", "canceled") and data.get("request_id") == request_id:
                    break
            t_end = time.time()
            results.append(_metrics(total, t0 or t_end, t_first or t_end, t_end))
        # send stop
        await ws.send(json.dumps({"type":"stop"}))
    return results

async def bench_ws(base_url: str, text: str, total_reqs: int, concurrency: int) -> List[Dict[str, float]]:
    """Run benchmark using persistent WS per worker; multiple speak() per connection."""
    per_worker = max(1, (total_reqs + concurrency - 1) // concurrency)
    workers = min(concurrency, total_reqs)
    voice_cycle = ["female", "male"]
    tasks = [asyncio.create_task(_ws_worker(base_url, text, voice_cycle, per_worker, i+1)) for i in range(workers)]
    results_nested = await asyncio.gather(*tasks, return_exceptions=True)
    results: List[Dict[str, float]] = []
    for r in results_nested:
        if isinstance(r, list):
            results.extend(r)
        else:
            print(f"Worker error: {r}")
    return results[:total_reqs]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.getenv("RUNPOD_TCP_HOST", "localhost"))
    ap.add_argument("--port", type=int, default=int(os.getenv("RUNPOD_TCP_PORT", "8000")))
    ap.add_argument("--proto", choices=["ws"], default="ws")
    ap.add_argument("--n", type=int, default=40, help="total requests")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--text", default=DEFAULT_TEXT)
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    print(f"Benchmark → WebSocket | n={args.n} | concurrency={args.concurrency} | host={args.host}:{args.port}")
    print(f"Text length: {len(args.text)} characters")
    print(f"Text preview: {args.text[:100]}...")
    # ~0.6s per 10 words is a rougher average than needed; let RTF report instead

    t0 = time.time()
    ws_res = asyncio.run(bench_ws(base_url, args.text, args.n, args.concurrency))
    summarize("WebSocket", ws_res)
    print(f"WebSocket elapsed: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()


