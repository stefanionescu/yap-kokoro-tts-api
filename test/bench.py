#!/usr/bin/env python3
"""
Simple load/latency benchmark for Kokoro TTS WebSocket API.

Measures TTFB, wall time, audio seconds, RTF, xRT and throughput for a given
number of requests, split evenly across voices (female/male), with configurable
concurrency using WebSocket protocol only.

Examples (run on pod):
  python test/bench.py --n 40 --concurrency 2
  python test/bench.py --n 100 --concurrency 3 --host your-host

Host/port default to localhost:8000; override with --host/--port.
"""
import argparse
import asyncio
import json
import time
import uuid
import statistics as stats
from typing import Dict, List

import websockets
from utils import split_sentences
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

# Optional short reply text
SHORT_TEXT = (
    "I'm not sure what's funnier, the fact that you're asking a 22-year-old woman to tell you something "
    "funny or that you think I can actually make you laugh. You're probably one of those guys who thinks "
    "laughing out loud is a valid form of humor, right? Or maybe you're into weirder stuff like fart jokes? "
    "Anyway, each to their own I guess."
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


def _is_primer_chunk(b: bytes) -> bool:
    try:
        prime_bytes = int(os.getenv("PRIME_BYTES", "512"))
    except Exception:
        prime_bytes = 512
    return len(b) <= prime_bytes and not any(b)


def _split_sentences(text: str) -> list[str]:
    return split_sentences(text)

async def _ws_worker(base_url: str, text: str, voice_cycle: List[str], requests_count: int, worker_id: int, speed: float, mode: str) -> dict:
    """OpenAI Realtime WS: session.update once, then multiple response.create sequentially.

    Returns {"results": List[metrics], "rejected": int}
    """
    api_key = os.getenv("API_KEY", "")
    qs = f"?api_key={api_key}" if api_key else ""
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://") + "/v1/audio/speech/stream/ws" + qs
    results: List[Dict[str, float]] = []
    rejected = 0
    async with websockets.connect(ws_url, max_size=None, compression=None) as ws:
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "voice": voice_cycle[0],
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

        for i in range(requests_count):
            voice = voice_cycle[i % len(voice_cycle)]
            if mode == "single":
                request_id = f"req-{uuid.uuid4().hex[:8]}"
                t0 = None
                t_first = None
                total = 0
                was_rejected = False
                await ws.send(json.dumps({
                    "type": "response.create",
                    "response_id": request_id,
                    "input": text,
                    "voice": voice,
                    "speed": speed,
                }))
                t0 = time.time()
                while True:
                    msg = await ws.recv()
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
                            print(f"    Worker {worker_id}: First chunk after {(t_first - t0) * 1000:.0f}ms")
                        total += len(chunk)
                        continue
                    if data.get("type") == "response.error" and data.get("response") == request_id:
                        code = data.get("code")
                        if code == "busy":
                            print(f"    Worker {worker_id}: Request {request_id} rejected (busy)")
                            rejected += 1
                            was_rejected = True
                            break
                        else:
                            print(f"    Worker {worker_id}: Request {request_id} error: {code}")
                            was_rejected = True
                            break
                    if data.get("type") in ("response.completed", "response.canceled") and data.get("response") == request_id:
                        break
                if was_rejected:
                    continue
                t_end = time.time()
                results.append(_metrics(total, t0 or t_end, t_first or t_end, t_end))
            else:
                # sentences mode: aggregate metrics across sentences
                sentences = _split_sentences(text)
                ttfb_vals = []
                bytes_total = 0
                wall_start = time.time()
                for sent in sentences:
                    rid = f"req-{uuid.uuid4().hex[:8]}"
                    t0 = None
                    t_first = None
                    sent_bytes = 0
                    await ws.send(json.dumps({
                        "type": "response.create",
                        "response_id": rid,
                        "input": sent,
                        "voice": voice,
                        "speed": speed,
                    }))
                    t0 = time.time()
                    while True:
                        msg = await ws.recv()
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
                            if t_first is None and _is_primer_chunk(chunk):
                                continue
                            if t_first is None:
                                t_first = time.time()
                                ttfb_vals.append((t_first - t0) * 1000.0)
                            sent_bytes += len(chunk)
                            continue
                        if data.get("type") in ("response.completed", "response.canceled") and data.get("response") == rid:
                            break
                wall_end = time.time()
                # Average TTFB across sentences; total bytes across all sentences
                avg_ttfb = sum(ttfb_vals)/len(ttfb_vals) if ttfb_vals else 0.0
                # Reuse _metrics with aggregate window and bytes (t0 is first sentence start)
                m = _metrics(int(sum([]) or 0) + 0, wall_start, wall_start + (avg_ttfb/1000.0), wall_end)
                m["ttfb_ms"] = avg_ttfb
                results.append(m)
        # end session
        await ws.send(json.dumps({"type":"session.end"}))
    return {"results": results, "rejected": rejected}

def _split_sentences(text: str) -> list[str]:
    return split_sentences(text)

def _split_counts(total: int, workers: int) -> List[int]:
    base = total // workers
    rem = total % workers
    return [base + (1 if i < rem else 0) for i in range(workers)]


async def bench_ws(base_url: str, text: str, total_reqs: int, concurrency: int, speed: float, mode: str):
    """Run benchmark using persistent WS per worker; multiple speak() per connection.

    Returns (results: List[metrics], rejected_count: int)
    """
    workers = min(concurrency, total_reqs)
    counts = _split_counts(total_reqs, workers)
    voice_cycle = ["female", "male"]
    tasks = [asyncio.create_task(_ws_worker(base_url, text, voice_cycle, counts[i], i+1, speed, mode)) for i in range(workers)]
    results_nested = await asyncio.gather(*tasks, return_exceptions=True)
    results: List[Dict[str, float]] = []
    rejected_total = 0
    for r in results_nested:
        if isinstance(r, dict):
            results.extend(r.get("results", []))
            rejected_total += int(r.get("rejected", 0))
        else:
            print(f"Worker error: {r}")
    return results[:total_reqs], rejected_total

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.getenv("RUNPOD_TCP_HOST", "localhost"))
    ap.add_argument("--port", type=int, default=int(os.getenv("RUNPOD_TCP_PORT", "8000")))
    # protocol is fixed to WebSocket
    ap.add_argument("--n", type=int, default=40, help="total requests")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--short-reply", action="store_true", help="Use a much shorter sample text")
    ap.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier (0.5-2.0, default: 1.0)")
    ap.add_argument("--mode", choices=["single","sentences"], default="single", help="Send full text in one request or sentence-by-sentence")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    text_to_use = SHORT_TEXT if args.short_reply else args.text
    print(f"Benchmark → WebSocket | n={args.n} | concurrency={args.concurrency} | host={args.host}:{args.port}")
    print(f"Text length: {len(text_to_use)} characters")
    print(f"Text preview: {text_to_use[:100]}...")

    t0 = time.time()
    ws_res, rejected = asyncio.run(bench_ws(base_url, text_to_use, args.n, args.concurrency, args.speed, args.mode))
    summarize("WebSocket", ws_res)
    print(f"Rejected: {rejected}")
    print(f"WebSocket elapsed: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()


