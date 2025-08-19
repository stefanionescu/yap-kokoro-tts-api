#!/usr/bin/env python3
"""
Transactions Per Minute (TPM) benchmark for Kokoro TTS WebSocket API.

Measures sustained throughput by sending MAX_CONCURRENT_JOBS concurrent requests
continuously for 60 seconds. When one request completes, immediately send another.
Tracks successful vs failed transactions and average TTFB.

Examples:
  # Full text per transaction (TX counted when single request completes)
  python test/tpm.py --mode single
  # Sentence-by-sentence per TX (TX counted when all sentences complete)
  python test/tpm.py --mode sentences --duration 120 --host your-host
  python test/tpm.py --short-reply --speed 1.5 --mode single
"""
import argparse
import asyncio
import json
import time
import uuid
import statistics as stats
from typing import Dict, List
import os
from dotenv import load_dotenv
import websockets
from utils import split_sentences

# Load .env to get MAX_CONCURRENT_JOBS and other defaults
load_dotenv(override=True)

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

class TPMWorker:
    """Sustained load worker that sends requests continuously until time expires."""
    
    def __init__(self, worker_id: int, ws_url: str, text: str, voice_cycle: List[str], 
                 speed: float, duration_s: float, mode: str):
        self.worker_id = worker_id
        self.ws_url = ws_url
        self.text = text
        self.voice_cycle = voice_cycle
        self.speed = speed
        self.duration_s = duration_s
        self.mode = mode
        
        # Results tracking
        self.total_attempts = 0
        self.successful = 0
        self.failed = 0
        self.rejected = 0
        self.ttfb_times = []
        
    async def run(self) -> Dict:
        """Run continuous requests for duration_s seconds."""
        start_time = time.time()
        end_time = start_time + self.duration_s
        
        async with websockets.connect(self.ws_url, max_size=None, compression=None) as ws:
            # OpenAI session
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "voice": self.voice_cycle[0],
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
            
            # Continuous request loop
            while time.time() < end_time:
                voice = self.voice_cycle[self.total_attempts % len(self.voice_cycle)]
                request_id = f"req-{uuid.uuid4().hex[:8]}"
                self.total_attempts += 1
                
                if self.mode == "single":
                    # Single-shot
                    t_first = None
                    await ws.send(json.dumps({
                        "type": "response.create",
                        "response_id": request_id,
                        "input": self.text,
                        "voice": voice,
                        "speed": self.speed,
                    }))
                    t_start_tx = time.time()
                    while True:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=30)
                        except asyncio.TimeoutError:
                            self.failed += 1
                            break
                        try:
                            data = json.loads(msg)
                        except Exception:
                            continue
                        if not isinstance(data, dict):
                            continue
                        if data.get("type") == "response.output_audio.delta" and data.get("response") == request_id:
                            b64 = data.get("delta") or ""
                            try:
                                chunk = __import__("base64").b64decode(b64) if isinstance(b64, str) else b""
                            except Exception:
                                chunk = b""
                            if t_first is None and not _is_primer_chunk(chunk):
                                t_first = time.time()
                                self.ttfb_times.append((t_first - t_start_tx) * 1000)
                            continue
                        if data.get("type") in ("response.completed", "response.canceled") and data.get("response") == request_id:
                            self.successful += 1
                            break
                else:
                    # Sentence-by-sentence mode
                    ok_all = True
                    sentences = _split_sentences(self.text)
                    per_sentence_ttfb: List[float] = []
                    for sent in sentences:
                        rid = f"req-{uuid.uuid4().hex[:8]}"
                        t0 = time.time()
                        t_first = None
                        await ws.send(json.dumps({
                            "type": "response.create",
                            "response_id": rid,
                            "input": sent,
                            "voice": voice,
                            "speed": self.speed,
                        }))
                        while True:
                            try:
                                msg = await asyncio.wait_for(ws.recv(), timeout=30)
                            except asyncio.TimeoutError:
                                ok_all = False
                                break
                            try:
                                data = json.loads(msg)
                            except Exception:
                                continue
                            if not isinstance(data, dict):
                                continue
                            if data.get("type") == "response.output_audio.delta" and data.get("response") == rid:
                                b64 = data.get("delta") or ""
                                try:
                                    chunk = __import__("base64").b64decode(b64) if isinstance(b64, str) else b""
                                except Exception:
                                    chunk = b""
                                if t_first is None and not _is_primer_chunk(chunk):
                                    t_first = time.time()
                                    per_sentence_ttfb.append((t_first - t0) * 1000)
                                continue
                            if data.get("type") == "response.error" and data.get("response") == rid:
                                ok_all = False
                                break
                            if data.get("type") in ("response.completed", "response.canceled") and data.get("response") == rid:
                                break
                    if ok_all:
                        self.successful += 1
                        if per_sentence_ttfb:
                            self.ttfb_times.append(sum(per_sentence_ttfb)/len(per_sentence_ttfb))
                    else:
                        self.failed += 1
                
            # End session
            await ws.send(json.dumps({"type": "session.end"}))
        
        return {
            "worker_id": self.worker_id,
            "total_attempts": self.total_attempts,
            "successful": self.successful,
            "failed": self.failed,
            "rejected": self.rejected,
            "ttfb_times": self.ttfb_times,
        }

async def run_tpm_test(base_url: str, text: str, concurrency: int, speed: float, duration_s: float, mode: str) -> None:
    """Run TPM test with specified concurrency for duration_s seconds."""
    api_key = os.getenv("API_KEY", "")
    qs = f"?api_key={api_key}" if api_key else ""
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://") + "/v1/audio/speech/stream/ws" + qs
    voice_cycle = ["female", "male"]
    
    print(f"Starting TPM test:")
    print(f"  Duration: {duration_s}s")
    print(f"  Concurrency: {concurrency}")
    print(f"  Text length: {len(text)} chars")
    print(f"  Speed: {speed}x")
    print(f"  URL: {ws_url}")
    print()
    
    # Create workers
    workers = []
    for i in range(concurrency):
        worker = TPMWorker(
            worker_id=i+1,
            ws_url=ws_url,
            text=text,
            voice_cycle=voice_cycle,
            speed=speed,
            duration_s=duration_s,
            mode=mode,
        )
        workers.append(worker)
    
    # Run all workers concurrently
    start_time = time.time()
    tasks = [asyncio.create_task(worker.run()) for worker in workers]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    actual_duration = time.time() - start_time
    
    # Aggregate results
    total_attempts = 0
    total_successful = 0
    total_failed = 0
    total_rejected = 0
    all_ttfb_times = []
    
    for result in results:
        if isinstance(result, dict):
            total_attempts += result["total_attempts"]
            total_successful += result["successful"]
            total_failed += result["failed"]
            total_rejected += result["rejected"]
            all_ttfb_times.extend(result["ttfb_times"])
        else:
            print(f"Worker error: {result}")
    
    # Calculate rates and stats
    tpm = (total_successful / actual_duration) * 60 if actual_duration > 0 else 0
    success_rate = (total_successful / total_attempts) * 100 if total_attempts > 0 else 0
    
    print(f"\n== TPM Test Results ==")
    print(f"Duration: {actual_duration:.1f}s")
    print(f"Total attempts: {total_attempts}")
    print(f"Successful: {total_successful}")
    print(f"Failed: {total_failed}")
    print(f"Rejected (busy): {total_rejected}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Transactions per minute: {tpm:.1f}")
    print(f"Transactions per second: {tpm/60:.1f}")
    
    if all_ttfb_times:
        avg_ttfb = stats.mean(all_ttfb_times)
        p50_ttfb = stats.median(all_ttfb_times)
        p95_ttfb = stats.quantiles(all_ttfb_times, n=20)[18] if len(all_ttfb_times) >= 20 else max(all_ttfb_times)
        print(f"TTFB ms | avg={avg_ttfb:.0f}  p50={p50_ttfb:.0f}  p95={p95_ttfb:.0f}")
    else:
        print("No TTFB data available")

def main() -> None:
    parser = argparse.ArgumentParser(description="Transactions Per Minute (TPM) benchmark for Kokoro TTS")
    parser.add_argument("--host", default=os.getenv("RUNPOD_TCP_HOST", "localhost"), 
                       help="API host (defaults to RUNPOD_TCP_HOST or localhost)")
    parser.add_argument("--port", type=int, default=int(os.getenv("RUNPOD_TCP_PORT", "8000")), 
                       help="API port (defaults to RUNPOD_TCP_PORT or 8000)")
    parser.add_argument("--duration", type=int, default=60, 
                       help="Test duration in seconds (default: 60)")
    parser.add_argument("--concurrency", type=int, 
                       default=int(os.getenv("MAX_CONCURRENT_JOBS", "4")),
                       help="Concurrent requests (defaults to MAX_CONCURRENT_JOBS from .env)")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Input text to synthesize")
    parser.add_argument("--short-reply", action="store_true", help="Use a much shorter sample text")
    parser.add_argument("--speed", type=float, default=1.0, 
                       help="Speech speed multiplier (0.5-2.0, default: 1.0)")
    parser.add_argument("--mode", choices=["single","sentences"], default="single",
                       help="Send full text at once (single) or sentence-by-sentence (sentences)")
    
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    text_to_use = SHORT_TEXT if args.short_reply else args.text
    
    print(f"TPM Benchmark - Kokoro TTS WebSocket API")
    print(f"Host: {args.host}:{args.port}")
    print(f"Text preview: {text_to_use[:100]}...")
    print()
    
    try:
        asyncio.run(run_tpm_test(
            base_url, 
            text_to_use, 
            args.concurrency, 
            args.speed, 
            args.duration,
            args.mode,
        ))
    except KeyboardInterrupt:
        print("\nInterrupted by user")

if __name__ == "__main__":
    main()
