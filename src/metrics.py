import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import json
import time
from typing import Dict, List
from constants import METRICS_LOG_PATH

def log_request_metrics(metrics: Dict) -> None:
    """Append per-request metrics as a JSON line.

    Expected keys: request_id, ttfb_ms, wall_s, audio_s, rtf, xrt, kbps, canceled(bool), test_mode
    Adds ts epoch seconds on write.
    """
    try:
        rec = {
            "ts": time.time(),
            "request_id": metrics.get("request_id"),
            "ttfb_ms": float(metrics.get("ttfb_ms", 0.0)),
            "wall_s": float(metrics.get("wall_s", 0.0)),
            "audio_s": float(metrics.get("audio_s", 0.0)),
            "rtf": float(metrics.get("rtf", 0.0)),
            "xrt": float(metrics.get("xrt", 0.0)),
            "kbps": float(metrics.get("kbps", 0.0)),
            "canceled": bool(metrics.get("canceled", False)),
            "test_mode": metrics.get("test_mode"),  # Track client test mode
        }
        METRICS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with METRICS_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _load_records() -> List[Dict]:
    if not METRICS_LOG_PATH.exists():
        return []
    out: List[Dict] = []
    try:
        with METRICS_LOG_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return out

def _window_seconds(spec: str) -> int:
    spec = spec.strip().lower()
    if spec.endswith("d"):
        return int(float(spec[:-1]) * 86400)
    if spec.endswith("h"):
        return int(float(spec[:-1]) * 3600)
    if spec.endswith("m"):
        return int(float(spec[:-1]) * 60)
    return int(spec)

def aggregate_window(records: List[Dict], window_spec: str) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics for records within a time window, bucketed by test mode."""
    now = time.time()
    win = _window_seconds(window_spec)
    cutoff = now - win
    sel = [r for r in records if float(r.get("ts", 0)) >= cutoff]
    
    def _stats(sub: List[Dict]) -> Dict[str, float]:
        n = len(sub)
        if n == 0:
            return {
                "count": 0,
                "avg_ttfb_ms": 0.0,
                "avg_wall_s": 0.0,
                "avg_audio_s": 0.0,
                "avg_rtf": 0.0,
                "avg_xrt": 0.0,
                "avg_kbps": 0.0,
                "canceled_rate": 0.0,
            }
        
        def avg(key: str) -> float:
            vals = [float(r.get(key, 0.0)) for r in sub]
            return sum(vals) / len(vals) if vals else 0.0
        
        canceled = sum(1 for r in sub if bool(r.get("canceled", False)))
        
        return {
            "count": float(n),
            "avg_ttfb_ms": avg("ttfb_ms"),
            "avg_wall_s": avg("wall_s"),
            "avg_audio_s": avg("audio_s"),
            "avg_rtf": avg("rtf"),
            "avg_xrt": avg("xrt"),
            "avg_kbps": avg("kbps"),
            "canceled_rate": (canceled / n) if n > 0 else 0.0,
        }
    
    # Bucket by test mode
    overall_stats = _stats(sel)
    single_stats = _stats([r for r in sel if r.get("test_mode") == "single"])
    sentences_stats = _stats([r for r in sel if r.get("test_mode") == "sentences"])
    
    return {
        "overall": overall_stats,
        "single": single_stats,
        "sentences": sentences_stats,
    }

def print_report(periods: List[str]) -> None:
    """Print TTS performance metrics bucketed by test mode for specified time periods."""
    recs = _load_records()
    print(f"TTS Performance Report - {len(recs)} total records in {METRICS_LOG_PATH}")
    
    for p in periods:
        buckets = aggregate_window(recs, p)
        overall = buckets.get("overall", {})
        single = buckets.get("single", {})
        sentences = buckets.get("sentences", {})
        
        total_count = int(overall.get('count', 0))
        single_count = int(single.get('count', 0))
        sentences_count = int(sentences.get('count', 0))
        
        if total_count == 0:
            print(f"\n== Last {p} ==")
            print("No requests in this period")
            continue
            
        print(f"\n== Last {p} ==")
        print(f"Total: {total_count} requests (Single: {single_count}, Sentences: {sentences_count})")
        
        def _print_bucket(name: str, stats: Dict[str, float]):
            count = int(stats.get('count', 0))
            if count == 0:
                print(f"  {name}: No data")
                return
            print(f"  {name} ({count} requests):")
            print(f"    TTFB: {stats.get('avg_ttfb_ms', 0.0):.0f} ms")
            print(f"    RTF: {stats.get('avg_rtf', 0.0):.2f}x")
            print(f"    xRT: {stats.get('avg_xrt', 0.0):.2f}x realtime")
            print(f"    Processing: {stats.get('avg_wall_s', 0.0):.2f} s")
            print(f"    Audio: {stats.get('avg_audio_s', 0.0):.2f} s")
            print(f"    Throughput: {stats.get('avg_kbps', 0.0):.0f} KB/s")
            cancel_rate = stats.get('canceled_rate', 0.0) * 100
            if cancel_rate > 0:
                print(f"    Canceled: {cancel_rate:.1f}%")
        
        _print_bucket("OVERALL", overall)
        _print_bucket("SINGLE MODE", single)
        _print_bucket("SENTENCES MODE", sentences)

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Print rolling TTS request metrics")
    ap.add_argument(
        "--periods",
        default="1h,6h,12h,24h,3d",
        help="Comma-separated windows, e.g. 30m,1h,6h,24h",
    )
    args = ap.parse_args()
    periods = [p.strip() for p in args.periods.split(",") if p.strip()]
    print_report(periods)


