import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

METRICS_LOG_PATH = Path("logs/metrics.log")

def log_request_metrics(metrics: Dict) -> None:
    """Append per-request metrics as a JSON line.

    Expected keys: request_id, ttfb_ms, wall_s, audio_s, rtf, xrt, kbps, canceled(bool)
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

def aggregate_window(records: List[Dict], window_spec: str) -> Tuple[int, Dict[str, float]]:
    now = time.time()
    win = _window_seconds(window_spec)
    cutoff = now - win
    sel = [r for r in records if float(r.get("ts", 0)) >= cutoff]
    n = len(sel)
    if n == 0:
        return 0, {
            "avg_ttfb_ms": 0.0,
            "avg_wall_s": 0.0,
            "avg_audio_s": 0.0,
            "avg_rtf": 0.0,
            "avg_xrt": 0.0,
            "avg_kbps": 0.0,
            "canceled_rate": 0.0,
        }
    def avg(key: str) -> float:
        vals = [float(r.get(key, 0.0)) for r in sel]
        return sum(vals) / len(vals) if vals else 0.0
    canceled = sum(1 for r in sel if bool(r.get("canceled", False)))
    stats = {
        "avg_ttfb_ms": avg("ttfb_ms"),
        "avg_wall_s": avg("wall_s"),
        "avg_audio_s": avg("audio_s"),
        "avg_rtf": avg("rtf"),
        "avg_xrt": avg("xrt"),
        "avg_kbps": avg("kbps"),
        "canceled_rate": (canceled / n) if n > 0 else 0.0,
    }
    return n, stats

def print_report(periods: List[str]) -> None:
    recs = _load_records()
    print(f"Records: {len(recs)} in {METRICS_LOG_PATH}")
    for p in periods:
        n, s = aggregate_window(recs, p)
        print(f"\n== Last {p} ==")
        print(f"count={n}")
        print(f"avg TTFB ms: {s['avg_ttfb_ms']:.0f}")
        print(f"avg wall s: {s['avg_wall_s']:.2f}")
        print(f"avg audio s: {s['avg_audio_s']:.2f}")
        print(f"avg RTF: {s['avg_rtf']:.2f}")
        print(f"avg xRT: {s['avg_xrt']:.2f}")
        print(f"avg throughput KB/s: {s['avg_kbps']:.0f}")
        print(f"canceled rate: {s['canceled_rate']*100:.1f}%")

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


