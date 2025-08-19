import json
import time
from typing import Dict, List
from src.constants import METRICS_LOG_PATH

def log_request_metrics(metrics: Dict) -> None:
    """Append per-request metrics as a JSON line.

    Expected keys: request_id, ttfb_ms, wall_s, audio_s, rtf, xrt, kbps, canceled(bool)
    Optional keys: n_sentences(int), n_ws_chunks(int), total_samples(int)
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
            # New metadata for richer aggregation (backfilled safely on read)
            "n_sentences": int(metrics.get("n_sentences", metrics.get("sentences", 1)) or 1),
            "n_ws_chunks": int(metrics.get("n_ws_chunks", 0) or 0),
            "total_samples": int(metrics.get("total_samples", 0) or 0),
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
    now = time.time()
    win = _window_seconds(window_spec)
    cutoff = now - win
    sel = [r for r in records if float(r.get("ts", 0)) >= cutoff]

    def _normalize_record(r: Dict) -> Dict:
        # Ensure new fields present for older logs
        r2 = dict(r)
        try:
            r2["n_sentences"] = int(r.get("n_sentences") or 1)
        except Exception:
            r2["n_sentences"] = 1
        try:
            r2["n_ws_chunks"] = int(r.get("n_ws_chunks") or 0)
        except Exception:
            r2["n_ws_chunks"] = 0
        return r2

    sel = [_normalize_record(r) for r in sel]

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

    # Per-request buckets
    all_stats = _stats(sel)
    single_stats = _stats([r for r in sel if int(r.get("n_sentences", 1)) == 1])
    block_stats = _stats([r for r in sel if int(r.get("n_sentences", 1)) > 1])

    # Normalized views (per sentence and per WebSocket chunk)
    def _normalized(sub: List[Dict], unit_key: str) -> Dict[str, float]:
        if not sub:
            return {
                "units": 0.0,
                "avg_ttfb_ms": 0.0,
                "avg_wall_s": 0.0,
                "avg_audio_s": 0.0,
                "avg_rtf": 0.0,
                "avg_xrt": 0.0,
                "avg_kbps": 0.0,
            }
        try:
            total_units = sum(max(0, int(r.get(unit_key, 0) or 0)) for r in sub)
        except Exception:
            total_units = 0
        if total_units <= 0:
            return {
                "units": 0.0,
                "avg_ttfb_ms": 0.0,
                "avg_wall_s": 0.0,
                "avg_audio_s": 0.0,
                "avg_rtf": 0.0,
                "avg_xrt": 0.0,
                "avg_kbps": 0.0,
            }
        sum_ttfb = sum(float(r.get("ttfb_ms", 0.0)) for r in sub)
        sum_wall = sum(float(r.get("wall_s", 0.0)) for r in sub)
        sum_audio = sum(float(r.get("audio_s", 0.0)) for r in sub)
        # Derived normalized values
        avg_wall_per_unit = sum_wall / total_units
        avg_audio_per_unit = sum_audio / total_units
        # Ratios are invariant to unit scaling when computed per-record, but for a population
        # we compute from sums for stability
        avg_rtf = (sum_wall / sum_audio) if sum_audio > 0 else 0.0
        avg_xrt = (sum_audio / sum_wall) if sum_wall > 0 else 0.0
        # Throughput as simple mean of per-request kbps (kept consistent with _stats)
        avg_kbps = sum(float(r.get("kbps", 0.0)) for r in sub) / len(sub)
        return {
            "units": float(total_units),
            "avg_ttfb_ms": (sum_ttfb / total_units),
            "avg_wall_s": avg_wall_per_unit,
            "avg_audio_s": avg_audio_per_unit,
            "avg_rtf": avg_rtf,
            "avg_xrt": avg_xrt,
            "avg_kbps": avg_kbps,
        }

    per_sentence_norm = _normalized(sel, "n_sentences")
    per_chunk_norm = _normalized([r for r in sel if int(r.get("n_ws_chunks", 0)) > 0], "n_ws_chunks")

    return {
        "all": all_stats,
        "single": single_stats,
        "block": block_stats,
        "per_sentence": per_sentence_norm,
        "per_chunk": per_chunk_norm,
    }

def print_report(periods: List[str]) -> None:
    recs = _load_records()
    print(f"Records: {len(recs)} in {METRICS_LOG_PATH}")
    for p in periods:
        s = aggregate_window(recs, p)
        all_s = s.get("all", {})
        sg = s.get("single", {})
        bl = s.get("block", {})
        ps = s.get("per_sentence", {})
        pc = s.get("per_chunk", {})
        print(f"\n== Last {p} ==")
        print(f"count(all/single/block)={int(all_s.get('count',0))}/{int(sg.get('count',0))}/{int(bl.get('count',0))}")
        print(
            f"avg TTFB ms: {all_s.get('avg_ttfb_ms',0.0):.0f} | single: {sg.get('avg_ttfb_ms',0.0):.0f} | block: {bl.get('avg_ttfb_ms',0.0):.0f} | per-sent: {ps.get('avg_ttfb_ms',0.0):.0f} | per-chunk: {pc.get('avg_ttfb_ms',0.0):.0f}"
        )
        print(
            f"avg wall s: {all_s.get('avg_wall_s',0.0):.2f} | single: {sg.get('avg_wall_s',0.0):.2f} | block: {bl.get('avg_wall_s',0.0):.2f} | per-sent: {ps.get('avg_wall_s',0.0):.2f} | per-chunk: {pc.get('avg_wall_s',0.0):.3f}"
        )
        print(
            f"avg audio s: {all_s.get('avg_audio_s',0.0):.2f} | single: {sg.get('avg_audio_s',0.0):.2f} | block: {bl.get('avg_audio_s',0.0):.2f} | per-sent: {ps.get('avg_audio_s',0.0):.2f} | per-chunk: {pc.get('avg_audio_s',0.0):.3f}"
        )
        print(
            f"avg RTF: {all_s.get('avg_rtf',0.0):.2f} | single: {sg.get('avg_rtf',0.0):.2f} | block: {bl.get('avg_rtf',0.0):.2f} | per-sent: {ps.get('avg_rtf',0.0):.2f} | per-chunk: {pc.get('avg_rtf',0.0):.2f}"
        )
        print(
            f"avg xRT: {all_s.get('avg_xrt',0.0):.2f} | single: {sg.get('avg_xrt',0.0):.2f} | block: {bl.get('avg_xrt',0.0):.2f} | per-sent: {ps.get('avg_xrt',0.0):.2f} | per-chunk: {pc.get('avg_xrt',0.0):.2f}"
        )
        print(
            f"avg throughput KB/s: {all_s.get('avg_kbps',0.0):.0f} | single: {sg.get('avg_kbps',0.0):.0f} | block: {bl.get('avg_kbps',0.0):.0f} | per-sent: {ps.get('avg_kbps',0.0):.0f} | per-chunk: {pc.get('avg_kbps',0.0):.0f}"
        )
        print(f"canceled rate: {all_s.get('canceled_rate',0.0)*100:.1f}% | single: {sg.get('canceled_rate',0.0)*100:.1f}% | block: {bl.get('canceled_rate',0.0)*100:.1f}%")

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


