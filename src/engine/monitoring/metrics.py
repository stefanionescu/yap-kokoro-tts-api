"""Metrics collection and system status monitoring."""

import json
import time
import shutil
import contextlib
from pathlib import Path
from typing import Optional

try:
    import torch
except ImportError:
    torch = None

from constants import SAMPLE_RATE


class MetricsCollector:
    """Collects and logs request metrics and system status."""
    
    def __init__(self, metrics_log_path: Optional[Path] = None):
        # Simple rolling metrics store (in-memory + file append)
        self._metrics_log_path = metrics_log_path or Path("logs/metrics.log")
        self._last_metrics: dict[str, float] = {}
    
    def log_request_metrics(self, metrics: dict) -> None:
        """Append request metrics to a log file and update memory cache.

        Expected keys: request_id, ttfb_ms, wall_s, audio_s, rtf, xrt, kbps, canceled(bool)
        """
        ts = time.time()
        try:
            self._last_metrics = {
                "ts": ts,
                "ttfb_ms": float(metrics.get("ttfb_ms", 0.0)),
                "wall_s": float(metrics.get("wall_s", 0.0)),
                "audio_s": float(metrics.get("audio_s", 0.0)),
                "rtf": float(metrics.get("rtf", 0.0)),
                "xrt": float(metrics.get("xrt", 0.0)),
                "kbps": float(metrics.get("kbps", 0.0)),
                "canceled": bool(metrics.get("canceled", False)),
            }
            rec = {
                "ts": ts,
                "request_id": metrics.get("request_id"),
                **self._last_metrics,
            }
            self._metrics_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._metrics_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass


class SystemStatus:
    """Provides system status information for diagnostics."""
    
    def __init__(
        self,
        device: str,
        voice_mapping: dict,
        speed: float,
        split_pattern: str,
        stream_chunk_samples: int
    ):
        self.device = device
        self._voice_mapping = voice_mapping
        self.speed = speed
        self.split_pattern = split_pattern
        self.stream_chunk_samples = stream_chunk_samples
    
    def get_status(self) -> dict:
        """Return runtime status for diagnostics."""
        gpu_name = None
        cuda_available = False
        device_index = None
        free_mem = None
        total_mem = None
        
        if torch:
            try:
                cuda_available = torch.cuda.is_available()
                if cuda_available:
                    device_index = torch.cuda.current_device()
                    gpu_name = torch.cuda.get_device_name(device_index)
                    try:
                        free_mem, total_mem = torch.cuda.mem_get_info(device_index)
                    except Exception:
                        free_mem = total_mem = None
            except Exception:
                pass
        
        ffmpeg = shutil.which("ffmpeg") is not None
        return {
            "device": self.device,
            "cuda_available": cuda_available,
            "gpu_name": gpu_name,
            "device_index": device_index,
            "gpu_mem_free_bytes": free_mem,
            "gpu_mem_total_bytes": total_mem,
            "ffmpeg_available": ffmpeg,
            "voices": self._voice_mapping,
            "speed": self.speed,
            "split_pattern": self.split_pattern,
            "stream_chunk_seconds": self.stream_chunk_samples / float(SAMPLE_RATE),
        }
