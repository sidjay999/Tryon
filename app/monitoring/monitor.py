"""
Pipeline monitoring – Phase 2.
Captures GPU info, timing, memory usage per request.
"""
import json
import logging
import time

import torch

logger = logging.getLogger(__name__)


class PipelineMonitor:
    """
    Context manager that captures performance metrics for a pipeline run.

    Usage:
        with PipelineMonitor(request_id="abc") as mon:
            # ... run pipeline ...
        metrics = mon.metrics
    """

    def __init__(self, request_id: str):
        self.request_id = request_id
        self._start_time = None
        self._start_vram = 0
        self.metrics: dict = {}

    def __enter__(self):
        self._start_time = time.time()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self._start_vram = torch.cuda.memory_allocated(0)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self._start_time

        gpu_name = ""
        vram_used_gb = 0
        vram_peak_gb = 0
        vram_total_gb = 0

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_name = torch.cuda.get_device_name(0)
            vram_used_gb = round(torch.cuda.memory_allocated(0) / 1e9, 2)
            vram_peak_gb = round(torch.cuda.max_memory_allocated(0) / 1e9, 2)
            vram_total_gb = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2
            )
            # Reset peak stats for next request
            torch.cuda.reset_peak_memory_stats(0)

        status = "SUCCESS" if exc_type is None else "FAILED"

        self.metrics = {
            "request_id": self.request_id,
            "gpu": gpu_name,
            "vram_used_gb": vram_used_gb,
            "vram_peak_gb": vram_peak_gb,
            "vram_total_gb": vram_total_gb,
            "inference_time_s": round(elapsed, 1),
            "status": status,
        }

        if exc_type is not None:
            self.metrics["error"] = str(exc_val)

        # Emit structured JSON log
        _log_metrics(self.metrics)

        return False  # Don't suppress exceptions


def _log_metrics(metrics: dict):
    """Emit a structured JSON log line for monitoring."""
    log_line = json.dumps(metrics, separators=(",", ":"))
    if metrics.get("status") == "SUCCESS":
        logger.info("PIPELINE_METRICS %s", log_line)
    else:
        logger.error("PIPELINE_METRICS %s", log_line)
