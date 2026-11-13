"""Timing, resource monitoring, and hardware preflight checks."""

from __future__ import annotations

from dataclasses import dataclass
import os
import platform
import threading
import time
from typing import Any


class Stopwatch:
    """Measure elapsed wall-clock seconds."""

    def __enter__(self) -> "Stopwatch":
        self.started = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, *_: object) -> None:
        self.seconds = time.perf_counter() - self.started


class ResourceMonitor:
    """Sample process RSS while tracking PyTorch peak GPU allocation."""

    def __init__(self, interval_seconds: float = 0.25) -> None:
        self.interval_seconds = interval_seconds
        self.peak_host_ram_bytes = 0
        self.peak_gpu_memory_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        import psutil

        process = psutil.Process()
        while not self._stop.is_set():
            try:
                rss = process.memory_info().rss
                rss += sum(child.memory_info().rss for child in process.children(recursive=True))
                self.peak_host_ram_bytes = max(self.peak_host_ram_bytes, int(rss))
            except (psutil.Error, OSError):
                pass
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> "ResourceMonitor":
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass
        self._sample_once()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample_once(self) -> None:
        import psutil

        process = psutil.Process()
        try:
            rss = process.memory_info().rss
            rss += sum(child.memory_info().rss for child in process.children(recursive=True))
            self.peak_host_ram_bytes = max(self.peak_host_ram_bytes, int(rss))
        except (psutil.Error, OSError):
            pass

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._sample_once()
        try:
            import torch

            if torch.cuda.is_available():
                self.peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated())
        except ImportError:
            pass


def hardware_info() -> dict[str, Any]:
    """Return portable CPU, RAM, GPU, and Python details."""
    import psutil

    result: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "system_ram_bytes": int(psutil.virtual_memory().total),
        "system_available_ram_bytes": int(psutil.virtual_memory().available),
        "gpu_name": None,
        "gpu_vram_bytes": 0,
        "cuda_available": False,
    }
    try:
        import torch

        result["torch_version"] = torch.__version__
        result["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            result["gpu_name"] = torch.cuda.get_device_name(0)
            result["gpu_vram_bytes"] = int(properties.total_memory)
            result["cuda_capability"] = list(torch.cuda.get_device_capability(0))
    except ImportError:
        result["torch_version"] = None
    return result


def require_cuda(label: str) -> dict[str, Any]:
    """Require a CUDA runtime and return its hardware details."""
    details = hardware_info()
    if not details["cuda_available"]:
        raise RuntimeError(f"{label} requires a CUDA GPU runtime")
    return details

def require_tabpfn_full_hardware() -> dict[str, Any]:
    """Require the supported full-run TabPFN hardware contract."""
    from .config import (
        MIN_HIGH_RAM_BYTES,
        MIN_TABPFN_VRAM_BYTES,
        SUPPORTED_TABPFN_GPU_NAMES,
    )

    details = require_cuda("The 700,000-row TabPFN run")
    gpu_name = str(details["gpu_name"])
    if not any(name in gpu_name for name in SUPPORTED_TABPFN_GPU_NAMES):
        raise RuntimeError(
            f"Unsupported GPU {gpu_name!r}. Use an A100 or a newer supported accelerator."
        )
    if int(details["gpu_vram_bytes"]) < MIN_TABPFN_VRAM_BYTES:
        raise RuntimeError("The 700,000-row TabPFN run requires at least 40 GB of VRAM")
    if int(details["system_ram_bytes"]) < MIN_HIGH_RAM_BYTES:
        raise RuntimeError("Enable a High-RAM runtime with at least 40 GiB of system RAM")
    return details
