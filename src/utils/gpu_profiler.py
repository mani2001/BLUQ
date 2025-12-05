"""
GPU Profiler Module
Provides detailed logging of GPU usage, timing, and performance metrics.
Helps identify bottlenecks and overhead in CPU-GPU communication.
"""

import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
from datetime import datetime
import json
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


@dataclass
class TimingRecord:
    """Record of a timed operation."""
    name: str
    start_time: float
    end_time: float
    duration_ms: float
    gpu_memory_before_mb: float
    gpu_memory_after_mb: float
    gpu_memory_delta_mb: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GPUSnapshot:
    """Snapshot of GPU state at a point in time."""
    timestamp: float
    memory_allocated_mb: float
    memory_reserved_mb: float
    memory_free_mb: float
    utilization_percent: float  # If available via pynvml


class GPUProfiler:
    """
    Profiles GPU usage, timing, and identifies performance bottlenecks.

    Usage:
        profiler = GPUProfiler()
        profiler.start_monitoring()

        with profiler.track("model_loading"):
            model = load_model()

        with profiler.track("inference", batch_size=32):
            results = model(inputs)

        profiler.stop_monitoring()
        profiler.print_summary()
        profiler.save_report("gpu_profile.json")
    """

    def __init__(self,
                 monitoring_interval: float = 1.0,
                 enable_nvml: bool = True):
        """
        Initialize the GPU profiler.

        Args:
            monitoring_interval: Interval in seconds for background monitoring
            enable_nvml: Whether to use NVIDIA Management Library for GPU utilization
        """
        self.monitoring_interval = monitoring_interval
        self.timing_records: List[TimingRecord] = []
        self.snapshots: List[GPUSnapshot] = []
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None

        # Check CUDA availability
        self.cuda_available = torch.cuda.is_available()
        if not self.cuda_available:
            logger.warning("CUDA not available. GPU profiling will be limited.")

        # Try to initialize pynvml for GPU utilization metrics
        self.nvml_available = False
        if enable_nvml and self.cuda_available:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.nvml_available = True
                logger.info("NVML initialized for GPU utilization monitoring")
            except ImportError:
                logger.debug("pynvml not installed. GPU utilization monitoring disabled. "
                           "Install with: pip install pynvml")
            except Exception as e:
                logger.debug(f"Failed to initialize NVML: {e}")

        # Performance counters
        self._total_gpu_time_ms = 0
        self._total_cpu_time_ms = 0
        self._data_transfer_time_ms = 0

        logger.info(f"GPUProfiler initialized (CUDA: {self.cuda_available}, NVML: {self.nvml_available})")

    def _get_gpu_memory_mb(self) -> float:
        """Get current GPU memory allocated in MB."""
        if self.cuda_available:
            return torch.cuda.memory_allocated() / (1024 ** 2)
        return 0.0

    def _get_gpu_memory_reserved_mb(self) -> float:
        """Get current GPU memory reserved in MB."""
        if self.cuda_available:
            return torch.cuda.memory_reserved() / (1024 ** 2)
        return 0.0

    def _get_gpu_memory_free_mb(self) -> float:
        """Get free GPU memory in MB."""
        if self.cuda_available:
            props = torch.cuda.get_device_properties(0)
            total = props.total_memory / (1024 ** 2)
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            return total - allocated
        return 0.0

    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage (requires pynvml)."""
        if self.nvml_available:
            try:
                import pynvml
                util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                return util.gpu
            except Exception:
                pass
        return -1.0  # -1 indicates not available

    def _take_snapshot(self) -> GPUSnapshot:
        """Take a snapshot of current GPU state."""
        return GPUSnapshot(
            timestamp=time.time(),
            memory_allocated_mb=self._get_gpu_memory_mb(),
            memory_reserved_mb=self._get_gpu_memory_reserved_mb(),
            memory_free_mb=self._get_gpu_memory_free_mb(),
            utilization_percent=self._get_gpu_utilization()
        )

    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            snapshot = self._take_snapshot()
            self.snapshots.append(snapshot)
            time.sleep(self.monitoring_interval)

    def start_monitoring(self):
        """Start background GPU monitoring."""
        if self._monitoring:
            logger.warning("Monitoring already running")
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Started GPU monitoring")

    def stop_monitoring(self):
        """Stop background GPU monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info(f"Stopped GPU monitoring. Collected {len(self.snapshots)} snapshots.")

    @contextmanager
    def track(self, name: str, **metadata):
        """
        Context manager to track timing and memory for an operation.

        Args:
            name: Name of the operation
            **metadata: Additional metadata to record

        Example:
            with profiler.track("inference", batch_size=32):
                results = model(inputs)
        """
        # Synchronize CUDA to get accurate timing
        if self.cuda_available:
            torch.cuda.synchronize()

        memory_before = self._get_gpu_memory_mb()
        start_time = time.perf_counter()

        try:
            yield
        finally:
            # Synchronize again to ensure all GPU ops complete
            if self.cuda_available:
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            memory_after = self._get_gpu_memory_mb()

            duration_ms = (end_time - start_time) * 1000

            record = TimingRecord(
                name=name,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                gpu_memory_before_mb=memory_before,
                gpu_memory_after_mb=memory_after,
                gpu_memory_delta_mb=memory_after - memory_before,
                metadata=metadata
            )
            self.timing_records.append(record)

            # Log the operation
            logger.info(
                f"[{name}] Duration: {duration_ms:.2f}ms | "
                f"GPU Memory: {memory_before:.1f}MB → {memory_after:.1f}MB "
                f"(Δ{memory_after - memory_before:+.1f}MB)"
            )

    def track_data_transfer(self, tensor_size_mb: float, direction: str = "to_gpu"):
        """
        Track data transfer between CPU and GPU.

        Args:
            tensor_size_mb: Size of transferred data in MB
            direction: 'to_gpu' or 'to_cpu'
        """
        if self.cuda_available:
            torch.cuda.synchronize()
            start = time.perf_counter()
            torch.cuda.synchronize()
            end = time.perf_counter()

            transfer_time_ms = (end - start) * 1000
            self._data_transfer_time_ms += transfer_time_ms

            bandwidth_gbps = (tensor_size_mb / 1024) / (transfer_time_ms / 1000) if transfer_time_ms > 0 else 0

            logger.debug(
                f"Data transfer ({direction}): {tensor_size_mb:.2f}MB in {transfer_time_ms:.2f}ms "
                f"({bandwidth_gbps:.2f} GB/s)"
            )

    def log_current_state(self, label: str = ""):
        """Log current GPU state."""
        if not self.cuda_available:
            logger.info(f"[{label}] GPU not available")
            return

        allocated = self._get_gpu_memory_mb()
        reserved = self._get_gpu_memory_reserved_mb()
        free = self._get_gpu_memory_free_mb()
        util = self._get_gpu_utilization()

        util_str = f"{util:.1f}%" if util >= 0 else "N/A"

        logger.info(
            f"[{label}] GPU State: "
            f"Allocated: {allocated:.1f}MB | "
            f"Reserved: {reserved:.1f}MB | "
            f"Free: {free:.1f}MB | "
            f"Utilization: {util_str}"
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of profiling results."""
        if not self.timing_records:
            return {"error": "No timing records collected"}

        # Group by operation name
        ops_by_name: Dict[str, List[TimingRecord]] = {}
        for record in self.timing_records:
            if record.name not in ops_by_name:
                ops_by_name[record.name] = []
            ops_by_name[record.name].append(record)

        # Calculate statistics
        summary = {
            "total_operations": len(self.timing_records),
            "total_time_ms": sum(r.duration_ms for r in self.timing_records),
            "operations": {},
            "memory": {},
            "bottlenecks": []
        }

        for name, records in ops_by_name.items():
            durations = [r.duration_ms for r in records]
            memory_deltas = [r.gpu_memory_delta_mb for r in records]

            summary["operations"][name] = {
                "count": len(records),
                "total_time_ms": sum(durations),
                "avg_time_ms": sum(durations) / len(durations),
                "min_time_ms": min(durations),
                "max_time_ms": max(durations),
                "avg_memory_delta_mb": sum(memory_deltas) / len(memory_deltas),
            }

        # Memory statistics from snapshots
        if self.snapshots:
            allocations = [s.memory_allocated_mb for s in self.snapshots]
            utils = [s.utilization_percent for s in self.snapshots if s.utilization_percent >= 0]

            summary["memory"] = {
                "peak_allocated_mb": max(allocations),
                "avg_allocated_mb": sum(allocations) / len(allocations),
                "avg_utilization_percent": sum(utils) / len(utils) if utils else -1,
                "snapshots_count": len(self.snapshots)
            }

        # Identify bottlenecks
        total_time = summary["total_time_ms"]
        for name, stats in summary["operations"].items():
            pct = (stats["total_time_ms"] / total_time * 100) if total_time > 0 else 0
            if pct > 20:  # Operation takes >20% of total time
                summary["bottlenecks"].append({
                    "operation": name,
                    "time_percent": pct,
                    "recommendation": self._get_recommendation(name, stats)
                })

        return summary

    def _get_recommendation(self, name: str, stats: Dict) -> str:
        """Get optimization recommendation for an operation."""
        name_lower = name.lower()

        if "load" in name_lower or "model" in name_lower:
            return "Model loading is expected to be slow. Consider caching or using quantization."
        elif "inference" in name_lower or "forward" in name_lower:
            avg_time = stats["avg_time_ms"]
            if avg_time > 1000:
                return "Inference is slow. Consider: larger batch size, Flash Attention, or quantization."
            return "Inference timing is reasonable."
        elif "transfer" in name_lower or "data" in name_lower:
            return "Data transfer overhead detected. Consider pinned memory or larger batch sizes."
        elif "tokeniz" in name_lower:
            return "Tokenization on CPU. Consider batching more tokens or using fast tokenizer."
        else:
            return "Review this operation for optimization opportunities."

    def print_summary(self):
        """Print a formatted summary of profiling results."""
        summary = self.get_summary()

        print("\n" + "=" * 80)
        print("GPU PROFILING SUMMARY")
        print("=" * 80)

        print(f"\nTotal Operations: {summary.get('total_operations', 0)}")
        print(f"Total Time: {summary.get('total_time_ms', 0):.2f}ms "
              f"({summary.get('total_time_ms', 0)/1000:.2f}s)")

        print("\n--- Operations Breakdown ---")
        ops = summary.get("operations", {})
        # Sort by total time
        sorted_ops = sorted(ops.items(), key=lambda x: x[1]["total_time_ms"], reverse=True)

        for name, stats in sorted_ops:
            total_time = summary.get('total_time_ms', 1)
            pct = (stats["total_time_ms"] / total_time * 100) if total_time > 0 else 0
            print(f"\n  [{name}]")
            print(f"    Count: {stats['count']}")
            print(f"    Total: {stats['total_time_ms']:.2f}ms ({pct:.1f}%)")
            print(f"    Avg: {stats['avg_time_ms']:.2f}ms | "
                  f"Min: {stats['min_time_ms']:.2f}ms | "
                  f"Max: {stats['max_time_ms']:.2f}ms")
            print(f"    Avg Memory Delta: {stats['avg_memory_delta_mb']:+.1f}MB")

        mem = summary.get("memory", {})
        if mem:
            print("\n--- Memory Statistics ---")
            print(f"  Peak Allocated: {mem.get('peak_allocated_mb', 0):.1f}MB")
            print(f"  Avg Allocated: {mem.get('avg_allocated_mb', 0):.1f}MB")
            util = mem.get('avg_utilization_percent', -1)
            if util >= 0:
                print(f"  Avg GPU Utilization: {util:.1f}%")

        bottlenecks = summary.get("bottlenecks", [])
        if bottlenecks:
            print("\n--- Bottlenecks & Recommendations ---")
            for b in bottlenecks:
                print(f"\n  [WARN] {b['operation']} ({b['time_percent']:.1f}% of total time)")
                print(f"     -> {b['recommendation']}")

        print("\n" + "=" * 80)

    def save_report(self, path: str):
        """Save profiling report to JSON file."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": self.get_summary(),
            "timing_records": [
                {
                    "name": r.name,
                    "duration_ms": r.duration_ms,
                    "gpu_memory_before_mb": r.gpu_memory_before_mb,
                    "gpu_memory_after_mb": r.gpu_memory_after_mb,
                    "gpu_memory_delta_mb": r.gpu_memory_delta_mb,
                    "metadata": r.metadata
                }
                for r in self.timing_records
            ],
            "snapshots_summary": {
                "count": len(self.snapshots),
                "memory_allocated_mb": [s.memory_allocated_mb for s in self.snapshots[-100:]],  # Last 100
                "utilization_percent": [s.utilization_percent for s in self.snapshots[-100:] if s.utilization_percent >= 0]
            }
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Saved GPU profiling report to {path}")

    def reset(self):
        """Reset all profiling data."""
        self.timing_records = []
        self.snapshots = []
        self._total_gpu_time_ms = 0
        self._total_cpu_time_ms = 0
        self._data_transfer_time_ms = 0

        if self.cuda_available:
            torch.cuda.reset_peak_memory_stats()

        logger.info("GPU profiler reset")


# Global profiler instance for convenience
_global_profiler: Optional[GPUProfiler] = None


def get_profiler() -> GPUProfiler:
    """Get or create global GPU profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = GPUProfiler()
    return _global_profiler


def profile(name: str, **metadata):
    """Decorator to profile a function."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            with profiler.track(name, **metadata):
                return func(*args, **kwargs)
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test the profiler
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    profiler = GPUProfiler()
    profiler.start_monitoring()

    print("\nTesting GPU Profiler...")

    # Simulate some operations
    with profiler.track("test_allocation"):
        if torch.cuda.is_available():
            x = torch.randn(1000, 1000, device='cuda')
            time.sleep(0.1)

    with profiler.track("test_computation", size=1000):
        if torch.cuda.is_available():
            y = torch.matmul(x, x.T)
            time.sleep(0.05)

    profiler.log_current_state("after_computation")

    with profiler.track("test_cleanup"):
        if torch.cuda.is_available():
            del x, y
            torch.cuda.empty_cache()

    profiler.stop_monitoring()
    profiler.print_summary()
    profiler.save_report("test_gpu_profile.json")
