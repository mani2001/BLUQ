"""
GPU Utilities Module

Provides combined functionality for GPU memory management, dynamic batch size
optimization, and detailed performance profiling.

Memory Management:
- DeviceInfo: Information about compute devices (CUDA, MPS, CPU)
- GPUMemoryManager: Dynamic batch size calculation based on available memory
- GPU tier detection for automatic optimization (A100, high-end, mid-range, low-end)

Profiling:
- GPUProfiler: Detailed timing and memory tracking for operations
- TimingRecord / GPUSnapshot: Data structures for profiling metrics
- profile() decorator and track() context manager for easy instrumentation
"""

import json
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class DeviceInfo:
    """Information about the compute device."""
    device_type: str  # 'cuda', 'mps', 'cpu'
    device_name: str
    total_memory_gb: float
    available_memory_gb: float
    compute_capability: Optional[Tuple[int, int]] = None

    def __str__(self) -> str:
        return (
            f"Device: {self.device_name} ({self.device_type})\n"
            f"  Total Memory: {self.total_memory_gb:.2f} GB\n"
            f"  Available Memory: {self.available_memory_gb:.2f} GB"
        )


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
    utilization_percent: float  # If available via nvidia-ml-py


# =============================================================================
# GPU Tier Constants
# =============================================================================

# GPU tier detection for automatic batch size optimization
# A100 gets its own tier with more aggressive settings for maximum utilization
HIGH_END_GPUS = ['H100', 'H200', 'A6000', 'RTX 4090', 'RTX 3090', 'L40', 'A40']
A100_GPUS = ['A100']  # Separate tier for A100 with optimized settings

GPU_TIER_CONFIG = {
    'a100': {
        'safety_margin': 0.93,       # More aggressive (vs 0.90 for high_end)
        'max_batch_size': 192,       # Higher batch size (vs 128)
        'min_vram_gb': 40,           # A100 comes in 40GB and 80GB variants
        'activation_multiplier': 0.6  # More aggressive activation estimate for larger batches
    },
    'high_end': {'safety_margin': 0.90, 'max_batch_size': 128, 'min_vram_gb': 48, 'activation_multiplier': 0.6},
    'mid_range': {'safety_margin': 0.85, 'max_batch_size': 64, 'min_vram_gb': 16, 'activation_multiplier': 0.6},
    'low_end': {'safety_margin': 0.80, 'max_batch_size': 32, 'min_vram_gb': 0, 'activation_multiplier': 0.6},
}


# =============================================================================
# Device Info Functions
# =============================================================================

def get_device_info() -> DeviceInfo:
    """
    Get information about the available compute device.

    Returns:
        DeviceInfo with device details
    """
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        total_memory = props.total_memory / (1024**3)
        available_memory = (props.total_memory - torch.cuda.memory_allocated(device)) / (1024**3)

        return DeviceInfo(
            device_type='cuda',
            device_name=props.name,
            total_memory_gb=total_memory,
            available_memory_gb=available_memory,
            compute_capability=(props.major, props.minor)
        )
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Apple Silicon MPS - estimate memory (MPS doesn't provide direct memory info)
        import subprocess
        try:
            result = subprocess.run(
                ['sysctl', '-n', 'hw.memsize'],
                capture_output=True, text=True
            )
            total_memory = int(result.stdout.strip()) / (1024**3)
            # Estimate available as 50% of system memory for MPS
            available_memory = total_memory * 0.5
        except Exception:
            total_memory = 8.0  # Default assumption
            available_memory = 4.0

        return DeviceInfo(
            device_type='mps',
            device_name='Apple Silicon',
            total_memory_gb=total_memory,
            available_memory_gb=available_memory
        )
    else:
        # CPU fallback
        import psutil
        total_memory = psutil.virtual_memory().total / (1024**3)
        available_memory = psutil.virtual_memory().available / (1024**3)

        return DeviceInfo(
            device_type='cpu',
            device_name='CPU',
            total_memory_gb=total_memory,
            available_memory_gb=available_memory
        )


def get_gpu_tier(device_info: Optional['DeviceInfo'] = None) -> str:
    """
    Detect GPU tier based on device name and available memory.

    Args:
        device_info: Optional DeviceInfo object. If not provided, will be fetched.

    Returns:
        GPU tier: 'a100', 'high_end', 'mid_range', or 'low_end'
    """
    if device_info is None:
        device_info = get_device_info()

    if device_info.device_type != 'cuda':
        return 'low_end'

    # Check by GPU name first
    gpu_name = device_info.device_name.upper()

    # Check for A100 first (dedicated tier with optimized settings)
    for a100_gpu in A100_GPUS:
        if a100_gpu.upper() in gpu_name:
            logger.info(f"Detected A100 GPU: {device_info.device_name} - using optimized A100 settings")
            return 'a100'

    # Check other high-end GPUs
    for high_end_gpu in HIGH_END_GPUS:
        if high_end_gpu.upper() in gpu_name:
            logger.info(f"Detected high-end GPU by name: {device_info.device_name}")
            return 'high_end'

    # Fallback to VRAM-based detection
    vram_gb = device_info.total_memory_gb
    if vram_gb >= 48:
        logger.info(f"Detected high-end GPU by VRAM: {vram_gb:.1f}GB")
        return 'high_end'
    elif vram_gb >= 16:
        logger.info(f"Detected mid-range GPU by VRAM: {vram_gb:.1f}GB")
        return 'mid_range'
    else:
        logger.info(f"Detected low-end GPU by VRAM: {vram_gb:.1f}GB")
        return 'low_end'


def get_gpu_config(device_info: Optional['DeviceInfo'] = None) -> Dict[str, Any]:
    """
    Get GPU configuration based on detected tier.

    Args:
        device_info: Optional DeviceInfo object.

    Returns:
        Dictionary with safety_margin and max_batch_size for the detected tier
    """
    tier = get_gpu_tier(device_info)
    config = GPU_TIER_CONFIG[tier].copy()
    config['tier'] = tier
    return config


# =============================================================================
# GPUMemoryManager Class
# =============================================================================

class GPUMemoryManager:
    """
    Manages GPU memory and provides dynamic batch size optimization.
    """

    # Approximate memory requirements per billion parameters (in GB)
    # Based on dtype: float32=4, float16/bfloat16=2, int8=1, int4=0.5
    MEMORY_PER_BILLION_PARAMS = {
        'float32': 4.0,
        'float16': 2.0,
        'bfloat16': 2.0,
        'int8': 1.0,
        'int4': 0.5,
    }

    # Default activation memory multiplier (can be overridden by tier config)
    DEFAULT_ACTIVATION_MULTIPLIER = 0.3

    # Default safety margin (will be overridden by GPU tier config)
    DEFAULT_SAFETY_MARGIN = 0.85

    def __init__(self):
        """Initialize the GPU memory manager."""
        self.device_info = get_device_info()
        self.gpu_config = get_gpu_config(self.device_info)
        self.safety_margin = self.gpu_config['safety_margin']
        # Use tier-specific activation multiplier if available (A100 uses 0.4)
        self.activation_multiplier = self.gpu_config.get(
            'activation_multiplier', self.DEFAULT_ACTIVATION_MULTIPLIER
        )
        logger.info(
            f"Initialized GPUMemoryManager:\n{self.device_info}\n"
            f"  GPU Tier: {self.gpu_config['tier']}\n"
            f"  Safety Margin: {self.safety_margin}\n"
            f"  Activation Multiplier: {self.activation_multiplier}\n"
            f"  Max Batch Size: {self.gpu_config['max_batch_size']}"
        )

    def estimate_model_memory(
        self,
        num_params_billions: float,
        dtype: str = 'float16'
    ) -> float:
        """
        Estimate memory required for model weights.

        Args:
            num_params_billions: Number of parameters in billions
            dtype: Data type for model weights

        Returns:
            Estimated memory in GB
        """
        dtype_key = dtype.replace('torch.', '')
        memory_per_b = self.MEMORY_PER_BILLION_PARAMS.get(dtype_key, 2.0)
        return num_params_billions * memory_per_b

    def estimate_batch_memory(
        self,
        num_params_billions: float,
        batch_size: int,
        seq_length: int = 2048,
        dtype: str = 'float16'
    ) -> float:
        """
        Estimate additional memory required for a batch during inference.

        Args:
            num_params_billions: Number of parameters in billions
            batch_size: Batch size
            seq_length: Sequence length
            dtype: Data type

        Returns:
            Estimated batch memory in GB
        """
        # Base activation memory scales with model size and batch
        model_memory = self.estimate_model_memory(num_params_billions, dtype)
        activation_per_item = model_memory * self.activation_multiplier * (seq_length / 2048)
        return activation_per_item * batch_size

    def get_optimal_batch_size(
        self,
        num_params_billions: float,
        dtype: str = 'float16',
        seq_length: int = 2048,
        min_batch_size: int = 1,
        max_batch_size: Optional[int] = None
    ) -> int:
        """
        Calculate optimal batch size based on available GPU memory.

        Args:
            num_params_billions: Number of parameters in billions
            dtype: Data type for model weights
            seq_length: Expected sequence length
            min_batch_size: Minimum batch size to return
            max_batch_size: Maximum batch size to consider (None = auto-detect from GPU tier)

        Returns:
            Optimal batch size
        """
        # Use GPU tier-based max if not specified
        if max_batch_size is None:
            max_batch_size = self.gpu_config['max_batch_size']

        available_memory = self.device_info.available_memory_gb * self.safety_margin
        model_memory = self.estimate_model_memory(num_params_billions, dtype)

        # Memory available for batches after loading model
        memory_for_batches = available_memory - model_memory

        if memory_for_batches <= 0:
            logger.warning(
                f"Model requires {model_memory:.2f}GB but only {available_memory:.2f}GB available. "
                f"Using minimum batch size {min_batch_size}."
            )
            return min_batch_size

        # Calculate batch size using tier-specific activation multiplier
        activation_per_item = model_memory * self.activation_multiplier * (seq_length / 2048)

        if activation_per_item <= 0:
            return max_batch_size

        optimal_batch = int(memory_for_batches / activation_per_item)
        optimal_batch = max(min_batch_size, min(optimal_batch, max_batch_size))

        logger.info(
            f"Optimal batch size for {num_params_billions:.1f}B model ({dtype}): {optimal_batch}\n"
            f"  Available memory: {available_memory:.2f}GB\n"
            f"  Model memory: {model_memory:.2f}GB\n"
            f"  Memory per batch item: {activation_per_item:.3f}GB"
        )

        return optimal_batch

    def get_current_memory_usage(self) -> Dict[str, float]:
        """
        Get current GPU memory usage.

        Returns:
            Dictionary with memory usage statistics
        """
        if self.device_info.device_type == 'cuda':
            return {
                'allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                'max_allocated_gb': torch.cuda.max_memory_allocated() / (1024**3),
                'total_gb': self.device_info.total_memory_gb,
                'utilization': torch.cuda.memory_allocated() / (self.device_info.total_memory_gb * 1024**3)
            }
        else:
            return {
                'allocated_gb': 0,
                'reserved_gb': 0,
                'max_allocated_gb': 0,
                'total_gb': self.device_info.total_memory_gb,
                'utilization': 0
            }

    def clear_cache(self) -> None:
        """Clear GPU cache to free memory."""
        if self.device_info.device_type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            logger.debug("Cleared CUDA cache")
        elif self.device_info.device_type == 'mps':
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            logger.debug("Cleared MPS cache")

    def check_memory_pressure(self, threshold: float = 0.85) -> bool:
        """
        Check if GPU memory usage exceeds threshold and log warning.

        Args:
            threshold: Memory utilization threshold (0-1). Default 0.85 (85%)

        Returns:
            True if memory usage exceeds threshold, False otherwise
        """
        usage = self.get_current_memory_usage()
        utilization = usage.get('utilization', 0)

        if utilization > threshold:
            logger.warning(
                f"GPU memory pressure: {utilization:.1%} utilization "
                f"(threshold: {threshold:.0%}). "
                f"Allocated: {usage['allocated_gb']:.2f}GB / {usage['total_gb']:.2f}GB"
            )
            return True
        return False

    def log_memory_status(self) -> None:
        """Log current GPU memory status."""
        usage = self.get_current_memory_usage()
        logger.info(
            f"GPU Memory: {usage['allocated_gb']:.2f}GB allocated, "
            f"{usage['reserved_gb']:.2f}GB reserved, "
            f"{usage['utilization']:.1%} utilization"
        )


# =============================================================================
# Memory Management Convenience Functions
# =============================================================================

def get_optimal_batch_size(
    num_params_billions: float,
    dtype: str = 'float16',
    seq_length: int = 2048,
    min_batch_size: int = 1,
    max_batch_size: Optional[int] = None
) -> int:
    """
    Convenience function to get optimal batch size.

    Args:
        num_params_billions: Number of parameters in billions
        dtype: Data type for model weights
        seq_length: Expected sequence length
        min_batch_size: Minimum batch size
        max_batch_size: Maximum batch size (None = auto-detect from GPU tier)

    Returns:
        Optimal batch size
    """
    manager = GPUMemoryManager()
    return manager.get_optimal_batch_size(
        num_params_billions=num_params_billions,
        dtype=dtype,
        seq_length=seq_length,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size
    )


# =============================================================================
# Model Size Utilities
# =============================================================================

# Model size lookup table (approximate parameter counts in billions)
MODEL_SIZES = {
    'tinyllama-1.1b': 1.1,
    'phi-1.5': 1.3,
    'phi-2': 2.7,
    'stablelm-2-1.6b': 1.6,
    'stablelm-2-zephyr-1.6b': 1.6,
    'qwen-1.8b': 1.8,
    'qwen-1.8b-chat': 1.8,
    'gemma-2b': 2.0,
    'gemma-2b-it': 2.0,
    'gemma-2-9b': 9.0,
    'gemma-2-9b-it': 9.0,
    'mistral-7b': 7.0,
    'mistral-7b-instruct': 7.0,
    'openelm-1.1b': 1.1,
    'smollm-135m': 0.135,
    'smollm-360m': 0.36,
    'smollm-1.7b': 1.7,
    'opt-1.3b': 1.3,
    'pythia-1.4b': 1.4,
    'gpt-neo-1.3b': 1.3,
}


def get_model_size(model_name: str) -> float:
    """
    Get approximate model size in billions of parameters.

    Args:
        model_name: Name of the model

    Returns:
        Number of parameters in billions
    """
    model_name_lower = model_name.lower()

    # Check direct match
    if model_name_lower in MODEL_SIZES:
        return MODEL_SIZES[model_name_lower]

    # Try to extract from name
    for key, size in MODEL_SIZES.items():
        if key in model_name_lower:
            return size

    # Default to 1B if unknown
    logger.warning(f"Unknown model size for {model_name}, defaulting to 1B")
    return 1.0


# =============================================================================
# GPUProfiler Class
# =============================================================================

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
        # Note: pynvml is provided by the nvidia-ml-py package
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
                           "Install with: pip install nvidia-ml-py")
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
        """Get GPU utilization percentage (requires nvidia-ml-py)."""
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
                f"GPU Memory: {memory_before:.1f}MB -> {memory_after:.1f}MB "
                f"(delta {memory_after - memory_before:+.1f}MB)"
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


# =============================================================================
# Profiler Convenience Functions
# =============================================================================

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


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Test device info
    print("\n" + "="*60)
    print("Device Information")
    print("="*60)
    device_info = get_device_info()
    print(device_info)

    # Test memory manager
    print("\n" + "="*60)
    print("GPU Memory Manager")
    print("="*60)
    manager = GPUMemoryManager()

    # Test optimal batch sizes for different models
    models = ['tinyllama-1.1b', 'phi-2', 'gemma-2b']
    dtypes = ['float16', 'float32']

    print("\nOptimal batch sizes:")
    for model in models:
        size = get_model_size(model)
        for dtype in dtypes:
            batch_size = manager.get_optimal_batch_size(
                num_params_billions=size,
                dtype=dtype,
                seq_length=2048
            )
            print(f"  {model} ({dtype}): batch_size={batch_size}")

    # Current memory usage
    print("\n" + "="*60)
    print("Current Memory Usage")
    print("="*60)
    usage = manager.get_current_memory_usage()
    for key, value in usage.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")

    # Test GPU Profiler
    print("\n" + "="*60)
    print("GPU Profiler Test")
    print("="*60)

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
