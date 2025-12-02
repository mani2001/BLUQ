"""
GPU Utility Module
Provides utilities for GPU memory management and dynamic batch size optimization.
"""

import logging
import torch
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


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


# GPU tier detection for automatic batch size optimization
HIGH_END_GPUS = ['A100', 'H100', 'H200', 'A6000', 'RTX 4090', 'RTX 3090', 'L40', 'A40']

GPU_TIER_CONFIG = {
    'high_end': {'safety_margin': 0.90, 'max_batch_size': 128, 'min_vram_gb': 48},
    'mid_range': {'safety_margin': 0.85, 'max_batch_size': 64, 'min_vram_gb': 16},
    'low_end': {'safety_margin': 0.80, 'max_batch_size': 32, 'min_vram_gb': 0},
}


def get_gpu_tier(device_info: Optional['DeviceInfo'] = None) -> str:
    """
    Detect GPU tier based on device name and available memory.

    Args:
        device_info: Optional DeviceInfo object. If not provided, will be fetched.

    Returns:
        GPU tier: 'high_end', 'mid_range', or 'low_end'
    """
    if device_info is None:
        device_info = get_device_info()

    if device_info.device_type != 'cuda':
        return 'low_end'

    # Check by GPU name first
    gpu_name = device_info.device_name.upper()
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

    # Approximate activation memory multiplier per batch item (relative to model size)
    ACTIVATION_MULTIPLIER = 0.3

    # Default safety margin (will be overridden by GPU tier config)
    DEFAULT_SAFETY_MARGIN = 0.85

    def __init__(self):
        """Initialize the GPU memory manager."""
        self.device_info = get_device_info()
        self.gpu_config = get_gpu_config(self.device_info)
        self.safety_margin = self.gpu_config['safety_margin']
        logger.info(
            f"Initialized GPUMemoryManager:\n{self.device_info}\n"
            f"  GPU Tier: {self.gpu_config['tier']}\n"
            f"  Safety Margin: {self.safety_margin}\n"
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
        activation_per_item = model_memory * self.ACTIVATION_MULTIPLIER * (seq_length / 2048)
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

        # Calculate batch size
        activation_per_item = model_memory * self.ACTIVATION_MULTIPLIER * (seq_length / 2048)

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
            logger.info("Cleared CUDA cache")
        elif self.device_info.device_type == 'mps':
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            logger.info("Cleared MPS cache")


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
