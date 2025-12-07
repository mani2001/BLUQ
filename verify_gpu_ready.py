"""
verify_gpu_ready.py
Script to verify that the compute environment is correctly set up and ready for benchmarking.
Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU fallback.
"""

import torch
import sys
import logging
import platform

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_device_type():
    """Determine the best available device type."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def check_gpu():
    logger.info("Checking compute device availability...")

    device_type = get_device_type()

    if device_type == "cuda":
        device_count = torch.cuda.device_count()
        logger.info(f"CUDA available with {device_count} GPU(s)")

        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"GPU {i}: {gpu_name} ({gpu_mem:.2f} GB)")

            # Check if it's an A100 (optional warning)
            if "A100" not in gpu_name:
                logger.warning(f"GPU {i} is not an A100. It is a {gpu_name}.")
        return True

    elif device_type == "mps":
        logger.info("Apple Silicon MPS backend available")

        # Get system memory info on Mac
        try:
            import subprocess
            result = subprocess.run(
                ['sysctl', '-n', 'hw.memsize'],
                capture_output=True, text=True
            )
            total_memory_gb = int(result.stdout.strip()) / (1024**3)
            logger.info(f"System memory: {total_memory_gb:.2f} GB (shared with GPU)")

            # Get chip info
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True, text=True
            )
            chip_info = result.stdout.strip()
            logger.info(f"Processor: {chip_info}")
        except Exception as e:
            logger.warning(f"Could not get system info: {e}")

        logger.info("MPS is ready for PyTorch acceleration")
        return True

    else:
        logger.warning("No GPU acceleration available. Running on CPU only.")
        logger.info("This will be significantly slower for model inference.")
        logger.info(f"Platform: {platform.platform()}")
        return True  # Still return True as CPU is valid

def check_imports():
    logger.info("Checking imports...")
    try:
        import transformers
        import datasets
        import accelerate
        logger.info("Core libraries imported successfully.")
        return True
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        return False

def run_minimal_inference():
    logger.info("Running minimal inference test...")
    try:
        from src.models.model_loader import ModelLoader
        from src.models.model_config import DefaultModelConfigs
        from src.models.inference_engine import InferenceEngine

        # Determine the best device
        device_type = get_device_type()
        logger.info(f"Using device: {device_type}")

        # Use a very small model for verification
        model_name = "smollm-135m"
        logger.info(f"Loading {model_name}...")

        model_config = DefaultModelConfigs.create_pipeline_config(model_name)

        # Set device based on availability
        if device_type == "cuda":
            model_config.load_config.device = "cuda"
            model_config.inference_config.device = "cuda"
        elif device_type == "mps":
            model_config.load_config.device = "mps"
            model_config.inference_config.device = "mps"
        else:
            model_config.load_config.device = "cpu"
            model_config.inference_config.device = "cpu"

        loader = ModelLoader()
        model, tokenizer, info = loader.load_model(model_config.load_config)

        logger.info("Model loaded. Running inference...")
        engine = InferenceEngine(model, tokenizer, info)

        prompt = "Question: What is 2+2?\nA. 3\nB. 4\nAnswer:"
        result = engine.infer_single(prompt)

        logger.info(f"Inference successful. Predicted option: {result.predicted_option}")
        return True

    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        return False

def main():
    logger.info("Starting Compute Environment Verification...")

    device_type = get_device_type()
    logger.info(f"Detected device type: {device_type}")

    checks = [
        ("Imports", check_imports),
        ("Compute Device", check_gpu),
        ("Inference", run_minimal_inference)
    ]

    all_passed = True
    for name, check_func in checks:
        logger.info(f"\n--- Checking {name} ---")
        if check_func():
            logger.info(f"[OK] {name} Passed")
        else:
            logger.error(f"[FAIL] {name} Failed")
            all_passed = False

    print("\n" + "="*50)
    if all_passed:
        device_msg = {
            "cuda": "GPU (CUDA)",
            "mps": "Apple Silicon (MPS)",
            "cpu": "CPU"
        }
        print(f"[OK] SYSTEM IS READY - Using {device_msg.get(device_type, device_type)}")
    else:
        print("[FAIL] SYSTEM IS NOT READY")
    print("="*50)

    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
