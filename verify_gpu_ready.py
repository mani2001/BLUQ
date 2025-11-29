"""
verify_gpu_ready.py
Script to verify that the GPU environment is correctly set up and ready for benchmarking.
"""

import torch
import sys
import logging

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_gpu():
    logger.info("Checking GPU availability...")
    
    if not torch.cuda.is_available():
        logger.error("CUDA is NOT available. Please check your driver and PyTorch installation.")
        return False
    
    device_count = torch.cuda.device_count()
    logger.info(f"Number of GPUs available: {device_count}")
    
    for i in range(device_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        logger.info(f"GPU {i}: {gpu_name} ({gpu_mem:.2f} GB)")
        
        # Check if it's an A100 (optional warning)
        if "A100" not in gpu_name:
            logger.warning(f"GPU {i} is not an A100. It is a {gpu_name}.")
            
    return True

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
        
        # Use a very small model for verification
        model_name = "smollm-135m"
        logger.info(f"Loading {model_name}...")
        
        model_config = DefaultModelConfigs.create_pipeline_config(model_name)
        # Force device to cuda if available
        if torch.cuda.is_available():
            model_config.load_config.device = "cuda"
            model_config.inference_config.device = "cuda"
            
        loader = ModelLoader()
        model, tokenizer, info = loader.load_model(model_config.load_config)
        
        logger.info("Model loaded. Running inference...")
        engine = InferenceEngine(model, tokenizer, info)
        
        prompt = "Question: What is 2+2?\nA. 3\nB. 4\nAnswer:"
        result = engine.infer(prompt)
        
        logger.info(f"Inference successful. Output: {result.generated_text}")
        return True
        
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        return False

def main():
    logger.info("Starting GPU Readiness Verification...")
    
    checks = [
        ("Imports", check_imports),
        ("GPU", check_gpu),
        ("Inference", run_minimal_inference)
    ]
    
    all_passed = True
    for name, check_func in checks:
        logger.info(f"\n--- Checking {name} ---")
        if check_func():
            logger.info(f"✅ {name} Passed")
        else:
            logger.error(f"❌ {name} Failed")
            all_passed = False
            
    print("\n" + "="*40)
    if all_passed:
        print("✅ SYSTEM IS READY FOR GPU DEPLOYMENT")
    else:
        print("❌ SYSTEM IS NOT READY")
    print("="*40)
    
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
