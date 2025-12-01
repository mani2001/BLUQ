#!/usr/bin/env python3
"""
Verification script to test the BLUQ benchmark setup.
Run this before running the full benchmark to ensure everything works.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_imports():
    """Check that all required imports work."""
    print("Checking imports...")

    try:
        import torch
        print(f"  [OK] PyTorch {torch.__version__}")
        print(f"       CUDA available: {torch.cuda.is_available()}")
        if hasattr(torch.backends, 'mps'):
            print(f"       MPS available: {torch.backends.mps.is_available()}")
    except ImportError as e:
        print(f"  [FAIL] PyTorch: {e}")
        return False

    try:
        import transformers
        print(f"  [OK] Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"  [FAIL] Transformers: {e}")
        return False

    try:
        import numpy as np
        print(f"  [OK] NumPy {np.__version__}")
    except ImportError as e:
        print(f"  [FAIL] NumPy: {e}")
        return False

    try:
        import matplotlib
        print(f"  [OK] Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"  [FAIL] Matplotlib: {e}")
        return False

    try:
        import seaborn as sns
        print(f"  [OK] Seaborn {sns.__version__}")
    except ImportError as e:
        print(f"  [FAIL] Seaborn: {e}")
        return False

    try:
        from datasets import load_dataset
        print("  [OK] Datasets (HuggingFace)")
    except ImportError as e:
        print(f"  [FAIL] Datasets: {e}")
        return False

    return True


def check_project_imports():
    """Check that project modules can be imported."""
    print("\nChecking project modules...")

    modules = [
        ("src.utils.gpu_utils", "GPU Utils"),
        ("src.visualization.result_visualizer", "Visualization"),
        ("src.data.dataset_loader", "Dataset Loader"),
        ("src.data.dataset_processor", "Dataset Processor"),
        ("src.data.data_splitter", "Data Splitter"),
        ("src.models.model_loader", "Model Loader"),
        ("src.models.model_config", "Model Config"),
        ("src.models.inference_engine", "Inference Engine"),
        ("src.models.probability_extractor", "Probability Extractor"),
        ("src.prompting.prompt_builder", "Prompt Builder"),
        ("src.prompting.demonstration_manager", "Demonstration Manager"),
        ("src.conformal.prediction_set_generator", "Conformal Prediction"),
        ("src.evaluation.evaluator", "Evaluator"),
    ]

    success = True
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"  [OK] {display_name}")
        except ImportError as e:
            print(f"  [FAIL] {display_name}: {e}")
            success = False

    return success


def check_device_info():
    """Check device information."""
    print("\nChecking device information...")

    try:
        from src.utils.gpu_utils import get_device_info, GPUMemoryManager

        device_info = get_device_info()
        print(f"  Device type: {device_info.device_type}")
        print(f"  Device name: {device_info.device_name}")
        print(f"  Total memory: {device_info.total_memory_gb:.2f} GB")
        print(f"  Available memory: {device_info.available_memory_gb:.2f} GB")

        # Test batch size calculation
        manager = GPUMemoryManager()
        batch_size = manager.get_optimal_batch_size(
            num_params_billions=1.1,
            dtype='float16',
            seq_length=2048
        )
        print(f"  Recommended batch size for 1.1B model: {batch_size}")

        return True
    except Exception as e:
        print(f"  [FAIL] Device check: {e}")
        return False


def check_dataset_loading():
    """Check that datasets can be loaded."""
    print("\nChecking dataset loading (quick test)...")

    try:
        from src.data.dataset_loader import DatasetLoaderFactory

        # Test with QA task (MMLU)
        loader = DatasetLoaderFactory.create_loader('qa')
        print("  [OK] Created QA loader")

        # Load just a few samples
        dataset = loader.load(num_samples=5)
        print(f"  [OK] Loaded {len(dataset.instances)} samples")

        if dataset.instances:
            inst = dataset.instances[0]
            print(f"  Sample: {inst.question[:50]}...")

        return True
    except Exception as e:
        print(f"  [FAIL] Dataset loading: {e}")
        return False


def check_visualization():
    """Check visualization module."""
    print("\nChecking visualization module...")

    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend

        from src.visualization.result_visualizer import ResultVisualizer, BenchmarkResult
        import numpy as np
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = ResultVisualizer(output_dir=tmpdir)

            # Add mock result
            result = BenchmarkResult(
                model_name='test-model',
                task_name='qa',
                dtype='float16',
                strategy='base',
                conformal_method='lac',
                accuracy=50.0,
                coverage_rate=90.0,
                avg_set_size=2.5,
                meets_guarantee=True,
                num_samples=100
            )
            visualizer.add_result(result)

            # Try to generate a heatmap
            fig = visualizer.plot_heatmap(metric='accuracy', save=True)

            # Check if file was created
            files = os.listdir(tmpdir)
            if files:
                print(f"  [OK] Visualization generated: {files[0]}")
            else:
                print("  [WARN] No visualization file created (but no error)")

        return True
    except Exception as e:
        print(f"  [FAIL] Visualization: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("BLUQ Benchmark Setup Verification")
    print("="*60)

    results = []

    results.append(("Imports", check_imports()))
    results.append(("Project Modules", check_project_imports()))
    results.append(("Device Info", check_device_info()))
    results.append(("Dataset Loading", check_dataset_loading()))
    results.append(("Visualization", check_visualization()))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll checks passed! You can run the benchmark.")
        print("\nQuick test command:")
        print("  python run_full_benchmark.py --mode short --tasks qa --models tinyllama-1.1b")
        return 0
    else:
        print("\nSome checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
