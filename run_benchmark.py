#!/usr/bin/env python3
"""
Full Benchmark Runner
Runs comprehensive benchmark across multiple models, tasks, and data types.
Follows the methodology from "Benchmarking LLMs via Uncertainty Quantification" (Ye et al., 2024).
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def setup_logging(output_dir: str, run_timestamp: str) -> Path:
    """
    Setup logging to both console and file.

    Args:
        output_dir: Directory to store log files
        run_timestamp: Timestamp string for the log filename

    Returns:
        Path to the log file
    """
    # Create logs directory
    logs_dir = Path(output_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / f"benchmark_{run_timestamp}.log"

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear any existing handlers
    root_logger.handlers = []

    # File handler - captures everything
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    return log_file


# Temporary basic logging until setup_logging is called
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Available configurations
AVAILABLE_MODELS = [
    'tinyllama-1.1b',
    'phi-2',
    'stablelm-2-1.6b',
    'gemma-2b',
    'qwen-1.8b',
]

AVAILABLE_TASKS = ['qa', 'rc', 'ci', 'drs', 'ds']

TASK_FULL_NAMES = {
    'qa': 'Question Answering (MMLU)',
    'rc': 'Reading Comprehension (CosmosQA)',
    'ci': 'Commonsense Inference (HellaSwag)',
    'drs': 'Dialogue Response Selection (HaluDial)',
    'ds': 'Document Summarization (HaluSum)',
}

DTYPES = ['float16', 'float32']


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark run."""
    models: List[str]
    tasks: List[str]
    dtypes: List[str]
    num_samples: int
    alpha: float
    calibration_ratio: float
    output_dir: str
    strategies: List[str]
    conformal_methods: List[str]
    seed: int
    use_dynamic_batch_size: bool
    max_batch_size: Optional[int]  # None = auto-detect from GPU tier
    resume: bool = False  # Resume from checkpoint if available
    enable_profiling: bool = True  # Enable GPU profiling

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SingleRunResult:
    """Result from a single model-task-dtype run."""
    model: str
    task: str
    dtype: str
    strategy: str
    conformal_method: str
    accuracy: float
    coverage_rate: float
    avg_set_size: float
    meets_guarantee: bool
    inference_time: float
    num_samples: int


class FullBenchmarkRunner:
    """Runs the complete benchmark suite."""

    CHECKPOINT_FILE = "checkpoint.json"

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.results: List[SingleRunResult] = []
        self.completed_combinations: set = set()  # Track completed (model, task, dtype) tuples

        # Check for resume
        if config.resume:
            self._load_checkpoint()
            if self.completed_combinations:
                logger.info(f"Resuming from checkpoint: {len(self.completed_combinations)} combinations already completed")
        else:
            self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize components
        self._init_components()

    def _get_checkpoint_path(self) -> Path:
        """Get path to checkpoint file."""
        return self.output_dir / self.CHECKPOINT_FILE

    def _save_checkpoint(self):
        """Save checkpoint with current progress."""
        checkpoint = {
            'timestamp': self.run_timestamp,
            'completed_combinations': list(self.completed_combinations),
            'results': [asdict(r) for r in self.results],
            'config': self.config.to_dict()
        }

        checkpoint_path = self._get_checkpoint_path()
        # Write to temp file first, then rename for atomicity
        temp_path = checkpoint_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        temp_path.replace(checkpoint_path)

        logger.debug(f"Checkpoint saved: {len(self.completed_combinations)} combinations completed")

    def _load_checkpoint(self):
        """Load checkpoint if it exists."""
        checkpoint_path = self._get_checkpoint_path()

        if not checkpoint_path.exists():
            logger.info("No checkpoint found, starting fresh")
            self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return

        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)

            self.run_timestamp = checkpoint.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
            self.completed_combinations = set(tuple(c) for c in checkpoint.get('completed_combinations', []))

            # Restore results
            for r in checkpoint.get('results', []):
                self.results.append(SingleRunResult(**r))

            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            logger.info(f"  Timestamp: {self.run_timestamp}")
            logger.info(f"  Completed: {len(self.completed_combinations)} combinations")
            logger.info(f"  Results: {len(self.results)} entries")

        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
            self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.completed_combinations = set()
            self.results = []

    def _is_combination_completed(self, model: str, task: str, dtype: str) -> bool:
        """Check if a combination has already been completed."""
        return (model, task, dtype) in self.completed_combinations

    def _mark_combination_completed(self, model: str, task: str, dtype: str):
        """Mark a combination as completed."""
        self.completed_combinations.add((model, task, dtype))

    def _init_components(self):
        """Initialize benchmark components."""
        from src.utils.gpu import GPUMemoryManager, get_device_info, get_gpu_config, GPUProfiler

        # Get device info
        self.device_info = get_device_info()
        logger.info(f"Running on: {self.device_info}")

        # Get GPU tier config for auto-detection
        self.gpu_config = get_gpu_config(self.device_info)
        logger.info(
            f"GPU Tier: {self.gpu_config['tier']} | "
            f"Auto max_batch_size: {self.gpu_config['max_batch_size']} | "
            f"Safety margin: {self.gpu_config['safety_margin']}"
        )

        # Resolve max_batch_size (auto-detect if None)
        if self.config.max_batch_size is None:
            self.effective_max_batch_size = self.gpu_config['max_batch_size']
            logger.info(f"Auto-detected max_batch_size: {self.effective_max_batch_size}")
        else:
            self.effective_max_batch_size = self.config.max_batch_size
            logger.info(f"Using user-specified max_batch_size: {self.effective_max_batch_size}")

        # Initialize GPU memory manager for dynamic batch sizing
        if self.config.use_dynamic_batch_size:
            self.gpu_manager = GPUMemoryManager()
        else:
            self.gpu_manager = None

        # Initialize GPU profiler
        if self.config.enable_profiling:
            self.profiler = GPUProfiler(monitoring_interval=2.0)
            self.profiler.start_monitoring()
            logger.info("GPU profiling enabled")
        else:
            self.profiler = None

    def get_batch_size(self, model_name: str, dtype: str) -> int:
        """Get optimal batch size for model and dtype."""
        if not self.config.use_dynamic_batch_size or self.gpu_manager is None:
            return self.effective_max_batch_size

        from src.utils.gpu import get_model_size
        model_size = get_model_size(model_name)

        batch_size = self.gpu_manager.get_optimal_batch_size(
            num_params_billions=model_size,
            dtype=dtype,
            seq_length=2048,
            min_batch_size=1,
            max_batch_size=self.effective_max_batch_size
        )
        return batch_size

    def run(self) -> Dict:
        """Run the complete benchmark."""
        start_time = time.time()

        # Print configuration
        self._print_config()

        # Save configuration
        config_path = self.output_dir / f"config_{self.run_timestamp}.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

        total_runs = (
            len(self.config.models) *
            len(self.config.tasks) *
            len(self.config.dtypes)
        )
        current_run = 0

        # Run all combinations
        skipped_count = 0
        for model_name in self.config.models:
            for dtype in self.config.dtypes:
                for task in self.config.tasks:
                    current_run += 1

                    # Check if already completed (for resume)
                    if self._is_combination_completed(model_name, task, dtype):
                        skipped_count += 1
                        logger.info(f"Skipping {current_run}/{total_runs}: {model_name} | {task} | {dtype} (already completed)")
                        continue

                    logger.info(f"\n{'='*80}")
                    logger.info(f"Run {current_run}/{total_runs}: {model_name} | {task} | {dtype}")
                    if skipped_count > 0:
                        logger.info(f"(Skipped {skipped_count} already-completed combinations)")
                    logger.info(f"{'='*80}")

                    try:
                        results = self._run_single_configuration(
                            model_name=model_name,
                            task=task,
                            dtype=dtype
                        )
                        self.results.extend(results)

                        # Mark as completed and save checkpoint
                        self._mark_combination_completed(model_name, task, dtype)
                        self._save_checkpoint()

                        # Save intermediate results
                        self._save_results()

                        logger.info(f"Checkpoint saved after completing: {model_name} | {task} | {dtype}")

                    except Exception as e:
                        logger.error(f"Failed: {model_name} | {task} | {dtype}")
                        logger.error(f"Error: {e}", exc_info=True)
                        # Still save checkpoint so we don't lose progress on other combinations
                        self._save_checkpoint()
                        continue

        # Generate summary and visualizations
        total_time = time.time() - start_time
        summary = self._generate_summary(total_time)

        # Generate visualizations
        self._generate_visualizations()

        # Stop profiling and generate report
        if self.profiler:
            self.profiler.stop_monitoring()
            self.profiler.print_summary()

            # Save profiling report
            profile_path = self.output_dir / f"gpu_profile_{self.run_timestamp}.json"
            self.profiler.save_report(str(profile_path))
            logger.info(f"GPU profiling report saved to: {profile_path}")

        logger.info(f"\n{'='*80}")
        logger.info("BENCHMARK COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total time: {total_time/60:.2f} minutes")
        logger.info(f"Results saved to: {self.output_dir}")

        return summary

    def _print_config(self):
        """Print benchmark configuration."""
        logger.info("\n" + "="*80)
        logger.info("FULL BENCHMARK CONFIGURATION")
        logger.info("="*80)
        logger.info(f"Models ({len(self.config.models)}): {self.config.models}")
        logger.info(f"Tasks ({len(self.config.tasks)}): {self.config.tasks}")
        logger.info(f"Data types: {self.config.dtypes}")
        logger.info(f"Samples per task: {self.config.num_samples}")
        logger.info(f"Alpha (error rate): {self.config.alpha}")
        logger.info(f"Strategies: {self.config.strategies}")
        logger.info(f"Conformal methods: {self.config.conformal_methods}")
        logger.info(f"Dynamic batch sizing: {self.config.use_dynamic_batch_size}")
        logger.info(f"Max batch size: {self.effective_max_batch_size} (GPU tier: {self.gpu_config['tier']})")
        logger.info(f"Output directory: {self.output_dir}")

    def _run_single_configuration(
        self,
        model_name: str,
        task: str,
        dtype: str
    ) -> List[SingleRunResult]:
        """Run benchmark for a single model-task-dtype configuration."""
        import torch
        from src.data.dataset_loader import DatasetLoaderFactory
        from src.data.dataset_processor import DatasetProcessor
        from src.data.data_splitter import DataSplitter
        from src.data.dataset_config import DefaultTaskConfigs
        from src.models.model_loader import ModelLoader, ModelLoadConfig
        from src.models.model_config import DefaultModelConfigs
        from src.models.inference_engine import InferenceEngine, InferenceConfig
        from src.models.probability_extractor import ProbabilityExtractor
        from src.prompting.prompt_builder import PromptBuilder
        from src.prompting.demonstration_manager import DemonstrationManager
        from src.conformal.prediction_set_generator import (
            PredictionSetGenerator, PredictionSetGeneratorConfig
        )

        results = []

        # Log GPU state at start
        if self.profiler:
            self.profiler.log_current_state(f"start_{model_name}_{task}")

        # Step 1: Load and process dataset
        logger.info(f"Loading {task} dataset ({self.config.num_samples} samples)...")

        if self.profiler:
            with self.profiler.track("dataset_loading", task=task, num_samples=self.config.num_samples):
                loader = DatasetLoaderFactory.create_loader(task)
                dataset = loader.load(num_samples=self.config.num_samples)
        else:
            loader = DatasetLoaderFactory.create_loader(task)
            dataset = loader.load(num_samples=self.config.num_samples)

        if self.profiler:
            with self.profiler.track("dataset_processing", task=task):
                processor = DatasetProcessor()
                processed = processor.process_dataset(dataset)
                splitter = DataSplitter()
                split = splitter.split_dataset(processed, calibration_ratio=self.config.calibration_ratio)
        else:
            processor = DatasetProcessor()
            processed = processor.process_dataset(dataset)
            splitter = DataSplitter()
            split = splitter.split_dataset(processed, calibration_ratio=self.config.calibration_ratio)

        # Step 2: Get demonstrations
        task_config = DefaultTaskConfigs.get_all_configs()[task]
        demo_manager = DemonstrationManager()
        demonstrations = demo_manager.get_demonstrations(
            task_name=task,
            dataset=processed,
            num_demonstrations=task_config.num_demonstrations
        )

        # Step 3: Load model with specified dtype
        logger.info(f"Loading model: {model_name} ({dtype})")

        if self.profiler:
            with self.profiler.track("model_loading", model=model_name, dtype=dtype):
                model_loader = ModelLoader()
                base_config = DefaultModelConfigs.create_pipeline_config(model_name)
                load_config = ModelLoadConfig(
                    model_id=base_config.load_config.model_id,
                    name=f"{model_name}_{dtype}",
                    dtype=dtype,
                    device="auto" if torch.cuda.is_available() else "cpu",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                model, tokenizer, model_info = model_loader.load_model(load_config)
            self.profiler.log_current_state(f"after_model_load_{model_name}")
        else:
            model_loader = ModelLoader()
            base_config = DefaultModelConfigs.create_pipeline_config(model_name)
            load_config = ModelLoadConfig(
                model_id=base_config.load_config.model_id,
                name=f"{model_name}_{dtype}",
                dtype=dtype,
                device="auto" if torch.cuda.is_available() else "cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            model, tokenizer, model_info = model_loader.load_model(load_config)

        # Get batch size
        batch_size = self.get_batch_size(model_name, dtype)
        logger.info(f"Using batch size: {batch_size}")

        # Step 4: Run for each strategy
        for strategy in self.config.strategies:
            logger.info(f"  Strategy: {strategy}")

            # Build prompts
            prompt_builder = PromptBuilder(task_type=task)
            all_instances = split.calibration.instances + split.test.instances

            prompts = prompt_builder.build_all_prompts(
                instances=all_instances,
                strategy=strategy,
                demonstrations=demonstrations
            )

            n_cal = len(split.calibration.instances)
            cal_prompts = prompts[:n_cal]
            test_prompts = prompts[n_cal:]

            # Run inference
            inference_config = InferenceConfig(
                batch_size=batch_size,
                max_length=2048,
                temperature=1.0
            )
            inference_engine = InferenceEngine(model, tokenizer, model_info, inference_config)

            inference_start = time.time()

            if self.profiler:
                with self.profiler.track("inference_calibration",
                                        model=model_name, strategy=strategy,
                                        num_samples=len(cal_prompts), batch_size=batch_size):
                    cal_results = inference_engine.infer_batch(cal_prompts, show_progress=True)

                with self.profiler.track("inference_test",
                                        model=model_name, strategy=strategy,
                                        num_samples=len(test_prompts), batch_size=batch_size):
                    test_results = inference_engine.infer_batch(test_prompts, show_progress=True)
            else:
                cal_results = inference_engine.infer_batch(cal_prompts, show_progress=True)
                test_results = inference_engine.infer_batch(test_prompts, show_progress=True)

            inference_time = time.time() - inference_start

            # Log throughput
            total_samples = len(cal_prompts) + len(test_prompts)
            throughput = total_samples / inference_time if inference_time > 0 else 0
            logger.info(f"    Inference: {inference_time:.2f}s for {total_samples} samples "
                       f"({throughput:.2f} samples/sec)")

            # Extract probabilities
            if self.profiler:
                with self.profiler.track("probability_extraction", model=model_name):
                    prob_extractor = ProbabilityExtractor()
                    cal_probs = prob_extractor.extract_batch(cal_results)
                    test_probs = prob_extractor.extract_batch(test_results)
            else:
                prob_extractor = ProbabilityExtractor()
                cal_probs = prob_extractor.extract_batch(cal_results)
                test_probs = prob_extractor.extract_batch(test_results)

            cal_prob_array = np.array([p.probabilities for p in cal_probs])
            test_prob_array = np.array([p.probabilities for p in test_probs])

            # Get true labels
            option_letters = ['A', 'B', 'C', 'D', 'E', 'F']
            cal_labels = np.array([
                option_letters.index(inst.answer)
                for inst in split.calibration.instances
            ])
            test_labels = np.array([
                option_letters.index(inst.answer)
                for inst in split.test.instances
            ])

            # Compute accuracy
            predictions = np.argmax(test_prob_array, axis=1)
            accuracy = (predictions == test_labels).mean()

            # Run conformal prediction
            for cp_method in self.config.conformal_methods:
                ps_config = PredictionSetGeneratorConfig(
                    methods=[cp_method],
                    alpha=self.config.alpha,
                    aggregation='separate'
                )

                ps_generator = PredictionSetGenerator(ps_config)
                ps_generator.calibrate(cal_prob_array, cal_labels, option_letters)

                cp_results = ps_generator.generate(
                    test_probabilities=test_prob_array,
                    test_labels=test_labels,
                    option_letters=option_letters
                )

                cp_result = cp_results[cp_method]

                result = SingleRunResult(
                    model=model_name,
                    task=task,
                    dtype=dtype,
                    strategy=strategy,
                    conformal_method=cp_method,
                    accuracy=float(accuracy),
                    coverage_rate=float(cp_result.coverage_rate),
                    avg_set_size=float(cp_result.average_set_size),
                    meets_guarantee=cp_result.meets_coverage_guarantee(),
                    inference_time=inference_time,
                    num_samples=len(test_labels)
                )
                results.append(result)

                logger.info(
                    f"    {cp_method.upper()}: Acc={accuracy:.2%}, "
                    f"CR={cp_result.coverage_rate:.2%}, "
                    f"SS={cp_result.average_set_size:.2f}"
                )

        # Unload model
        model_loader.unload_model(f"{model_name}_{dtype}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    def _save_results(self):
        """Save current results to file."""
        results_path = self.output_dir / f"results_{self.run_timestamp}.json"

        results_data = [asdict(r) for r in self.results]
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)

    def _generate_summary(self, total_time: float) -> Dict:
        """Generate summary statistics."""
        summary = {
            'timestamp': self.run_timestamp,
            'total_time_seconds': total_time,
            'total_runs': len(self.results),
            'models': list(set(r.model for r in self.results)),
            'tasks': list(set(r.task for r in self.results)),
            'dtypes': list(set(r.dtype for r in self.results)),
            'overall_accuracy': np.mean([r.accuracy for r in self.results]),
            'overall_coverage': np.mean([r.coverage_rate for r in self.results]),
            'overall_set_size': np.mean([r.avg_set_size for r in self.results]),
            'guarantee_met_ratio': np.mean([r.meets_guarantee for r in self.results]),
        }

        summary_path = self.output_dir / f"summary_{self.run_timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return summary

    def _generate_visualizations(self):
        """Generate visualization plots."""
        try:
            from src.visualization.result_visualizer import ResultVisualizer, BenchmarkResult

            # Convert results to BenchmarkResult format
            benchmark_results = []
            for r in self.results:
                br = BenchmarkResult(
                    model_name=r.model,
                    task_name=r.task,
                    dtype=r.dtype,
                    strategy=r.strategy,
                    conformal_method=r.conformal_method,
                    accuracy=r.accuracy,
                    coverage_rate=r.coverage_rate,
                    avg_set_size=r.avg_set_size,
                    meets_guarantee=r.meets_guarantee
                )
                benchmark_results.append(br)

            # Create visualizer and generate all plots
            figures_dir = self.output_dir / "figures"
            figures_dir.mkdir(exist_ok=True)

            visualizer = ResultVisualizer(benchmark_results, str(figures_dir))
            visualizer.generate_all_visualizations()

            logger.info(f"Visualizations saved to: {figures_dir}")

        except ImportError as e:
            logger.warning(f"Could not generate visualizations: {e}")
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run full benchmark across models, tasks, and data types"
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='short',
        choices=['short', 'long', 'custom'],
        help='Run mode: short (100 samples), long (10000 samples), or custom'
    )

    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Number of samples per task (overrides mode)'
    )

    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        help=f'Models to evaluate. Options: {AVAILABLE_MODELS}'
    )

    parser.add_argument(
        '--tasks',
        nargs='+',
        default=None,
        help=f'Tasks to evaluate. Options: {AVAILABLE_TASKS}'
    )

    parser.add_argument(
        '--dtypes',
        nargs='+',
        default=['float16'],
        choices=DTYPES,
        help='Data types to evaluate'
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=0.1,
        help='Error rate for conformal prediction (default: 0.1 for 90%% coverage)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs/results',
        help='Output directory for results'
    )

    parser.add_argument(
        '--no-dynamic-batch',
        action='store_true',
        help='Disable dynamic batch size optimization'
    )

    parser.add_argument(
        '--max-batch-size',
        type=int,
        default=None,
        help='Maximum batch size. Defaults: auto-detect based on GPU tier. '
             'Recommendations for A100 80GB with 2048 tokens: '
             '<500M params: 32, 500M-1.5B: 16, 1.5B-3B: 8, 3B-7B: 4, >7B: 2. '
             'Reduce if encountering OOM errors.'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    parser.add_argument(
        '--strategies',
        nargs='+',
        default=['base'],
        choices=['base', 'shared_instruction', 'task_specific'],
        help='Prompting strategies to evaluate'
    )

    parser.add_argument(
        '--conformal-methods',
        nargs='+',
        default=['lac', 'aps'],
        choices=['lac', 'aps'],
        help='Conformal prediction methods'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint if available. Skips already-completed (model, task, dtype) combinations.'
    )

    parser.add_argument(
        '--no-profiling',
        action='store_true',
        help='Disable GPU profiling (reduces overhead slightly)'
    )

    args = parser.parse_args()

    # Determine number of samples based on mode
    if args.num_samples is not None:
        num_samples = args.num_samples
    elif args.mode == 'short':
        num_samples = 100
    elif args.mode == 'long':
        num_samples = 10000
    else:
        num_samples = 100

    # Set models and tasks
    models = args.models if args.models else AVAILABLE_MODELS
    tasks = args.tasks if args.tasks else AVAILABLE_TASKS

    # Validate models and tasks
    for model in models:
        if model not in AVAILABLE_MODELS:
            logger.warning(f"Model '{model}' not in predefined list, attempting anyway...")

    for task in tasks:
        if task not in AVAILABLE_TASKS:
            logger.error(f"Unknown task: {task}. Available: {AVAILABLE_TASKS}")
            sys.exit(1)

    # Setup logging to file
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = setup_logging(args.output_dir, run_timestamp)

    # Create configuration
    config = BenchmarkConfig(
        models=models,
        tasks=tasks,
        dtypes=args.dtypes,
        num_samples=num_samples,
        alpha=args.alpha,
        calibration_ratio=0.5,
        output_dir=args.output_dir,
        strategies=args.strategies,
        conformal_methods=args.conformal_methods,
        seed=args.seed,
        use_dynamic_batch_size=not args.no_dynamic_batch,
        max_batch_size=args.max_batch_size,
        resume=args.resume,
        enable_profiling=not args.no_profiling
    )

    # Print and log banner
    banner = f"""
{'='*80}
BLUQ: Benchmarking Language models via Uncertainty Quantification
{'='*80}
Mode: {args.mode} ({num_samples} samples per task)
Models: {len(models)} | Tasks: {len(tasks)} | Dtypes: {len(args.dtypes)}
Total configurations: {len(models) * len(tasks) * len(args.dtypes)}
Log file: {log_file}
{'='*80}
"""
    print(banner)
    logger.info(f"Starting benchmark run")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Mode: {args.mode}, Samples: {num_samples}")
    logger.info(f"Models: {models}")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Dtypes: {args.dtypes}")

    # Run benchmark
    runner = FullBenchmarkRunner(config)
    summary = runner.run()

    # Print and log final summary
    summary_text = f"""
{'='*80}
FINAL SUMMARY
{'='*80}
Total runs: {summary['total_runs']}
Overall accuracy: {summary['overall_accuracy']:.2%}
Overall coverage: {summary['overall_coverage']:.2%}
Overall set size: {summary['overall_set_size']:.2f}
Guarantee met: {summary['guarantee_met_ratio']:.2%}
Total time: {summary['total_time_seconds']/60:.2f} minutes
Log file: {log_file}
{'='*80}
"""
    print(summary_text)
    logger.info("Benchmark completed successfully")
    logger.info(f"Total runs: {summary['total_runs']}")
    logger.info(f"Overall accuracy: {summary['overall_accuracy']:.2%}")
    logger.info(f"Overall coverage: {summary['overall_coverage']:.2%}")
    logger.info(f"Log file saved to: {log_file}")


if __name__ == "__main__":
    main()
