"""
Evaluator Module
Main orchestrator for evaluating LLMs with uncertainty quantification.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np

from src.data.dataset_loader import TaskDataset
from src.data.data_splitter import DataSplit
from src.models.model_loader import ModelInfo
from src.models.inference_engine import InferenceEngine, BatchInferenceResult
from src.models.probability_extractor import ProbabilityExtractor, ExtractedProbabilities
from src.conformal.prediction_set_generator import (
    PredictionSetGenerator,
    PredictionSetGeneratorConfig
)
from src.conformal.conformal_base import ConformalPredictionResult
from src.evaluation.metrics import MetricsCalculator, EvaluationMetrics

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    # Conformal prediction settings
    alpha: float = 0.1
    conformal_methods: List[str] = field(default_factory=lambda: ['lac', 'aps'])
    
    # Prompting settings
    prompting_strategies: List[str] = field(
        default_factory=lambda: ['base', 'shared_instruction', 'task_specific']
    )
    
    # Aggregation
    aggregate_across_prompts: bool = True
    aggregate_across_methods: bool = True
    
    # Computation settings
    batch_size: int = 1
    
    # Output settings
    save_predictions: bool = True
    save_intermediate: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'alpha': self.alpha,
            'conformal_methods': self.conformal_methods,
            'prompting_strategies': self.prompting_strategies,
            'aggregate_across_prompts': self.aggregate_across_prompts,
            'aggregate_across_methods': self.aggregate_across_methods,
            'batch_size': self.batch_size,
            'save_predictions': self.save_predictions,
            'save_intermediate': self.save_intermediate
        }


@dataclass
class TaskEvaluationResult:
    """Results from evaluating a single task."""
    task_name: str
    model_name: str
    
    # Results by prompting strategy and conformal method
    results_by_strategy: Dict[str, Dict[str, Any]]  # strategy -> method -> result
    
    # Aggregated results
    aggregated_metrics: EvaluationMetrics
    
    # Metadata
    num_calibration: int
    num_test: int
    evaluation_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'task_name': self.task_name,
            'model_name': self.model_name,
            'results_by_strategy': self.results_by_strategy,
            'aggregated_metrics': self.aggregated_metrics.to_dict(),
            'num_calibration': self.num_calibration,
            'num_test': self.num_test,
            'evaluation_time': self.evaluation_time
        }


@dataclass
class ModelEvaluationResult:
    """Results from evaluating a model across all tasks."""
    model_name: str
    task_results: Dict[str, TaskEvaluationResult]  # task_name -> result
    
    # Average metrics across tasks
    average_metrics: EvaluationMetrics
    
    # Total evaluation time
    total_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'task_results': {
                task_name: result.to_dict()
                for task_name, result in self.task_results.items()
            },
            'average_metrics': self.average_metrics.to_dict(),
            'total_time': self.total_time
        }


class TaskEvaluator:
    """Evaluator for a single task."""
    
    def __init__(
        self,
        task_name: str,
        config: Optional[EvaluationConfig] = None
    ):
        """
        Initialize task evaluator.
        
        Args:
            task_name: Name of the task
            config: Evaluation configuration
        """
        self.task_name = task_name
        self.config = config or EvaluationConfig()
        
        logger.info(f"Initialized TaskEvaluator for {task_name}")
    
    def evaluate(
        self,
        model_name: str,
        data_split: DataSplit,
        inference_engine: InferenceEngine,
        probability_extractor: ProbabilityExtractor,
        prompts_dict: Dict[str, List[str]]  # strategy -> list of prompts
    ) -> TaskEvaluationResult:
        """
        Evaluate a model on this task.
        
        Args:
            model_name: Name of the model
            data_split: Data split with calibration and test sets
            inference_engine: Inference engine for the model
            probability_extractor: Probability extractor
            prompts_dict: Dictionary mapping strategy names to prompt lists
            
        Returns:
            TaskEvaluationResult
        """
        start_time = time.time()
        
        logger.info(f"Evaluating {model_name} on {self.task_name}...")
        
        # Results storage
        results_by_strategy = {}
        
        # Get option letters
        option_letters = ['A', 'B', 'C', 'D', 'E', 'F']
        
        # Process each prompting strategy
        for strategy_name, prompts in prompts_dict.items():
            logger.info(f"  Processing prompting strategy: {strategy_name}")
            
            # Split prompts into calibration and test
            n_cal = len(data_split.calibration.instances)
            n_test = len(data_split.test.instances)
            
            cal_prompts = prompts[:n_cal]
            test_prompts = prompts[n_cal:n_cal + n_test]
            
            # Get true labels
            cal_labels = np.array([
                option_letters.index(inst.answer)
                for inst in data_split.calibration.instances
            ])
            test_labels = np.array([
                option_letters.index(inst.answer)
                for inst in data_split.test.instances
            ])
            
            # Run inference on calibration set
            logger.info(f"    Running inference on calibration set ({n_cal} samples)...")
            cal_inference = inference_engine.infer_batch(
                prompts=cal_prompts,
                instance_ids=[inst.id for inst in data_split.calibration.instances],
                show_progress=False
            )
            
            # Extract probabilities from calibration
            cal_extracted = probability_extractor.extract_batch(cal_inference)
            cal_probabilities = np.array([ep.probabilities for ep in cal_extracted])
            
            # Run inference on test set
            logger.info(f"    Running inference on test set ({n_test} samples)...")
            test_inference = inference_engine.infer_batch(
                prompts=test_prompts,
                instance_ids=[inst.id for inst in data_split.test.instances],
                show_progress=False
            )
            
            # Extract probabilities from test
            test_extracted = probability_extractor.extract_batch(test_inference)
            test_probabilities = np.array([ep.probabilities for ep in test_extracted])
            
            # Generate prediction sets using conformal prediction
            logger.info(f"    Generating prediction sets...")
            
            ps_config = PredictionSetGeneratorConfig(
                methods=self.config.conformal_methods,
                alpha=self.config.alpha,
                aggregation='separate',  # Get results for each method separately
                random_seed=42
            )
            
            ps_generator = PredictionSetGenerator(ps_config)
            
            # Calibrate
            ps_generator.calibrate(
                calibration_probabilities=cal_probabilities,
                calibration_labels=cal_labels,
                option_letters=option_letters
            )
            
            # Generate prediction sets
            conformal_results = ps_generator.generate(
                test_probabilities=test_probabilities,
                test_labels=test_labels,
                option_letters=option_letters,
                instance_ids=[inst.id for inst in data_split.test.instances]
            )
            
            # Compute metrics for each conformal method
            strategy_results = {}
            
            for method_name, cp_result in conformal_results.items():
                logger.info(f"    Computing metrics for {method_name.upper()}...")
                
                # Convert prediction sets to list format
                prediction_sets = [
                    [option_letters.index(opt) for opt in ps.options]
                    for ps in cp_result.prediction_sets
                ]
                
                # Compute all metrics
                metrics = MetricsCalculator.compute_all(
                    probabilities=test_probabilities,
                    true_labels=test_labels,
                    prediction_sets=prediction_sets,
                    class_names=option_letters,
                    compute_calibration=True
                )
                
                strategy_results[method_name] = {
                    'conformal_result': cp_result,
                    'metrics': metrics,
                    'probabilities': test_probabilities,
                    'predictions': np.argmax(test_probabilities, axis=1),
                    'true_labels': test_labels
                }
            
            results_by_strategy[strategy_name] = strategy_results
        
        # Aggregate results
        aggregated_metrics = self._aggregate_results(results_by_strategy)
        
        evaluation_time = time.time() - start_time
        
        logger.info(f"Evaluation complete for {self.task_name} in {evaluation_time:.2f}s")
        
        return TaskEvaluationResult(
            task_name=self.task_name,
            model_name=model_name,
            results_by_strategy=results_by_strategy,
            aggregated_metrics=aggregated_metrics,
            num_calibration=n_cal,
            num_test=n_test,
            evaluation_time=evaluation_time
        )
    
    def _aggregate_results(
        self,
        results_by_strategy: Dict[str, Dict[str, Any]]
    ) -> EvaluationMetrics:
        """
        Aggregate results across strategies and methods.
        
        Args:
            results_by_strategy: Results dictionary
            
        Returns:
            Aggregated EvaluationMetrics
        """
        # Collect all metrics
        all_metrics = []
        
        for strategy_results in results_by_strategy.values():
            for method_results in strategy_results.values():
                all_metrics.append(method_results['metrics'])
        
        # Compute averages
        avg_accuracy = np.mean([m.accuracy for m in all_metrics])
        avg_set_size = np.mean([m.set_size for m in all_metrics])
        avg_coverage = np.mean([m.coverage_rate for m in all_metrics])
        
        # Optional metrics
        avg_ece = None
        if all(m.ece is not None for m in all_metrics):
            avg_ece = np.mean([m.ece for m in all_metrics])
        
        avg_entropy = None
        if all(m.mean_entropy is not None for m in all_metrics):
            avg_entropy = np.mean([m.mean_entropy for m in all_metrics])
        
        avg_confidence = None
        if all(m.mean_confidence is not None for m in all_metrics):
            avg_confidence = np.mean([m.mean_confidence for m in all_metrics])
        
        return EvaluationMetrics(
            accuracy=avg_accuracy,
            set_size=avg_set_size,
            coverage_rate=avg_coverage,
            ece=avg_ece,
            mean_entropy=avg_entropy,
            mean_confidence=avg_confidence
        )


class ModelEvaluator:
    """Evaluator for a model across multiple tasks."""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize model evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config or EvaluationConfig()
        logger.info("Initialized ModelEvaluator")
    
    def evaluate(
        self,
        model_name: str,
        task_data_splits: Dict[str, DataSplit],
        inference_engine: InferenceEngine,
        probability_extractor: ProbabilityExtractor,
        prompts_by_task: Dict[str, Dict[str, List[str]]]
    ) -> ModelEvaluationResult:
        """
        Evaluate a model across multiple tasks.
        
        Args:
            model_name: Name of the model
            task_data_splits: Dictionary mapping task names to data splits
            inference_engine: Inference engine
            probability_extractor: Probability extractor
            prompts_by_task: Nested dict: task -> strategy -> prompts
            
        Returns:
            ModelEvaluationResult
        """
        start_time = time.time()
        
        logger.info(f"Evaluating model: {model_name}")
        logger.info(f"  Tasks: {list(task_data_splits.keys())}")
        
        task_results = {}
        
        # Evaluate each task
        for task_name, data_split in task_data_splits.items():
            logger.info(f"\n{'='*80}")
            logger.info(f"Task: {task_name.upper()}")
            logger.info(f"{'='*80}")
            
            task_evaluator = TaskEvaluator(task_name, self.config)
            
            task_result = task_evaluator.evaluate(
                model_name=model_name,
                data_split=data_split,
                inference_engine=inference_engine,
                probability_extractor=probability_extractor,
                prompts_dict=prompts_by_task[task_name]
            )
            
            task_results[task_name] = task_result
            
            # Print summary
            logger.info(f"\nTask {task_name} Results:")
            logger.info(f"  Accuracy: {task_result.aggregated_metrics.accuracy:.2%}")
            logger.info(f"  Avg Set Size: {task_result.aggregated_metrics.set_size:.2f}")
            logger.info(f"  Coverage Rate: {task_result.aggregated_metrics.coverage_rate:.2%}")
        
        # Compute average metrics across tasks
        average_metrics = self._compute_average_metrics(task_results)
        
        total_time = time.time() - start_time
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Model Evaluation Complete: {model_name}")
        logger.info(f"{'='*80}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"\nAverage Metrics Across Tasks:")
        logger.info(f"  Accuracy: {average_metrics.accuracy:.2%}")
        logger.info(f"  Avg Set Size: {average_metrics.set_size:.2f}")
        logger.info(f"  Coverage Rate: {average_metrics.coverage_rate:.2%}")
        
        return ModelEvaluationResult(
            model_name=model_name,
            task_results=task_results,
            average_metrics=average_metrics,
            total_time=total_time
        )
    
    def _compute_average_metrics(
        self,
        task_results: Dict[str, TaskEvaluationResult]
    ) -> EvaluationMetrics:
        """
        Compute average metrics across all tasks.
        
        Args:
            task_results: Dictionary of task results
            
        Returns:
            Average EvaluationMetrics
        """
        metrics_list = [result.aggregated_metrics for result in task_results.values()]
        
        avg_accuracy = np.mean([m.accuracy for m in metrics_list])
        avg_set_size = np.mean([m.set_size for m in metrics_list])
        avg_coverage = np.mean([m.coverage_rate for m in metrics_list])
        
        avg_ece = None
        if all(m.ece is not None for m in metrics_list):
            avg_ece = np.mean([m.ece for m in metrics_list])
        
        avg_entropy = None
        if all(m.mean_entropy is not None for m in metrics_list):
            avg_entropy = np.mean([m.mean_entropy for m in metrics_list])
        
        avg_confidence = None
        if all(m.mean_confidence is not None for m in metrics_list):
            avg_confidence = np.mean([m.mean_confidence for m in metrics_list])
        
        return EvaluationMetrics(
            accuracy=avg_accuracy,
            set_size=avg_set_size,
            coverage_rate=avg_coverage,
            ece=avg_ece,
            mean_entropy=avg_entropy,
            mean_confidence=avg_confidence
        )
    
    def save_results(
        self,
        result: ModelEvaluationResult,
        output_dir: str
    ) -> None:
        """
        Save evaluation results.
        
        Args:
            result: ModelEvaluationResult to save
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        result_file = output_path / f"{result.model_name}_results.json"
        with open(result_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        logger.info(f"Saved results to {result_file}")
        
        # Save summary
        summary_file = output_path / f"{result.model_name}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Model: {result.model_name}\n")
            f.write(f"{'='*80}\n\n")
            
            f.write("Average Metrics Across Tasks:\n")
            f.write(result.average_metrics.get_summary())
            f.write("\n\n")
            
            f.write("Per-Task Results:\n")
            f.write("-" * 80 + "\n")
            for task_name, task_result in result.task_results.items():
                f.write(f"\n{task_name.upper()}:\n")
                f.write(task_result.aggregated_metrics.get_summary())
                f.write("\n")
        
        logger.info(f"Saved summary to {summary_file}")


# Example usage
if __name__ == "__main__":
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*80)
    print("Evaluator Module Test")
    print("="*80)
    
    # This module orchestrates the entire evaluation pipeline
    # It requires properly initialized components from other modules
    
    print("\nThe evaluator module provides:")
    print("  - TaskEvaluator: Evaluates a single task")
    print("  - ModelEvaluator: Evaluates a model across multiple tasks")
    print("  - Integration of all components:")
    print("    * Data loading and splitting")
    print("    * Model inference")
    print("    * Probability extraction")
    print("    * Conformal prediction")
    print("    * Metrics computation")
    
    print("\nFor a complete end-to-end example, see the main benchmarking script.")