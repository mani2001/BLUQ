"""
Main Benchmark Script
Orchestrates the complete benchmarking pipeline for SLMs with uncertainty quantification.
"""

import logging
import argparse
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

from src.data.dataset_loader import load_all_datasets
from src.data.dataset_processor import process_all_datasets
from src.data.data_splitter import split_all_datasets
from src.data.dataset_config import DatasetConfigManager

from src.models.model_loader import ModelLoader
from src.models.model_config import ModelConfigManager
from src.models.inference_engine import InferenceEngine, InferenceConfig
from src.models.probability_extractor import ProbabilityExtractor, ProbabilityExtractionConfig

from src.prompting.prompt_builder import PromptBuilder
from src.prompting.demonstration_manager import DemonstrationManager
from src.prompting.prompt_formatter import PromptFormatter, FormattingConfig

from src.conformal.prediction_set_generator import PredictionSetGenerator, PredictionSetGeneratorConfig

from src.evaluation.evaluator import ModelEvaluator, TaskEvaluator, EvaluationConfig
from src.evaluation.result_aggregator import ResultAggregator, ResultExporter
from src.evaluation.statistical_analyzer import StatisticalAnalyzer

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BenchmarkPipeline:
    """Main pipeline for running the complete benchmark."""
    
    def __init__(
        self,
        data_config_path: Optional[str] = None,
        model_config_path: Optional[str] = None,
        output_dir: str = "./results"
    ):
        """
        Initialize the benchmark pipeline.
        
        Args:
            data_config_path: Path to data configuration YAML
            model_config_path: Path to model configuration YAML
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configurations
        if data_config_path:
            self.data_config_manager = DatasetConfigManager.load(data_config_path)
        else:
            self.data_config_manager = DatasetConfigManager()
        
        if model_config_path:
            self.model_config_manager = ModelConfigManager.load(model_config_path)
        else:
            self.model_config_manager = ModelConfigManager()
        
        # Initialize components
        self.model_loader = ModelLoader(
            cache_dir=self.model_config_manager.config.cache_dir
        )
        
        logger.info("="*80)
        logger.info("Initialized Benchmark Pipeline")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Data tasks: {self.data_config_manager.get_enabled_tasks()}")
        logger.info(f"Models: {self.model_config_manager.list_available_models()}")
    
    def run(
        self,
        tasks: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        quick_test: bool = False
    ):
        """
        Run the complete benchmark.
        
        Args:
            tasks: List of tasks to run (None = all)
            models: List of models to evaluate (None = all)
            quick_test: If True, use small samples for quick testing
        """
        start_time = time.time()
        
        logger.info("\n" + "="*80)
        logger.info("STARTING BENCHMARK")
        logger.info("="*80)
        
        # Step 1: Prepare data
        logger.info("\n" + "="*80)
        logger.info("STEP 1: DATA PREPARATION")
        logger.info("="*80)
        
        datasets, data_splits, demonstrations = self._prepare_data(tasks, quick_test)
        
        # Step 2: Prepare models
        logger.info("\n" + "="*80)
        logger.info("STEP 2: MODEL PREPARATION")
        logger.info("="*80)
        
        models_to_evaluate = models or self.model_config_manager.list_available_models()
        logger.info(f"Models to evaluate: {models_to_evaluate}")
        
        # Step 3: Run evaluation for each model
        logger.info("\n" + "="*80)
        logger.info("STEP 3: MODEL EVALUATION")
        logger.info("="*80)
        
        all_results = {}
        
        for model_name in models_to_evaluate:
            logger.info(f"\n{'#'*80}")
            logger.info(f"EVALUATING MODEL: {model_name}")
            logger.info(f"{'#'*80}")
            
            try:
                # Evaluate model
                model_result = self._evaluate_model(
                    model_name=model_name,
                    data_splits=data_splits,
                    demonstrations=demonstrations
                )
                
                all_results[model_name] = model_result
                
                # Save intermediate results
                self._save_model_results(model_result)
                
                # Unload model to free memory
                self.model_loader.unload_model(model_name)
                logger.info(f"Unloaded {model_name} from memory")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}", exc_info=True)
                continue
        
        # Step 4: Aggregate and analyze results
        logger.info("\n" + "="*80)
        logger.info("STEP 4: RESULT AGGREGATION AND ANALYSIS")
        logger.info("="*80)
        
        self._aggregate_and_analyze(all_results)
        
        # Complete
        total_time = time.time() - start_time
        
        logger.info("\n" + "="*80)
        logger.info("BENCHMARK COMPLETE")
        logger.info("="*80)
        logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        logger.info(f"Results saved to: {self.output_dir}")
    
    def _prepare_data(
        self,
        tasks: Optional[List[str]],
        quick_test: bool
    ) -> Tuple[Dict, Dict, Dict]:
        """Prepare datasets, splits, and demonstrations."""
        # Determine tasks to load
        tasks_to_load = tasks or self.data_config_manager.get_enabled_tasks()
        
        # Determine number of samples
        num_samples = 100 if quick_test else 10000
        
        logger.info(f"Loading {len(tasks_to_load)} tasks with {num_samples} samples each...")
        
        # Load datasets
        datasets = load_all_datasets(
            tasks=tasks_to_load,
            num_samples=num_samples,
            cache_dir=self.data_config_manager.config.cache_dir,
            seed=self.data_config_manager.config.seed
        )
        
        # Process datasets to 6-option format
        logger.info("Processing datasets to 6-option format...")
        processed_datasets = process_all_datasets(
            datasets=datasets,
            seed=self.data_config_manager.config.seed,
            validate=True
        )
        
        # Split datasets
        logger.info("Splitting datasets into calibration and test sets...")
        data_splits = split_all_datasets(
            datasets=processed_datasets,
            calibration_ratio=self.data_config_manager.config.calibration_ratio,
            stratify_by_answer=self.data_config_manager.config.stratify_split,
            seed=self.data_config_manager.config.seed
        )
        
        # Select demonstrations
        logger.info("Selecting demonstrations for in-context learning...")
        demo_manager = DemonstrationManager()
        
        demonstrations = {}
        for task_name, dataset in processed_datasets.items():
            task_config = self.data_config_manager.get_task_config(task_name)
            demos = demo_manager.get_demonstrations(
                task_name=task_name,
                dataset=dataset,
                num_demonstrations=task_config.num_demonstrations
            )
            demonstrations[task_name] = demos
        
        logger.info("Data preparation complete")
        return datasets, data_splits, demonstrations
    
    def _evaluate_model(
        self,
        model_name: str,
        data_splits: Dict,
        demonstrations: Dict
    ):
        """Evaluate a single model."""
        # Load model
        logger.info(f"Loading model: {model_name}")
        model_pipeline_config = self.model_config_manager.get_model_config(model_name)
        
        model, tokenizer, model_info = self.model_loader.load_model(
            model_pipeline_config.load_config
        )
        
        # Initialize inference components
        inference_engine = InferenceEngine(
            model=model,
            tokenizer=tokenizer,
            model_info=model_info,
            config=model_pipeline_config.inference_config
        )
        
        probability_extractor = ProbabilityExtractor(
            config=model_pipeline_config.probability_config
        )
        
        # Build prompts for all tasks and strategies
        logger.info("Building prompts for all tasks and strategies...")
        prompts_by_task = self._build_all_prompts(data_splits, demonstrations)
        
        # Run evaluation
        eval_config = EvaluationConfig(
            alpha=0.1,
            conformal_methods=['lac', 'aps'],
            prompting_strategies=['base', 'shared_instruction', 'task_specific'],
            batch_size=model_pipeline_config.inference_config.batch_size
        )
        
        evaluator = ModelEvaluator(config=eval_config)
        
        result = evaluator.evaluate(
            model_name=model_name,
            task_data_splits=data_splits,
            inference_engine=inference_engine,
            probability_extractor=probability_extractor,
            prompts_by_task=prompts_by_task
        )
        
        return result
    
    def _build_all_prompts(
        self,
        data_splits: Dict,
        demonstrations: Dict
    ) -> Dict[str, Dict[str, List[str]]]:
        """Build prompts for all tasks and strategies."""
        prompts_by_task = {}
        
        strategies = ['base', 'shared_instruction', 'task_specific']
        
        for task_name, data_split in data_splits.items():
            logger.info(f"Building prompts for {task_name}...")
            
            # Get task type
            task_type = data_split.calibration.task_type
            
            # Initialize prompt builder
            prompt_builder = PromptBuilder(task_type=task_type)
            
            # Combine calibration and test instances
            all_instances = (
                data_split.calibration.instances + 
                data_split.test.instances
            )
            
            # Build prompts for each strategy
            task_prompts = {}
            for strategy in strategies:
                prompts = prompt_builder.build_all_prompts(
                    instances=all_instances,
                    strategy=strategy,
                    demonstrations=demonstrations.get(task_name, [])
                )
                task_prompts[strategy] = prompts
            
            prompts_by_task[task_name] = task_prompts
        
        return prompts_by_task
    
    def _save_model_results(self, model_result):
        """Save individual model results."""
        model_output_dir = self.output_dir / "models" / model_result.model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        evaluator = ModelEvaluator()
        evaluator.save_results(model_result, str(model_output_dir))
        
        logger.info(f"Saved results for {model_result.model_name}")
    
    def _aggregate_and_analyze(self, all_results: Dict):
        """Aggregate and analyze all results."""
        if not all_results:
            logger.warning("No results to aggregate")
            return
        
        # Extract metrics for aggregation
        results_for_aggregation = {}
        for model_name, model_result in all_results.items():
            results_for_aggregation[model_name] = {
                task_name: task_result.aggregated_metrics
                for task_name, task_result in model_result.task_results.items()
            }
        
        # Aggregate
        aggregator = ResultAggregator()
        aggregated = aggregator.aggregate(results_for_aggregation)
        
        # Create summary tables
        logger.info("Creating summary tables...")
        
        for metric in ['accuracy', 'set_size', 'coverage_rate']:
            summary_table = aggregator.create_summary_table(aggregated, metric=metric)
            
            # Save to CSV
            csv_path = self.output_dir / f"summary_{metric}.csv"
            summary_table.to_csv(csv_path)
            logger.info(f"Saved {metric} summary to {csv_path}")
            
            # Print to console
            print(f"\n{'='*80}")
            print(f"{metric.upper()} SUMMARY")
            print(f"{'='*80}")
            print(summary_table)
        
        # Create comparison report
        report = aggregator.create_comparison_report(aggregated)
        
        report_path = self.output_dir / "comparison_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(report)
        
        # Statistical analysis (only if we have multiple models)
        logger.info("Performing statistical analysis...")
        analyzer = StatisticalAnalyzer()
        
        # Analyze accuracy-uncertainty tradeoff
        tradeoff = analyzer.analyze_accuracy_uncertainty_tradeoff(results_for_aggregation)
        
        # Save analysis
        analysis_path = self.output_dir / "statistical_analysis.json"
        import json
        
        # Convert aggregated results to JSON-serializable format
        def make_json_serializable(obj):
            """Convert objects with tuple keys to JSON-serializable format."""
            if isinstance(obj, dict):
                return {
                    (str(k) if isinstance(k, tuple) else k): make_json_serializable(v)
                    for k, v in obj.items()
                }
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'to_dict'):
                return make_json_serializable(obj.to_dict())
            else:
                return obj
        
        with open(analysis_path, 'w') as f:
            json.dump({
                'accuracy_uncertainty_tradeoff': make_json_serializable(tradeoff),
                'aggregated_results': make_json_serializable(aggregated.to_dict())
            }, f, indent=2)
        
        logger.info(f"Saved statistical analysis to {analysis_path}")
        
        # Export results
        exporter = ResultExporter()
        exporter.export_to_csv(aggregated, str(self.output_dir / "csv_exports"))
        
        logger.info("Aggregation and analysis complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark Small Language Models with Uncertainty Quantification"
    )
    
    parser.add_argument(
        '--data-config',
        type=str,
        default=None,
        help='Path to data configuration YAML file'
    )
    
    parser.add_argument(
        '--model-config',
        type=str,
        default=None,
        help='Path to model configuration YAML file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=None,
        help='Tasks to evaluate (default: all). Options: qa rc ci drs ds'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        help='Models to evaluate (default: all)'
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test with 100 samples per task'
    )
    
    parser.add_argument(
        '--save-configs',
        action='store_true',
        help='Save default configurations and exit'
    )
    
    args = parser.parse_args()
    
    # Save default configs if requested
    if args.save_configs:
        logger.info("Saving default configurations...")
        
        config_dir = Path("./configs")
        config_dir.mkdir(exist_ok=True)
        
        # Save data config
        data_manager = DatasetConfigManager()
        data_manager.save(config_dir / "dataset_config.yaml", format='yaml')
        
        # Save model config
        model_manager = ModelConfigManager()
        model_manager.save(config_dir / "model_config.yaml", format='yaml')
        
        logger.info(f"Configurations saved to {config_dir}")
        return
    
    # Run benchmark
    pipeline = BenchmarkPipeline(
        data_config_path=args.data_config,
        model_config_path=args.model_config,
        output_dir=args.output_dir
    )
    
    pipeline.run(
        tasks=args.tasks,
        models=args.models,
        quick_test=args.quick_test
    )
    
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK FINISHED SUCCESSFULLY")
    logger.info("="*80)


if __name__ == "__main__":
    main()