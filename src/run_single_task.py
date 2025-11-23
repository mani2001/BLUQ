"""
run_single_task.py
Script for evaluating models on a single task (useful for debugging and testing).
"""

import logging
import argparse
from pathlib import Path

from src.data.dataset_loader import DatasetLoaderFactory
from src.data.dataset_processor import DatasetProcessor
from src.data.data_splitter import DataSplitter
from src.data.dataset_config import DefaultTaskConfigs

from src.models.model_loader import ModelLoader
from src.models.model_config import DefaultModelConfigs
from src.models.inference_engine import InferenceEngine
from src.models.probability_extractor import ProbabilityExtractor

from src.prompting.prompt_builder import PromptBuilder
from src.prompting.demonstration_manager import DemonstrationManager

from src.conformal.prediction_set_generator import PredictionSetGenerator, PredictionSetGeneratorConfig
from src.evaluation.metrics import MetricsCalculator

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a single model on a single task"
    )
    
    parser.add_argument(
        '--task',
        type=str,
        required=True,
        choices=['qa', 'rc', 'ci', 'drs', 'ds'],
        help='Task to evaluate'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name (e.g., tinyllama-1.1b, phi-2)'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of samples to use'
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        default='base',
        choices=['base', 'shared_instruction', 'task_specific'],
        help='Prompting strategy'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.1,
        help='Error rate for conformal prediction'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results/single_task',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info(f"Single Task Evaluation: {args.task.upper()} on {args.model}")
    logger.info("="*80)
    
    # Load dataset
    logger.info(f"\nLoading {args.task} dataset...")
    loader = DatasetLoaderFactory.create_loader(args.task)
    dataset = loader.load(num_samples=args.num_samples)
    
    # Process dataset
    logger.info("Processing dataset...")
    processor = DatasetProcessor()
    processed = processor.process_dataset(dataset)
    
    # Split dataset
    logger.info("Splitting dataset...")
    splitter = DataSplitter()
    split = splitter.split_dataset(processed, calibration_ratio=0.5)
    
    # Select demonstrations
    logger.info("Selecting demonstrations...")
    task_config = DefaultTaskConfigs.get_all_configs()[args.task]
    demo_manager = DemonstrationManager()
    demonstrations = demo_manager.get_demonstrations(
        task_name=args.task,
        dataset=processed,
        num_demonstrations=task_config.num_demonstrations
    )
    
    # Load model
    logger.info(f"\nLoading model: {args.model}")
    model_loader = ModelLoader()
    model_config = DefaultModelConfigs.create_pipeline_config(args.model)
    
    model, tokenizer, info = model_loader.load_model(model_config.load_config)
    
    # Build prompts
    logger.info(f"Building prompts with '{args.strategy}' strategy...")
    prompt_builder = PromptBuilder(task_type=args.task)
    
    all_instances = split.calibration.instances + split.test.instances
    prompts = prompt_builder.build_all_prompts(
        instances=all_instances,
        strategy=args.strategy,
        demonstrations=demonstrations
    )
    
    n_cal = len(split.calibration.instances)
    cal_prompts = prompts[:n_cal]
    test_prompts = prompts[n_cal:]
    
    # Run inference
    logger.info("\nRunning inference...")
    inference_engine = InferenceEngine(model, tokenizer, info)
    
    logger.info("  Calibration set...")
    cal_results = inference_engine.infer_batch(cal_prompts, show_progress=True)
    
    logger.info("  Test set...")
    test_results = inference_engine.infer_batch(test_prompts, show_progress=True)
    
    # Extract probabilities
    logger.info("Extracting probabilities...")
    prob_extractor = ProbabilityExtractor()
    
    cal_probs = prob_extractor.extract_batch(cal_results)
    test_probs = prob_extractor.extract_batch(test_results)
    
    import numpy as np
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
    
    # Run conformal prediction
    logger.info("\nRunning conformal prediction...")
    ps_config = PredictionSetGeneratorConfig(
        methods=['lac', 'aps'],
        alpha=args.alpha,
        aggregation='separate'
    )
    
    ps_generator = PredictionSetGenerator(ps_config)
    ps_generator.calibrate(cal_prob_array, cal_labels, option_letters)
    
    conformal_results = ps_generator.generate(
        test_probabilities=test_prob_array,
        test_labels=test_labels,
        option_letters=option_letters
    )
    
    # Compute metrics
    logger.info("\nComputing metrics...")
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    for method_name, cp_result in conformal_results.items():
        print(f"\n{method_name.upper()}:")
        print(f"  Coverage Rate: {cp_result.coverage_rate:.2%}")
        print(f"  Average Set Size: {cp_result.average_set_size:.2f}")
        print(f"  Meets Guarantee: {cp_result.meets_coverage_guarantee()}")
        
        # Compute accuracy
        predictions = np.argmax(test_prob_array, axis=1)
        accuracy = (predictions == test_labels).mean()
        print(f"  Accuracy: {accuracy:.2%}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ps_generator.save_results(
        conformal_results,
        str(output_dir / f"{args.model}_{args.task}_results.json")
    )
    
    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()