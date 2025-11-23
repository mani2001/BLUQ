"""
Result Aggregator Module
Aggregates and analyzes evaluation results across multiple models and tasks.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from pathlib import Path
import json

from src.evaluation.metrics import EvaluationMetrics

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class AggregatedResults:
    """Container for aggregated results across models and tasks."""
    # Results matrix: model x task -> metrics
    results_matrix: pd.DataFrame
    
    # Rankings
    model_rankings: Dict[str, int]  # model -> rank (by average performance)
    task_rankings: Dict[str, Dict[str, int]]  # task -> model -> rank
    
    # Summary statistics
    summary_stats: Dict[str, Any]
    
    # Best performers
    best_overall: str  # Best model overall
    best_per_task: Dict[str, str]  # task -> best model
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'results_matrix': self.results_matrix.to_dict(),
            'model_rankings': self.model_rankings,
            'task_rankings': self.task_rankings,
            'summary_stats': self.summary_stats,
            'best_overall': self.best_overall,
            'best_per_task': self.best_per_task
        }


class ResultAggregator:
    """Aggregates results across multiple models and tasks."""
    
    def __init__(self):
        """Initialize the result aggregator."""
        logger.info("Initialized ResultAggregator")
    
    def aggregate(
        self,
        results: Dict[str, Dict[str, EvaluationMetrics]]
    ) -> AggregatedResults:
        """
        Aggregate results across models and tasks.
        
        Args:
            results: Nested dict: model_name -> task_name -> metrics
            
        Returns:
            AggregatedResults object
        """
        logger.info("Aggregating results...")
        
        # Extract all models and tasks
        models = sorted(results.keys())
        tasks = sorted(set(
            task for model_results in results.values()
            for task in model_results.keys()
        ))
        
        logger.info(f"  Models: {len(models)}")
        logger.info(f"  Tasks: {len(tasks)}")
        
        # Create results matrix
        results_matrix = self._create_results_matrix(results, models, tasks)
        
        # Compute rankings
        model_rankings = self._compute_model_rankings(results, models, tasks)
        task_rankings = self._compute_task_rankings(results, models, tasks)
        
        # Compute summary statistics
        summary_stats = self._compute_summary_stats(results, models, tasks)
        
        # Identify best performers
        best_overall = self._identify_best_overall(model_rankings)
        best_per_task = self._identify_best_per_task(task_rankings)
        
        logger.info("Aggregation complete")
        
        return AggregatedResults(
            results_matrix=results_matrix,
            model_rankings=model_rankings,
            task_rankings=task_rankings,
            summary_stats=summary_stats,
            best_overall=best_overall,
            best_per_task=best_per_task
        )
    
    def _create_results_matrix(
        self,
        results: Dict[str, Dict[str, EvaluationMetrics]],
        models: List[str],
        tasks: List[str]
    ) -> pd.DataFrame:
        """Create a results matrix as a DataFrame."""
        # Create multi-index columns: (task, metric)
        metrics_columns = ['accuracy', 'set_size', 'coverage_rate']
        
        index_tuples = []
        for task in tasks:
            for metric in metrics_columns:
                index_tuples.append((task, metric))
        
        multi_index = pd.MultiIndex.from_tuples(
            index_tuples,
            names=['Task', 'Metric']
        )
        
        # Create DataFrame
        df = pd.DataFrame(index=models, columns=multi_index)
        
        # Fill in values
        for model in models:
            for task in tasks:
                if task in results[model]:
                    metrics = results[model][task]
                    df.loc[model, (task, 'accuracy')] = metrics.accuracy
                    df.loc[model, (task, 'set_size')] = metrics.set_size
                    df.loc[model, (task, 'coverage_rate')] = metrics.coverage_rate
        
        return df
    
    def _compute_model_rankings(
        self,
        results: Dict[str, Dict[str, EvaluationMetrics]],
        models: List[str],
        tasks: List[str]
    ) -> Dict[str, int]:
        """
        Compute overall rankings for models.
        
        Models are ranked by average accuracy across tasks.
        """
        # Compute average accuracy for each model
        avg_accuracies = {}
        for model in models:
            accuracies = [
                results[model][task].accuracy
                for task in tasks
                if task in results[model]
            ]
            avg_accuracies[model] = np.mean(accuracies) if accuracies else 0.0
        
        # Rank models (higher accuracy = better rank = lower number)
        sorted_models = sorted(
            avg_accuracies.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        rankings = {
            model: rank + 1
            for rank, (model, _) in enumerate(sorted_models)
        }
        
        return rankings
    
    def _compute_task_rankings(
        self,
        results: Dict[str, Dict[str, EvaluationMetrics]],
        models: List[str],
        tasks: List[str]
    ) -> Dict[str, Dict[str, int]]:
        """
        Compute rankings for each task separately.
        
        For each task, rank models by accuracy.
        """
        task_rankings = {}
        
        for task in tasks:
            # Get accuracy for each model on this task
            task_accuracies = {}
            for model in models:
                if task in results[model]:
                    task_accuracies[model] = results[model][task].accuracy
            
            # Rank models for this task
            sorted_models = sorted(
                task_accuracies.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            rankings = {
                model: rank + 1
                for rank, (model, _) in enumerate(sorted_models)
            }
            
            task_rankings[task] = rankings
        
        return task_rankings
    
    def _compute_summary_stats(
        self,
        results: Dict[str, Dict[str, EvaluationMetrics]],
        models: List[str],
        tasks: List[str]
    ) -> Dict[str, Any]:
        """Compute summary statistics across all results."""
        all_accuracies = []
        all_set_sizes = []
        all_coverage_rates = []
        
        for model in models:
            for task in tasks:
                if task in results[model]:
                    metrics = results[model][task]
                    all_accuracies.append(metrics.accuracy)
                    all_set_sizes.append(metrics.set_size)
                    all_coverage_rates.append(metrics.coverage_rate)
        
        return {
            'accuracy': {
                'mean': float(np.mean(all_accuracies)),
                'std': float(np.std(all_accuracies)),
                'min': float(np.min(all_accuracies)),
                'max': float(np.max(all_accuracies))
            },
            'set_size': {
                'mean': float(np.mean(all_set_sizes)),
                'std': float(np.std(all_set_sizes)),
                'min': float(np.min(all_set_sizes)),
                'max': float(np.max(all_set_sizes))
            },
            'coverage_rate': {
                'mean': float(np.mean(all_coverage_rates)),
                'std': float(np.std(all_coverage_rates)),
                'min': float(np.min(all_coverage_rates)),
                'max': float(np.max(all_coverage_rates))
            },
            'num_models': len(models),
            'num_tasks': len(tasks),
            'total_evaluations': len(all_accuracies)
        }
    
    def _identify_best_overall(
        self,
        model_rankings: Dict[str, int]
    ) -> str:
        """Identify the best overall model."""
        return min(model_rankings.items(), key=lambda x: x[1])[0]
    
    def _identify_best_per_task(
        self,
        task_rankings: Dict[str, Dict[str, int]]
    ) -> Dict[str, str]:
        """Identify the best model for each task."""
        best_per_task = {}
        
        for task, rankings in task_rankings.items():
            best_model = min(rankings.items(), key=lambda x: x[1])[0]
            best_per_task[task] = best_model
        
        return best_per_task
    
    def create_summary_table(
        self,
        aggregated_results: AggregatedResults,
        metric: str = 'accuracy'
    ) -> pd.DataFrame:
        """
        Create a summary table for a specific metric.
        
        Args:
            aggregated_results: AggregatedResults object
            metric: Metric to display ('accuracy', 'set_size', 'coverage_rate')
            
        Returns:
            DataFrame with models as rows and tasks as columns
        """
        df = aggregated_results.results_matrix
        
        # Extract only the specified metric
        tasks = df.columns.get_level_values('Task').unique()
        
        summary = pd.DataFrame(index=df.index)
        
        for task in tasks:
            if (task, metric) in df.columns:
                summary[task] = df[(task, metric)]
        
        # Add average column
        summary['Average'] = summary.mean(axis=1)
        
        # Add rank column
        summary['Rank'] = summary['Average'].rank(ascending=False, method='min').astype(int)
        
        # Sort by rank
        summary = summary.sort_values('Rank')
        
        return summary
    
    def create_comparison_report(
        self,
        aggregated_results: AggregatedResults
    ) -> str:
        """
        Create a human-readable comparison report.
        
        Args:
            aggregated_results: AggregatedResults object
            
        Returns:
            Report string
        """
        lines = []
        
        lines.append("="*80)
        lines.append("MODEL COMPARISON REPORT")
        lines.append("="*80)
        
        # Best overall
        lines.append(f"\nBest Overall Model: {aggregated_results.best_overall}")
        lines.append(f"  Rank: {aggregated_results.model_rankings[aggregated_results.best_overall]}")
        
        # Best per task
        lines.append("\nBest Model Per Task:")
        for task, model in aggregated_results.best_per_task.items():
            lines.append(f"  {task.upper()}: {model}")
        
        # Summary statistics
        lines.append("\nOverall Statistics:")
        stats = aggregated_results.summary_stats
        lines.append(f"  Accuracy: {stats['accuracy']['mean']:.2%} ± {stats['accuracy']['std']:.2%}")
        lines.append(f"  Set Size: {stats['set_size']['mean']:.2f} ± {stats['set_size']['std']:.2f}")
        lines.append(f"  Coverage: {stats['coverage_rate']['mean']:.2%} ± {stats['coverage_rate']['std']:.2%}")
        
        # Rankings
        lines.append("\nModel Rankings (by average accuracy):")
        sorted_rankings = sorted(
            aggregated_results.model_rankings.items(),
            key=lambda x: x[1]
        )
        for model, rank in sorted_rankings:
            lines.append(f"  {rank}. {model}")
        
        lines.append("\n" + "="*80)
        
        return "\n".join(lines)


class ComparisonAnalyzer:
    """Analyzer for comparing specific aspects of results."""
    
    @staticmethod
    def compare_base_vs_instruct(
        results: Dict[str, Dict[str, EvaluationMetrics]],
        base_instruct_pairs: List[Tuple[str, str]]
    ) -> pd.DataFrame:
        """
        Compare base models with their instruct-tuned versions.
        
        Args:
            results: Results dictionary
            base_instruct_pairs: List of (base_model, instruct_model) tuples
            
        Returns:
            DataFrame with comparison results
        """
        comparisons = []
        
        for base_model, instruct_model in base_instruct_pairs:
            if base_model not in results or instruct_model not in results:
                continue
            
            # Get all common tasks
            common_tasks = set(results[base_model].keys()) & set(results[instruct_model].keys())
            
            for task in common_tasks:
                base_metrics = results[base_model][task]
                instruct_metrics = results[instruct_model][task]
                
                comparisons.append({
                    'model_family': base_model.split('-')[0],
                    'task': task,
                    'base_accuracy': base_metrics.accuracy,
                    'instruct_accuracy': instruct_metrics.accuracy,
                    'accuracy_diff': instruct_metrics.accuracy - base_metrics.accuracy,
                    'base_set_size': base_metrics.set_size,
                    'instruct_set_size': instruct_metrics.set_size,
                    'set_size_diff': instruct_metrics.set_size - base_metrics.set_size
                })
        
        return pd.DataFrame(comparisons)
    
    @staticmethod
    def compare_model_scales(
        results: Dict[str, Dict[str, EvaluationMetrics]],
        model_families: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """
        Compare models of different scales within the same family.
        
        Args:
            results: Results dictionary
            model_families: Dict mapping family name to list of model names
            
        Returns:
            DataFrame with scale comparison results
        """
        comparisons = []
        
        for family, models in model_families.items():
            # Get all tasks covered by all models in family
            all_tasks = set()
            for model in models:
                if model in results:
                    all_tasks.update(results[model].keys())
            
            for task in all_tasks:
                family_results = []
                
                for model in models:
                    if model in results and task in results[model]:
                        metrics = results[model][task]
                        
                        # Extract model size from name (e.g., "1.8b" from "qwen-1.8b")
                        size_str = model.split('-')[-1]
                        
                        family_results.append({
                            'family': family,
                            'task': task,
                            'model': model,
                            'size': size_str,
                            'accuracy': metrics.accuracy,
                            'set_size': metrics.set_size,
                            'coverage_rate': metrics.coverage_rate
                        })
                
                comparisons.extend(family_results)
        
        return pd.DataFrame(comparisons)
    
    @staticmethod
    def analyze_uncertainty_vs_accuracy(
        results: Dict[str, Dict[str, EvaluationMetrics]]
    ) -> pd.DataFrame:
        """
        Analyze the relationship between accuracy and uncertainty (set size).
        
        Args:
            results: Results dictionary
            
        Returns:
            DataFrame with correlation analysis
        """
        data = []
        
        for model, task_results in results.items():
            for task, metrics in task_results.items():
                data.append({
                    'model': model,
                    'task': task,
                    'accuracy': metrics.accuracy,
                    'set_size': metrics.set_size,
                    'coverage_rate': metrics.coverage_rate
                })
        
        df = pd.DataFrame(data)
        
        # Compute correlation
        if len(df) > 0:
            corr = df[['accuracy', 'set_size']].corr().iloc[0, 1]
            logger.info(f"Accuracy-SetSize correlation: {corr:.4f}")
        
        return df


class ResultExporter:
    """Export results in various formats."""
    
    @staticmethod
    def export_to_csv(
        aggregated_results: AggregatedResults,
        output_dir: str
    ) -> None:
        """
        Export results to CSV files.
        
        Args:
            aggregated_results: AggregatedResults to export
            output_dir: Directory to save CSV files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export results matrix
        matrix_file = output_path / "results_matrix.csv"
        aggregated_results.results_matrix.to_csv(matrix_file)
        logger.info(f"Exported results matrix to {matrix_file}")
        
        # Export rankings
        rankings_file = output_path / "model_rankings.csv"
        rankings_df = pd.DataFrame.from_dict(
            aggregated_results.model_rankings,
            orient='index',
            columns=['Rank']
        )
        rankings_df.to_csv(rankings_file)
        logger.info(f"Exported rankings to {rankings_file}")
    
    @staticmethod
    def export_to_latex(
        summary_table: pd.DataFrame,
        output_file: str,
        caption: str = "Model Comparison Results"
    ) -> None:
        """
        Export summary table to LaTeX format.
        
        Args:
            summary_table: Summary table DataFrame
            output_file: Output file path
            caption: Table caption
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to LaTeX
        latex_str = summary_table.to_latex(
            float_format="%.2f",
            caption=caption,
            label="tab:results"
        )
        
        with open(output_path, 'w') as f:
            f.write(latex_str)
        
        logger.info(f"Exported LaTeX table to {output_path}")


# Example usage
if __name__ == "__main__":
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*80)
    print("Result Aggregator Test")
    print("="*80)
    
    # Create sample results
    from src.evaluation.metrics import EvaluationMetrics
    
    # Simulate results for 3 models and 2 tasks
    results = {
        'model-a': {
            'qa': EvaluationMetrics(accuracy=0.75, set_size=2.1, coverage_rate=0.92),
            'rc': EvaluationMetrics(accuracy=0.82, set_size=1.8, coverage_rate=0.93)
        },
        'model-b': {
            'qa': EvaluationMetrics(accuracy=0.71, set_size=2.3, coverage_rate=0.91),
            'rc': EvaluationMetrics(accuracy=0.79, set_size=2.0, coverage_rate=0.92)
        },
        'model-c': {
            'qa': EvaluationMetrics(accuracy=0.78, set_size=2.0, coverage_rate=0.93),
            'rc': EvaluationMetrics(accuracy=0.85, set_size=1.7, coverage_rate=0.94)
        }
    }
    
    # Aggregate results
    print("\nAggregating results...")
    aggregator = ResultAggregator()
    aggregated = aggregator.aggregate(results)
    
    print(f"\nBest overall model: {aggregated.best_overall}")
    print(f"Model rankings: {aggregated.model_rankings}")
    print(f"\nBest per task:")
    for task, model in aggregated.best_per_task.items():
        print(f"  {task}: {model}")
    
    # Create summary table
    print("\n" + "="*80)
    print("Accuracy Summary Table")
    print("="*80)
    summary = aggregator.create_summary_table(aggregated, metric='accuracy')
    print(summary)
    
    # Create comparison report
    print("\n" + "="*80)
    report = aggregator.create_comparison_report(aggregated)
    print(report)
    
    # Export results
    print("\n" + "="*80)
    print("Exporting results...")
    print("="*80)
    
    output_dir = Path("./results/aggregated")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exporter = ResultExporter()
    exporter.export_to_csv(aggregated, str(output_dir))
    exporter.export_to_latex(summary, str(output_dir / "summary.tex"))
    
    print(f"\nResults exported to {output_dir}")