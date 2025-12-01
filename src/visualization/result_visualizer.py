"""
Result Visualization Module for BLUQ Benchmark
Creates comprehensive visualizations for model vs task performance matrices.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

# Visualization imports
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Setup logger
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    model_name: str
    task_name: str
    dtype: str
    strategy: str
    conformal_method: str  # 'lac' or 'aps'
    accuracy: float
    coverage_rate: float
    avg_set_size: float
    meets_guarantee: bool = True
    num_samples: int = 0
    inference_time: float = 0.0

    # Aliases for compatibility
    @property
    def task(self) -> str:
        return self.task_name

    @property
    def set_size(self) -> float:
        return self.avg_set_size


class ResultVisualizer:
    """Visualizer for BLUQ benchmark results."""

    TASKS = ['qa', 'rc', 'ci', 'drs', 'ds']
    TASK_NAMES = {
        'qa': 'Question\nAnswering',
        'rc': 'Reading\nComprehension',
        'ci': 'Commonsense\nInference',
        'drs': 'Dialogue\nResponse',
        'ds': 'Document\nSummarization'
    }

    METRICS = ['accuracy', 'coverage_rate', 'avg_set_size']
    METRIC_NAMES = {
        'accuracy': 'Accuracy (%)',
        'coverage_rate': 'Coverage Rate (%)',
        'avg_set_size': 'Avg Set Size',
        'set_size_lac': 'Set Size (LAC)',
        'set_size_aps': 'Set Size (APS)'
    }

    # Color schemes
    COLORS = {
        'accuracy': 'RdYlGn',
        'coverage_rate': 'RdYlGn',
        'avg_set_size': 'RdYlGn_r',  # Reversed - smaller is better
        'set_size_lac': 'RdYlGn_r',
        'set_size_aps': 'RdYlGn_r'
    }

    def __init__(
        self,
        results: Optional[List[BenchmarkResult]] = None,
        output_dir: str = "./outputs/figures"
    ):
        """Initialize visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = results or []

    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result."""
        self.results.append(result)

    def add_results_from_dict(self, results_dict: Dict[str, Any]):
        """Add results from a dictionary format (from JSON results file)."""
        # Handle list format from run_full_benchmark.py
        if isinstance(results_dict, list):
            for item in results_dict:
                result = BenchmarkResult(
                    model_name=item.get('model', ''),
                    task_name=item.get('task', ''),
                    dtype=item.get('dtype', 'float16'),
                    strategy=item.get('strategy', 'base'),
                    conformal_method=item.get('conformal_method', 'lac'),
                    accuracy=item.get('accuracy', 0) * 100,
                    coverage_rate=item.get('coverage_rate', 0) * 100,
                    avg_set_size=item.get('avg_set_size', 0),
                    meets_guarantee=item.get('meets_guarantee', True),
                    num_samples=item.get('num_samples', 0),
                    inference_time=item.get('inference_time', 0)
                )
                self.results.append(result)
        # Handle nested dict format
        else:
            for model_name, model_data in results_dict.items():
                for task, task_data in model_data.items():
                    result = BenchmarkResult(
                        model_name=model_name,
                        task_name=task,
                        dtype=task_data.get('dtype', 'float16'),
                        strategy=task_data.get('strategy', 'base'),
                        conformal_method=task_data.get('conformal_method', 'lac'),
                        accuracy=task_data.get('accuracy', 0) * 100,
                        coverage_rate=task_data.get('coverage_rate', 0) * 100,
                        avg_set_size=task_data.get('avg_set_size', 0),
                        meets_guarantee=task_data.get('meets_guarantee', True),
                        num_samples=task_data.get('num_samples', 0),
                        inference_time=task_data.get('inference_time', 0)
                    )
                    self.results.append(result)

    def load_results_from_json(self, json_path: str):
        """Load results from a JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.add_results_from_dict(data)

    def _get_result_matrix(
        self,
        metric: str,
        dtype_filter: Optional[str] = None,
        conformal_method_filter: Optional[str] = None,
        strategy_filter: Optional[str] = None
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Create a result matrix for a given metric.

        Args:
            metric: Metric to extract (accuracy, coverage_rate, avg_set_size)
            dtype_filter: Filter by dtype (float16, float32)
            conformal_method_filter: Filter by conformal method (lac, aps)
            strategy_filter: Filter by prompting strategy

        Returns:
            matrix: numpy array of shape (num_models, num_tasks)
            models: list of model names
            tasks: list of task names
        """
        # Filter results
        filtered_results = self.results
        if dtype_filter:
            filtered_results = [r for r in filtered_results if r.dtype == dtype_filter]
        if conformal_method_filter:
            filtered_results = [r for r in filtered_results if r.conformal_method == conformal_method_filter]
        if strategy_filter:
            filtered_results = [r for r in filtered_results if r.strategy == strategy_filter]

        # Get unique models and tasks
        models = sorted(list(set(r.model_name for r in filtered_results)))
        tasks = [t for t in self.TASKS if any(r.task_name == t for r in filtered_results)]

        # Create matrix - aggregate by averaging if multiple results per cell
        matrix = np.full((len(models), len(tasks)), np.nan)
        counts = np.zeros((len(models), len(tasks)))

        for r in filtered_results:
            if r.model_name in models and r.task_name in tasks:
                i = models.index(r.model_name)
                j = tasks.index(r.task_name)
                value = getattr(r, metric, None)
                if value is not None:
                    if np.isnan(matrix[i, j]):
                        matrix[i, j] = value
                    else:
                        matrix[i, j] += value
                    counts[i, j] += 1

        # Average the values where we had multiple results
        for i in range(len(models)):
            for j in range(len(tasks)):
                if counts[i, j] > 1:
                    matrix[i, j] /= counts[i, j]

        return matrix, models, tasks

    def plot_heatmap(
        self,
        metric: str = 'accuracy',
        dtype_filter: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 8),
        title: Optional[str] = None,
        save: bool = True,
        show_values: bool = True,
        cmap: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a heatmap of model vs task performance.

        Args:
            metric: Metric to visualize
            dtype_filter: Filter by dtype (e.g., 'float16', 'float32')
            figsize: Figure size
            title: Custom title
            save: Whether to save the figure
            show_values: Whether to annotate cells with values
            cmap: Custom colormap
        """
        matrix, models, tasks = self._get_result_matrix(metric, dtype_filter)

        if len(models) == 0 or len(tasks) == 0:
            logger.warning("No data available for heatmap")
            return None

        fig, ax = plt.subplots(figsize=figsize)

        # Use appropriate colormap
        if cmap is None:
            cmap = self.COLORS.get(metric, 'viridis')

        # Create heatmap
        task_labels = [self.TASK_NAMES.get(t, t) for t in tasks]

        # Determine format string based on metric
        fmt = '.1f' if 'size' in metric else '.1f'

        sns.heatmap(
            matrix,
            annot=show_values,
            fmt=fmt,
            cmap=cmap,
            xticklabels=task_labels,
            yticklabels=models,
            ax=ax,
            cbar_kws={'label': self.METRIC_NAMES.get(metric, metric)},
            linewidths=0.5,
            linecolor='white'
        )

        # Styling
        dtype_str = f" ({dtype_filter})" if dtype_filter else ""
        if title is None:
            title = f"{self.METRIC_NAMES.get(metric, metric)} by Model and Task{dtype_str}"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Task', fontsize=12)
        ax.set_ylabel('Model', fontsize=12)

        plt.xticks(rotation=0, ha='center')
        plt.yticks(rotation=0)

        plt.tight_layout()

        if save:
            dtype_suffix = f"_{dtype_filter}" if dtype_filter else ""
            filename = f"heatmap_{metric}{dtype_suffix}.png"
            fig.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved heatmap to {self.output_dir / filename}")

        return fig

    def plot_comprehensive_dashboard(
        self,
        dtype_filter: Optional[str] = None,
        figsize: Tuple[int, int] = (20, 16),
        save: bool = True
    ) -> plt.Figure:
        """
        Create a comprehensive dashboard with multiple metrics.
        Shows: Accuracy, Coverage, Set Size (LAC), Set Size (APS)
        """
        fig = plt.figure(figsize=figsize)

        # Create grid for subplots
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

        # Define metrics and their configurations
        configs = [
            ('accuracy', 'Accuracy (%)', None),
            ('coverage_rate', 'Coverage Rate (%)', None),
            ('avg_set_size', 'Set Size (LAC)', 'lac'),
            ('avg_set_size', 'Set Size (APS)', 'aps'),
        ]

        for idx, (metric, title, cp_method) in enumerate(configs):
            ax = fig.add_subplot(gs[idx // 2, idx % 2])

            matrix, models, tasks = self._get_result_matrix(
                metric, dtype_filter, conformal_method_filter=cp_method
            )

            if len(models) == 0 or len(tasks) == 0:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
                ax.set_title(title)
                continue

            task_labels = [self.TASK_NAMES.get(t, t) for t in tasks]
            cmap = self.COLORS.get(metric, 'viridis')

            sns.heatmap(
                matrix,
                annot=True,
                fmt='.1f',
                cmap=cmap,
                xticklabels=task_labels,
                yticklabels=models,
                ax=ax,
                cbar_kws={'shrink': 0.8},
                linewidths=0.5,
                linecolor='white'
            )

            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('')

        dtype_str = f" ({dtype_filter})" if dtype_filter else ""
        fig.suptitle(f'BLUQ Benchmark Results Dashboard{dtype_str}',
                    fontsize=16, fontweight='bold', y=1.02)

        if save:
            dtype_suffix = f"_{dtype_filter}" if dtype_filter else ""
            filename = f"dashboard{dtype_suffix}.png"
            fig.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved dashboard to {self.output_dir / filename}")

        return fig

    def plot_model_comparison_bars(
        self,
        metric: str = 'accuracy',
        dtype_filter: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 8),
        save: bool = True
    ) -> plt.Figure:
        """
        Create grouped bar chart comparing models across tasks.
        """
        matrix, models, tasks = self._get_result_matrix(metric, dtype_filter)

        if len(models) == 0 or len(tasks) == 0:
            logger.warning("No data available for bar chart")
            return None

        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(tasks))
        width = 0.8 / len(models)

        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

        for i, (model, color) in enumerate(zip(models, colors)):
            offset = (i - len(models)/2 + 0.5) * width
            values = matrix[i, :]
            bars = ax.bar(x + offset, values, width, label=model, color=color, edgecolor='white')

            # Add value labels on bars
            for bar, val in zip(bars, values):
                if not np.isnan(val):
                    height = bar.get_height()
                    ax.annotate(f'{val:.1f}',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3),
                              textcoords="offset points",
                              ha='center', va='bottom', fontsize=8, rotation=90)

        task_labels = [self.TASK_NAMES.get(t, t).replace('\n', ' ') for t in tasks]
        ax.set_xticks(x)
        ax.set_xticklabels(task_labels, fontsize=10)

        dtype_str = f" ({dtype_filter})" if dtype_filter else ""
        ax.set_title(f'{self.METRIC_NAMES.get(metric, metric)} by Model and Task{dtype_str}',
                    fontsize=14, fontweight='bold')
        ax.set_ylabel(self.METRIC_NAMES.get(metric, metric), fontsize=12)
        ax.set_xlabel('Task', fontsize=12)

        ax.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.set_ylim(bottom=0)

        plt.tight_layout()

        if save:
            dtype_suffix = f"_{dtype_filter}" if dtype_filter else ""
            filename = f"bar_comparison_{metric}{dtype_suffix}.png"
            fig.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved bar chart to {self.output_dir / filename}")

        return fig

    def plot_dtype_comparison(
        self,
        metric: str = 'accuracy',
        figsize: Tuple[int, int] = (16, 8),
        save: bool = True
    ) -> plt.Figure:
        """
        Create side-by-side comparison of FP16 vs FP32 results.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        for ax, dtype in zip(axes, ['float16', 'float32']):
            matrix, models, tasks = self._get_result_matrix(metric, dtype)

            if len(models) == 0 or len(tasks) == 0:
                ax.text(0.5, 0.5, f'No {dtype} Data', ha='center', va='center', fontsize=14)
                ax.set_title(f'{dtype.upper()}')
                continue

            task_labels = [self.TASK_NAMES.get(t, t) for t in tasks]
            cmap = self.COLORS.get(metric, 'viridis')

            sns.heatmap(
                matrix,
                annot=True,
                fmt='.1f',
                cmap=cmap,
                xticklabels=task_labels,
                yticklabels=models,
                ax=ax,
                linewidths=0.5,
                linecolor='white'
            )

            ax.set_title(f'{dtype.upper()}', fontsize=12, fontweight='bold')

        fig.suptitle(f'{self.METRIC_NAMES.get(metric, metric)}: FP16 vs FP32 Comparison',
                    fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()

        if save:
            filename = f"dtype_comparison_{metric}.png"
            fig.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved dtype comparison to {self.output_dir / filename}")

        return fig

    def plot_radar_chart(
        self,
        model_names: Optional[List[str]] = None,
        dtype_filter: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10),
        save: bool = True
    ) -> plt.Figure:
        """
        Create radar chart showing model performance across tasks.
        """
        matrix, models, tasks = self._get_result_matrix('accuracy', dtype_filter)

        if len(models) == 0 or len(tasks) == 0:
            logger.warning("No data available for radar chart")
            return None

        if model_names:
            # Filter to specified models
            indices = [i for i, m in enumerate(models) if m in model_names]
            if not indices:
                logger.warning("No matching models found")
                return None
            matrix = matrix[indices, :]
            models = [models[i] for i in indices]

        # Number of tasks
        num_tasks = len(tasks)

        # Compute angle for each task
        angles = np.linspace(0, 2 * np.pi, num_tasks, endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop

        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))

        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

        for i, (model, color) in enumerate(zip(models, colors)):
            values = matrix[i, :].tolist()
            values += values[:1]  # Complete the loop

            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)

        task_labels = [self.TASK_NAMES.get(t, t).replace('\n', ' ') for t in tasks]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(task_labels, fontsize=10)

        dtype_str = f" ({dtype_filter})" if dtype_filter else ""
        ax.set_title(f'Model Performance Radar Chart{dtype_str}',
                    fontsize=14, fontweight='bold', y=1.1)

        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        if save:
            dtype_suffix = f"_{dtype_filter}" if dtype_filter else ""
            filename = f"radar_chart{dtype_suffix}.png"
            fig.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved radar chart to {self.output_dir / filename}")

        return fig

    def plot_uncertainty_analysis(
        self,
        dtype_filter: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10),
        save: bool = True
    ) -> plt.Figure:
        """
        Create scatter plot showing accuracy vs uncertainty (set size).
        """
        filtered_results = self.results
        if dtype_filter:
            filtered_results = [r for r in self.results if r.dtype == dtype_filter]

        if not filtered_results:
            logger.warning("No data available for uncertainty analysis")
            return None

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Group by model
        models = list(set(r.model_name for r in filtered_results))
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        model_colors = dict(zip(models, colors))

        for ax, cp_method in zip(axes, ['lac', 'aps']):
            # Filter by conformal method
            method_results = [r for r in filtered_results if r.conformal_method == cp_method]

            for r in method_results:
                ax.scatter(
                    r.accuracy,
                    r.avg_set_size,
                    c=[model_colors[r.model_name]],
                    s=100,
                    alpha=0.7,
                    label=r.model_name
                )

            ax.set_xlabel('Accuracy (%)', fontsize=12)
            ax.set_ylabel(f'Set Size ({cp_method.upper()})', fontsize=12)
            ax.set_title(f'Accuracy vs Set Size ({cp_method.upper()})',
                        fontsize=12, fontweight='bold')

            # Add trend line
            if len(method_results) > 1:
                accs = [r.accuracy for r in method_results]
                sizes = [r.avg_set_size for r in method_results]
                z = np.polyfit(accs, sizes, 1)
                p = np.poly1d(z)
                x_line = np.linspace(min(accs), max(accs), 100)
                ax.plot(x_line, p(x_line), "r--", alpha=0.5, label='Trend')

        # Create legend
        handles = [mpatches.Patch(color=c, label=m) for m, c in model_colors.items()]
        fig.legend(handles=handles, loc='center right', bbox_to_anchor=(1.15, 0.5))

        dtype_str = f" ({dtype_filter})" if dtype_filter else ""
        fig.suptitle(f'Uncertainty Analysis: Accuracy vs Prediction Set Size{dtype_str}',
                    fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()

        if save:
            dtype_suffix = f"_{dtype_filter}" if dtype_filter else ""
            filename = f"uncertainty_analysis{dtype_suffix}.png"
            fig.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved uncertainty analysis to {self.output_dir / filename}")

        return fig

    def generate_summary_table(
        self,
        dtype_filter: Optional[str] = None,
        conformal_method_filter: Optional[str] = None,
        save: bool = True
    ) -> str:
        """Generate a markdown summary table."""
        filtered_results = self.results
        if dtype_filter:
            filtered_results = [r for r in filtered_results if r.dtype == dtype_filter]
        if conformal_method_filter:
            filtered_results = [r for r in filtered_results if r.conformal_method == conformal_method_filter]

        if not filtered_results:
            return "No results available."

        # Get unique models and tasks
        models = sorted(list(set(r.model_name for r in filtered_results)))
        tasks = [t for t in self.TASKS if any(r.task_name == t for r in filtered_results)]

        # Build table
        lines = []
        dtype_str = f" ({dtype_filter})" if dtype_filter else ""
        cp_str = f" - {conformal_method_filter.upper()}" if conformal_method_filter else ""
        lines.append(f"# BLUQ Benchmark Results{dtype_str}{cp_str}\n")

        # Accuracy table
        lines.append("## Accuracy (%)\n")
        header = "| Model | " + " | ".join(self.TASK_NAMES.get(t, t).replace('\n', ' ') for t in tasks) + " | Avg |"
        lines.append(header)
        lines.append("|" + "---|" * (len(tasks) + 2))

        for model in models:
            row = f"| {model} |"
            values = []
            for task in tasks:
                task_results = [r for r in filtered_results
                               if r.model_name == model and r.task_name == task]
                if task_results:
                    avg_acc = np.mean([r.accuracy for r in task_results])
                    row += f" {avg_acc:.1f} |"
                    values.append(avg_acc)
                else:
                    row += " - |"
            avg = np.mean(values) if values else 0
            row += f" **{avg:.1f}** |"
            lines.append(row)

        lines.append("")

        # Coverage table
        lines.append("## Coverage Rate (%)\n")
        header = "| Model | " + " | ".join(self.TASK_NAMES.get(t, t).replace('\n', ' ') for t in tasks) + " | Avg |"
        lines.append(header)
        lines.append("|" + "---|" * (len(tasks) + 2))

        for model in models:
            row = f"| {model} |"
            values = []
            for task in tasks:
                task_results = [r for r in filtered_results
                               if r.model_name == model and r.task_name == task]
                if task_results:
                    avg_cr = np.mean([r.coverage_rate for r in task_results])
                    row += f" {avg_cr:.1f} |"
                    values.append(avg_cr)
                else:
                    row += " - |"
            avg = np.mean(values) if values else 0
            row += f" **{avg:.1f}** |"
            lines.append(row)

        lines.append("")

        # Set size table
        lines.append("## Average Set Size\n")
        header = "| Model | " + " | ".join(self.TASK_NAMES.get(t, t).replace('\n', ' ') for t in tasks) + " | Avg |"
        lines.append(header)
        lines.append("|" + "---|" * (len(tasks) + 2))

        for model in models:
            row = f"| {model} |"
            values = []
            for task in tasks:
                task_results = [r for r in filtered_results
                               if r.model_name == model and r.task_name == task]
                if task_results:
                    avg_ss = np.mean([r.avg_set_size for r in task_results])
                    row += f" {avg_ss:.2f} |"
                    values.append(avg_ss)
                else:
                    row += " - |"
            avg = np.mean(values) if values else 0
            row += f" **{avg:.2f}** |"
            lines.append(row)

        summary = "\n".join(lines)

        if save:
            dtype_suffix = f"_{dtype_filter}" if dtype_filter else ""
            cp_suffix = f"_{conformal_method_filter}" if conformal_method_filter else ""
            filename = f"results_summary{dtype_suffix}{cp_suffix}.md"
            with open(self.output_dir / filename, 'w') as f:
                f.write(summary)
            logger.info(f"Saved summary table to {self.output_dir / filename}")

        return summary

    def generate_all_visualizations(self, dtype_filter: Optional[str] = None):
        """Generate all visualizations."""
        logger.info("Generating all visualizations...")

        # Heatmaps for main metrics
        for metric in ['accuracy', 'coverage_rate', 'avg_set_size']:
            self.plot_heatmap(metric=metric, dtype_filter=dtype_filter)

        # Dashboard
        self.plot_comprehensive_dashboard(dtype_filter=dtype_filter)

        # Bar charts
        for metric in ['accuracy', 'avg_set_size']:
            self.plot_model_comparison_bars(metric=metric, dtype_filter=dtype_filter)

        # Radar chart
        self.plot_radar_chart(dtype_filter=dtype_filter)

        # Uncertainty analysis
        self.plot_uncertainty_analysis(dtype_filter=dtype_filter)

        # Summary tables for each conformal method
        self.generate_summary_table(dtype_filter=dtype_filter)
        for cp_method in ['lac', 'aps']:
            self.generate_summary_table(dtype_filter=dtype_filter, conformal_method_filter=cp_method)

        # If we have both dtypes, generate comparison
        dtypes = list(set(r.dtype for r in self.results))
        if len(dtypes) > 1:
            for metric in ['accuracy', 'avg_set_size']:
                self.plot_dtype_comparison(metric=metric)

        logger.info(f"All visualizations saved to {self.output_dir}")


def create_benchmark_report(
    results_path: str,
    output_dir: str = "./outputs/figures"
) -> ResultVisualizer:
    """
    Create a complete benchmark report from results.

    Args:
        results_path: Path to JSON results file
        output_dir: Directory to save visualizations

    Returns:
        ResultVisualizer instance
    """
    visualizer = ResultVisualizer(output_dir=output_dir)
    visualizer.load_results_from_json(results_path)
    visualizer.generate_all_visualizations()
    return visualizer


if __name__ == "__main__":
    # Example usage with mock data
    import logging
    logging.basicConfig(level=logging.INFO)

    visualizer = ResultVisualizer(output_dir="./outputs/figures")

    # Add some mock results for testing
    models = ['tinyllama-1.1b', 'phi-2', 'stablelm-2-1.6b', 'qwen-1.8b', 'gemma-2b']
    tasks = ['qa', 'rc', 'ci', 'drs', 'ds']
    strategies = ['base']
    conformal_methods = ['lac', 'aps']

    np.random.seed(42)
    for model in models:
        for task in tasks:
            for dtype in ['float16', 'float32']:
                for strategy in strategies:
                    for cp_method in conformal_methods:
                        # Generate realistic-ish results
                        base_acc = np.random.uniform(25, 75)
                        result = BenchmarkResult(
                            model_name=model,
                            task_name=task,
                            dtype=dtype,
                            strategy=strategy,
                            conformal_method=cp_method,
                            accuracy=base_acc,
                            coverage_rate=np.random.uniform(88, 95),
                            avg_set_size=np.random.uniform(1.5, 4.5) if cp_method == 'lac' else np.random.uniform(2.0, 5.0),
                            meets_guarantee=np.random.random() > 0.1,
                            num_samples=100
                        )
                        visualizer.add_result(result)

    # Generate all visualizations
    visualizer.generate_all_visualizations()

    print("\nSample summary:")
    print(visualizer.generate_summary_table(save=False))
