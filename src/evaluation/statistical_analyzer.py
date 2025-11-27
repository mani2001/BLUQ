"""
Statistical Analyzer Module
Provides statistical analysis and significance testing for evaluation results.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from scipy import stats
from scipy.stats import spearmanr, kendalltau

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class SignificanceTestResult:
    """Result from a statistical significance test."""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    significance_level: float
    interpretation: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'test_name': self.test_name,
            'statistic': float(self.statistic),
            'p_value': float(self.p_value),
            'is_significant': self.is_significant,
            'significance_level': self.significance_level,
            'interpretation': self.interpretation,
            'metadata': self.metadata or {}
        }


@dataclass
class CorrelationAnalysis:
    """Results from correlation analysis."""
    metric1: str
    metric2: str
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    kendall_tau: float
    kendall_p: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'metric1': self.metric1,
            'metric2': self.metric2,
            'pearson': {
                'r': float(self.pearson_r),
                'p_value': float(self.pearson_p)
            },
            'spearman': {
                'r': float(self.spearman_r),
                'p_value': float(self.spearman_p)
            },
            'kendall': {
                'tau': float(self.kendall_tau),
                'p_value': float(self.kendall_p)
            }
        }
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        return (
            f"Correlation between {self.metric1} and {self.metric2}:\n"
            f"  Pearson r: {self.pearson_r:.4f} (p={self.pearson_p:.4f})\n"
            f"  Spearman ρ: {self.spearman_r:.4f} (p={self.spearman_p:.4f})\n"
            f"  Kendall τ: {self.kendall_tau:.4f} (p={self.kendall_p:.4f})"
        )


class StatisticalAnalyzer:
    """Main class for statistical analysis of results."""
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize the statistical analyzer.
        
        Args:
            significance_level: Significance level for hypothesis tests
        """
        self.significance_level = significance_level
        logger.info(f"Initialized StatisticalAnalyzer (alpha={significance_level})")
    
    def compare_two_models(
        self,
        model1_metrics: List[float],
        model2_metrics: List[float],
        metric_name: str = "accuracy",
        paired: bool = True
    ) -> SignificanceTestResult:
        """
        Compare two models on a metric using statistical test.
        
        Args:
            model1_metrics: Metric values for model 1
            model2_metrics: Metric values for model 2
            metric_name: Name of the metric being compared
            paired: Whether to use paired test (same test instances)
            
        Returns:
            SignificanceTestResult
        """
        model1_metrics = np.array(model1_metrics)
        model2_metrics = np.array(model2_metrics)
        
        if paired:
            # Paired t-test
            statistic, p_value = stats.ttest_rel(model1_metrics, model2_metrics)
            test_name = "Paired t-test"
        else:
            # Independent t-test
            statistic, p_value = stats.ttest_ind(model1_metrics, model2_metrics)
            test_name = "Independent t-test"
        
        is_significant = p_value < self.significance_level
        
        # Interpretation
        mean_diff = np.mean(model1_metrics) - np.mean(model2_metrics)
        if is_significant:
            if mean_diff > 0:
                interpretation = f"Model 1 significantly outperforms Model 2 on {metric_name}"
            else:
                interpretation = f"Model 2 significantly outperforms Model 1 on {metric_name}"
        else:
            interpretation = f"No significant difference between models on {metric_name}"
        
        return SignificanceTestResult(
            test_name=test_name,
            statistic=float(statistic),
            p_value=float(p_value),
            is_significant=is_significant,
            significance_level=self.significance_level,
            interpretation=interpretation,
            metadata={
                'model1_mean': float(np.mean(model1_metrics)),
                'model2_mean': float(np.mean(model2_metrics)),
                'mean_difference': float(mean_diff)
            }
        )
    
    def compute_correlation(
        self,
        metric1_values: np.ndarray,
        metric2_values: np.ndarray,
        metric1_name: str = "metric1",
        metric2_name: str = "metric2"
    ) -> CorrelationAnalysis:
        """
        Compute correlation between two metrics.
        
        Args:
            metric1_values: Values for first metric
            metric2_values: Values for second metric
            metric1_name: Name of first metric
            metric2_name: Name of second metric
            
        Returns:
            CorrelationAnalysis object
        """
        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(metric1_values, metric2_values)
        
        # Spearman correlation
        spearman_r, spearman_p = spearmanr(metric1_values, metric2_values)
        
        # Kendall's tau
        kendall_tau, kendall_p = kendalltau(metric1_values, metric2_values)
        
        return CorrelationAnalysis(
            metric1=metric1_name,
            metric2=metric2_name,
            pearson_r=float(pearson_r),
            pearson_p=float(pearson_p),
            spearman_r=float(spearman_r),
            spearman_p=float(spearman_p),
            kendall_tau=float(kendall_tau),
            kendall_p=float(kendall_p)
        )
    
    def analyze_accuracy_uncertainty_tradeoff(
        self,
        results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze the tradeoff between accuracy and uncertainty (set size).
        
        Args:
            results: Results dict (model -> task -> metrics)
            
        Returns:
            Analysis dictionary
        """
        logger.info("Analyzing accuracy-uncertainty tradeoff...")
        
        # Collect data points
        accuracies = []
        set_sizes = []
        model_task_pairs = []
        
        for model, task_results in results.items():
            for task, metrics in task_results.items():
                accuracies.append(metrics.accuracy)
                set_sizes.append(metrics.set_size)
                model_task_pairs.append((model, task))
        
        accuracies = np.array(accuracies)
        set_sizes = np.array(set_sizes)
        
        # Compute correlation (need at least 2 data points)
        if len(accuracies) < 2:
            logger.warning("Not enough data points for correlation analysis (need at least 2, got %d)", len(accuracies))
            corr_analysis = None
        else:
            corr_analysis = self.compute_correlation(
                accuracies,
                set_sizes,
                "accuracy",
                "set_size"
            )
        
        # Find counterexamples (high accuracy but high uncertainty)
        # Define thresholds
        high_acc_threshold = np.percentile(accuracies, 75)
        high_size_threshold = np.percentile(set_sizes, 75)
        
        counterexamples = []
        for i, (model, task) in enumerate(model_task_pairs):
            if accuracies[i] >= high_acc_threshold and set_sizes[i] >= high_size_threshold:
                counterexamples.append({
                    'model': model,
                    'task': task,
                    'accuracy': float(accuracies[i]),
                    'set_size': float(set_sizes[i])
                })
        
        # Build result dict
        result = {
            'num_samples': len(accuracies),
            'counterexamples': counterexamples,
            'num_counterexamples': len(counterexamples),
        }
        
        if corr_analysis is not None:
            result['correlation'] = corr_analysis.to_dict()
            result['interpretation'] = self._interpret_accuracy_uncertainty_correlation(
                corr_analysis.pearson_r
            )
        else:
            result['correlation'] = None
            result['interpretation'] = "Insufficient data points for correlation analysis"
        
        return result
    
    def _interpret_accuracy_uncertainty_correlation(self, correlation: float) -> str:
        """Interpret the correlation between accuracy and uncertainty."""
        if correlation < -0.5:
            return "Strong negative correlation: Higher accuracy → Lower uncertainty (expected)"
        elif correlation < -0.2:
            return "Moderate negative correlation: Higher accuracy tends toward lower uncertainty"
        elif correlation < 0.2:
            return "Weak/no correlation: Accuracy and uncertainty are largely independent"
        else:
            return "Positive correlation: Higher accuracy → Higher uncertainty (unexpected)"
    
    def compare_conformal_methods(
        self,
        lac_results: Dict[str, Any],
        aps_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare LAC and APS conformal methods statistically.
        
        Args:
            lac_results: Results using LAC method
            aps_results: Results using APS method
            
        Returns:
            Comparison statistics
        """
        logger.info("Comparing LAC vs APS...")
        
        # Extract metrics
        lac_set_sizes = []
        aps_set_sizes = []
        lac_coverages = []
        aps_coverages = []
        
        for model in lac_results.keys():
            for task in lac_results[model].keys():
                if task in aps_results.get(model, {}):
                    lac_set_sizes.append(lac_results[model][task].set_size)
                    aps_set_sizes.append(aps_results[model][task].set_size)
                    lac_coverages.append(lac_results[model][task].coverage_rate)
                    aps_coverages.append(aps_results[model][task].coverage_rate)
        
        # Compare set sizes
        size_test = self.compare_two_models(
            lac_set_sizes,
            aps_set_sizes,
            metric_name="set_size",
            paired=True
        )
        
        # Compare coverage
        coverage_test = self.compare_two_models(
            lac_coverages,
            aps_coverages,
            metric_name="coverage_rate",
            paired=True
        )
        
        return {
            'set_size_comparison': size_test.to_dict(),
            'coverage_comparison': coverage_test.to_dict(),
            'lac_avg_size': float(np.mean(lac_set_sizes)),
            'aps_avg_size': float(np.mean(aps_set_sizes)),
            'lac_avg_coverage': float(np.mean(lac_coverages)),
            'aps_avg_coverage': float(np.mean(aps_coverages))
        }
    
    def analyze_model_scale_effects(
        self,
        results: Dict[str, Dict[str, Any]],
        model_sizes: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Analyze the effect of model scale on performance and uncertainty.
        
        Args:
            results: Results dict (model -> task -> metrics)
            model_sizes: Dict mapping model names to parameter counts (in billions)
            
        Returns:
            Scale analysis
        """
        logger.info("Analyzing model scale effects...")
        
        # Collect data
        sizes = []
        accuracies = []
        set_sizes_list = []
        
        for model, task_results in results.items():
            if model in model_sizes:
                size = model_sizes[model]
                
                for task, metrics in task_results.items():
                    sizes.append(size)
                    accuracies.append(metrics.accuracy)
                    set_sizes_list.append(metrics.set_size)
        
        sizes = np.array(sizes)
        accuracies = np.array(accuracies)
        set_sizes_arr = np.array(set_sizes_list)
        
        # Correlation between size and accuracy
        size_acc_corr = self.compute_correlation(
            sizes, accuracies, "model_size", "accuracy"
        )
        
        # Correlation between size and uncertainty
        size_unc_corr = self.compute_correlation(
            sizes, set_sizes_arr, "model_size", "set_size"
        )
        
        return {
            'size_accuracy_correlation': size_acc_corr.to_dict(),
            'size_uncertainty_correlation': size_unc_corr.to_dict(),
            'num_datapoints': len(sizes),
            'size_range': [float(sizes.min()), float(sizes.max())]
        }
    
    def compute_confidence_intervals(
        self,
        metric_values: np.ndarray,
        confidence_level: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Compute confidence intervals for a metric.
        
        Args:
            metric_values: Array of metric values
            confidence_level: Confidence level (default: 0.95)
            
        Returns:
            Tuple of (mean, lower_bound, upper_bound)
        """
        mean = np.mean(metric_values)
        std_error = stats.sem(metric_values)
        
        # Compute confidence interval
        ci = stats.t.interval(
            confidence_level,
            len(metric_values) - 1,
            loc=mean,
            scale=std_error
        )
        
        return mean, ci[0], ci[1]
    
    def bootstrap_confidence_interval(
        self,
        metric_values: np.ndarray,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Compute bootstrap confidence interval.
        
        Args:
            metric_values: Array of metric values
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level
            
        Returns:
            Tuple of (mean, lower_bound, upper_bound)
        """
        means = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(metric_values, size=len(metric_values), replace=True)
            means.append(np.mean(sample))
        
        means = np.array(means)
        mean = np.mean(metric_values)
        
        # Compute percentile-based CI
        alpha = 1 - confidence_level
        lower = np.percentile(means, alpha / 2 * 100)
        upper = np.percentile(means, (1 - alpha / 2) * 100)
        
        return mean, lower, upper


class RankingAnalyzer:
    """Analyzer for model rankings."""
    
    @staticmethod
    def compute_ranking_stability(
        rankings_by_task: Dict[str, Dict[str, int]]
    ) -> Dict[str, Any]:
        """
        Analyze stability of rankings across tasks.
        
        Args:
            rankings_by_task: Dict mapping task to model rankings
            
        Returns:
            Stability analysis
        """
        # Convert to matrix: tasks x models
        tasks = list(rankings_by_task.keys())
        all_models = set()
        for task_ranks in rankings_by_task.values():
            all_models.update(task_ranks.keys())
        models = sorted(all_models)
        
        # Create ranking matrix
        rank_matrix = np.zeros((len(tasks), len(models)))
        for i, task in enumerate(tasks):
            for j, model in enumerate(models):
                rank_matrix[i, j] = rankings_by_task[task].get(model, len(models) + 1)
        
        # Compute variance in rankings for each model
        ranking_variance = {}
        for j, model in enumerate(models):
            variance = np.var(rank_matrix[:, j])
            ranking_variance[model] = float(variance)
        
        # Most stable model (lowest variance)
        most_stable = min(ranking_variance.items(), key=lambda x: x[1])[0]
        
        # Compute average rank for each model across tasks
        avg_ranks = {}
        for j, model in enumerate(models):
            avg_ranks[model] = float(np.mean(rank_matrix[:, j]))
        
        return {
            'ranking_variance': ranking_variance,
            'most_stable_model': most_stable,
            'average_ranks': avg_ranks,
            'rank_matrix': rank_matrix.tolist(),
            'tasks': tasks,
            'models': models
        }
    
    @staticmethod
    def compute_ranking_correlation(
        ranking1: Dict[str, int],
        ranking2: Dict[str, int]
    ) -> Tuple[float, float]:
        """
        Compute correlation between two rankings.
        
        Args:
            ranking1: First ranking (model -> rank)
            ranking2: Second ranking (model -> rank)
            
        Returns:
            Tuple of (spearman_rho, kendall_tau)
        """
        # Get common models
        common_models = set(ranking1.keys()) & set(ranking2.keys())
        common_models = sorted(common_models)
        
        ranks1 = np.array([ranking1[m] for m in common_models])
        ranks2 = np.array([ranking2[m] for m in common_models])
        
        # Spearman correlation
        spearman_rho, _ = spearmanr(ranks1, ranks2)
        
        # Kendall's tau
        kendall_tau, _ = kendalltau(ranks1, ranks2)
        
        return float(spearman_rho), float(kendall_tau)


class EffectSizeAnalyzer:
    """Analyzer for effect sizes."""
    
    @staticmethod
    def compute_cohens_d(
        group1: np.ndarray,
        group2: np.ndarray
    ) -> float:
        """
        Compute Cohen's d effect size.
        
        Args:
            group1: First group values
            group2: Second group values
            
        Returns:
            Cohen's d
        """
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)
        
        # Pooled standard deviation
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        cohens_d = (mean1 - mean2) / pooled_std
        
        return float(cohens_d)
    
    @staticmethod
    def interpret_cohens_d(d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"


class CoverageAnalyzer:
    """Specialized analyzer for coverage guarantee validation."""
    
    @staticmethod
    def validate_coverage_guarantee(
        coverage_rates: List[float],
        target_coverage: float,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Validate that coverage guarantee is met with statistical confidence.
        
        Args:
            coverage_rates: List of empirical coverage rates
            target_coverage: Target coverage (1 - alpha)
            confidence_level: Confidence level for validation
            
        Returns:
            Validation results
        """
        coverage_rates = np.array(coverage_rates)
        
        # Test if mean coverage is at least target_coverage
        # One-sided t-test
        statistic, p_value = stats.ttest_1samp(
            coverage_rates,
            target_coverage,
            alternative='greater'
        )
        
        mean_coverage = np.mean(coverage_rates)
        std_coverage = np.std(coverage_rates)
        
        # Compute confidence interval
        ci_lower, ci_upper = stats.t.interval(
            confidence_level,
            len(coverage_rates) - 1,
            loc=mean_coverage,
            scale=stats.sem(coverage_rates)
        )
        
        passes_validation = ci_lower >= target_coverage
        
        return {
            'mean_coverage': float(mean_coverage),
            'std_coverage': float(std_coverage),
            'target_coverage': float(target_coverage),
            'confidence_interval': [float(ci_lower), float(ci_upper)],
            'confidence_level': confidence_level,
            'passes_validation': passes_validation,
            'p_value': float(p_value),
            'num_samples': len(coverage_rates)
        }
    
    @staticmethod
    def analyze_coverage_by_difficulty(
        coverage_indicators: np.ndarray,
        difficulty_scores: np.ndarray,
        num_bins: int = 5
    ) -> Dict[str, Any]:
        """
        Analyze coverage stratified by instance difficulty.
        
        Args:
            coverage_indicators: Binary array (1 if covered, 0 otherwise)
            difficulty_scores: Difficulty score for each instance (e.g., entropy)
            num_bins: Number of difficulty bins
            
        Returns:
            Stratified coverage analysis
        """
        # Create difficulty bins
        bins = np.linspace(
            difficulty_scores.min(),
            difficulty_scores.max(),
            num_bins + 1
        )
        
        bin_indices = np.digitize(difficulty_scores, bins) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        
        # Compute coverage per bin
        coverage_by_bin = {}
        for i in range(num_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_coverage = coverage_indicators[mask].mean()
                bin_count = mask.sum()
                bin_range = (float(bins[i]), float(bins[i + 1]))
                
                coverage_by_bin[i] = {
                    'coverage': float(bin_coverage),
                    'count': int(bin_count),
                    'difficulty_range': bin_range
                }
        
        return {
            'coverage_by_difficulty': coverage_by_bin,
            'num_bins': num_bins
        }


# Example usage
if __name__ == "__main__":
    import logging
    from src.evaluation.metrics import EvaluationMetrics
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*80)
    print("Statistical Analyzer Test")
    print("="*80)
    
    # Create sample data
    np.random.seed(42)
    
    # Simulate results for 3 models on 2 tasks
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
    
    analyzer = StatisticalAnalyzer(significance_level=0.05)
    
    # Test comparing two models
    print("\n" + "="*80)
    print("Comparing two models...")
    print("="*80)
    
    model_a_acc = [0.75, 0.82]
    model_b_acc = [0.71, 0.79]
    
    comparison = analyzer.compare_two_models(
        model_a_acc, model_b_acc, metric_name="accuracy", paired=True
    )
    
    print(f"\n{comparison.interpretation}")
    print(f"  Statistic: {comparison.statistic:.4f}")
    print(f"  P-value: {comparison.p_value:.4f}")
    print(f"  Significant: {comparison.is_significant}")
    
    # Test correlation analysis
    print("\n" + "="*80)
    print("Analyzing accuracy-uncertainty tradeoff...")
    print("="*80)
    
    tradeoff = analyzer.analyze_accuracy_uncertainty_tradeoff(results)
    
    print(f"\nCorrelation: {tradeoff['correlation']['pearson']['r']:.4f}")
    print(f"Interpretation: {tradeoff['interpretation']}")
    print(f"Counterexamples: {tradeoff['num_counterexamples']}")
    
    # Test coverage validation
    print("\n" + "="*80)
    print("Validating coverage guarantee...")
    print("="*80)
    
    coverage_analyzer = CoverageAnalyzer()
    
    # Simulate coverage rates
    coverage_rates = [0.92, 0.93, 0.91, 0.92, 0.94, 0.91]
    
    validation = coverage_analyzer.validate_coverage_guarantee(
        coverage_rates=coverage_rates,
        target_coverage=0.9,
        confidence_level=0.95
    )
    
    print(f"\nCoverage Validation:")
    print(f"  Mean coverage: {validation['mean_coverage']:.2%}")
    print(f"  Target coverage: {validation['target_coverage']:.2%}")
    print(f"  95% CI: [{validation['confidence_interval'][0]:.2%}, "
          f"{validation['confidence_interval'][1]:.2%}]")
    print(f"  Passes validation: {validation['passes_validation']}")
    
    # Test ranking analysis
    print("\n" + "="*80)
    print("Analyzing ranking stability...")
    print("="*80)
    
    rankings_by_task = {
        'qa': {'model-a': 2, 'model-b': 3, 'model-c': 1},
        'rc': {'model-a': 2, 'model-b': 3, 'model-c': 1}
    }
    
    rank_analyzer = RankingAnalyzer()
    stability = rank_analyzer.compute_ranking_stability(rankings_by_task)
    
    print(f"\nRanking Stability:")
    print(f"  Most stable model: {stability['most_stable_model']}")
    print(f"  Ranking variance:")
    for model, variance in stability['ranking_variance'].items():
        print(f"    {model}: {variance:.4f}")
    
    # Save analysis results
    print("\n" + "="*80)
    print("Saving analysis results...")
    print("="*80)
    
    output_dir = Path("./results/statistical_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "analysis.json", 'w') as f:
        json.dump({
            'tradeoff_analysis': tradeoff,
            'coverage_validation': validation,
            'ranking_stability': stability
        }, f, indent=2)
    
    print(f"Analysis saved to {output_dir}")