#!/usr/bin/env python3
"""
Claims Verification Script

Verifies the three main claims from "Benchmarking LLMs via Uncertainty Quantification"
(Ye et al., 2024) using benchmark results from SLMs.

Claims:
1. Higher accuracy may exhibit lower certainty (negative correlation)
2. Larger models may display greater uncertainty (positive correlation with size)
3. Instruction-tuning increases uncertainty (instruct > base set sizes)
"""

import json
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Model size mapping (in billions of parameters)
MODEL_SIZES = {
    'tinyllama-1.1b': 1.1,
    'stablelm-2-1.6b': 1.6,
    'phi-2': 2.7,
    'gemma-2b': 2.0,
    'gemma-2b-it': 2.0,
    'gemma-2-2b-it': 2.0,
    'mistral-7b': 7.0,
    'mistral-7b-instruct': 7.0,
    'gemma-2-9b-it': 9.0,
}

# Base vs Instruct model pairs
MODEL_PAIRS = {
    'mistral-7b': 'mistral-7b-instruct',
    # Add more pairs as they become available
    # 'gemma-2b': 'gemma-2b-it',
}

TASK_NAMES = {
    'qa': 'Question Answering (MMLU)',
    'rc': 'Reading Comprehension (CosmosQA)',
    'ci': 'Commonsense Inference (HellaSwag)',
    'drs': 'Dialogue Response Selection (HaluDial)',
    'ds': 'Document Summarization (HaluSum)',
}


@dataclass
class ClaimResult:
    """Result of a single claim verification."""
    claim_id: int
    claim_description: str
    supported: bool
    evidence: Dict[str, Any]
    interpretation: str
    statistical_significance: Optional[float] = None


@dataclass
class VerificationReport:
    """Complete verification report for all claims."""
    claim1: ClaimResult
    claim2: ClaimResult
    claim3: ClaimResult
    summary: Dict[str, Any]
    methodology_notes: List[str]


class ClaimsVerifier:
    """Verifies paper claims using benchmark results."""

    def __init__(self, results_path: str):
        """
        Initialize verifier with results file.

        Args:
            results_path: Path to all_results.json or similar
        """
        self.results_path = Path(results_path)
        self.results = self._load_results()
        logger.info(f"Loaded {len(self.results)} result entries from {results_path}")

    def _load_results(self) -> List[Dict]:
        """Load results from JSON file."""
        with open(self.results_path, 'r') as f:
            return json.load(f)

    def _filter_results(
        self,
        conformal_method: Optional[str] = None,
        dtype: Optional[str] = None,
        models: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None
    ) -> List[Dict]:
        """Filter results by criteria."""
        filtered = self.results

        if conformal_method:
            filtered = [r for r in filtered if r.get('conformal_method') == conformal_method]
        if dtype:
            filtered = [r for r in filtered if r.get('dtype') == dtype]
        if models:
            filtered = [r for r in filtered if r.get('model') in models]
        if tasks:
            filtered = [r for r in filtered if r.get('task') in tasks]

        return filtered

    def verify_claim1(self, conformal_method: str = 'lac', dtype: str = 'float16') -> ClaimResult:
        """
        Verify Claim 1: Higher accuracy may exhibit lower certainty.

        Tests correlation between accuracy and average set size.
        Lower set size = higher certainty, so we expect negative correlation.

        Args:
            conformal_method: 'lac' or 'aps'
            dtype: 'float16' or 'float32'

        Returns:
            ClaimResult with correlation analysis
        """
        logger.info(f"Verifying Claim 1 with {conformal_method.upper()}, {dtype}")

        filtered = self._filter_results(conformal_method=conformal_method, dtype=dtype)

        if len(filtered) < 3:
            return ClaimResult(
                claim_id=1,
                claim_description="Higher accuracy may exhibit lower certainty",
                supported=False,
                evidence={'error': 'Insufficient data points', 'count': len(filtered)},
                interpretation="Cannot verify - need more data points"
            )

        accuracies = np.array([r['accuracy'] for r in filtered])
        set_sizes = np.array([r['avg_set_size'] for r in filtered])

        # Pearson correlation
        correlation, p_value = stats.pearsonr(accuracies, set_sizes)

        # Spearman correlation (more robust to outliers)
        spearman_corr, spearman_p = stats.spearmanr(accuracies, set_sizes)

        # Per-task analysis
        task_correlations = {}
        tasks = list(set(r['task'] for r in filtered))
        for task in tasks:
            task_results = [r for r in filtered if r['task'] == task]
            if len(task_results) >= 3:
                task_acc = [r['accuracy'] for r in task_results]
                task_ss = [r['avg_set_size'] for r in task_results]
                t_corr, t_p = stats.pearsonr(task_acc, task_ss)
                task_correlations[task] = {
                    'correlation': float(t_corr),
                    'p_value': float(t_p),
                    'n': len(task_results)
                }

        # Per-model analysis
        model_stats = {}
        models = list(set(r['model'] for r in filtered))
        for model in models:
            model_results = [r for r in filtered if r['model'] == model]
            model_stats[model] = {
                'mean_accuracy': float(np.mean([r['accuracy'] for r in model_results])),
                'mean_set_size': float(np.mean([r['avg_set_size'] for r in model_results])),
                'n_tasks': len(model_results)
            }

        # Claim is supported if negative correlation with p < 0.05
        supported = correlation < 0 and p_value < 0.05

        # Alternative: supported if trend is negative even if not significant
        trend_negative = correlation < 0

        if supported:
            interpretation = (
                f"Claim SUPPORTED: Significant negative correlation (r={correlation:.3f}, p={p_value:.4f}). "
                f"Higher accuracy is associated with smaller prediction sets (higher certainty)."
            )
        elif trend_negative:
            interpretation = (
                f"Claim PARTIALLY SUPPORTED: Negative correlation (r={correlation:.3f}) but not significant (p={p_value:.4f}). "
                f"Trend suggests higher accuracy -> higher certainty, but more data needed."
            )
        else:
            interpretation = (
                f"Claim NOT SUPPORTED: Correlation is positive (r={correlation:.3f}, p={p_value:.4f}). "
                f"This suggests higher accuracy is associated with larger sets (lower certainty) in this data."
            )

        return ClaimResult(
            claim_id=1,
            claim_description="Higher accuracy may exhibit lower certainty (smaller prediction sets)",
            supported=supported,
            statistical_significance=float(p_value),
            evidence={
                'pearson_correlation': float(correlation),
                'pearson_p_value': float(p_value),
                'spearman_correlation': float(spearman_corr),
                'spearman_p_value': float(spearman_p),
                'n_data_points': len(filtered),
                'accuracy_range': [float(accuracies.min()), float(accuracies.max())],
                'set_size_range': [float(set_sizes.min()), float(set_sizes.max())],
                'per_task_correlations': task_correlations,
                'per_model_stats': model_stats,
                'conformal_method': conformal_method,
                'dtype': dtype
            },
            interpretation=interpretation
        )

    def verify_claim2(self, conformal_method: str = 'lac', dtype: str = 'float16') -> ClaimResult:
        """
        Verify Claim 2: Larger models may display greater uncertainty.

        Tests if model size correlates with average set size.

        Args:
            conformal_method: 'lac' or 'aps'
            dtype: 'float16' or 'float32'

        Returns:
            ClaimResult with size vs uncertainty analysis
        """
        logger.info(f"Verifying Claim 2 with {conformal_method.upper()}, {dtype}")

        filtered = self._filter_results(conformal_method=conformal_method, dtype=dtype)

        # Get unique models and their sizes
        model_data = {}
        for r in filtered:
            model = r['model']
            if model not in MODEL_SIZES:
                continue
            if model not in model_data:
                model_data[model] = {
                    'size': MODEL_SIZES[model],
                    'set_sizes': [],
                    'accuracies': [],
                    'tasks': []
                }
            model_data[model]['set_sizes'].append(r['avg_set_size'])
            model_data[model]['accuracies'].append(r['accuracy'])
            model_data[model]['tasks'].append(r['task'])

        if len(model_data) < 3:
            return ClaimResult(
                claim_id=2,
                claim_description="Larger models may display greater uncertainty",
                supported=False,
                evidence={'error': 'Insufficient models', 'count': len(model_data)},
                interpretation="Cannot verify - need more models with known sizes"
            )

        # Aggregate per model
        models = list(model_data.keys())
        sizes = np.array([model_data[m]['size'] for m in models])
        mean_set_sizes = np.array([np.mean(model_data[m]['set_sizes']) for m in models])
        mean_accuracies = np.array([np.mean(model_data[m]['accuracies']) for m in models])

        # Correlation between size and set size
        correlation, p_value = stats.pearsonr(sizes, mean_set_sizes)
        spearman_corr, spearman_p = stats.spearmanr(sizes, mean_set_sizes)

        # Size vs accuracy correlation (for context)
        size_acc_corr, size_acc_p = stats.pearsonr(sizes, mean_accuracies)

        # Model-by-model breakdown
        model_breakdown = {}
        for model in models:
            model_breakdown[model] = {
                'size_billions': MODEL_SIZES[model],
                'mean_set_size': float(np.mean(model_data[model]['set_sizes'])),
                'std_set_size': float(np.std(model_data[model]['set_sizes'])),
                'mean_accuracy': float(np.mean(model_data[model]['accuracies'])),
                'n_tasks': len(model_data[model]['tasks'])
            }

        # Sort by size for easier interpretation
        sorted_models = sorted(model_breakdown.items(), key=lambda x: x[1]['size_billions'])

        # Claim is supported if positive correlation (larger -> more uncertain)
        supported = correlation > 0 and p_value < 0.05
        trend_positive = correlation > 0

        if supported:
            interpretation = (
                f"Claim SUPPORTED: Significant positive correlation (r={correlation:.3f}, p={p_value:.4f}). "
                f"Larger models show greater uncertainty (larger prediction sets)."
            )
        elif trend_positive:
            interpretation = (
                f"Claim PARTIALLY SUPPORTED: Positive correlation (r={correlation:.3f}) but not significant (p={p_value:.4f}). "
                f"Trend suggests larger models -> more uncertainty, but more data needed."
            )
        else:
            interpretation = (
                f"Claim NOT SUPPORTED: Correlation is negative or zero (r={correlation:.3f}, p={p_value:.4f}). "
                f"Larger models do not show greater uncertainty in this data."
            )

        return ClaimResult(
            claim_id=2,
            claim_description="Larger models may display greater uncertainty (larger prediction sets)",
            supported=supported,
            statistical_significance=float(p_value),
            evidence={
                'size_setsize_pearson_correlation': float(correlation),
                'size_setsize_pearson_p_value': float(p_value),
                'size_setsize_spearman_correlation': float(spearman_corr),
                'size_setsize_spearman_p_value': float(spearman_p),
                'size_accuracy_correlation': float(size_acc_corr),
                'size_accuracy_p_value': float(size_acc_p),
                'n_models': len(models),
                'model_breakdown': model_breakdown,
                'models_sorted_by_size': [m[0] for m in sorted_models],
                'conformal_method': conformal_method,
                'dtype': dtype
            },
            interpretation=interpretation
        )

    def verify_claim3(self, conformal_method: str = 'lac', dtype: str = 'float16') -> ClaimResult:
        """
        Verify Claim 3: Instruction-tuning increases uncertainty.

        Compares base vs instruction-tuned model variants.

        Args:
            conformal_method: 'lac' or 'aps'
            dtype: 'float16' or 'float32'

        Returns:
            ClaimResult with base vs instruct comparison
        """
        logger.info(f"Verifying Claim 3 with {conformal_method.upper()}, {dtype}")

        filtered = self._filter_results(conformal_method=conformal_method, dtype=dtype)

        # Find model pairs in the data
        available_pairs = {}
        available_models = set(r['model'] for r in filtered)

        for base_model, instruct_model in MODEL_PAIRS.items():
            if base_model in available_models and instruct_model in available_models:
                available_pairs[base_model] = instruct_model

        if not available_pairs:
            return ClaimResult(
                claim_id=3,
                claim_description="Instruction-tuning increases uncertainty",
                supported=False,
                evidence={
                    'error': 'No base/instruct pairs found',
                    'available_models': list(available_models),
                    'expected_pairs': MODEL_PAIRS
                },
                interpretation="Cannot verify - no matching base/instruct model pairs in data"
            )

        # Compare each pair
        pair_comparisons = {}
        all_base_set_sizes = []
        all_instruct_set_sizes = []

        for base_model, instruct_model in available_pairs.items():
            base_results = [r for r in filtered if r['model'] == base_model]
            instruct_results = [r for r in filtered if r['model'] == instruct_model]

            # Match by task
            base_by_task = {r['task']: r for r in base_results}
            instruct_by_task = {r['task']: r for r in instruct_results}

            common_tasks = set(base_by_task.keys()) & set(instruct_by_task.keys())

            if not common_tasks:
                continue

            task_comparisons = {}
            for task in common_tasks:
                base_ss = base_by_task[task]['avg_set_size']
                instruct_ss = instruct_by_task[task]['avg_set_size']
                base_acc = base_by_task[task]['accuracy']
                instruct_acc = instruct_by_task[task]['accuracy']

                all_base_set_sizes.append(base_ss)
                all_instruct_set_sizes.append(instruct_ss)

                task_comparisons[task] = {
                    'base_set_size': base_ss,
                    'instruct_set_size': instruct_ss,
                    'difference': instruct_ss - base_ss,
                    'instruct_larger': instruct_ss > base_ss,
                    'base_accuracy': base_acc,
                    'instruct_accuracy': instruct_acc
                }

            pair_comparisons[f"{base_model}_vs_{instruct_model}"] = {
                'base_model': base_model,
                'instruct_model': instruct_model,
                'n_tasks_compared': len(common_tasks),
                'tasks': task_comparisons,
                'mean_base_set_size': float(np.mean([t['base_set_size'] for t in task_comparisons.values()])),
                'mean_instruct_set_size': float(np.mean([t['instruct_set_size'] for t in task_comparisons.values()])),
                'tasks_where_instruct_larger': sum(1 for t in task_comparisons.values() if t['instruct_larger'])
            }

        if not all_base_set_sizes:
            return ClaimResult(
                claim_id=3,
                claim_description="Instruction-tuning increases uncertainty",
                supported=False,
                evidence={'error': 'No overlapping tasks between model pairs'},
                interpretation="Cannot verify - no common tasks between base and instruct models"
            )

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(all_instruct_set_sizes, all_base_set_sizes)

        # Wilcoxon signed-rank test (non-parametric)
        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
                all_instruct_set_sizes, all_base_set_sizes, alternative='greater'
            )
        except ValueError:
            wilcoxon_stat, wilcoxon_p = None, None

        # Summary statistics
        mean_diff = np.mean(np.array(all_instruct_set_sizes) - np.array(all_base_set_sizes))
        instruct_larger_count = sum(1 for i, b in zip(all_instruct_set_sizes, all_base_set_sizes) if i > b)

        # Claim supported if instruct has larger sets (more uncertainty) with significance
        supported = mean_diff > 0 and p_value < 0.05
        trend_positive = mean_diff > 0

        if supported:
            interpretation = (
                f"Claim SUPPORTED: Instruction-tuned models have significantly larger prediction sets "
                f"(mean diff={mean_diff:.3f}, p={p_value:.4f}). "
                f"Instruct models show greater uncertainty in {instruct_larger_count}/{len(all_base_set_sizes)} comparisons."
            )
        elif trend_positive:
            interpretation = (
                f"Claim PARTIALLY SUPPORTED: Instruction-tuned models have larger sets (mean diff={mean_diff:.3f}) "
                f"but not significant (p={p_value:.4f}). "
                f"Instruct larger in {instruct_larger_count}/{len(all_base_set_sizes)} comparisons."
            )
        else:
            interpretation = (
                f"Claim NOT SUPPORTED: Instruction-tuned models do NOT have larger prediction sets "
                f"(mean diff={mean_diff:.3f}, p={p_value:.4f}). "
                f"Base models may actually show more uncertainty."
            )

        return ClaimResult(
            claim_id=3,
            claim_description="Instruction-tuning increases uncertainty (larger prediction sets)",
            supported=supported,
            statistical_significance=float(p_value),
            evidence={
                'paired_ttest_t_stat': float(t_stat),
                'paired_ttest_p_value': float(p_value),
                'wilcoxon_stat': float(wilcoxon_stat) if wilcoxon_stat else None,
                'wilcoxon_p_value': float(wilcoxon_p) if wilcoxon_p else None,
                'mean_difference_instruct_minus_base': float(mean_diff),
                'n_comparisons': len(all_base_set_sizes),
                'instruct_larger_count': instruct_larger_count,
                'pair_comparisons': pair_comparisons,
                'conformal_method': conformal_method,
                'dtype': dtype
            },
            interpretation=interpretation
        )

    def verify_all_claims(
        self,
        conformal_method: str = 'lac',
        dtype: str = 'float16'
    ) -> VerificationReport:
        """
        Verify all three claims.

        Args:
            conformal_method: 'lac' or 'aps'
            dtype: 'float16' or 'float32'

        Returns:
            Complete verification report
        """
        logger.info(f"Running full claims verification ({conformal_method.upper()}, {dtype})")

        claim1 = self.verify_claim1(conformal_method, dtype)
        claim2 = self.verify_claim2(conformal_method, dtype)
        claim3 = self.verify_claim3(conformal_method, dtype)

        # Summary
        claims_supported = sum([claim1.supported, claim2.supported, claim3.supported])

        summary = {
            'conformal_method': conformal_method,
            'dtype': dtype,
            'claims_supported': claims_supported,
            'claims_total': 3,
            'claim1_supported': claim1.supported,
            'claim2_supported': claim2.supported,
            'claim3_supported': claim3.supported,
            'overall_assessment': self._get_overall_assessment(claims_supported)
        }

        methodology_notes = [
            f"Analysis performed using {conformal_method.upper()} conformal method",
            f"Data type: {dtype}",
            f"Statistical significance threshold: p < 0.05",
            "Claim 1 uses Pearson/Spearman correlation between accuracy and set size",
            "Claim 2 uses correlation between model size (billions params) and mean set size",
            "Claim 3 uses paired t-test comparing base vs instruction-tuned models",
        ]

        return VerificationReport(
            claim1=claim1,
            claim2=claim2,
            claim3=claim3,
            summary=summary,
            methodology_notes=methodology_notes
        )

    def _get_overall_assessment(self, claims_supported: int) -> str:
        """Generate overall assessment based on number of claims supported."""
        if claims_supported == 3:
            return "FULLY CONSISTENT: All paper claims verified on SLMs"
        elif claims_supported == 2:
            return "MOSTLY CONSISTENT: 2/3 claims verified on SLMs"
        elif claims_supported == 1:
            return "PARTIALLY CONSISTENT: 1/3 claims verified on SLMs"
        else:
            return "NOT CONSISTENT: Paper claims do not hold for SLMs"

    def save_report(self, report: VerificationReport, output_path: str):
        """Save verification report to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def convert_numpy(obj):
            """Convert numpy types to Python native types."""
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj

        report_dict = convert_numpy({
            'claim1': asdict(report.claim1),
            'claim2': asdict(report.claim2),
            'claim3': asdict(report.claim3),
            'summary': report.summary,
            'methodology_notes': report.methodology_notes
        })

        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)

        logger.info(f"Report saved to {output_path}")

    def print_report(self, report: VerificationReport):
        """Print human-readable report."""
        print("\n" + "=" * 80)
        print("PAPER CLAIMS VERIFICATION REPORT")
        print("=" * 80)
        print(f"Method: {report.summary['conformal_method'].upper()}")
        print(f"Data type: {report.summary['dtype']}")
        print()

        for claim in [report.claim1, report.claim2, report.claim3]:
            status = "SUPPORTED" if claim.supported else "NOT SUPPORTED"
            print(f"Claim {claim.claim_id}: {status}")
            print(f"  Description: {claim.claim_description}")
            print(f"  P-value: {claim.statistical_significance:.4f}" if claim.statistical_significance else "  P-value: N/A")
            print(f"  {claim.interpretation}")
            print()

        print("-" * 80)
        print(f"OVERALL: {report.summary['overall_assessment']}")
        print(f"Claims supported: {report.summary['claims_supported']}/3")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Verify paper claims using benchmark results")
    parser.add_argument(
        '--results-path',
        type=str,
        default='./outputs/results/all_results.json',
        help='Path to results JSON file'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='./outputs/results/claims_analysis.json',
        help='Path to save analysis report'
    )
    parser.add_argument(
        '--conformal-method',
        type=str,
        default='lac',
        choices=['lac', 'aps'],
        help='Conformal prediction method to analyze'
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='float16',
        choices=['float16', 'float32'],
        help='Data type to analyze'
    )
    parser.add_argument(
        '--all-methods',
        action='store_true',
        help='Run analysis for both LAC and APS methods'
    )

    args = parser.parse_args()

    verifier = ClaimsVerifier(args.results_path)

    if args.all_methods:
        # Run for both methods
        for method in ['lac', 'aps']:
            for dtype in ['float16', 'float32']:
                try:
                    report = verifier.verify_all_claims(method, dtype)
                    verifier.print_report(report)

                    output_path = args.output_path.replace('.json', f'_{method}_{dtype}.json')
                    verifier.save_report(report, output_path)
                except Exception as e:
                    logger.warning(f"Failed for {method}/{dtype}: {e}")
    else:
        report = verifier.verify_all_claims(args.conformal_method, args.dtype)
        verifier.print_report(report)
        verifier.save_report(report, args.output_path)


if __name__ == "__main__":
    main()
