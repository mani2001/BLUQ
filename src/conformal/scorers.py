"""
Conformal Score Functions Module
Implements LAC (Least Ambiguous set-valued Classifiers) and APS (Adaptive Prediction Sets).

References:
- LAC: Sadinle et al. (2019). Least ambiguous set-valued classifiers with bounded error levels.
- APS: Romano et al. (2020). Classification with valid and adaptive coverage.
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional

from src.conformal.conformal_base import (
    BaseConformalPredictor,
    ConformalConfig,
    PredictionSet
)

logger = logging.getLogger(__name__)


class LACScorer(BaseConformalPredictor):
    """
    Least Ambiguous set-valued Classifiers (LAC) conformal predictor.

    Score function: s(X, Y) = 1 - P(Y)

    This produces the smallest average set size among valid conformal methods.
    However, it may undercover hard instances and overcover easy ones.
    """

    def __init__(self, config: Optional[ConformalConfig] = None):
        """
        Initialize LAC scorer.

        Args:
            config: Configuration for conformal prediction
        """
        if config is None:
            config = ConformalConfig(score_function='lac')
        elif config.score_function != 'lac':
            logger.warning(
                f"Config has score_function='{config.score_function}', "
                f"but LAC scorer requires 'lac'. Overriding."
            )
            config.score_function = 'lac'

        super().__init__(config)
        logger.info("Initialized LAC (Least Ambiguous set-valued Classifiers) scorer")

        self._empty_set_count = 0
        self._total_predictions = 0

    def reset_empty_set_stats(self) -> None:
        """Reset empty set statistics."""
        self._empty_set_count = 0
        self._total_predictions = 0

    def get_empty_set_stats(self) -> Dict[str, Any]:
        """Get statistics about empty prediction set occurrences."""
        rate = self._empty_set_count / self._total_predictions if self._total_predictions > 0 else 0.0
        return {
            'empty_set_count': self._empty_set_count,
            'total_predictions': self._total_predictions,
            'empty_set_rate': rate
        }

    def compute_score(
        self,
        probabilities: np.ndarray,
        true_label_idx: int
    ) -> float:
        """
        Compute LAC score: s(X, Y) = 1 - P(Y)

        Args:
            probabilities: Predicted probabilities for all options [num_options]
            true_label_idx: Index of the true label

        Returns:
            Conformal score (higher score = worse fit)
        """
        score = 1.0 - probabilities[true_label_idx]
        return float(score)

    def create_prediction_set(
        self,
        probabilities: np.ndarray,
        threshold: float,
        option_letters: List[str],
        instance_id: str = "unknown"
    ) -> PredictionSet:
        """
        Create prediction set using LAC method.

        Includes all options where: s(X, Y) <= threshold, i.e., P(Y) >= 1 - threshold

        Args:
            probabilities: Predicted probabilities for all options [num_options]
            threshold: Conformal threshold
            option_letters: Letters for each option (A, B, C, D, E, F)
            instance_id: ID for the instance

        Returns:
            PredictionSet object
        """
        self._total_predictions += 1

        scores = {}
        included_options = []
        min_prob = 1.0 - threshold

        for i, (letter, prob) in enumerate(zip(option_letters, probabilities)):
            score = 1.0 - prob
            scores[letter] = float(score)

            if score <= threshold:
                included_options.append(letter)

        # Handle empty set case
        empty_set_fallback = False
        if len(included_options) == 0:
            best_idx = np.argmax(probabilities)
            included_options = [option_letters[best_idx]]
            empty_set_fallback = True
            self._empty_set_count += 1
            logger.debug(
                f"Empty prediction set for {instance_id}, "
                f"adding highest probability option: {included_options[0]}"
            )

        return PredictionSet(
            instance_id=instance_id,
            options=included_options,
            scores=scores,
            threshold=threshold,
            size=len(included_options),
            metadata={
                'method': 'lac',
                'min_required_prob': min_prob,
                'empty_set_fallback': empty_set_fallback
            }
        )

    def get_score_statistics(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute statistics about LAC scores for a dataset."""
        scores = []
        for probs, label in zip(probabilities, labels):
            score = self.compute_score(probs, label)
            scores.append(score)

        scores = np.array(scores)

        return {
            'mean': float(np.mean(scores)),
            'median': float(np.median(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'q25': float(np.percentile(scores, 25)),
            'q75': float(np.percentile(scores, 75))
        }

    def analyze_threshold_sensitivity(
        self,
        probabilities: np.ndarray,
        thresholds: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Analyze how set sizes change with different thresholds."""
        if thresholds is None:
            thresholds = np.linspace(0, 1, 50)

        avg_set_sizes = []

        for threshold in thresholds:
            set_sizes = []
            for probs in probabilities:
                scores = 1.0 - probs
                n_included = np.sum(scores <= threshold)
                if n_included == 0:
                    n_included = 1
                set_sizes.append(n_included)

            avg_set_sizes.append(np.mean(set_sizes))

        return {
            'thresholds': thresholds,
            'avg_set_sizes': np.array(avg_set_sizes)
        }


class APSScorer(BaseConformalPredictor):
    """
    Adaptive Prediction Sets (APS) conformal predictor.

    Score function: s(X, Y) = sum of P(Y') for all Y' with P(Y') >= P(Y)

    This method produces prediction sets that are more adaptive to the
    probability distribution compared to LAC. It addresses LAC's limitation
    of potentially undercovering hard instances, but typically produces
    larger average set sizes.
    """

    def __init__(self, config: Optional[ConformalConfig] = None):
        """
        Initialize APS scorer.

        Args:
            config: Configuration for conformal prediction
        """
        if config is None:
            config = ConformalConfig(score_function='aps')
        elif config.score_function != 'aps':
            logger.warning(
                f"Config has score_function='{config.score_function}', "
                f"but APS scorer requires 'aps'. Overriding."
            )
            config.score_function = 'aps'

        super().__init__(config)
        logger.info("Initialized APS (Adaptive Prediction Sets) scorer")

        self._empty_set_count = 0
        self._total_predictions = 0

    def reset_empty_set_stats(self) -> None:
        """Reset empty set statistics."""
        self._empty_set_count = 0
        self._total_predictions = 0

    def get_empty_set_stats(self) -> Dict[str, Any]:
        """Get statistics about empty prediction set occurrences."""
        rate = self._empty_set_count / self._total_predictions if self._total_predictions > 0 else 0.0
        return {
            'empty_set_count': self._empty_set_count,
            'total_predictions': self._total_predictions,
            'empty_set_rate': rate
        }

    def compute_score(
        self,
        probabilities: np.ndarray,
        true_label_idx: int
    ) -> float:
        """
        Compute APS score: sum of probabilities ranked higher than or equal to true label.

        s(X, Y) = sum of P(Y') for all Y' where P(Y') >= P(Y)

        Args:
            probabilities: Predicted probabilities for all options [num_options]
            true_label_idx: Index of the true label

        Returns:
            Conformal score (higher score = worse fit)
        """
        true_prob = probabilities[true_label_idx]
        score = np.sum(probabilities[probabilities >= true_prob])
        return float(score)

    def create_prediction_set(
        self,
        probabilities: np.ndarray,
        threshold: float,
        option_letters: List[str],
        instance_id: str = "unknown"
    ) -> PredictionSet:
        """
        Create prediction set using APS method.

        The prediction set is constructed by:
        1. Sort options by probability (descending)
        2. Add options starting from highest probability
        3. Stop when cumulative probability > threshold

        Args:
            probabilities: Predicted probabilities for all options [num_options]
            threshold: Conformal threshold
            option_letters: Letters for each option (A, B, C, D, E, F)
            instance_id: ID for the instance

        Returns:
            PredictionSet object
        """
        self._total_predictions += 1

        sorted_indices = np.argsort(probabilities)[::-1]

        # Compute scores for all options (for metadata)
        scores = {}
        for i, letter in enumerate(option_letters):
            true_prob = probabilities[i]
            score = np.sum(probabilities[probabilities >= true_prob])
            scores[letter] = float(score)

        # Build prediction set by adding options in order of probability
        included_options = []
        cumulative_prob = 0.0

        for idx in sorted_indices:
            letter = option_letters[idx]
            prob = probabilities[idx]

            included_options.append(letter)
            cumulative_prob += prob

            if cumulative_prob > threshold and len(included_options) > 0:
                break

        # Ensure at least one option is included
        empty_set_fallback = False
        if len(included_options) == 0:
            best_idx = sorted_indices[0]
            included_options = [option_letters[best_idx]]
            empty_set_fallback = True
            self._empty_set_count += 1
            logger.debug(
                f"Empty prediction set for {instance_id}, "
                f"adding highest probability option: {included_options[0]}"
            )

        return PredictionSet(
            instance_id=instance_id,
            options=included_options,
            scores=scores,
            threshold=threshold,
            size=len(included_options),
            metadata={
                'method': 'aps',
                'empty_set_fallback': empty_set_fallback,
                'cumulative_prob': float(cumulative_prob),
                'sorted_order': [option_letters[i] for i in sorted_indices]
            }
        )

    def get_score_statistics(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute statistics about APS scores for a dataset."""
        scores = []
        for probs, label in zip(probabilities, labels):
            score = self.compute_score(probs, label)
            scores.append(score)

        scores = np.array(scores)

        return {
            'mean': float(np.mean(scores)),
            'median': float(np.median(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'q25': float(np.percentile(scores, 25)),
            'q75': float(np.percentile(scores, 75))
        }

    def analyze_threshold_sensitivity(
        self,
        probabilities: np.ndarray,
        thresholds: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Analyze how set sizes change with different thresholds."""
        if thresholds is None:
            thresholds = np.linspace(0, 1, 50)

        avg_set_sizes = []

        for threshold in thresholds:
            set_sizes = []
            for probs in probabilities:
                sorted_probs = np.sort(probs)[::-1]
                cumsum = np.cumsum(sorted_probs)
                n_included = np.searchsorted(cumsum, threshold, side='right') + 1
                n_included = max(1, min(n_included, len(probs)))
                set_sizes.append(n_included)

            avg_set_sizes.append(np.mean(set_sizes))

        return {
            'thresholds': thresholds,
            'avg_set_sizes': np.array(avg_set_sizes)
        }

    def analyze_rank_distribution(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze the rank distribution of true labels."""
        ranks = []

        for probs, label in zip(probabilities, labels):
            sorted_indices = np.argsort(probs)[::-1]
            rank = np.where(sorted_indices == label)[0][0] + 1
            ranks.append(rank)

        ranks = np.array(ranks)

        rank_distribution = {}
        for rank in range(1, len(probabilities[0]) + 1):
            count = np.sum(ranks == rank)
            rank_distribution[rank] = int(count)

        return {
            'mean_rank': float(np.mean(ranks)),
            'median_rank': float(np.median(ranks)),
            'rank_distribution': rank_distribution,
            'top1_rate': float(np.mean(ranks == 1)),
            'top3_rate': float(np.mean(ranks <= 3)),
            'worst_rank': int(np.max(ranks))
        }


class ConformalAnalyzer:
    """Unified analyzer for conformal prediction results."""

    @staticmethod
    def compare_methods(
        lac_result,
        aps_result
    ) -> Dict[str, Any]:
        """
        Compare LAC and APS results.

        Args:
            lac_result: Result from LAC predictor
            aps_result: Result from APS predictor

        Returns:
            Comparison statistics
        """
        comparison = {
            'lac': {
                'coverage': lac_result.coverage_rate,
                'avg_set_size': lac_result.average_set_size,
                'meets_guarantee': lac_result.meets_coverage_guarantee()
            },
            'aps': {
                'coverage': aps_result.coverage_rate,
                'avg_set_size': aps_result.average_set_size,
                'meets_guarantee': aps_result.meets_coverage_guarantee()
            }
        }

        comparison['differences'] = {
            'coverage_diff': aps_result.coverage_rate - lac_result.coverage_rate,
            'size_diff': aps_result.average_set_size - lac_result.average_set_size,
            'aps_larger_sets': aps_result.average_set_size > lac_result.average_set_size
        }

        # Instance-level comparison
        size_differences = []
        for aps_ps, lac_ps in zip(aps_result.prediction_sets, lac_result.prediction_sets):
            size_diff = aps_ps.size - lac_ps.size
            size_differences.append(size_diff)

        size_differences = np.array(size_differences)

        comparison['instance_level'] = {
            'aps_larger_count': int(np.sum(size_differences > 0)),
            'lac_larger_count': int(np.sum(size_differences < 0)),
            'equal_count': int(np.sum(size_differences == 0)),
            'mean_size_diff': float(np.mean(size_differences)),
            'max_size_diff': int(np.max(np.abs(size_differences)))
        }

        return comparison

    @staticmethod
    def analyze_probability_distribution(
        result,
        probabilities: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze relationship between probability distribution and set sizes."""
        max_probs = np.max(probabilities, axis=1)
        entropies = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
        set_sizes = result.get_set_sizes()

        # Bin by max probability
        prob_bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
        binned_sizes = {f'{prob_bins[i]:.1f}-{prob_bins[i+1]:.1f}': []
                        for i in range(len(prob_bins)-1)}

        for max_prob, size in zip(max_probs, set_sizes):
            for i in range(len(prob_bins)-1):
                if prob_bins[i] <= max_prob < prob_bins[i+1]:
                    key = f'{prob_bins[i]:.1f}-{prob_bins[i+1]:.1f}'
                    binned_sizes[key].append(size)
                    break

        avg_sizes_per_bin = {
            bin_name: np.mean(sizes) if sizes else 0
            for bin_name, sizes in binned_sizes.items()
        }

        correlation = np.corrcoef(max_probs, set_sizes)[0, 1]

        return {
            'max_prob_range': [float(max_probs.min()), float(max_probs.max())],
            'entropy_range': [float(entropies.min()), float(entropies.max())],
            'avg_sizes_by_confidence': avg_sizes_per_bin,
            'confidence_size_correlation': float(correlation),
            'high_confidence_avg_size': float(np.mean(set_sizes[max_probs > 0.9]))
                if np.any(max_probs > 0.9) else None,
            'low_confidence_avg_size': float(np.mean(set_sizes[max_probs < 0.5]))
                if np.any(max_probs < 0.5) else None
        }

    @staticmethod
    def compare_with_naive(
        result,
        probabilities: np.ndarray,
        labels: np.ndarray,
        option_letters: List[str]
    ) -> Dict[str, Any]:
        """Compare conformal prediction with naive approach (always predict highest probability)."""
        naive_predictions = np.argmax(probabilities, axis=1)
        naive_correct = (naive_predictions == labels).mean()

        coverage = result.coverage_rate
        avg_size = result.average_set_size

        # Calculate singleton set statistics
        singleton_sets = sum(1 for ps in result.prediction_sets if ps.size == 1)
        singleton_rate = singleton_sets / len(result.prediction_sets)

        singleton_correct = 0
        for ps, label in zip(result.prediction_sets, labels):
            if ps.size == 1:
                predicted = option_letters.index(ps.options[0])
                if predicted == label:
                    singleton_correct += 1

        singleton_accuracy = singleton_correct / singleton_sets if singleton_sets > 0 else 0

        return {
            'naive_accuracy': float(naive_correct),
            'conformal_coverage': float(coverage),
            'conformal_avg_set_size': float(avg_size),
            'singleton_rate': float(singleton_rate),
            'singleton_accuracy': float(singleton_accuracy),
            'coverage_improvement': float(coverage - naive_correct)
        }
