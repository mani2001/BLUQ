"""
LAC (Least Ambiguous set-valued Classifiers) Scorer Module
Implements the LAC conformal score function.
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from src.conformal.conformal_base import (
    BaseConformalPredictor,
    ConformalConfig,
    PredictionSet
)

# Setup logger
logger = logging.getLogger(__name__)


class LACScorer(BaseConformalPredictor):
    """
    Least Ambiguous set-valued Classifiers (LAC) conformal predictor.

    Score function: s(X, Y) = 1 - P(Y)

    This produces the smallest average set size among valid conformal methods.
    However, it may undercover hard instances and overcover easy ones.

    Reference:
    Sadinle, M., Lei, J., & Wasserman, L. (2019). Least ambiguous set-valued
    classifiers with bounded error levels. Journal of the American Statistical
    Association, 114(525), 223-234.
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

        # Track empty set fallbacks
        self._empty_set_count = 0
        self._total_predictions = 0

    def reset_empty_set_stats(self) -> None:
        """Reset empty set statistics."""
        self._empty_set_count = 0
        self._total_predictions = 0

    def get_empty_set_stats(self) -> Dict[str, Any]:
        """
        Get statistics about empty prediction set occurrences.

        Returns:
            Dictionary with empty set statistics
        """
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
            probabilities: Predicted probabilities for all options
                Shape: [num_options]
            true_label_idx: Index of the true label
            
        Returns:
            Conformal score (higher score = worse fit)
        """
        # LAC score is simply 1 minus the probability of the true label
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
        
        The prediction set includes all options where:
        s(X, Y) ≤ threshold
        ⟺ 1 - P(Y) ≤ threshold
        ⟺ P(Y) ≥ 1 - threshold
        
        Args:
            probabilities: Predicted probabilities for all options
                Shape: [num_options]
            threshold: Conformal threshold
            option_letters: Letters for each option (A, B, C, D, E, F)
            instance_id: ID for the instance
            
        Returns:
            PredictionSet object
        """
        # Track total predictions
        self._total_predictions += 1

        # Compute scores for all options
        scores = {}
        included_options = []

        min_prob = 1.0 - threshold

        for i, (letter, prob) in enumerate(zip(option_letters, probabilities)):
            score = 1.0 - prob
            scores[letter] = float(score)

            # Include option if score ≤ threshold (i.e., prob ≥ 1 - threshold)
            if score <= threshold:
                included_options.append(letter)

        # Handle empty set case
        # If no options meet the threshold, include the option with highest probability
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
        """
        Compute statistics about LAC scores for a dataset.
        
        Args:
            probabilities: Predicted probabilities
                Shape: [num_samples, num_options]
            labels: True labels
                Shape: [num_samples]
            
        Returns:
            Dictionary with score statistics
        """
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
        """
        Analyze how set sizes change with different thresholds.
        
        Args:
            probabilities: Test probabilities
                Shape: [num_samples, num_options]
            thresholds: Array of thresholds to test
                If None, uses linspace from 0 to 1
            
        Returns:
            Dictionary with threshold analysis
        """
        if thresholds is None:
            thresholds = np.linspace(0, 1, 50)
        
        avg_set_sizes = []
        
        for threshold in thresholds:
            set_sizes = []
            for probs in probabilities:
                # Count how many options have score ≤ threshold
                scores = 1.0 - probs
                n_included = np.sum(scores <= threshold)
                # Handle empty set
                if n_included == 0:
                    n_included = 1
                set_sizes.append(n_included)
            
            avg_set_sizes.append(np.mean(set_sizes))
        
        return {
            'thresholds': thresholds,
            'avg_set_sizes': np.array(avg_set_sizes)
        }


class LACAnalyzer:
    """Analyzer for LAC-specific characteristics."""
    
    @staticmethod
    def compare_with_naive(
        lac_result,
        probabilities: np.ndarray,
        labels: np.ndarray,
        option_letters: List[str]
    ) -> Dict[str, Any]:
        """
        Compare LAC with naive approach (always predict highest probability).
        
        Args:
            lac_result: Result from LAC predictor
            probabilities: Test probabilities
            labels: True labels
            option_letters: Option letters
            
        Returns:
            Comparison statistics
        """
        # Naive approach: always predict the option with highest probability
        naive_predictions = np.argmax(probabilities, axis=1)
        naive_correct = (naive_predictions == labels).mean()
        
        # LAC approach: check if true label is in prediction set
        lac_coverage = lac_result.coverage_rate
        lac_avg_size = lac_result.average_set_size
        
        # Calculate how often LAC set size is 1 (making a definite prediction)
        singleton_sets = sum(1 for ps in lac_result.prediction_sets if ps.size == 1)
        singleton_rate = singleton_sets / len(lac_result.prediction_sets)
        
        # For singleton sets, check accuracy
        singleton_correct = 0
        for ps, label in zip(lac_result.prediction_sets, labels):
            if ps.size == 1:
                predicted = option_letters.index(ps.options[0])
                if predicted == label:
                    singleton_correct += 1
        
        singleton_accuracy = singleton_correct / singleton_sets if singleton_sets > 0 else 0
        
        return {
            'naive_accuracy': float(naive_correct),
            'lac_coverage': float(lac_coverage),
            'lac_avg_set_size': float(lac_avg_size),
            'singleton_rate': float(singleton_rate),
            'singleton_accuracy': float(singleton_accuracy),
            'coverage_improvement': float(lac_coverage - naive_correct)
        }
    
    @staticmethod
    def analyze_probability_distribution(
        lac_result,
        probabilities: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze relationship between probability distribution and set sizes.
        
        Args:
            lac_result: Result from LAC predictor
            probabilities: Test probabilities
            
        Returns:
            Analysis dictionary
        """
        max_probs = np.max(probabilities, axis=1)
        entropies = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
        set_sizes = lac_result.get_set_sizes()
        
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
        
        # Compute average set size per bin
        avg_sizes_per_bin = {
            bin_name: np.mean(sizes) if sizes else 0
            for bin_name, sizes in binned_sizes.items()
        }
        
        # Correlation between confidence and set size
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


# Example usage
if __name__ == "__main__":
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*80)
    print("LAC Scorer Test")
    print("="*80)
    
    # Create sample data
    np.random.seed(42)
    
    # Calibration data (50 samples, 6 options)
    n_cal = 50
    n_options = 6
    cal_probs = np.random.dirichlet(np.ones(n_options), size=n_cal)
    cal_labels = np.random.randint(0, n_options, size=n_cal)
    
    # Test data (20 samples)
    n_test = 20
    test_probs = np.random.dirichlet(np.ones(n_options), size=n_test)
    test_labels = np.random.randint(0, n_options, size=n_test)
    
    option_letters = ['A', 'B', 'C', 'D', 'E', 'F']
    
    # Initialize LAC scorer
    config = ConformalConfig(alpha=0.1, score_function='lac')
    lac = LACScorer(config)
    
    # Calibrate
    print("\n" + "="*80)
    print("Calibration")
    print("="*80)
    
    threshold = lac.calibrate(
        calibration_probabilities=cal_probs,
        calibration_labels=cal_labels,
        option_letters=option_letters
    )
    
    # Test single prediction
    print("\n" + "="*80)
    print("Single Prediction Test")
    print("="*80)
    
    sample_probs = test_probs[0]
    sample_label = option_letters[test_labels[0]]
    
    print(f"\nSample probabilities:")
    for letter, prob in zip(option_letters, sample_probs):
        print(f"  {letter}: {prob:.4f}")
    
    pred_set = lac.predict_single(
        probabilities=sample_probs,
        true_label=sample_label,
        option_letters=option_letters,
        instance_id="test_0"
    )
    
    print(f"\nPrediction set: {pred_set.options}")
    print(f"Set size: {pred_set.size}")
    print(f"True label: {pred_set.true_label}")
    print(f"Contains true: {pred_set.contains_true}")
    
    print(f"\nScores for all options:")
    for letter, score in pred_set.scores.items():
        in_set = "✓" if letter in pred_set.options else " "
        print(f"  [{in_set}] {letter}: {score:.4f}")
    
    # Test batch prediction
    print("\n" + "="*80)
    print("Batch Prediction Test")
    print("="*80)
    
    result = lac.predict(
        test_probabilities=test_probs,
        test_labels=test_labels,
        option_letters=option_letters
    )
    
    print(f"\nResults:")
    print(f"  Test instances: {result.num_test}")
    print(f"  Coverage rate: {result.coverage_rate:.2%}")
    print(f"  Target coverage: {result.config.target_coverage:.2%}")
    print(f"  Meets guarantee: {result.meets_coverage_guarantee()}")
    print(f"  Average set size: {result.average_set_size:.2f}")
    
    print(f"\nSet size distribution:")
    for size, count in sorted(result.set_size_distribution.items()):
        print(f"  Size {size}: {count} instances")
    
    # Score statistics
    print("\n" + "="*80)
    print("Score Statistics")
    print("="*80)
    
    score_stats = lac.get_score_statistics(test_probs, test_labels)
    print(f"\nLAC Score Statistics:")
    for key, value in score_stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Threshold sensitivity analysis
    print("\n" + "="*80)
    print("Threshold Sensitivity Analysis")
    print("="*80)
    
    sensitivity = lac.analyze_threshold_sensitivity(test_probs)
    
    print(f"\nThreshold range: [{sensitivity['thresholds'].min():.2f}, "
          f"{sensitivity['thresholds'].max():.2f}]")
    print(f"Set size range: [{sensitivity['avg_set_sizes'].min():.2f}, "
          f"{sensitivity['avg_set_sizes'].max():.2f}]")
    print(f"At threshold {threshold:.4f}: "
          f"avg size = {result.average_set_size:.2f}")
    
    # LAC analysis
    print("\n" + "="*80)
    print("LAC Analysis")
    print("="*80)
    
    analyzer = LACAnalyzer()
    
    comparison = analyzer.compare_with_naive(
        result, test_probs, test_labels, option_letters
    )
    
    print(f"\nComparison with naive approach:")
    print(f"  Naive accuracy: {comparison['naive_accuracy']:.2%}")
    print(f"  LAC coverage: {comparison['lac_coverage']:.2%}")
    print(f"  LAC avg set size: {comparison['lac_avg_set_size']:.2f}")
    print(f"  Singleton rate: {comparison['singleton_rate']:.2%}")
    print(f"  Singleton accuracy: {comparison['singleton_accuracy']:.2%}")
    
    prob_analysis = analyzer.analyze_probability_distribution(result, test_probs)
    
    print(f"\nProbability distribution analysis:")
    print(f"  Confidence-size correlation: {prob_analysis['confidence_size_correlation']:.4f}")
    print(f"  Avg sizes by confidence bin:")
    for bin_name, avg_size in prob_analysis['avg_sizes_by_confidence'].items():
        print(f"    {bin_name}: {avg_size:.2f}")