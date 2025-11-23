"""
APS (Adaptive Prediction Sets) Scorer Module
Implements the APS conformal score function.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

from src.conformal.conformal_base import (
    BaseConformalPredictor,
    ConformalConfig,
    PredictionSet
)

# Setup logger
logger = logging.getLogger(__name__)


class APSScorer(BaseConformalPredictor):
    """
    Adaptive Prediction Sets (APS) conformal predictor.
    
    Score function: s(X, Y) = Σ P(Y') for all Y' with P(Y') ≥ P(Y)
    
    This method produces prediction sets that are more adaptive to the 
    probability distribution compared to LAC. It addresses LAC's limitation
    of potentially undercovering hard instances, but typically produces 
    larger average set sizes.
    
    Reference:
    Romano, Y., Sesia, M., & Candes, E. (2020). Classification with valid 
    and adaptive coverage. Advances in Neural Information Processing Systems, 
    33, 3581-3591.
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
    
    def compute_score(
        self,
        probabilities: np.ndarray,
        true_label_idx: int
    ) -> float:
        """
        Compute APS score: sum of probabilities ranked higher than or equal to true label.
        
        s(X, Y) = Σ P(Y') for all Y' where P(Y') ≥ P(Y)
        
        This is equivalent to summing probabilities in descending order until
        reaching (and including) the true label.
        
        Args:
            probabilities: Predicted probabilities for all options
                Shape: [num_options]
            true_label_idx: Index of the true label
            
        Returns:
            Conformal score (higher score = worse fit)
        """
        # Get probability of the true label
        true_prob = probabilities[true_label_idx]
        
        # Sum probabilities of all options with probability >= true_prob
        # This is equivalent to summing from highest to lowest until reaching true label
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
            probabilities: Predicted probabilities for all options
                Shape: [num_options]
            threshold: Conformal threshold
            option_letters: Letters for each option (A, B, C, D, E, F)
            instance_id: ID for the instance
            
        Returns:
            PredictionSet object
        """
        # Sort indices by probability (descending)
        sorted_indices = np.argsort(probabilities)[::-1]
        
        # Compute scores for all options (for metadata)
        scores = {}
        for i, letter in enumerate(option_letters):
            # Score is cumulative sum up to and including this option
            true_prob = probabilities[i]
            score = np.sum(probabilities[probabilities >= true_prob])
            scores[letter] = float(score)
        
        # Build prediction set by adding options in order of probability
        included_options = []
        cumulative_prob = 0.0
        
        for idx in sorted_indices:
            letter = option_letters[idx]
            prob = probabilities[idx]
            
            # Add this option
            included_options.append(letter)
            cumulative_prob += prob
            
            # Check if we've exceeded the threshold
            # We need: cumulative_prob > threshold
            # But we include at least one option
            if cumulative_prob > threshold and len(included_options) > 0:
                break
        
        # Ensure at least one option is included
        if len(included_options) == 0:
            best_idx = sorted_indices[0]
            included_options = [option_letters[best_idx]]
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
                'cumulative_prob': float(cumulative_prob),
                'sorted_order': [option_letters[i] for i in sorted_indices]
            }
        )
    
    def get_score_statistics(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute statistics about APS scores for a dataset.
        
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
                # Sort probabilities descending
                sorted_probs = np.sort(probs)[::-1]
                
                # Count how many to include
                cumsum = np.cumsum(sorted_probs)
                n_included = np.searchsorted(cumsum, threshold, side='right') + 1
                
                # At least 1, at most all options
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
        """
        Analyze the rank distribution of true labels.
        
        Args:
            probabilities: Predicted probabilities
                Shape: [num_samples, num_options]
            labels: True labels
                Shape: [num_samples]
            
        Returns:
            Dictionary with rank analysis
        """
        ranks = []
        
        for probs, label in zip(probabilities, labels):
            # Get rank of true label (1 = highest probability)
            sorted_indices = np.argsort(probs)[::-1]
            rank = np.where(sorted_indices == label)[0][0] + 1
            ranks.append(rank)
        
        ranks = np.array(ranks)
        
        # Distribution of ranks
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


class APSAnalyzer:
    """Analyzer for APS-specific characteristics."""
    
    @staticmethod
    def compare_with_lac(
        aps_result,
        lac_result
    ) -> Dict[str, Any]:
        """
        Compare APS with LAC results.
        
        Args:
            aps_result: Result from APS predictor
            lac_result: Result from LAC predictor
            
        Returns:
            Comparison statistics
        """
        comparison = {
            'aps': {
                'coverage': aps_result.coverage_rate,
                'avg_set_size': aps_result.average_set_size,
                'meets_guarantee': aps_result.meets_coverage_guarantee()
            },
            'lac': {
                'coverage': lac_result.coverage_rate,
                'avg_set_size': lac_result.average_set_size,
                'meets_guarantee': lac_result.meets_coverage_guarantee()
            }
        }
        
        # Compute differences
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
    def analyze_adaptive_behavior(
        aps_result,
        probabilities: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze how APS adapts to probability distributions.
        
        Args:
            aps_result: Result from APS predictor
            probabilities: Test probabilities
            
        Returns:
            Analysis dictionary
        """
        set_sizes = aps_result.get_set_sizes()
        
        # Compute uncertainty measures
        max_probs = np.max(probabilities, axis=1)
        entropies = -np.sum(
            probabilities * np.log(probabilities + 1e-10), 
            axis=1
        )
        
        # Analyze relationship between uncertainty and set size
        # Higher entropy (more uncertain) should lead to larger sets
        entropy_size_corr = np.corrcoef(entropies, set_sizes)[0, 1]
        
        # Higher max probability (more confident) should lead to smaller sets
        confidence_size_corr = np.corrcoef(max_probs, set_sizes)[0, 1]
        
        # Bin by entropy
        entropy_bins = [0, 0.5, 1.0, 1.5, 2.0, np.max(entropies)]
        binned_sizes = {}
        
        for i in range(len(entropy_bins) - 1):
            mask = (entropies >= entropy_bins[i]) & (entropies < entropy_bins[i+1])
            if np.any(mask):
                bin_name = f'{entropy_bins[i]:.1f}-{entropy_bins[i+1]:.1f}'
                binned_sizes[bin_name] = float(np.mean(set_sizes[mask]))
        
        return {
            'entropy_size_correlation': float(entropy_size_corr),
            'confidence_size_correlation': float(confidence_size_corr),
            'avg_sizes_by_entropy': binned_sizes,
            'high_entropy_avg_size': float(np.mean(set_sizes[entropies > np.median(entropies)])),
            'low_entropy_avg_size': float(np.mean(set_sizes[entropies <= np.median(entropies)]))
        }
    
    @staticmethod
    def analyze_set_composition(
        aps_result,
        probabilities: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze the composition of prediction sets.
        
        Args:
            aps_result: Result from APS predictor
            probabilities: Test probabilities
            
        Returns:
            Composition analysis
        """
        # Analyze what options are typically included
        option_inclusion_count = {
            'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0
        }
        
        for ps in aps_result.prediction_sets:
            for option in ps.options:
                option_inclusion_count[option] += 1
        
        # Normalize to rates
        n = len(aps_result.prediction_sets)
        option_inclusion_rate = {
            opt: count / n 
            for opt, count in option_inclusion_count.items()
        }
        
        # Analyze probability mass in prediction sets
        prob_masses = []
        for ps, probs in zip(aps_result.prediction_sets, probabilities):
            # Get indices of included options
            option_letters = ['A', 'B', 'C', 'D', 'E', 'F']
            indices = [option_letters.index(opt) for opt in ps.options]
            prob_mass = np.sum(probs[indices])
            prob_masses.append(prob_mass)
        
        prob_masses = np.array(prob_masses)
        
        return {
            'option_inclusion_rates': option_inclusion_rate,
            'avg_probability_mass': float(np.mean(prob_masses)),
            'median_probability_mass': float(np.median(prob_masses)),
            'min_probability_mass': float(np.min(prob_masses)),
            'max_probability_mass': float(np.max(prob_masses))
        }


# Example usage
if __name__ == "__main__":
    import logging
    from src.conformal.lac_scorer import LACScorer
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*80)
    print("APS Scorer Test")
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
    
    # Initialize APS scorer
    config = ConformalConfig(alpha=0.1, score_function='aps')
    aps = APSScorer(config)
    
    # Calibrate
    print("\n" + "="*80)
    print("Calibration")
    print("="*80)
    
    threshold = aps.calibrate(
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
    sorted_indices = np.argsort(sample_probs)[::-1]
    for idx in sorted_indices:
        letter = option_letters[idx]
        prob = sample_probs[idx]
        print(f"  {letter}: {prob:.4f}")
    
    pred_set = aps.predict_single(
        probabilities=sample_probs,
        true_label=sample_label,
        option_letters=option_letters,
        instance_id="test_0"
    )
    
    print(f"\nPrediction set: {pred_set.options}")
    print(f"Set size: {pred_set.size}")
    print(f"True label: {pred_set.true_label}")
    print(f"Contains true: {pred_set.contains_true}")
    print(f"Cumulative prob: {pred_set.metadata['cumulative_prob']:.4f}")
    print(f"Sorted order: {pred_set.metadata['sorted_order']}")
    
    # Test batch prediction
    print("\n" + "="*80)
    print("Batch Prediction Test")
    print("="*80)
    
    result = aps.predict(
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
    
    score_stats = aps.get_score_statistics(test_probs, test_labels)
    print(f"\nAPS Score Statistics:")
    for key, value in score_stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Rank distribution analysis
    print("\n" + "="*80)
    print("Rank Distribution Analysis")
    print("="*80)
    
    rank_analysis = aps.analyze_rank_distribution(test_probs, test_labels)
    print(f"\nTrue label rank statistics:")
    print(f"  Mean rank: {rank_analysis['mean_rank']:.2f}")
    print(f"  Top-1 rate: {rank_analysis['top1_rate']:.2%}")
    print(f"  Top-3 rate: {rank_analysis['top3_rate']:.2%}")
    print(f"  Rank distribution: {rank_analysis['rank_distribution']}")
    
    # Compare with LAC
    print("\n" + "="*80)
    print("Comparison with LAC")
    print("="*80)
    
    # Run LAC for comparison
    lac_config = ConformalConfig(alpha=0.1, score_function='lac')
    lac = LACScorer(lac_config)
    lac.calibrate(cal_probs, cal_labels, option_letters)
    lac_result = lac.predict(test_probs, test_labels, option_letters)
    
    analyzer = APSAnalyzer()
    comparison = analyzer.compare_with_lac(result, lac_result)
    
    print(f"\nMethod comparison:")
    print(f"  APS coverage: {comparison['aps']['coverage']:.2%}")
    print(f"  LAC coverage: {comparison['lac']['coverage']:.2%}")
    print(f"  APS avg size: {comparison['aps']['avg_set_size']:.2f}")
    print(f"  LAC avg size: {comparison['lac']['avg_set_size']:.2f}")
    print(f"  Size difference: {comparison['differences']['size_diff']:.2f}")
    
    print(f"\nInstance-level comparison:")
    for key, value in comparison['instance_level'].items():
        print(f"  {key}: {value}")
    
    # Adaptive behavior analysis
    print("\n" + "="*80)
    print("Adaptive Behavior Analysis")
    print("="*80)
    
    adaptive_analysis = analyzer.analyze_adaptive_behavior(result, test_probs)
    print(f"\nAdaptivity analysis:")
    print(f"  Entropy-size correlation: {adaptive_analysis['entropy_size_correlation']:.4f}")
    print(f"  Confidence-size correlation: {adaptive_analysis['confidence_size_correlation']:.4f}")
    print(f"  High entropy avg size: {adaptive_analysis['high_entropy_avg_size']:.2f}")
    print(f"  Low entropy avg size: {adaptive_analysis['low_entropy_avg_size']:.2f}")
    
    # Set composition analysis
    composition = analyzer.analyze_set_composition(result, test_probs)
    print(f"\nSet composition:")
    print(f"  Avg probability mass: {composition['avg_probability_mass']:.2%}")
    print(f"  Option inclusion rates:")
    for opt, rate in sorted(composition['option_inclusion_rates'].items()):
        print(f"    {opt}: {rate:.2%}")