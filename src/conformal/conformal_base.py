"""
Conformal Prediction Base Module
Abstract base class and core utilities for conformal prediction methods.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class ConformalConfig:
    """Configuration for conformal prediction."""
    # Error rate (alpha)
    alpha: float = 0.1  # Coverage should be at least 1 - alpha
    
    # Score function type
    score_function: str = "lac"  # 'lac' or 'aps'
    
    # Randomization (for exact coverage)
    use_randomization: bool = False
    random_seed: Optional[int] = None
    
    # Validation
    validate_coverage: bool = True
    
    def __post_init__(self):
        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1, got {self.alpha}")
        
        if self.score_function not in ['lac', 'aps']:
            raise ValueError(
                f"score_function must be 'lac' or 'aps', got {self.score_function}"
            )
        
        self.target_coverage = 1 - self.alpha
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'alpha': self.alpha,
            'score_function': self.score_function,
            'use_randomization': self.use_randomization,
            'random_seed': self.random_seed,
            'validate_coverage': self.validate_coverage,
            'target_coverage': self.target_coverage
        }


@dataclass
class PredictionSet:
    """Represents a prediction set for a single instance."""
    instance_id: str
    options: List[str]  # Options included in the prediction set
    scores: Dict[str, float]  # Conformal scores for each option
    threshold: float  # The conformal threshold used
    size: int  # Size of the prediction set
    contains_true: Optional[bool] = None  # Whether true label is in the set
    true_label: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __len__(self) -> int:
        return self.size
    
    def __contains__(self, option: str) -> bool:
        return option in self.options
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'instance_id': self.instance_id,
            'options': self.options,
            'scores': self.scores,
            'threshold': self.threshold,
            'size': self.size,
            'contains_true': self.contains_true,
            'true_label': self.true_label,
            'metadata': self.metadata or {}
        }


@dataclass
class ConformalPredictionResult:
    """Results from conformal prediction on a dataset."""
    prediction_sets: List[PredictionSet]
    config: ConformalConfig
    threshold: float
    
    # Calibration statistics
    calibration_scores: np.ndarray
    num_calibration: int
    
    # Test statistics
    num_test: int
    coverage_rate: float
    average_set_size: float
    
    # Per-set-size statistics
    set_size_distribution: Dict[int, int]
    
    def __len__(self) -> int:
        return len(self.prediction_sets)
    
    def __getitem__(self, idx: int) -> PredictionSet:
        return self.prediction_sets[idx]
    
    def get_coverage(self) -> float:
        """Get empirical coverage rate."""
        return self.coverage_rate
    
    def get_average_size(self) -> float:
        """Get average prediction set size."""
        return self.average_set_size
    
    def get_set_sizes(self) -> np.ndarray:
        """Get array of all set sizes."""
        return np.array([ps.size for ps in self.prediction_sets])
    
    def get_predictions(self) -> List[List[str]]:
        """Get all prediction sets as lists of options."""
        return [ps.options for ps in self.prediction_sets]
    
    def meets_coverage_guarantee(self) -> bool:
        """Check if empirical coverage meets the theoretical guarantee."""
        return self.coverage_rate >= self.config.target_coverage
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'config': self.config.to_dict(),
            'threshold': float(self.threshold),
            'num_calibration': self.num_calibration,
            'num_test': self.num_test,
            'coverage_rate': float(self.coverage_rate),
            'average_set_size': float(self.average_set_size),
            'set_size_distribution': {
                int(k): int(v) for k, v in self.set_size_distribution.items()
            },
            'prediction_sets': [ps.to_dict() for ps in self.prediction_sets]
        }


class BaseConformalPredictor(ABC):
    """Abstract base class for conformal prediction methods."""
    
    def __init__(self, config: Optional[ConformalConfig] = None):
        """
        Initialize the conformal predictor.
        
        Args:
            config: Configuration for conformal prediction
        """
        self.config = config or ConformalConfig()
        self.threshold = None
        self.calibration_scores = None
        self.is_calibrated = False
        
        # Set random seed if specified
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        logger.info(f"Initialized {self.__class__.__name__}")
        logger.info(f"  Alpha: {self.config.alpha}")
        logger.info(f"  Target coverage: {self.config.target_coverage:.1%}")
    
    @abstractmethod
    def compute_score(
        self,
        probabilities: np.ndarray,
        true_label_idx: int
    ) -> float:
        """
        Compute conformal score for a single instance.
        
        Args:
            probabilities: Predicted probabilities for all options
            true_label_idx: Index of the true label
            
        Returns:
            Conformal score (higher is worse)
        """
        pass
    
    @abstractmethod
    def create_prediction_set(
        self,
        probabilities: np.ndarray,
        threshold: float,
        option_letters: List[str],
        instance_id: str = "unknown"
    ) -> PredictionSet:
        """
        Create prediction set for a single instance.
        
        Args:
            probabilities: Predicted probabilities for all options
            threshold: Conformal threshold
            option_letters: Letters for each option (A, B, C, D, E, F)
            instance_id: ID for the instance
            
        Returns:
            PredictionSet object
        """
        pass
    
    def calibrate(
        self,
        calibration_probabilities: np.ndarray,
        calibration_labels: np.ndarray,
        option_letters: Optional[List[str]] = None
    ) -> float:
        """
        Calibrate the conformal predictor using calibration data.
        
        Args:
            calibration_probabilities: Probabilities for calibration set
                Shape: [num_calibration, num_options]
            calibration_labels: True labels for calibration set
                Shape: [num_calibration]
            option_letters: Letters for options (default: A-F)
            
        Returns:
            Computed threshold
        """
        if option_letters is None:
            option_letters = ['A', 'B', 'C', 'D', 'E', 'F']
        
        n = len(calibration_probabilities)
        logger.info(f"Calibrating with {n} samples...")
        
        # Compute scores for calibration set
        scores = []
        for probs, label in zip(calibration_probabilities, calibration_labels):
            score = self.compute_score(probs, label)
            scores.append(score)
        
        self.calibration_scores = np.array(scores)
        
        # Compute threshold as quantile
        self.threshold = self._compute_threshold(self.calibration_scores, n)
        
        self.is_calibrated = True
        
        logger.info(f"Calibration complete")
        logger.info(f"  Threshold: {self.threshold:.4f}")
        logger.info(f"  Score range: [{self.calibration_scores.min():.4f}, "
                   f"{self.calibration_scores.max():.4f}]")
        
        return self.threshold
    
    def _compute_threshold(self, scores: np.ndarray, n: int) -> float:
        """
        Compute conformal threshold from calibration scores.
        
        Uses the formula: quantile at ⌈(n+1)(1-α)⌉/n
        
        Args:
            scores: Calibration scores
            n: Number of calibration samples
            
        Returns:
            Threshold value
        """
        # Compute the quantile level
        quantile_level = np.ceil((n + 1) * (1 - self.config.alpha)) / n
        
        # Ensure quantile level is in valid range [0, 1]
        quantile_level = min(quantile_level, 1.0)
        
        # Compute threshold
        threshold = np.quantile(scores, quantile_level)
        
        logger.debug(f"Quantile level: {quantile_level:.4f}")
        logger.debug(f"Threshold: {threshold:.4f}")
        
        return threshold
    
    def predict(
        self,
        test_probabilities: np.ndarray,
        test_labels: Optional[np.ndarray] = None,
        option_letters: Optional[List[str]] = None,
        instance_ids: Optional[List[str]] = None
    ) -> ConformalPredictionResult:
        """
        Generate prediction sets for test data.
        
        Args:
            test_probabilities: Probabilities for test set
                Shape: [num_test, num_options]
            test_labels: True labels for test set (optional, for evaluation)
                Shape: [num_test]
            option_letters: Letters for options (default: A-F)
            instance_ids: IDs for test instances
            
        Returns:
            ConformalPredictionResult with all prediction sets
        """
        if not self.is_calibrated:
            raise RuntimeError(
                "Must calibrate the predictor before making predictions. "
                "Call calibrate() first."
            )
        
        if option_letters is None:
            option_letters = ['A', 'B', 'C', 'D', 'E', 'F']
        
        if instance_ids is None:
            instance_ids = [f"test_{i}" for i in range(len(test_probabilities))]
        
        num_test = len(test_probabilities)
        logger.info(f"Generating prediction sets for {num_test} test instances...")
        
        # Generate prediction sets
        prediction_sets = []
        for i, probs in enumerate(test_probabilities):
            pred_set = self.create_prediction_set(
                probabilities=probs,
                threshold=self.threshold,
                option_letters=option_letters,
                instance_id=instance_ids[i]
            )
            
            # Add true label info if available
            if test_labels is not None:
                true_label = option_letters[test_labels[i]]
                pred_set.true_label = true_label
                pred_set.contains_true = true_label in pred_set.options
            
            prediction_sets.append(pred_set)
        
        # Compute statistics
        set_sizes = np.array([ps.size for ps in prediction_sets])
        average_set_size = np.mean(set_sizes)
        
        # Coverage rate (if labels available)
        coverage_rate = 0.0
        if test_labels is not None:
            coverage_rate = np.mean([ps.contains_true for ps in prediction_sets])
        
        # Set size distribution
        set_size_distribution = {}
        for size in set_sizes:
            set_size_distribution[int(size)] = set_size_distribution.get(int(size), 0) + 1
        
        result = ConformalPredictionResult(
            prediction_sets=prediction_sets,
            config=self.config,
            threshold=self.threshold,
            calibration_scores=self.calibration_scores,
            num_calibration=len(self.calibration_scores),
            num_test=num_test,
            coverage_rate=coverage_rate,
            average_set_size=average_set_size,
            set_size_distribution=set_size_distribution
        )
        
        logger.info(f"Prediction complete")
        logger.info(f"  Average set size: {average_set_size:.2f}")
        if test_labels is not None:
            logger.info(f"  Coverage rate: {coverage_rate:.2%}")
            if self.config.validate_coverage:
                meets_guarantee = result.meets_coverage_guarantee()
                logger.info(f"  Meets coverage guarantee: {meets_guarantee}")
        
        return result
    
    def predict_single(
        self,
        probabilities: np.ndarray,
        true_label: Optional[str] = None,
        option_letters: Optional[List[str]] = None,
        instance_id: str = "unknown"
    ) -> PredictionSet:
        """
        Generate prediction set for a single instance.
        
        Args:
            probabilities: Predicted probabilities
            true_label: True label (optional)
            option_letters: Letters for options (default: A-F)
            instance_id: Instance ID
            
        Returns:
            PredictionSet
        """
        if not self.is_calibrated:
            raise RuntimeError("Must calibrate before making predictions")
        
        if option_letters is None:
            option_letters = ['A', 'B', 'C', 'D', 'E', 'F']
        
        pred_set = self.create_prediction_set(
            probabilities=probabilities,
            threshold=self.threshold,
            option_letters=option_letters,
            instance_id=instance_id
        )
        
        if true_label is not None:
            pred_set.true_label = true_label
            pred_set.contains_true = true_label in pred_set.options
        
        return pred_set
    
    def update_alpha(self, new_alpha: float) -> None:
        """
        Update the error rate (requires recalibration).
        
        Args:
            new_alpha: New error rate
        """
        if not 0 < new_alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1, got {new_alpha}")
        
        self.config.alpha = new_alpha
        self.config.target_coverage = 1 - new_alpha
        
        # Recalibrate if we have calibration scores
        if self.calibration_scores is not None:
            n = len(self.calibration_scores)
            self.threshold = self._compute_threshold(self.calibration_scores, n)
            logger.info(f"Updated alpha to {new_alpha}, new threshold: {self.threshold:.4f}")
        else:
            logger.info(f"Updated alpha to {new_alpha}, recalibration needed")
            self.is_calibrated = False


class ConformalPredictionValidator:
    """Validator for conformal prediction results."""
    
    @staticmethod
    def validate_coverage(
        result: ConformalPredictionResult,
        tolerance: float = 0.01
    ) -> Dict[str, Any]:
        """
        Validate that coverage guarantee is met.
        
        Args:
            result: Conformal prediction result
            tolerance: Acceptable deviation from target coverage
            
        Returns:
            Dictionary with validation results
        """
        target = result.config.target_coverage
        actual = result.coverage_rate
        
        meets_guarantee = actual >= target
        deviation = actual - target
        within_tolerance = abs(deviation) <= tolerance
        
        validation_result = {
            'meets_guarantee': meets_guarantee,
            'within_tolerance': within_tolerance,
            'target_coverage': target,
            'actual_coverage': actual,
            'deviation': deviation,
            'num_test': result.num_test
        }
        
        if meets_guarantee:
            logger.info(f"✓ Coverage guarantee met: {actual:.2%} >= {target:.2%}")
        else:
            logger.warning(f"✗ Coverage guarantee NOT met: {actual:.2%} < {target:.2%}")
        
        return validation_result
    
    @staticmethod
    def analyze_set_sizes(
        result: ConformalPredictionResult
    ) -> Dict[str, Any]:
        """
        Analyze distribution of prediction set sizes.
        
        Args:
            result: Conformal prediction result
            
        Returns:
            Dictionary with size statistics
        """
        sizes = result.get_set_sizes()
        
        analysis = {
            'mean': float(np.mean(sizes)),
            'median': float(np.median(sizes)),
            'std': float(np.std(sizes)),
            'min': int(np.min(sizes)),
            'max': int(np.max(sizes)),
            'distribution': result.set_size_distribution,
            'percentiles': {
                '25': float(np.percentile(sizes, 25)),
                '50': float(np.percentile(sizes, 50)),
                '75': float(np.percentile(sizes, 75)),
                '90': float(np.percentile(sizes, 90)),
                '95': float(np.percentile(sizes, 95))
            }
        }
        
        logger.info(f"Set size statistics:")
        logger.info(f"  Mean: {analysis['mean']:.2f}")
        logger.info(f"  Median: {analysis['median']:.2f}")
        logger.info(f"  Range: [{analysis['min']}, {analysis['max']}]")
        
        return analysis
    
    @staticmethod
    def compare_methods(
        results: Dict[str, ConformalPredictionResult]
    ) -> Dict[str, Any]:
        """
        Compare multiple conformal prediction methods.
        
        Args:
            results: Dictionary mapping method names to results
            
        Returns:
            Comparison statistics
        """
        comparison = {}
        
        for method_name, result in results.items():
            comparison[method_name] = {
                'coverage': result.coverage_rate,
                'avg_set_size': result.average_set_size,
                'meets_guarantee': result.meets_coverage_guarantee()
            }
        
        # Find best method (smallest average size while meeting guarantee)
        valid_methods = {
            name: stats for name, stats in comparison.items()
            if stats['meets_guarantee']
        }
        
        if valid_methods:
            best_method = min(
                valid_methods.items(),
                key=lambda x: x[1]['avg_set_size']
            )[0]
            comparison['best_method'] = best_method
        
        return comparison


# Example usage
if __name__ == "__main__":
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*80)
    print("Conformal Prediction Base Module Test")
    print("="*80)
    
    # This is an abstract class, so we can't instantiate it directly
    # The actual implementations (LAC, APS) will be in separate files
    
    # Test configuration
    config = ConformalConfig(alpha=0.1, score_function='lac')
    print(f"\nConfiguration:")
    print(f"  Alpha: {config.alpha}")
    print(f"  Target coverage: {config.target_coverage:.1%}")
    print(f"  Score function: {config.score_function}")
    
    # Test data structures
    print("\n" + "="*80)
    print("Testing data structures...")
    print("="*80)
    
    # Create a sample prediction set
    pred_set = PredictionSet(
        instance_id="test_1",
        options=['A', 'B'],
        scores={'A': 0.1, 'B': 0.3, 'C': 0.6, 'D': 0.8, 'E': 0.9, 'F': 1.0},
        threshold=0.5,
        size=2,
        contains_true=True,
        true_label='A'
    )
    
    print(f"\nSample Prediction Set:")
    print(f"  Instance: {pred_set.instance_id}")
    print(f"  Options in set: {pred_set.options}")
    print(f"  Set size: {len(pred_set)}")
    print(f"  Contains true label: {pred_set.contains_true}")
    print(f"  'A' in set: {'A' in pred_set}")
    print(f"  'C' in set: {'C' in pred_set}")
    
    print("\n" + "="*80)
    print("Base module loaded successfully!")
    print("Actual implementations (LAC, APS) will be in separate files.")
    print("="*80)