"""
Metrics Module
Implements evaluation metrics for LLM uncertainty quantification benchmarking.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, confusion_matrix

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Container for a single metric result."""
    name: str
    value: float
    description: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'description': self.description,
            'metadata': self.metadata or {}
        }


@dataclass
class EvaluationMetrics:
    """Container for all evaluation metrics."""
    accuracy: float
    set_size: float
    coverage_rate: float
    
    # Optional additional metrics
    ece: Optional[float] = None  # Expected Calibration Error
    mce: Optional[float] = None  # Maximum Calibration Error
    brier_score: Optional[float] = None
    nll: Optional[float] = None  # Negative Log Likelihood
    
    # Per-class metrics
    per_class_accuracy: Optional[Dict[str, float]] = None
    
    # Uncertainty metrics
    mean_entropy: Optional[float] = None
    mean_confidence: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'accuracy': float(self.accuracy),
            'set_size': float(self.set_size),
            'coverage_rate': float(self.coverage_rate)
        }
        
        # Add optional metrics if present
        if self.ece is not None:
            result['ece'] = float(self.ece)
        if self.mce is not None:
            result['mce'] = float(self.mce)
        if self.brier_score is not None:
            result['brier_score'] = float(self.brier_score)
        if self.nll is not None:
            result['nll'] = float(self.nll)
        if self.per_class_accuracy is not None:
            result['per_class_accuracy'] = {
                k: float(v) for k, v in self.per_class_accuracy.items()
            }
        if self.mean_entropy is not None:
            result['mean_entropy'] = float(self.mean_entropy)
        if self.mean_confidence is not None:
            result['mean_confidence'] = float(self.mean_confidence)
        
        return result
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Accuracy: {self.accuracy:.2%}",
            f"Average Set Size: {self.set_size:.2f}",
            f"Coverage Rate: {self.coverage_rate:.2%}"
        ]
        
        if self.ece is not None:
            lines.append(f"ECE: {self.ece:.4f}")
        if self.mean_entropy is not None:
            lines.append(f"Mean Entropy: {self.mean_entropy:.4f}")
        if self.mean_confidence is not None:
            lines.append(f"Mean Confidence: {self.mean_confidence:.4f}")
        
        return "\n".join(lines)


class AccuracyMetric:
    """Compute prediction accuracy."""
    
    @staticmethod
    def compute(
        predictions: np.ndarray,
        true_labels: np.ndarray
    ) -> MetricResult:
        """
        Compute accuracy.
        
        Args:
            predictions: Predicted labels (indices)
                Shape: [num_samples]
            true_labels: True labels (indices)
                Shape: [num_samples]
            
        Returns:
            MetricResult with accuracy
        """
        accuracy = accuracy_score(true_labels, predictions)
        
        return MetricResult(
            name="accuracy",
            value=float(accuracy),
            description="Proportion of correct predictions"
        )
    
    @staticmethod
    def compute_from_probabilities(
        probabilities: np.ndarray,
        true_labels: np.ndarray
    ) -> MetricResult:
        """
        Compute accuracy from probabilities.
        
        Args:
            probabilities: Predicted probabilities
                Shape: [num_samples, num_classes]
            true_labels: True labels (indices)
                Shape: [num_samples]
            
        Returns:
            MetricResult with accuracy
        """
        predictions = np.argmax(probabilities, axis=1)
        return AccuracyMetric.compute(predictions, true_labels)
    
    @staticmethod
    def compute_per_class(
        predictions: np.ndarray,
        true_labels: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute per-class accuracy.
        
        Args:
            predictions: Predicted labels
            true_labels: True labels
            class_names: Names for each class
            
        Returns:
            Dictionary mapping class names to accuracies
        """
        unique_labels = np.unique(true_labels)
        
        if class_names is None:
            class_names = [f"Class_{i}" for i in unique_labels]
        
        per_class_acc = {}
        for label, name in zip(unique_labels, class_names):
            mask = true_labels == label
            if mask.sum() > 0:
                acc = (predictions[mask] == true_labels[mask]).mean()
                per_class_acc[name] = float(acc)
        
        return per_class_acc


class SetSizeMetric:
    """Compute average prediction set size."""
    
    @staticmethod
    def compute(set_sizes: np.ndarray) -> MetricResult:
        """
        Compute average set size.
        
        Args:
            set_sizes: Array of set sizes
                Shape: [num_samples]
            
        Returns:
            MetricResult with average set size
        """
        avg_size = np.mean(set_sizes)
        
        metadata = {
            'min': float(np.min(set_sizes)),
            'max': float(np.max(set_sizes)),
            'median': float(np.median(set_sizes)),
            'std': float(np.std(set_sizes))
        }
        
        return MetricResult(
            name="set_size",
            value=float(avg_size),
            description="Average prediction set size",
            metadata=metadata
        )
    
    @staticmethod
    def compute_distribution(set_sizes: np.ndarray) -> Dict[int, int]:
        """
        Compute distribution of set sizes.
        
        Args:
            set_sizes: Array of set sizes
            
        Returns:
            Dictionary mapping size to count
        """
        distribution = {}
        for size in set_sizes:
            size = int(size)
            distribution[size] = distribution.get(size, 0) + 1
        
        return distribution


class CoverageRateMetric:
    """Compute coverage rate (proportion of true labels in prediction sets)."""
    
    @staticmethod
    def compute(
        prediction_sets: List[List[int]],
        true_labels: np.ndarray
    ) -> MetricResult:
        """
        Compute coverage rate.
        
        Args:
            prediction_sets: List of prediction sets (each is a list of indices)
            true_labels: True labels (indices)
                Shape: [num_samples]
            
        Returns:
            MetricResult with coverage rate
        """
        n = len(prediction_sets)
        covered = sum(
            1 for pred_set, true_label in zip(prediction_sets, true_labels)
            if true_label in pred_set
        )
        
        coverage = covered / n if n > 0 else 0.0
        
        return MetricResult(
            name="coverage_rate",
            value=float(coverage),
            description="Proportion of instances where true label is in prediction set",
            metadata={'covered_count': covered, 'total_count': n}
        )


class CalibrationMetric:
    """Compute calibration metrics (ECE, MCE)."""
    
    @staticmethod
    def compute_ece(
        probabilities: np.ndarray,
        true_labels: np.ndarray,
        num_bins: int = 10
    ) -> MetricResult:
        """
        Compute Expected Calibration Error (ECE).
        
        ECE measures the difference between confidence and accuracy.
        
        Args:
            probabilities: Predicted probabilities
                Shape: [num_samples, num_classes]
            true_labels: True labels (indices)
                Shape: [num_samples]
            num_bins: Number of bins for calibration
            
        Returns:
            MetricResult with ECE
        """
        predictions = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1)
        accuracies = (predictions == true_labels).astype(float)
        
        # Create bins
        bins = np.linspace(0, 1, num_bins + 1)
        bin_indices = np.digitize(confidences, bins) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        
        # Compute ECE
        ece = 0.0
        bin_stats = []
        
        for i in range(num_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_acc = accuracies[mask].mean()
                bin_conf = confidences[mask].mean()
                bin_count = mask.sum()
                
                weight = bin_count / len(confidences)
                ece += weight * abs(bin_acc - bin_conf)
                
                bin_stats.append({
                    'bin': i,
                    'accuracy': float(bin_acc),
                    'confidence': float(bin_conf),
                    'count': int(bin_count)
                })
        
        return MetricResult(
            name="ece",
            value=float(ece),
            description="Expected Calibration Error",
            metadata={'bin_stats': bin_stats, 'num_bins': num_bins}
        )
    
    @staticmethod
    def compute_mce(
        probabilities: np.ndarray,
        true_labels: np.ndarray,
        num_bins: int = 10
    ) -> MetricResult:
        """
        Compute Maximum Calibration Error (MCE).
        
        MCE is the maximum difference between confidence and accuracy across bins.
        
        Args:
            probabilities: Predicted probabilities
            true_labels: True labels
            num_bins: Number of bins
            
        Returns:
            MetricResult with MCE
        """
        predictions = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1)
        accuracies = (predictions == true_labels).astype(float)
        
        # Create bins
        bins = np.linspace(0, 1, num_bins + 1)
        bin_indices = np.digitize(confidences, bins) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        
        # Compute MCE
        max_error = 0.0
        
        for i in range(num_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_acc = accuracies[mask].mean()
                bin_conf = confidences[mask].mean()
                error = abs(bin_acc - bin_conf)
                max_error = max(max_error, error)
        
        return MetricResult(
            name="mce",
            value=float(max_error),
            description="Maximum Calibration Error"
        )


class BrierScoreMetric:
    """Compute Brier score (mean squared error of probabilities)."""
    
    @staticmethod
    def compute(
        probabilities: np.ndarray,
        true_labels: np.ndarray
    ) -> MetricResult:
        """
        Compute Brier score.
        
        Args:
            probabilities: Predicted probabilities
                Shape: [num_samples, num_classes]
            true_labels: True labels (indices)
                Shape: [num_samples]
            
        Returns:
            MetricResult with Brier score
        """
        n_samples, n_classes = probabilities.shape
        
        # Create one-hot encoded labels
        one_hot = np.zeros((n_samples, n_classes))
        one_hot[np.arange(n_samples), true_labels] = 1
        
        # Compute Brier score
        brier = np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
        
        return MetricResult(
            name="brier_score",
            value=float(brier),
            description="Brier score (lower is better)"
        )


class NegativeLogLikelihoodMetric:
    """Compute negative log-likelihood."""
    
    @staticmethod
    def compute(
        probabilities: np.ndarray,
        true_labels: np.ndarray
    ) -> MetricResult:
        """
        Compute negative log-likelihood (cross-entropy loss).
        
        Args:
            probabilities: Predicted probabilities
                Shape: [num_samples, num_classes]
            true_labels: True labels (indices)
                Shape: [num_samples]
            
        Returns:
            MetricResult with NLL
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        probs_clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
        
        # Get probabilities for true labels
        n_samples = len(true_labels)
        true_probs = probs_clipped[np.arange(n_samples), true_labels]
        
        # Compute NLL
        nll = -np.mean(np.log(true_probs))
        
        return MetricResult(
            name="nll",
            value=float(nll),
            description="Negative Log-Likelihood (lower is better)"
        )


class UncertaintyMetric:
    """Compute uncertainty-related metrics."""
    
    @staticmethod
    def compute_entropy(probabilities: np.ndarray) -> np.ndarray:
        """
        Compute entropy for each prediction.
        
        Args:
            probabilities: Predicted probabilities
                Shape: [num_samples, num_classes]
            
        Returns:
            Array of entropies
        """
        epsilon = 1e-10
        probs_clipped = np.clip(probabilities, epsilon, 1.0)
        entropy = -np.sum(probs_clipped * np.log(probs_clipped), axis=1)
        return entropy
    
    @staticmethod
    def compute_mean_entropy(probabilities: np.ndarray) -> MetricResult:
        """
        Compute mean entropy across all predictions.
        
        Args:
            probabilities: Predicted probabilities
            
        Returns:
            MetricResult with mean entropy
        """
        entropies = UncertaintyMetric.compute_entropy(probabilities)
        mean_entropy = np.mean(entropies)
        
        return MetricResult(
            name="mean_entropy",
            value=float(mean_entropy),
            description="Mean entropy (uncertainty measure)",
            metadata={
                'min': float(np.min(entropies)),
                'max': float(np.max(entropies)),
                'std': float(np.std(entropies))
            }
        )
    
    @staticmethod
    def compute_mean_confidence(probabilities: np.ndarray) -> MetricResult:
        """
        Compute mean confidence (max probability).
        
        Args:
            probabilities: Predicted probabilities
            
        Returns:
            MetricResult with mean confidence
        """
        confidences = np.max(probabilities, axis=1)
        mean_conf = np.mean(confidences)
        
        return MetricResult(
            name="mean_confidence",
            value=float(mean_conf),
            description="Mean confidence (max probability)",
            metadata={
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences)),
                'std': float(np.std(confidences))
            }
        )


class MetricsCalculator:
    """Main calculator for all metrics."""
    
    @staticmethod
    def compute_all(
        probabilities: np.ndarray,
        true_labels: np.ndarray,
        prediction_sets: Optional[List[List[int]]] = None,
        class_names: Optional[List[str]] = None,
        compute_calibration: bool = True
    ) -> EvaluationMetrics:
        """
        Compute all evaluation metrics.
        
        Args:
            probabilities: Predicted probabilities
                Shape: [num_samples, num_classes]
            true_labels: True labels (indices)
                Shape: [num_samples]
            prediction_sets: List of prediction sets (for coverage and set size)
            class_names: Names for each class
            compute_calibration: Whether to compute calibration metrics (ECE, MCE)
            
        Returns:
            EvaluationMetrics object with all metrics
        """
        # Accuracy
        acc_result = AccuracyMetric.compute_from_probabilities(
            probabilities, true_labels
        )
        accuracy = acc_result.value
        
        # Per-class accuracy
        predictions = np.argmax(probabilities, axis=1)
        per_class_acc = AccuracyMetric.compute_per_class(
            predictions, true_labels, class_names
        )
        
        # Set size and coverage (if prediction sets provided)
        set_size = None
        coverage_rate = None
        
        if prediction_sets is not None:
            set_sizes = np.array([len(ps) for ps in prediction_sets])
            set_size_result = SetSizeMetric.compute(set_sizes)
            set_size = set_size_result.value
            
            coverage_result = CoverageRateMetric.compute(
                prediction_sets, true_labels
            )
            coverage_rate = coverage_result.value
        
        # Calibration metrics
        ece = None
        mce = None
        if compute_calibration:
            ece_result = CalibrationMetric.compute_ece(probabilities, true_labels)
            ece = ece_result.value
            
            mce_result = CalibrationMetric.compute_mce(probabilities, true_labels)
            mce = mce_result.value
        
        # Brier score
        brier_result = BrierScoreMetric.compute(probabilities, true_labels)
        brier_score = brier_result.value
        
        # NLL
        nll_result = NegativeLogLikelihoodMetric.compute(probabilities, true_labels)
        nll = nll_result.value
        
        # Uncertainty metrics
        entropy_result = UncertaintyMetric.compute_mean_entropy(probabilities)
        mean_entropy = entropy_result.value
        
        conf_result = UncertaintyMetric.compute_mean_confidence(probabilities)
        mean_confidence = conf_result.value
        
        return EvaluationMetrics(
            accuracy=accuracy,
            set_size=set_size if set_size is not None else 0.0,
            coverage_rate=coverage_rate if coverage_rate is not None else 0.0,
            ece=ece,
            mce=mce,
            brier_score=brier_score,
            nll=nll,
            per_class_accuracy=per_class_acc,
            mean_entropy=mean_entropy,
            mean_confidence=mean_confidence
        )


# Example usage
if __name__ == "__main__":
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*80)
    print("Metrics Module Test")
    print("="*80)
    
    # Create sample data
    np.random.seed(42)
    
    n_samples = 100
    n_classes = 6
    
    # Generate probabilities (slightly biased towards correct labels)
    probabilities = np.random.dirichlet(np.ones(n_classes), size=n_samples)
    true_labels = np.random.randint(0, n_classes, size=n_samples)
    
    # Boost probability of true label
    for i in range(n_samples):
        probabilities[i, true_labels[i]] += 0.3
    
    # Renormalize
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    
    # Generate prediction sets (simulate conformal prediction)
    prediction_sets = []
    for i, probs in enumerate(probabilities):
        sorted_indices = np.argsort(probs)[::-1]
        # Include top 2-3 predictions
        n_include = np.random.choice([2, 3])
        pred_set = sorted_indices[:n_include].tolist()
        prediction_sets.append(pred_set)
    
    # Compute all metrics
    print("\n" + "="*80)
    print("Computing all metrics...")
    print("="*80)
    
    metrics = MetricsCalculator.compute_all(
        probabilities=probabilities,
        true_labels=true_labels,
        prediction_sets=prediction_sets,
        class_names=['A', 'B', 'C', 'D', 'E', 'F'],
        compute_calibration=True
    )
    
    print("\n" + metrics.get_summary())
    
    print(f"\nAdditional metrics:")
    print(f"  Brier Score: {metrics.brier_score:.4f}")
    print(f"  NLL: {metrics.nll:.4f}")
    
    print(f"\nPer-class accuracy:")
    for class_name, acc in metrics.per_class_accuracy.items():
        print(f"  {class_name}: {acc:.2%}")
    
    # Test individual metrics
    print("\n" + "="*80)
    print("Testing individual metrics...")
    print("="*80)
    
    # Accuracy
    acc_result = AccuracyMetric.compute_from_probabilities(probabilities, true_labels)
    print(f"\nAccuracy: {acc_result.value:.2%}")
    
    # Set size
    set_sizes = np.array([len(ps) for ps in prediction_sets])
    ss_result = SetSizeMetric.compute(set_sizes)
    print(f"\nSet Size:")
    print(f"  Average: {ss_result.value:.2f}")
    print(f"  Min: {ss_result.metadata['min']:.0f}")
    print(f"  Max: {ss_result.metadata['max']:.0f}")
    
    # Coverage
    cov_result = CoverageRateMetric.compute(prediction_sets, true_labels)
    print(f"\nCoverage Rate: {cov_result.value:.2%}")
    print(f"  Covered: {cov_result.metadata['covered_count']}/{cov_result.metadata['total_count']}")
    
    # ECE
    ece_result = CalibrationMetric.compute_ece(probabilities, true_labels)
    print(f"\nECE: {ece_result.value:.4f}")
    
    # Save results
    print("\n" + "="*80)
    print("Saving metrics to JSON...")
    print("="*80)
    
    import json
    from pathlib import Path
    
    output_dir = Path("./results/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "metrics_example.json", 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2)
    
    print(f"Metrics saved to {output_dir / 'metrics_example.json'}")