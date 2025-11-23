"""
Probability Extractor Module
Extracts and processes probabilities from model logits with various calibration methods.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from scipy.special import softmax as scipy_softmax

from src.models.inference_engine import InferenceResult, BatchInferenceResult

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class ProbabilityExtractionConfig:
    """Configuration for probability extraction."""
    # Softmax temperature
    temperature: float = 1.0
    
    # Calibration method
    calibration_method: Optional[str] = None  # None, 'temperature', 'platt', 'isotonic'
    
    # Normalization
    normalize: bool = True
    
    # Clipping to avoid numerical issues
    min_prob: float = 1e-10
    max_prob: float = 1.0 - 1e-10
    
    # Option handling
    option_letters: List[str] = None
    
    def __post_init__(self):
        if self.option_letters is None:
            self.option_letters = ['A', 'B', 'C', 'D', 'E', 'F']
        
        if self.calibration_method and self.calibration_method not in [
            'temperature', 'platt', 'isotonic', None
        ]:
            raise ValueError(
                f"Unknown calibration method: {self.calibration_method}. "
                f"Must be one of: None, 'temperature', 'platt', 'isotonic'"
            )


@dataclass
class ExtractedProbabilities:
    """Container for extracted probabilities."""
    instance_id: str
    probabilities: np.ndarray  # Shape: [num_options]
    logits: np.ndarray  # Original logits
    predicted_option: str
    predicted_probability: float
    entropy: float  # Uncertainty measure
    confidence: float  # Max probability
    option_letters: List[str]
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'instance_id': self.instance_id,
            'probabilities': {
                letter: float(prob) 
                for letter, prob in zip(self.option_letters, self.probabilities)
            },
            'predicted_option': self.predicted_option,
            'predicted_probability': float(self.predicted_probability),
            'entropy': float(self.entropy),
            'confidence': float(self.confidence),
            'metadata': self.metadata or {}
        }
    
    def get_probability(self, option: str) -> float:
        """Get probability for a specific option."""
        if option not in self.option_letters:
            raise ValueError(f"Invalid option: {option}")
        idx = self.option_letters.index(option)
        return float(self.probabilities[idx])
    
    def get_top_k_options(self, k: int = 3) -> List[Tuple[str, float]]:
        """Get top-k most probable options."""
        indices = np.argsort(self.probabilities)[::-1][:k]
        return [
            (self.option_letters[idx], float(self.probabilities[idx]))
            for idx in indices
        ]


class ProbabilityExtractor:
    """Extracts and processes probabilities from inference results."""
    
    def __init__(self, config: Optional[ProbabilityExtractionConfig] = None):
        """
        Initialize the probability extractor.
        
        Args:
            config: Configuration for probability extraction
        """
        self.config = config or ProbabilityExtractionConfig()
        self.calibrator = None
        
        logger.info("Initialized ProbabilityExtractor")
        logger.info(f"  Temperature: {self.config.temperature}")
        logger.info(f"  Calibration method: {self.config.calibration_method}")
    
    def extract_from_inference_result(
        self,
        result: InferenceResult
    ) -> ExtractedProbabilities:
        """
        Extract probabilities from an InferenceResult.
        
        Args:
            result: InferenceResult from inference engine
            
        Returns:
            ExtractedProbabilities object
        """
        # Get logits
        logits = result.logits
        
        # Apply temperature scaling
        if self.config.temperature != 1.0:
            logits = logits / self.config.temperature
        
        # Compute probabilities
        probabilities = self._compute_probabilities(logits)
        
        # Apply calibration if configured
        if self.config.calibration_method and self.calibrator:
            probabilities = self.calibrator.calibrate(probabilities)
        
        # Normalize
        if self.config.normalize:
            probabilities = probabilities / probabilities.sum()
        
        # Clip probabilities
        probabilities = np.clip(
            probabilities,
            self.config.min_prob,
            self.config.max_prob
        )
        
        # Get prediction
        predicted_idx = np.argmax(probabilities)
        predicted_option = self.config.option_letters[predicted_idx]
        predicted_probability = probabilities[predicted_idx]
        
        # Compute uncertainty metrics
        entropy = self._compute_entropy(probabilities)
        confidence = np.max(probabilities)
        
        return ExtractedProbabilities(
            instance_id=result.instance_id,
            probabilities=probabilities,
            logits=result.logits,
            predicted_option=predicted_option,
            predicted_probability=predicted_probability,
            entropy=entropy,
            confidence=confidence,
            option_letters=self.config.option_letters,
            metadata=result.metadata
        )
    
    def extract_batch(
        self,
        batch_result: BatchInferenceResult
    ) -> List[ExtractedProbabilities]:
        """
        Extract probabilities from batch inference results.
        
        Args:
            batch_result: BatchInferenceResult from inference engine
            
        Returns:
            List of ExtractedProbabilities
        """
        extracted = []
        for result in batch_result.results:
            extracted_prob = self.extract_from_inference_result(result)
            extracted.append(extracted_prob)
        
        return extracted
    
    def extract_from_logits(
        self,
        logits: np.ndarray,
        instance_id: str = "unknown"
    ) -> ExtractedProbabilities:
        """
        Extract probabilities directly from logits.
        
        Args:
            logits: Logits array (shape: [num_options])
            instance_id: ID for tracking
            
        Returns:
            ExtractedProbabilities object
        """
        # Apply temperature
        if self.config.temperature != 1.0:
            logits = logits / self.config.temperature
        
        # Compute probabilities
        probabilities = self._compute_probabilities(logits)
        
        # Apply calibration
        if self.config.calibration_method and self.calibrator:
            probabilities = self.calibrator.calibrate(probabilities)
        
        # Normalize
        if self.config.normalize:
            probabilities = probabilities / probabilities.sum()
        
        # Clip
        probabilities = np.clip(
            probabilities,
            self.config.min_prob,
            self.config.max_prob
        )
        
        # Get prediction
        predicted_idx = np.argmax(probabilities)
        predicted_option = self.config.option_letters[predicted_idx]
        predicted_probability = probabilities[predicted_idx]
        
        # Compute metrics
        entropy = self._compute_entropy(probabilities)
        confidence = np.max(probabilities)
        
        return ExtractedProbabilities(
            instance_id=instance_id,
            probabilities=probabilities,
            logits=logits,
            predicted_option=predicted_option,
            predicted_probability=predicted_probability,
            entropy=entropy,
            confidence=confidence,
            option_letters=self.config.option_letters
        )
    
    def _compute_probabilities(self, logits: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities from logits."""
        return scipy_softmax(logits)
    
    def _compute_entropy(self, probabilities: np.ndarray) -> float:
        """Compute Shannon entropy of probability distribution."""
        # Add small epsilon to avoid log(0)
        probs = probabilities + 1e-10
        entropy = -np.sum(probs * np.log(probs))
        return float(entropy)
    
    def set_temperature(self, temperature: float) -> None:
        """Update temperature for scaling."""
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        self.config.temperature = temperature
        logger.info(f"Set temperature to {temperature}")
    
    def fit_calibrator(
        self,
        logits: np.ndarray,
        true_labels: np.ndarray,
        method: str = 'temperature'
    ) -> None:
        """
        Fit a probability calibrator on validation data.
        
        Args:
            logits: Array of logits (shape: [num_samples, num_options])
            true_labels: Array of true labels (shape: [num_samples])
            method: Calibration method ('temperature', 'platt', 'isotonic')
        """
        logger.info(f"Fitting {method} calibrator...")
        
        if method == 'temperature':
            self.calibrator = TemperatureCalibrator()
        elif method == 'platt':
            self.calibrator = PlattCalibrator()
        elif method == 'isotonic':
            self.calibrator = IsotonicCalibrator()
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        
        # Compute initial probabilities
        probabilities = np.array([self._compute_probabilities(l) for l in logits])
        
        # Fit calibrator
        self.calibrator.fit(probabilities, true_labels)
        self.config.calibration_method = method
        
        logger.info(f"Calibrator fitted successfully")


class TemperatureCalibrator:
    """Temperature scaling calibrator."""
    
    def __init__(self):
        self.temperature = 1.0
    
    def fit(self, probabilities: np.ndarray, true_labels: np.ndarray) -> None:
        """
        Fit temperature parameter to minimize negative log-likelihood.
        
        Args:
            probabilities: Predicted probabilities (shape: [num_samples, num_options])
            true_labels: True labels (shape: [num_samples])
        """
        from scipy.optimize import minimize
        
        # Convert probabilities back to logits
        logits = np.log(probabilities + 1e-10)
        
        def nll(temp):
            """Negative log-likelihood loss."""
            scaled_logits = logits / temp
            scaled_probs = scipy_softmax(scaled_logits, axis=1)
            nll_loss = -np.mean(
                np.log(scaled_probs[np.arange(len(true_labels)), true_labels] + 1e-10)
            )
            return nll_loss
        
        # Optimize temperature
        result = minimize(nll, x0=1.0, bounds=[(0.01, 10.0)], method='L-BFGS-B')
        self.temperature = result.x[0]
        
        logger.info(f"Optimal temperature: {self.temperature:.4f}")
    
    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to probabilities."""
        logits = np.log(probabilities + 1e-10)
        scaled_logits = logits / self.temperature
        return scipy_softmax(scaled_logits)


class PlattCalibrator:
    """Platt scaling calibrator (logistic regression)."""
    
    def __init__(self):
        self.a = 1.0
        self.b = 0.0
    
    def fit(self, probabilities: np.ndarray, true_labels: np.ndarray) -> None:
        """
        Fit Platt scaling parameters.
        
        Args:
            probabilities: Predicted probabilities (shape: [num_samples, num_options])
            true_labels: True labels (shape: [num_samples])
        """
        from sklearn.linear_model import LogisticRegression
        
        # Get max probabilities
        max_probs = np.max(probabilities, axis=1).reshape(-1, 1)
        
        # Binary labels (correct vs incorrect)
        binary_labels = (np.argmax(probabilities, axis=1) == true_labels).astype(int)
        
        # Fit logistic regression
        lr = LogisticRegression()
        lr.fit(max_probs, binary_labels)
        
        self.a = lr.coef_[0][0]
        self.b = lr.intercept_[0]
        
        logger.info(f"Platt parameters: a={self.a:.4f}, b={self.b:.4f}")
    
    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply Platt scaling."""
        # This is a simplified version for multi-class
        # Scale the entire distribution
        max_prob = np.max(probabilities)
        scaled_max = 1.0 / (1.0 + np.exp(-(self.a * max_prob + self.b)))
        scale_factor = scaled_max / max_prob
        
        scaled_probs = probabilities * scale_factor
        return scaled_probs / scaled_probs.sum()


class IsotonicCalibrator:
    """Isotonic regression calibrator."""
    
    def __init__(self):
        self.calibrator = None
    
    def fit(self, probabilities: np.ndarray, true_labels: np.ndarray) -> None:
        """
        Fit isotonic regression.
        
        Args:
            probabilities: Predicted probabilities (shape: [num_samples, num_options])
            true_labels: True labels (shape: [num_samples])
        """
        from sklearn.isotonic import IsotonicRegression
        
        # Get predicted probabilities for true labels
        predicted_probs = probabilities[np.arange(len(true_labels)), true_labels]
        
        # Binary correctness
        correct = (np.argmax(probabilities, axis=1) == true_labels).astype(float)
        
        # Fit isotonic regression
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(predicted_probs, correct)
        
        logger.info("Isotonic regression fitted")
    
    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration."""
        if self.calibrator is None:
            return probabilities
        
        # This is simplified - calibrate the max probability
        max_prob = np.max(probabilities)
        calibrated_max = self.calibrator.predict([max_prob])[0]
        
        scale_factor = calibrated_max / max_prob
        scaled_probs = probabilities * scale_factor
        return scaled_probs / scaled_probs.sum()


class ProbabilityAnalyzer:
    """Analyzer for probability distributions."""
    
    @staticmethod
    def analyze_calibration(
        predicted_probs: np.ndarray,
        true_labels: np.ndarray,
        num_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze calibration of predicted probabilities.
        
        Args:
            predicted_probs: Predicted probabilities (shape: [num_samples, num_options])
            true_labels: True labels (shape: [num_samples])
            num_bins: Number of bins for calibration analysis
            
        Returns:
            Dictionary with calibration metrics
        """
        # Get predicted class and its probability
        predicted_classes = np.argmax(predicted_probs, axis=1)
        predicted_confidences = np.max(predicted_probs, axis=1)
        
        # Compute accuracy
        accuracies = (predicted_classes == true_labels).astype(float)
        
        # Bin probabilities
        bins = np.linspace(0, 1, num_bins + 1)
        bin_indices = np.digitize(predicted_confidences, bins) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        
        # Compute bin statistics
        bin_stats = []
        for i in range(num_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_accuracy = accuracies[mask].mean()
                bin_confidence = predicted_confidences[mask].mean()
                bin_count = mask.sum()
                
                bin_stats.append({
                    'bin': i,
                    'accuracy': float(bin_accuracy),
                    'confidence': float(bin_confidence),
                    'count': int(bin_count),
                    'calibration_error': abs(bin_accuracy - bin_confidence)
                })
        
        # Compute ECE (Expected Calibration Error)
        ece = 0.0
        for stat in bin_stats:
            weight = stat['count'] / len(predicted_probs)
            ece += weight * stat['calibration_error']
        
        # Compute MCE (Maximum Calibration Error)
        mce = max([stat['calibration_error'] for stat in bin_stats]) if bin_stats else 0.0
        
        return {
            'ece': float(ece),
            'mce': float(mce),
            'bin_stats': bin_stats,
            'num_samples': len(predicted_probs)
        }
    
    @staticmethod
    def compute_confidence_statistics(
        extracted_probs: List[ExtractedProbabilities]
    ) -> Dict[str, float]:
        """
        Compute statistics about confidence/uncertainty.
        
        Args:
            extracted_probs: List of ExtractedProbabilities
            
        Returns:
            Dictionary with statistics
        """
        confidences = [ep.confidence for ep in extracted_probs]
        entropies = [ep.entropy for ep in extracted_probs]
        
        return {
            'mean_confidence': float(np.mean(confidences)),
            'std_confidence': float(np.std(confidences)),
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences)),
            'mean_entropy': float(np.mean(entropies)),
            'std_entropy': float(np.std(entropies)),
            'min_entropy': float(np.min(entropies)),
            'max_entropy': float(np.max(entropies))
        }


# Example usage
if __name__ == "__main__":
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create sample logits
    print("\n" + "="*80)
    print("Testing ProbabilityExtractor")
    print("="*80)
    
    sample_logits = np.array([2.5, 3.0, 1.5, 0.5, -1.0, -2.0])
    
    # Initialize extractor
    config = ProbabilityExtractionConfig(temperature=1.0)
    extractor = ProbabilityExtractor(config)
    
    # Extract probabilities
    extracted = extractor.extract_from_logits(sample_logits, instance_id="test_1")
    
    print("\nExtracted Probabilities:")
    print(f"  Predicted option: {extracted.predicted_option}")
    print(f"  Confidence: {extracted.confidence:.4f}")
    print(f"  Entropy: {extracted.entropy:.4f}")
    print(f"\n  Probabilities:")
    for letter, prob in zip(extracted.option_letters, extracted.probabilities):
        print(f"    {letter}: {prob:.4f}")
    
    print(f"\n  Top-3 options:")
    for option, prob in extracted.get_top_k_options(k=3):
        print(f"    {option}: {prob:.4f}")
    
    # Test temperature scaling
    print("\n" + "="*80)
    print("Testing temperature scaling")
    print("="*80)
    
    for temp in [0.5, 1.0, 2.0]:
        extractor.set_temperature(temp)
        extracted = extractor.extract_from_logits(sample_logits, instance_id=f"test_temp_{temp}")
        print(f"\nTemperature {temp}:")
        print(f"  Confidence: {extracted.confidence:.4f}")
        print(f"  Entropy: {extracted.entropy:.4f}")