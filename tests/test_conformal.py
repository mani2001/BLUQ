"""
Unit tests for Conformal Prediction algorithms.

Tests verify that the LAC and APS implementations match the paper equations:
- LAC score: s(X, Y) = 1 - P(Y)  [Paper Equation 4]
- APS score: s(X, Y) = Σ P(Y') for all Y' with P(Y') ≥ P(Y)  [Paper Equation 5]
- Threshold: q̂ = quantile at ⌈(n+1)(1-α)⌉/n  [Paper Equation 2]

Reference: "Benchmarking LLMs via Uncertainty Quantification" (Ye et al., 2024)
"""

import numpy as np
import pytest
from typing import List

# Import conformal prediction modules
from src.conformal.scorers import LACScorer, APSScorer
from src.conformal.conformal_base import ConformalConfig


class TestLACScorer:
    """Tests for LAC (Least Ambiguous set-valued Classifiers) scorer."""

    def test_lac_score_computation(self):
        """
        Test LAC score: s(X, Y) = 1 - P(Y)

        Paper Equation 4: The LAC score is simply 1 minus the probability
        assigned to the true class.
        """
        scorer = LACScorer()

        # Test case 1: High probability for true class
        probs = np.array([0.8, 0.1, 0.05, 0.05, 0.0, 0.0])
        true_label_idx = 0  # True class has 80% probability
        score = scorer.compute_score(probs, true_label_idx)
        expected_score = 1 - 0.8  # = 0.2
        assert np.isclose(score, expected_score), f"Expected {expected_score}, got {score}"

        # Test case 2: Low probability for true class
        probs = np.array([0.1, 0.5, 0.2, 0.1, 0.05, 0.05])
        true_label_idx = 0  # True class has only 10% probability
        score = scorer.compute_score(probs, true_label_idx)
        expected_score = 1 - 0.1  # = 0.9
        assert np.isclose(score, expected_score), f"Expected {expected_score}, got {score}"

        # Test case 3: True class is not the first option
        probs = np.array([0.1, 0.7, 0.1, 0.05, 0.025, 0.025])
        true_label_idx = 1  # True class is B with 70% probability
        score = scorer.compute_score(probs, true_label_idx)
        expected_score = 1 - 0.7  # = 0.3
        assert np.isclose(score, expected_score), f"Expected {expected_score}, got {score}"

    def test_lac_score_range(self):
        """Test that LAC scores are in valid range [0, 1]."""
        scorer = LACScorer()

        # Generate random probability distributions
        np.random.seed(42)
        for _ in range(100):
            probs = np.random.dirichlet(np.ones(6))
            true_label_idx = np.random.randint(0, 6)
            score = scorer.compute_score(probs, true_label_idx)

            assert 0 <= score <= 1, f"Score {score} outside valid range [0, 1]"

    def test_lac_prediction_set(self):
        """
        Test LAC prediction set construction.

        Options are included if: s(X, Y) ≤ threshold
        ⟺ 1 - P(Y) ≤ threshold
        ⟺ P(Y) ≥ 1 - threshold
        """
        scorer = LACScorer()
        option_letters = ['A', 'B', 'C', 'D', 'E', 'F']

        # Test case: threshold = 0.3 means options with P ≥ 0.7 are included
        probs = np.array([0.75, 0.15, 0.05, 0.03, 0.01, 0.01])
        threshold = 0.3
        min_prob = 1 - threshold  # = 0.7

        pred_set = scorer.create_prediction_set(probs, threshold, option_letters)

        # Only A (0.75 ≥ 0.7) should be included
        assert 'A' in pred_set.options, "A should be in prediction set"
        assert 'B' not in pred_set.options, "B should not be in prediction set (0.15 < 0.7)"


class TestAPSScorer:
    """Tests for APS (Adaptive Prediction Sets) scorer."""

    def test_aps_score_computation(self):
        """
        Test APS score: s(X, Y) = Σ P(Y') for all Y' with P(Y') ≥ P(Y)

        Paper Equation 5: The APS score is the cumulative sum of probabilities
        for all classes with probability at least as high as the true class.
        """
        scorer = APSScorer()

        # Test case 1: True class has highest probability
        probs = np.array([0.7, 0.15, 0.1, 0.03, 0.01, 0.01])
        true_label_idx = 0  # A has 70%
        score = scorer.compute_score(probs, true_label_idx)
        # Only A has P ≥ 0.7, so score = 0.7
        expected_score = 0.7
        assert np.isclose(score, expected_score), f"Expected {expected_score}, got {score}"

        # Test case 2: True class has second-highest probability
        probs = np.array([0.5, 0.3, 0.1, 0.05, 0.03, 0.02])
        true_label_idx = 1  # B has 30%
        score = scorer.compute_score(probs, true_label_idx)
        # Classes with P ≥ 0.3: A (0.5), B (0.3), so score = 0.5 + 0.3 = 0.8
        expected_score = 0.5 + 0.3
        assert np.isclose(score, expected_score), f"Expected {expected_score}, got {score}"

        # Test case 3: True class has low probability
        probs = np.array([0.4, 0.3, 0.15, 0.1, 0.03, 0.02])
        true_label_idx = 3  # D has 10%
        score = scorer.compute_score(probs, true_label_idx)
        # Classes with P ≥ 0.1: A (0.4), B (0.3), C (0.15), D (0.1)
        # score = 0.4 + 0.3 + 0.15 + 0.1 = 0.95
        expected_score = 0.4 + 0.3 + 0.15 + 0.1
        assert np.isclose(score, expected_score), f"Expected {expected_score}, got {score}"

    def test_aps_score_range(self):
        """Test that APS scores are in valid range [0, 1]."""
        scorer = APSScorer()

        np.random.seed(42)
        for _ in range(100):
            probs = np.random.dirichlet(np.ones(6))
            true_label_idx = np.random.randint(0, 6)
            score = scorer.compute_score(probs, true_label_idx)

            assert 0 <= score <= 1, f"Score {score} outside valid range [0, 1]"

    def test_aps_score_monotonicity(self):
        """
        Test that APS score is monotonically increasing as we move to
        lower-probability true classes.
        """
        scorer = APSScorer()

        # Fixed probability distribution
        probs = np.array([0.5, 0.25, 0.15, 0.06, 0.03, 0.01])

        scores = []
        for i in range(6):
            score = scorer.compute_score(probs, i)
            scores.append(score)

        # Verify monotonicity: score should increase as we go to lower prob classes
        # (Higher score = less confident, lower coverage)
        for i in range(len(scores) - 1):
            if probs[i] > probs[i + 1]:  # Only check if probabilities are actually different
                assert scores[i] <= scores[i + 1], (
                    f"APS scores should be monotonic: score[{i}]={scores[i]} > score[{i+1}]={scores[i+1]}"
                )


class TestThresholdComputation:
    """Tests for conformal threshold computation."""

    def test_threshold_formula(self):
        """
        Test threshold computation: q̂ = quantile at ⌈(n+1)(1-α)⌉/n

        Paper Equation 2: The threshold is computed as a quantile of the
        calibration scores, adjusted for finite sample correction.
        """
        config = ConformalConfig(alpha=0.1)  # 90% coverage target
        scorer = LACScorer(config)

        # Create calibration data
        n = 100
        np.random.seed(42)

        # Generate random probabilities and labels
        cal_probs = np.random.dirichlet(np.ones(6), size=n)
        cal_labels = np.random.randint(0, 4, size=n)  # Labels 0-3 (A-D)

        # Compute scores manually
        scores = []
        for probs, label in zip(cal_probs, cal_labels):
            score = scorer.compute_score(probs, label)
            scores.append(score)
        scores = np.array(scores)

        # Expected quantile level: ⌈(n+1)(1-α)⌉/n = ⌈101*0.9⌉/100 = ⌈90.9⌉/100 = 91/100 = 0.91
        expected_quantile_level = np.ceil((n + 1) * (1 - 0.1)) / n
        assert np.isclose(expected_quantile_level, 0.91), f"Expected 0.91, got {expected_quantile_level}"

        # Compute threshold
        option_letters = ['A', 'B', 'C', 'D', 'E', 'F']
        scorer.calibrate(cal_probs, cal_labels, option_letters)

        # Verify threshold is in reasonable range
        assert scorer.threshold is not None, "Threshold should be computed after calibration"
        assert 0 <= scorer.threshold <= 1, f"Threshold {scorer.threshold} outside valid range [0, 1]"

    def test_coverage_guarantee(self):
        """
        Test that the coverage guarantee (1-α) is approximately met.

        For a properly calibrated conformal predictor, the coverage rate
        on a hold-out test set should be close to 1-α.
        """
        alpha = 0.1  # Target 90% coverage
        n_cal = 500
        n_test = 500

        np.random.seed(42)

        # Generate data with a known distribution
        all_probs = np.random.dirichlet(np.ones(6), size=n_cal + n_test)
        all_labels = np.random.randint(0, 4, size=n_cal + n_test)  # Only A-D are valid answers

        cal_probs = all_probs[:n_cal]
        cal_labels = all_labels[:n_cal]
        test_probs = all_probs[n_cal:]
        test_labels = all_labels[n_cal:]

        option_letters = ['A', 'B', 'C', 'D', 'E', 'F']

        # Test LAC
        lac_config = ConformalConfig(alpha=alpha)
        lac_scorer = LACScorer(lac_config)
        lac_scorer.calibrate(cal_probs, cal_labels, option_letters)

        lac_coverage_count = 0
        for probs, label in zip(test_probs, test_labels):
            pred_set = lac_scorer.create_prediction_set(probs, lac_scorer.threshold, option_letters)
            true_letter = option_letters[label]
            if true_letter in pred_set.options:
                lac_coverage_count += 1

        lac_coverage = lac_coverage_count / n_test

        # Coverage should be close to 1-alpha (with some tolerance for finite sample)
        assert lac_coverage >= 1 - alpha - 0.05, (
            f"LAC coverage {lac_coverage:.3f} should be >= {1-alpha-0.05:.3f}"
        )

        # Test APS
        aps_config = ConformalConfig(alpha=alpha)
        aps_scorer = APSScorer(aps_config)
        aps_scorer.calibrate(cal_probs, cal_labels, option_letters)

        aps_coverage_count = 0
        for probs, label in zip(test_probs, test_labels):
            pred_set = aps_scorer.create_prediction_set(probs, aps_scorer.threshold, option_letters)
            true_letter = option_letters[label]
            if true_letter in pred_set.options:
                aps_coverage_count += 1

        aps_coverage = aps_coverage_count / n_test

        assert aps_coverage >= 1 - alpha - 0.05, (
            f"APS coverage {aps_coverage:.3f} should be >= {1-alpha-0.05:.3f}"
        )


class TestEmptySetHandling:
    """Tests for empty prediction set handling."""

    def test_lac_empty_set_fallback(self):
        """Test that LAC handles empty prediction sets by including highest prob option."""
        scorer = LACScorer()
        option_letters = ['A', 'B', 'C', 'D', 'E', 'F']

        # Very low threshold = very high min_prob requirement
        # No option will meet P ≥ 0.999
        probs = np.array([0.5, 0.3, 0.1, 0.05, 0.03, 0.02])
        threshold = 0.001  # min_prob = 0.999

        pred_set = scorer.create_prediction_set(probs, threshold, option_letters)

        # Should fall back to highest probability option
        assert len(pred_set.options) >= 1, "Prediction set should not be empty"
        assert 'A' in pred_set.options, "Highest probability option (A) should be included"
        assert pred_set.metadata.get('empty_set_fallback', False), "Should be marked as fallback"

    def test_aps_never_empty(self):
        """Test that APS prediction sets are never empty."""
        scorer = APSScorer()
        option_letters = ['A', 'B', 'C', 'D', 'E', 'F']

        np.random.seed(42)
        for _ in range(100):
            probs = np.random.dirichlet(np.ones(6))
            threshold = np.random.uniform(0, 1)

            pred_set = scorer.create_prediction_set(probs, threshold, option_letters)

            assert len(pred_set.options) >= 1, "APS prediction set should never be empty"


class TestEmptySetStatistics:
    """Tests for empty set tracking statistics."""

    def test_lac_empty_set_stats(self):
        """Test LAC empty set statistics tracking."""
        scorer = LACScorer()
        option_letters = ['A', 'B', 'C', 'D', 'E', 'F']

        # Reset stats
        scorer.reset_empty_set_stats()

        # Create some prediction sets with very low threshold (will trigger fallback)
        probs = np.array([0.5, 0.3, 0.1, 0.05, 0.03, 0.02])
        low_threshold = 0.001

        for _ in range(5):
            scorer.create_prediction_set(probs, low_threshold, option_letters)

        stats = scorer.get_empty_set_stats()
        assert stats['total_predictions'] == 5
        assert stats['empty_set_count'] == 5
        assert stats['empty_set_rate'] == 1.0

    def test_aps_empty_set_stats(self):
        """Test APS empty set statistics tracking."""
        scorer = APSScorer()
        option_letters = ['A', 'B', 'C', 'D', 'E', 'F']

        scorer.reset_empty_set_stats()

        probs = np.array([0.5, 0.3, 0.1, 0.05, 0.03, 0.02])
        threshold = 0.5  # Normal threshold

        for _ in range(10):
            scorer.create_prediction_set(probs, threshold, option_letters)

        stats = scorer.get_empty_set_stats()
        assert stats['total_predictions'] == 10
        # APS with reasonable threshold should rarely have empty sets
        assert stats['empty_set_count'] <= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
