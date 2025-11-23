"""
Prediction Set Generator Module
Unified interface for generating prediction sets using different conformal methods.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import json

from src.conformal.conformal_base import (
    ConformalConfig,
    PredictionSet,
    ConformalPredictionResult,
    ConformalPredictionValidator
)
from src.conformal.lac_scorer import LACScorer
from src.conformal.aps_scorer import APSScorer

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class PredictionSetGeneratorConfig:
    """Configuration for prediction set generation."""
    # Conformal methods to use
    methods: List[str] = None  # ['lac', 'aps'] or subset
    
    # Error rate
    alpha: float = 0.1
    
    # Aggregation strategy when using multiple methods
    aggregation: str = "average"  # 'average', 'union', 'intersection', 'separate'
    
    # Validation
    validate_results: bool = True
    
    # Random seed
    random_seed: Optional[int] = None
    
    def __post_init__(self):
        if self.methods is None:
            self.methods = ['lac', 'aps']
        
        for method in self.methods:
            if method not in ['lac', 'aps']:
                raise ValueError(
                    f"Unknown method: {method}. Must be 'lac' or 'aps'"
                )
        
        if self.aggregation not in ['average', 'union', 'intersection', 'separate']:
            raise ValueError(
                f"Unknown aggregation: {self.aggregation}. "
                f"Must be 'average', 'union', 'intersection', or 'separate'"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'methods': self.methods,
            'alpha': self.alpha,
            'aggregation': self.aggregation,
            'validate_results': self.validate_results,
            'random_seed': self.random_seed
        }


class PredictionSetGenerator:
    """
    Unified generator for prediction sets using conformal prediction.
    Supports multiple methods (LAC, APS) and aggregation strategies.
    """
    
    def __init__(self, config: Optional[PredictionSetGeneratorConfig] = None):
        """
        Initialize the prediction set generator.
        
        Args:
            config: Configuration for generation
        """
        self.config = config or PredictionSetGeneratorConfig()
        
        # Initialize conformal predictors
        self.predictors: Dict[str, Any] = {}
        self._initialize_predictors()
        
        logger.info("Initialized PredictionSetGenerator")
        logger.info(f"  Methods: {self.config.methods}")
        logger.info(f"  Alpha: {self.config.alpha}")
        logger.info(f"  Aggregation: {self.config.aggregation}")
    
    def _initialize_predictors(self) -> None:
        """Initialize conformal predictors for each method."""
        for method in self.config.methods:
            conf_config = ConformalConfig(
                alpha=self.config.alpha,
                score_function=method,
                random_seed=self.config.random_seed
            )
            
            if method == 'lac':
                self.predictors['lac'] = LACScorer(conf_config)
            elif method == 'aps':
                self.predictors['aps'] = APSScorer(conf_config)
    
    def calibrate(
        self,
        calibration_probabilities: np.ndarray,
        calibration_labels: np.ndarray,
        option_letters: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calibrate all conformal predictors.
        
        Args:
            calibration_probabilities: Probabilities for calibration set
                Shape: [num_calibration, num_options]
            calibration_labels: True labels for calibration set
                Shape: [num_calibration]
            option_letters: Letters for options (default: A-F)
            
        Returns:
            Dictionary mapping method names to thresholds
        """
        if option_letters is None:
            option_letters = ['A', 'B', 'C', 'D', 'E', 'F']
        
        logger.info(f"Calibrating {len(self.predictors)} conformal predictors...")
        
        thresholds = {}
        for method_name, predictor in self.predictors.items():
            logger.info(f"  Calibrating {method_name.upper()}...")
            threshold = predictor.calibrate(
                calibration_probabilities=calibration_probabilities,
                calibration_labels=calibration_labels,
                option_letters=option_letters
            )
            thresholds[method_name] = threshold
        
        logger.info("Calibration complete")
        return thresholds
    
    def generate(
        self,
        test_probabilities: np.ndarray,
        test_labels: Optional[np.ndarray] = None,
        option_letters: Optional[List[str]] = None,
        instance_ids: Optional[List[str]] = None
    ) -> Union[ConformalPredictionResult, Dict[str, ConformalPredictionResult]]:
        """
        Generate prediction sets for test data.
        
        Args:
            test_probabilities: Probabilities for test set
                Shape: [num_test, num_options]
            test_labels: True labels for test set (optional)
                Shape: [num_test]
            option_letters: Letters for options (default: A-F)
            instance_ids: IDs for test instances
            
        Returns:
            If aggregation='separate': Dict mapping methods to results
            Otherwise: Single aggregated result
        """
        if option_letters is None:
            option_letters = ['A', 'B', 'C', 'D', 'E', 'F']
        
        if instance_ids is None:
            instance_ids = [f"test_{i}" for i in range(len(test_probabilities))]
        
        # Generate prediction sets using each method
        results = {}
        for method_name, predictor in self.predictors.items():
            logger.info(f"Generating prediction sets using {method_name.upper()}...")
            result = predictor.predict(
                test_probabilities=test_probabilities,
                test_labels=test_labels,
                option_letters=option_letters,
                instance_ids=instance_ids
            )
            results[method_name] = result
            
            # Validate if configured
            if self.config.validate_results and test_labels is not None:
                validator = ConformalPredictionValidator()
                validation = validator.validate_coverage(result)
                if not validation['meets_guarantee']:
                    logger.warning(
                        f"{method_name.upper()} does not meet coverage guarantee: "
                        f"{validation['actual_coverage']:.2%} < {validation['target_coverage']:.2%}"
                    )
        
        # Return based on aggregation strategy
        if self.config.aggregation == 'separate':
            return results
        else:
            return self._aggregate_results(
                results,
                test_labels,
                option_letters,
                instance_ids
            )
    
    def _aggregate_results(
        self,
        results: Dict[str, ConformalPredictionResult],
        test_labels: Optional[np.ndarray],
        option_letters: List[str],
        instance_ids: List[str]
    ) -> ConformalPredictionResult:
        """
        Aggregate results from multiple methods.
        
        Args:
            results: Dictionary of results from each method
            test_labels: True labels (optional)
            option_letters: Option letters
            instance_ids: Instance IDs
            
        Returns:
            Aggregated ConformalPredictionResult
        """
        if self.config.aggregation == 'average':
            return self._aggregate_average(
                results, test_labels, option_letters, instance_ids
            )
        elif self.config.aggregation == 'union':
            return self._aggregate_union(
                results, test_labels, option_letters, instance_ids
            )
        elif self.config.aggregation == 'intersection':
            return self._aggregate_intersection(
                results, test_labels, option_letters, instance_ids
            )
        else:
            raise ValueError(f"Unknown aggregation: {self.config.aggregation}")
    
    def _aggregate_average(
        self,
        results: Dict[str, ConformalPredictionResult],
        test_labels: Optional[np.ndarray],
        option_letters: List[str],
        instance_ids: List[str]
    ) -> ConformalPredictionResult:
        """Aggregate by averaging metrics across methods."""
        # Compute average metrics
        avg_coverage = np.mean([r.coverage_rate for r in results.values()])
        avg_set_size = np.mean([r.average_set_size for r in results.values()])
        
        # For prediction sets, take the first method's sets
        # (This is mainly for reporting; actual usage should use separate results)
        first_result = list(results.values())[0]
        
        # Create aggregated result
        config = ConformalConfig(alpha=self.config.alpha, score_function='aggregated')
        
        return ConformalPredictionResult(
            prediction_sets=first_result.prediction_sets,
            config=config,
            threshold=np.mean([r.threshold for r in results.values()]),
            calibration_scores=first_result.calibration_scores,
            num_calibration=first_result.num_calibration,
            num_test=first_result.num_test,
            coverage_rate=avg_coverage,
            average_set_size=avg_set_size,
            set_size_distribution=first_result.set_size_distribution
        )
    
    def _aggregate_union(
        self,
        results: Dict[str, ConformalPredictionResult],
        test_labels: Optional[np.ndarray],
        option_letters: List[str],
        instance_ids: List[str]
    ) -> ConformalPredictionResult:
        """Aggregate by taking union of prediction sets."""
        # Get all prediction sets
        all_pred_sets = {
            method: result.prediction_sets 
            for method, result in results.items()
        }
        
        # Create union prediction sets
        union_sets = []
        for i in range(len(instance_ids)):
            # Get all options from all methods for this instance
            all_options = set()
            all_scores = {}
            
            for method, pred_sets in all_pred_sets.items():
                ps = pred_sets[i]
                all_options.update(ps.options)
                all_scores.update({f"{method}_{k}": v for k, v in ps.scores.items()})
            
            # Create union set
            union_options = sorted(list(all_options))
            
            # Check if true label is in union
            contains_true = None
            true_label = None
            if test_labels is not None:
                true_label = option_letters[test_labels[i]]
                contains_true = true_label in union_options
            
            union_set = PredictionSet(
                instance_id=instance_ids[i],
                options=union_options,
                scores=all_scores,
                threshold=np.mean([r.threshold for r in results.values()]),
                size=len(union_options),
                contains_true=contains_true,
                true_label=true_label,
                metadata={'aggregation': 'union'}
            )
            union_sets.append(union_set)
        
        # Compute statistics
        set_sizes = np.array([ps.size for ps in union_sets])
        avg_set_size = np.mean(set_sizes)
        
        coverage_rate = 0.0
        if test_labels is not None:
            coverage_rate = np.mean([ps.contains_true for ps in union_sets])
        
        set_size_distribution = {}
        for size in set_sizes:
            set_size_distribution[int(size)] = set_size_distribution.get(int(size), 0) + 1
        
        first_result = list(results.values())[0]
        config = ConformalConfig(alpha=self.config.alpha, score_function='union')
        
        return ConformalPredictionResult(
            prediction_sets=union_sets,
            config=config,
            threshold=np.mean([r.threshold for r in results.values()]),
            calibration_scores=first_result.calibration_scores,
            num_calibration=first_result.num_calibration,
            num_test=len(union_sets),
            coverage_rate=coverage_rate,
            average_set_size=avg_set_size,
            set_size_distribution=set_size_distribution
        )
    
    def _aggregate_intersection(
        self,
        results: Dict[str, ConformalPredictionResult],
        test_labels: Optional[np.ndarray],
        option_letters: List[str],
        instance_ids: List[str]
    ) -> ConformalPredictionResult:
        """Aggregate by taking intersection of prediction sets."""
        # Get all prediction sets
        all_pred_sets = {
            method: result.prediction_sets 
            for method, result in results.items()
        }
        
        # Create intersection prediction sets
        intersection_sets = []
        for i in range(len(instance_ids)):
            # Get intersection of options from all methods
            method_options = [
                set(pred_sets[i].options)
                for pred_sets in all_pred_sets.values()
            ]
            
            common_options = set.intersection(*method_options)
            
            # If intersection is empty, take union (fallback)
            if not common_options:
                logger.debug(f"Empty intersection for {instance_ids[i]}, using union")
                common_options = set.union(*method_options)
            
            # Collect scores
            all_scores = {}
            for method, pred_sets in all_pred_sets.items():
                ps = pred_sets[i]
                all_scores.update({f"{method}_{k}": v for k, v in ps.scores.items()})
            
            # Create intersection set
            intersection_options = sorted(list(common_options))
            
            # Check if true label is in intersection
            contains_true = None
            true_label = None
            if test_labels is not None:
                true_label = option_letters[test_labels[i]]
                contains_true = true_label in intersection_options
            
            intersection_set = PredictionSet(
                instance_id=instance_ids[i],
                options=intersection_options,
                scores=all_scores,
                threshold=np.mean([r.threshold for r in results.values()]),
                size=len(intersection_options),
                contains_true=contains_true,
                true_label=true_label,
                metadata={'aggregation': 'intersection'}
            )
            intersection_sets.append(intersection_set)
        
        # Compute statistics
        set_sizes = np.array([ps.size for ps in intersection_sets])
        avg_set_size = np.mean(set_sizes)
        
        coverage_rate = 0.0
        if test_labels is not None:
            coverage_rate = np.mean([ps.contains_true for ps in intersection_sets])
        
        set_size_distribution = {}
        for size in set_sizes:
            set_size_distribution[int(size)] = set_size_distribution.get(int(size), 0) + 1
        
        first_result = list(results.values())[0]
        config = ConformalConfig(alpha=self.config.alpha, score_function='intersection')
        
        return ConformalPredictionResult(
            prediction_sets=intersection_sets,
            config=config,
            threshold=np.mean([r.threshold for r in results.values()]),
            calibration_scores=first_result.calibration_scores,
            num_calibration=first_result.num_calibration,
            num_test=len(intersection_sets),
            coverage_rate=coverage_rate,
            average_set_size=avg_set_size,
            set_size_distribution=set_size_distribution
        )
    
    def save_results(
        self,
        results: Union[ConformalPredictionResult, Dict[str, ConformalPredictionResult]],
        output_path: str
    ) -> None:
        """
        Save prediction results to file.
        
        Args:
            results: Results to save
            output_path: Path to save file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to dict
        if isinstance(results, dict):
            data = {
                'aggregation': 'separate',
                'methods': list(results.keys()),
                'results': {
                    method: result.to_dict()
                    for method, result in results.items()
                }
            }
        else:
            data = {
                'aggregation': self.config.aggregation,
                'result': results.to_dict()
            }
        
        # Add config
        data['config'] = self.config.to_dict()
        
        # Save
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved results to {output_path}")
    
    def update_alpha(self, new_alpha: float) -> None:
        """
        Update error rate for all predictors.
        
        Args:
            new_alpha: New error rate
        """
        self.config.alpha = new_alpha
        
        for predictor in self.predictors.values():
            predictor.update_alpha(new_alpha)
        
        logger.info(f"Updated alpha to {new_alpha} for all predictors")


class PredictionSetComparator:
    """Utility for comparing prediction sets from different methods."""
    
    @staticmethod
    def compare_methods(
        results: Dict[str, ConformalPredictionResult]
    ) -> Dict[str, Any]:
        """
        Compare results from multiple methods.
        
        Args:
            results: Dictionary mapping method names to results
            
        Returns:
            Comparison statistics
        """
        comparison = {}
        
        for method, result in results.items():
            comparison[method] = {
                'coverage_rate': result.coverage_rate,
                'average_set_size': result.average_set_size,
                'meets_guarantee': result.meets_coverage_guarantee(),
                'set_size_distribution': result.set_size_distribution
            }
        
        # Find most efficient method (smallest size with coverage guarantee)
        valid_methods = {
            name: stats for name, stats in comparison.items()
            if stats['meets_guarantee']
        }
        
        if valid_methods:
            best_method = min(
                valid_methods.items(),
                key=lambda x: x[1]['average_set_size']
            )[0]
            comparison['best_method'] = best_method
            comparison['best_method_size'] = valid_methods[best_method]['average_set_size']
        
        return comparison
    
    @staticmethod
    def analyze_agreement(
        results: Dict[str, ConformalPredictionResult]
    ) -> Dict[str, Any]:
        """
        Analyze agreement between different methods.
        
        Args:
            results: Dictionary mapping method names to results
            
        Returns:
            Agreement statistics
        """
        method_names = list(results.keys())
        n_instances = results[method_names[0]].num_test
        
        # Check exact agreement (same prediction sets)
        exact_agreement_count = 0
        
        for i in range(n_instances):
            sets = [
                set(results[method].prediction_sets[i].options)
                for method in method_names
            ]
            
            # Check if all sets are equal
            if all(s == sets[0] for s in sets):
                exact_agreement_count += 1
        
        exact_agreement_rate = exact_agreement_count / n_instances
        
        # Compute pairwise overlaps
        pairwise_overlaps = {}
        for i, method1 in enumerate(method_names):
            for method2 in method_names[i+1:]:
                overlaps = []
                for j in range(n_instances):
                    set1 = set(results[method1].prediction_sets[j].options)
                    set2 = set(results[method2].prediction_sets[j].options)
                    
                    if len(set1.union(set2)) > 0:
                        overlap = len(set1.intersection(set2)) / len(set1.union(set2))
                    else:
                        overlap = 1.0
                    
                    overlaps.append(overlap)
                
                pair_name = f"{method1}-{method2}"
                pairwise_overlaps[pair_name] = {
                    'mean_overlap': float(np.mean(overlaps)),
                    'min_overlap': float(np.min(overlaps)),
                    'max_overlap': float(np.max(overlaps))
                }
        
        return {
            'exact_agreement_rate': float(exact_agreement_rate),
            'exact_agreement_count': int(exact_agreement_count),
            'pairwise_overlaps': pairwise_overlaps
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
    print("Prediction Set Generator Test")
    print("="*80)
    
    # Create sample data
    np.random.seed(42)
    
    # Calibration data
    n_cal = 100
    n_options = 6
    cal_probs = np.random.dirichlet(np.ones(n_options), size=n_cal)
    cal_labels = np.random.randint(0, n_options, size=n_cal)
    
    # Test data
    n_test = 30
    test_probs = np.random.dirichlet(np.ones(n_options), size=n_test)
    test_labels = np.random.randint(0, n_options, size=n_test)
    
    option_letters = ['A', 'B', 'C', 'D', 'E', 'F']
    
    # Test with separate results
    print("\n" + "="*80)
    print("Test 1: Separate Results (LAC and APS)")
    print("="*80)
    
    config1 = PredictionSetGeneratorConfig(
        methods=['lac', 'aps'],
        alpha=0.1,
        aggregation='separate'
    )
    
    generator1 = PredictionSetGenerator(config1)
    thresholds = generator1.calibrate(cal_probs, cal_labels, option_letters)
    
    print(f"\nThresholds:")
    for method, threshold in thresholds.items():
        print(f"  {method.upper()}: {threshold:.4f}")
    
    results_separate = generator1.generate(
        test_probabilities=test_probs,
        test_labels=test_labels,
        option_letters=option_letters
    )
    
    print(f"\nResults:")
    for method, result in results_separate.items():
        print(f"\n{method.upper()}:")
        print(f"  Coverage: {result.coverage_rate:.2%}")
        print(f"  Avg set size: {result.average_set_size:.2f}")
        print(f"  Meets guarantee: {result.meets_coverage_guarantee()}")
    
    # Compare methods
    comparator = PredictionSetComparator()
    comparison = comparator.compare_methods(results_separate)
    
    print(f"\nBest method: {comparison.get('best_method', 'None')}")
    if 'best_method' in comparison:
        print(f"  Avg set size: {comparison['best_method_size']:.2f}")
    
    # Analyze agreement
    agreement = comparator.analyze_agreement(results_separate)
    print(f"\nAgreement analysis:")
    print(f"  Exact agreement rate: {agreement['exact_agreement_rate']:.2%}")
    print(f"  Pairwise overlaps:")
    for pair, stats in agreement['pairwise_overlaps'].items():
        print(f"    {pair}: {stats['mean_overlap']:.2%}")
    
    # Test with aggregation
    print("\n" + "="*80)
    print("Test 2: Union Aggregation")
    print("="*80)
    
    config2 = PredictionSetGeneratorConfig(
        methods=['lac', 'aps'],
        alpha=0.1,
        aggregation='union'
    )
    
    generator2 = PredictionSetGenerator(config2)
    generator2.calibrate(cal_probs, cal_labels, option_letters)
    
    result_union = generator2.generate(
        test_probabilities=test_probs,
        test_labels=test_labels,
        option_letters=option_letters
    )
    
    print(f"\nUnion results:")
    print(f"  Coverage: {result_union.coverage_rate:.2%}")
    print(f"  Avg set size: {result_union.average_set_size:.2f}")
    print(f"  Meets guarantee: {result_union.meets_coverage_guarantee()}")
    
    # Save results
    print("\n" + "="*80)
    print("Saving results...")
    print("="*80)
    
    output_dir = Path("./results/conformal")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator1.save_results(results_separate, output_dir / "results_separate.json")
    generator2.save_results(result_union, output_dir / "results_union.json")
    
    print(f"\nResults saved to {output_dir}")