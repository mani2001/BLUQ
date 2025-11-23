"""
Data Splitter Module
Handles splitting datasets into calibration and test sets for conformal prediction.
"""

import logging
import random
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import json

from src.data.dataset_loader import DataInstance, TaskDataset

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class DataSplit:
    """Container for calibration and test splits."""
    calibration: TaskDataset
    test: TaskDataset
    split_ratio: float
    seed: int
    
    def get_split_info(self) -> Dict:
        """Get information about the split."""
        return {
            'task_name': self.calibration.task_name,
            'total_instances': len(self.calibration.instances) + len(self.test.instances),
            'calibration_size': len(self.calibration.instances),
            'test_size': len(self.test.instances),
            'split_ratio': self.split_ratio,
            'seed': self.seed
        }


class DataSplitter:
    """Splits datasets into calibration and test sets."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize the data splitter.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
    
    def split_dataset(
        self,
        dataset: TaskDataset,
        calibration_ratio: float = 0.5,
        stratify_by_answer: bool = True
    ) -> DataSplit:
        """
        Split dataset into calibration and test sets.
        
        Args:
            dataset: TaskDataset to split
            calibration_ratio: Ratio of data to use for calibration (0 to 1)
            stratify_by_answer: Whether to stratify split by answer distribution
            
        Returns:
            DataSplit containing calibration and test datasets
        """
        if not 0 < calibration_ratio < 1:
            raise ValueError(f"calibration_ratio must be between 0 and 1, got {calibration_ratio}")
        
        logger.info(
            f"Splitting {dataset.task_name} dataset "
            f"(calibration: {calibration_ratio:.0%}, test: {1-calibration_ratio:.0%})"
        )
        
        if stratify_by_answer:
            cal_instances, test_instances = self._stratified_split(
                dataset.instances,
                calibration_ratio
            )
        else:
            cal_instances, test_instances = self._simple_split(
                dataset.instances,
                calibration_ratio
            )
        
        # Create calibration dataset
        calibration_dataset = TaskDataset(
            task_name=dataset.task_name,
            instances=cal_instances,
            task_type=dataset.task_type,
            num_original_options=dataset.num_original_options
        )
        
        # Create test dataset
        test_dataset = TaskDataset(
            task_name=dataset.task_name,
            instances=test_instances,
            task_type=dataset.task_type,
            num_original_options=dataset.num_original_options
        )
        
        split = DataSplit(
            calibration=calibration_dataset,
            test=test_dataset,
            split_ratio=calibration_ratio,
            seed=self.seed
        )
        
        # Log split information
        info = split.get_split_info()
        logger.info(f"Split complete:")
        logger.info(f"  Calibration: {info['calibration_size']} instances")
        logger.info(f"  Test: {info['test_size']} instances")
        
        # Verify split
        self._verify_split(split, stratify_by_answer)
        
        return split
    
    def _simple_split(
        self,
        instances: List[DataInstance],
        calibration_ratio: float
    ) -> Tuple[List[DataInstance], List[DataInstance]]:
        """
        Perform simple random split.
        
        Args:
            instances: List of DataInstance objects
            calibration_ratio: Ratio for calibration set
            
        Returns:
            Tuple of (calibration_instances, test_instances)
        """
        # Shuffle instances
        shuffled = instances.copy()
        random.shuffle(shuffled)
        
        # Split
        split_idx = int(len(shuffled) * calibration_ratio)
        calibration = shuffled[:split_idx]
        test = shuffled[split_idx:]
        
        return calibration, test
    
    def _stratified_split(
        self,
        instances: List[DataInstance],
        calibration_ratio: float
    ) -> Tuple[List[DataInstance], List[DataInstance]]:
        """
        Perform stratified split maintaining answer distribution.
        
        Args:
            instances: List of DataInstance objects
            calibration_ratio: Ratio for calibration set
            
        Returns:
            Tuple of (calibration_instances, test_instances)
        """
        # Group instances by answer
        answer_groups = {'A': [], 'B': [], 'C': [], 'D': []}
        
        for instance in instances:
            if instance.answer in answer_groups:
                answer_groups[instance.answer].append(instance)
            else:
                logger.warning(
                    f"Instance {instance.id} has unexpected answer: {instance.answer}"
                )
        
        calibration = []
        test = []
        
        # Split each group
        for answer, group in answer_groups.items():
            if not group:
                continue
            
            # Shuffle group
            shuffled_group = group.copy()
            random.shuffle(shuffled_group)
            
            # Split group
            split_idx = int(len(shuffled_group) * calibration_ratio)
            calibration.extend(shuffled_group[:split_idx])
            test.extend(shuffled_group[split_idx:])
        
        # Shuffle final lists to mix answers
        random.shuffle(calibration)
        random.shuffle(test)
        
        return calibration, test
    
    def _verify_split(self, split: DataSplit, check_stratification: bool = True) -> None:
        """
        Verify that the split is valid.
        
        Args:
            split: DataSplit to verify
            check_stratification: Whether to check answer distribution
        """
        cal_size = len(split.calibration.instances)
        test_size = len(split.test.instances)
        total = cal_size + test_size
        
        # Check sizes
        expected_cal_size = int(total * split.split_ratio)
        if abs(cal_size - expected_cal_size) > 1:
            logger.warning(
                f"Calibration size {cal_size} differs from expected {expected_cal_size}"
            )
        
        # Check for duplicates
        cal_ids = {inst.id for inst in split.calibration.instances}
        test_ids = {inst.id for inst in split.test.instances}
        
        overlap = cal_ids & test_ids
        if overlap:
            raise ValueError(f"Found {len(overlap)} duplicate IDs between splits: {list(overlap)[:5]}")
        
        # Check answer distribution if stratified
        if check_stratification:
            cal_dist = self._get_answer_distribution(split.calibration.instances)
            test_dist = self._get_answer_distribution(split.test.instances)
            
            logger.info("Answer distribution:")
            logger.info(f"  Calibration: {cal_dist}")
            logger.info(f"  Test: {test_dist}")
            
            # Check if distributions are similar (within 5% for each answer)
            for answer in ['A', 'B', 'C', 'D']:
                cal_pct = cal_dist.get(answer, 0) / cal_size if cal_size > 0 else 0
                test_pct = test_dist.get(answer, 0) / test_size if test_size > 0 else 0
                diff = abs(cal_pct - test_pct)
                
                if diff > 0.05:
                    logger.warning(
                        f"Answer {answer} distribution differs by {diff:.1%} "
                        f"between calibration and test sets"
                    )
    
    def _get_answer_distribution(self, instances: List[DataInstance]) -> Dict[str, int]:
        """Get distribution of answers in a list of instances."""
        distribution = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        for instance in instances:
            if instance.answer in distribution:
                distribution[instance.answer] += 1
        return distribution
    
    def create_k_fold_splits(
        self,
        dataset: TaskDataset,
        k: int = 5,
        stratify_by_answer: bool = True
    ) -> List[DataSplit]:
        """
        Create k-fold cross-validation splits.
        
        Args:
            dataset: TaskDataset to split
            k: Number of folds
            stratify_by_answer: Whether to stratify by answer
            
        Returns:
            List of DataSplit objects (one per fold)
        """
        logger.info(f"Creating {k}-fold splits for {dataset.task_name}")
        
        if stratify_by_answer:
            # Group by answer
            answer_groups = {'A': [], 'B': [], 'C': [], 'D': []}
            for instance in dataset.instances:
                if instance.answer in answer_groups:
                    answer_groups[instance.answer].append(instance)
            
            # Create folds for each answer group
            folds = [[] for _ in range(k)]
            for answer, group in answer_groups.items():
                shuffled = group.copy()
                random.shuffle(shuffled)
                
                # Distribute to folds
                for i, instance in enumerate(shuffled):
                    fold_idx = i % k
                    folds[fold_idx].append(instance)
        else:
            # Simple k-fold split
            shuffled = dataset.instances.copy()
            random.shuffle(shuffled)
            
            fold_size = len(shuffled) // k
            folds = []
            for i in range(k):
                start_idx = i * fold_size
                end_idx = start_idx + fold_size if i < k - 1 else len(shuffled)
                folds.append(shuffled[start_idx:end_idx])
        
        # Create DataSplit objects
        splits = []
        for i in range(k):
            # Use fold i as test, rest as calibration
            test_instances = folds[i]
            cal_instances = []
            for j in range(k):
                if j != i:
                    cal_instances.extend(folds[j])
            
            # Shuffle calibration set
            random.shuffle(cal_instances)
            
            cal_dataset = TaskDataset(
                task_name=dataset.task_name,
                instances=cal_instances,
                task_type=dataset.task_type,
                num_original_options=dataset.num_original_options
            )
            
            test_dataset = TaskDataset(
                task_name=dataset.task_name,
                instances=test_instances,
                task_type=dataset.task_type,
                num_original_options=dataset.num_original_options
            )
            
            split = DataSplit(
                calibration=cal_dataset,
                test=test_dataset,
                split_ratio=(k - 1) / k,
                seed=self.seed
            )
            
            splits.append(split)
            logger.info(f"Fold {i+1}: {len(cal_instances)} cal, {len(test_instances)} test")
        
        return splits


def split_all_datasets(
    datasets: Dict[str, TaskDataset],
    calibration_ratio: float = 0.5,
    stratify_by_answer: bool = True,
    seed: int = 42
) -> Dict[str, DataSplit]:
    """
    Split multiple datasets into calibration and test sets.
    
    Args:
        datasets: Dictionary of TaskDataset objects
        calibration_ratio: Ratio for calibration set
        stratify_by_answer: Whether to stratify by answer
        seed: Random seed
        
    Returns:
        Dictionary mapping task names to DataSplit objects
    """
    splitter = DataSplitter(seed=seed)
    splits = {}
    
    for task_name, dataset in datasets.items():
        logger.info(f"\nSplitting {task_name} dataset...")
        split = splitter.split_dataset(
            dataset=dataset,
            calibration_ratio=calibration_ratio,
            stratify_by_answer=stratify_by_answer
        )
        splits[task_name] = split
    
    return splits


def save_split(split: DataSplit, output_dir: str) -> None:
    """
    Save data split to files.
    
    Args:
        split: DataSplit to save
        output_dir: Directory to save files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    task_name = split.calibration.task_name
    
    # Save calibration set
    cal_path = output_path / f"{task_name}_calibration.json"
    _save_dataset_to_json(split.calibration, cal_path)
    
    # Save test set
    test_path = output_path / f"{task_name}_test.json"
    _save_dataset_to_json(split.test, test_path)
    
    # Save split info
    info_path = output_path / f"{task_name}_split_info.json"
    with open(info_path, 'w') as f:
        json.dump(split.get_split_info(), f, indent=2)
    
    logger.info(f"Saved split for {task_name} to {output_dir}")


def load_split(input_dir: str, task_name: str) -> DataSplit:
    """
    Load data split from files.
    
    Args:
        input_dir: Directory containing split files
        task_name: Name of task to load
        
    Returns:
        DataSplit object
    """
    input_path = Path(input_dir)
    
    # Load calibration set
    cal_path = input_path / f"{task_name}_calibration.json"
    calibration = _load_dataset_from_json(cal_path)
    
    # Load test set
    test_path = input_path / f"{task_name}_test.json"
    test = _load_dataset_from_json(test_path)
    
    # Load split info
    info_path = input_path / f"{task_name}_split_info.json"
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    split = DataSplit(
        calibration=calibration,
        test=test,
        split_ratio=info['split_ratio'],
        seed=info['seed']
    )
    
    logger.info(f"Loaded split for {task_name} from {input_dir}")
    return split


def _save_dataset_to_json(dataset: TaskDataset, path: Path) -> None:
    """Save dataset to JSON file."""
    data = {
        'task_name': dataset.task_name,
        'task_type': dataset.task_type,
        'num_original_options': dataset.num_original_options,
        'instances': [
            {
                'id': inst.id,
                'question': inst.question,
                'options': inst.options,
                'answer': inst.answer,
                'context': inst.context,
                'dialogue': inst.dialogue,
                'document': inst.document,
                'metadata': inst.metadata
            }
            for inst in dataset.instances
        ]
    }
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _load_dataset_from_json(path: Path) -> TaskDataset:
    """Load dataset from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    instances = [
        DataInstance(
            id=inst['id'],
            question=inst['question'],
            options=inst['options'],
            answer=inst['answer'],
            context=inst.get('context'),
            dialogue=inst.get('dialogue'),
            document=inst.get('document'),
            metadata=inst.get('metadata')
        )
        for inst in data['instances']
    ]
    
    return TaskDataset(
        task_name=data['task_name'],
        instances=instances,
        task_type=data['task_type'],
        num_original_options=data.get('num_original_options', 6)
    )


class ExperimentSplitter:
    """Advanced splitter for experimental setups."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.splitter = DataSplitter(seed=seed)
    
    def create_varying_calibration_splits(
        self,
        dataset: TaskDataset,
        calibration_ratios: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5]
    ) -> Dict[float, DataSplit]:
        """
        Create splits with varying calibration set sizes.
        Used for analyzing effect of calibration data amount.
        
        Args:
            dataset: TaskDataset to split
            calibration_ratios: List of calibration ratios to try
            
        Returns:
            Dictionary mapping ratios to DataSplit objects
        """
        logger.info(
            f"Creating splits with varying calibration ratios for {dataset.task_name}"
        )
        
        splits = {}
        for ratio in calibration_ratios:
            split = self.splitter.split_dataset(
                dataset=dataset,
                calibration_ratio=ratio,
                stratify_by_answer=True
            )
            splits[ratio] = split
            logger.info(
                f"Ratio {ratio:.1%}: {len(split.calibration.instances)} cal, "
                f"{len(split.test.instances)} test"
            )
        
        return splits
    
    def create_repeated_splits(
        self,
        dataset: TaskDataset,
        num_repeats: int = 5,
        calibration_ratio: float = 0.5
    ) -> List[DataSplit]:
        """
        Create multiple random splits for statistical robustness.
        
        Args:
            dataset: TaskDataset to split
            num_repeats: Number of different random splits
            calibration_ratio: Ratio for calibration
            
        Returns:
            List of DataSplit objects
        """
        logger.info(
            f"Creating {num_repeats} repeated splits for {dataset.task_name}"
        )
        
        splits = []
        for i in range(num_repeats):
            # Use different seed for each split
            splitter = DataSplitter(seed=self.seed + i)
            split = splitter.split_dataset(
                dataset=dataset,
                calibration_ratio=calibration_ratio,
                stratify_by_answer=True
            )
            splits.append(split)
            logger.info(f"Repeat {i+1}: Created split")
        
        return splits


# Example usage
if __name__ == "__main__":
    import logging
    from src.data.dataset_loader import load_all_datasets
    from src.data.dataset_processor import process_all_datasets
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load and process datasets
    logger.info("Loading datasets...")
    datasets = load_all_datasets(
        tasks=["qa", "rc", "ci"],
        num_samples=1000,  # Small sample for testing
        seed=42
    )
    
    logger.info("\nProcessing datasets...")
    processed_datasets = process_all_datasets(
        datasets=datasets,
        seed=42
    )
    
    # Split datasets
    logger.info("\nSplitting datasets...")
    splits = split_all_datasets(
        datasets=processed_datasets,
        calibration_ratio=0.5,
        stratify_by_answer=True,
        seed=42
    )
    
    # Save splits
    logger.info("\nSaving splits...")
    for task_name, split in splits.items():
        save_split(split, f"./data/splits/{task_name}")
    
    # Test k-fold splitting
    logger.info("\n" + "="*80)
    logger.info("Testing k-fold splitting...")
    splitter = DataSplitter(seed=42)
    sample_dataset = list(processed_datasets.values())[0]
    k_folds = splitter.create_k_fold_splits(sample_dataset, k=5)
    
    print(f"\nCreated {len(k_folds)} folds for {sample_dataset.task_name}")
    for i, fold in enumerate(k_folds):
        info = fold.get_split_info()
        print(f"Fold {i+1}: {info['calibration_size']} cal, {info['test_size']} test")
    
    # Test varying calibration sizes
    logger.info("\n" + "="*80)
    logger.info("Testing varying calibration sizes...")
    exp_splitter = ExperimentSplitter(seed=42)
    var_splits = exp_splitter.create_varying_calibration_splits(
        sample_dataset,
        calibration_ratios=[0.1, 0.2, 0.3, 0.4, 0.5]
    )
    
    print(f"\nCreated {len(var_splits)} splits with different calibration ratios")
    for ratio, split in var_splits.items():
        info = split.get_split_info()
        print(
            f"Ratio {ratio:.1%}: {info['calibration_size']} cal, "
            f"{info['test_size']} test"
        )