"""
Dataset Processor Module
Processes loaded datasets to 6-option multiple choice format and handles various transformations.
"""

import logging
import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, replace
from copy import deepcopy

from src.data.dataset_loader import DataInstance, TaskDataset

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class ProcessedDataInstance:
    """Represents a processed data instance with 6 options."""
    id: str
    question: str
    options: List[str]  # Exactly 6 options [A, B, C, D, E, F]
    answer: str  # Correct answer letter (A, B, C, or D only)
    context: Optional[str] = None
    dialogue: Optional[str] = None
    document: Optional[str] = None
    metadata: Optional[Dict] = None
    original_num_options: int = 4


def _compute_string_similarity(s1: str, s2: str) -> float:
    """
    Compute simple string similarity ratio between two strings.
    Uses character-level comparison for efficiency.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Similarity ratio between 0 and 1
    """
    if not s1 or not s2:
        return 0.0

    # Normalize strings
    s1_lower = s1.lower().strip()
    s2_lower = s2.lower().strip()

    # Exact match
    if s1_lower == s2_lower:
        return 1.0

    # Compute Jaccard similarity on character trigrams
    def get_trigrams(s):
        if len(s) < 3:
            return {s}
        return {s[i:i+3] for i in range(len(s) - 2)}

    trigrams1 = get_trigrams(s1_lower)
    trigrams2 = get_trigrams(s2_lower)

    if not trigrams1 or not trigrams2:
        return 0.0

    intersection = len(trigrams1 & trigrams2)
    union = len(trigrams1 | trigrams2)

    return intersection / union if union > 0 else 0.0


class DatasetProcessor:
    """Processes datasets to standardized 6-option format."""

    # Similarity threshold for considering options as duplicates
    OPTION_SIMILARITY_THRESHOLD = 0.8

    def __init__(self, seed: int = 42):
        """
        Initialize the dataset processor.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)

        # Standard additional options
        self.option_e = "I don't know"
        self.option_f = "None of the above"

        # Statistics tracking for option expansion
        self._expansion_stats = {
            'instances_expanded': 0,
            'total_options_sampled': 0,
            'duplicate_options_avoided': 0,
            'fallback_options_used': 0
        }
    
    def process_dataset(self, dataset: TaskDataset) -> TaskDataset:
        """
        Process a TaskDataset to 6-option format.

        Args:
            dataset: Original TaskDataset with 4 or 2 options

        Returns:
            TaskDataset with all instances converted to 6 options
        """
        logger.info(f"Processing {dataset.task_name} dataset to 6-option format...")

        # Reset expansion stats for this dataset
        self._expansion_stats = {
            'instances_expanded': 0,
            'total_options_sampled': 0,
            'duplicate_options_avoided': 0,
            'fallback_options_used': 0
        }

        processed_instances = []

        for instance in dataset.instances:
            processed = self._process_instance(instance, dataset)
            processed_instances.append(processed)

        # Create new dataset with processed instances
        processed_dataset = TaskDataset(
            task_name=dataset.task_name,
            instances=processed_instances,
            task_type=dataset.task_type,
            num_original_options=6  # Now all have 6 options
        )

        logger.info(f"Processed {len(processed_instances)} instances for {dataset.task_name}")

        # Log expansion statistics if any expansion occurred
        if self._expansion_stats['instances_expanded'] > 0:
            logger.info(f"Option expansion statistics for {dataset.task_name}:")
            logger.info(f"  Instances expanded (2→4 options): {self._expansion_stats['instances_expanded']}")
            logger.info(f"  Total options sampled: {self._expansion_stats['total_options_sampled']}")
            logger.info(f"  Duplicate options avoided: {self._expansion_stats['duplicate_options_avoided']}")
            logger.info(f"  Fallback options used: {self._expansion_stats['fallback_options_used']}")

        return processed_dataset

    def get_expansion_stats(self) -> Dict:
        """Get statistics about option expansion from the last process_dataset call."""
        return self._expansion_stats.copy()
    
    def _process_instance(
        self,
        instance: DataInstance,
        dataset: TaskDataset
    ) -> DataInstance:
        """
        Process a single instance to 6-option format.

        Args:
            instance: Original DataInstance
            dataset: Parent TaskDataset for context

        Returns:
            Processed DataInstance with 6 options

        Raises:
            ValueError: If the answer is not A, B, C, or D (per paper specification)
        """
        # Start with existing options
        options = instance.options.copy()
        num_original = len(options)

        # If original has only 2 options (HaluDial, HaluSum), expand to 4 first
        if num_original == 2 or (num_original > 2 and any(not opt for opt in options[:4])):
            options = self._expand_to_four_options(instance, dataset)
            self._expansion_stats['instances_expanded'] += 1

        # Ensure we have exactly 4 options before adding E and F
        if len(options) != 4:
            logger.warning(
                f"Instance {instance.id} has {len(options)} options, "
                f"expected 4. Adjusting..."
            )
            options = self._normalize_to_four_options(options)

        # Add options E and F
        options.append(self.option_e)
        options.append(self.option_f)

        # CRITICAL: Validate answer is A, B, C, or D (per paper specification)
        # The correct answer should NEVER be E ("I don't know") or F ("None of the above")
        if instance.answer not in ['A', 'B', 'C', 'D']:
            error_msg = (
                f"Instance {instance.id} has invalid answer: '{instance.answer}'. "
                f"Per paper specification, correct answers must be A, B, C, or D only "
                f"(never E or F). This indicates a data loading or processing issue."
            )
            logger.error(error_msg)
            # Raise error instead of silently defaulting - this is a critical issue
            raise ValueError(error_msg)

        answer = instance.answer

        # Create new instance with updated options
        return DataInstance(
            id=instance.id,
            question=instance.question,
            options=options,
            answer=answer,
            context=instance.context,
            dialogue=instance.dialogue,
            document=instance.document,
            metadata={
                **(instance.metadata or {}),
                'original_num_options': num_original,
                'processed': True
            }
        )
    
    def _expand_to_four_options(
        self,
        instance: DataInstance,
        dataset: TaskDataset
    ) -> List[str]:
        """
        Expand 2-option instances to 4 options by sampling from other instances.
        Used for HaluDial and HaluSum datasets.
        
        Args:
            instance: Instance with 2 options
            dataset: Parent dataset to sample additional options from
            
        Returns:
            List of 4 options
        """
        # Start with existing 2 options
        options = [opt for opt in instance.options if opt]  # Filter empty strings
        
        if len(options) < 2:
            logger.warning(
                f"Instance {instance.id} has less than 2 valid options. "
                f"This may cause issues."
            )
        
        # Sample 2 additional options from other instances
        additional_options = self._sample_additional_options(
            instance=instance,
            dataset=dataset,
            num_needed=2
        )
        
        options.extend(additional_options)
        
        # Shuffle to randomize position of correct answer
        # But first, remember which is correct
        correct_option = options[0]  # Assuming first option is correct
        
        # Create shuffled version
        shuffled_options = options.copy()
        random.shuffle(shuffled_options)
        
        # Update the answer based on new position
        new_answer_idx = shuffled_options.index(correct_option)
        instance.answer = chr(65 + new_answer_idx)  # Convert to A, B, C, D
        
        return shuffled_options
    
    def _sample_additional_options(
        self,
        instance: DataInstance,
        dataset: TaskDataset,
        num_needed: int
    ) -> List[str]:
        """
        Sample additional options from other instances in the dataset.
        Validates that sampled options are not too similar to existing options.

        Args:
            instance: Current instance
            dataset: Parent dataset
            num_needed: Number of additional options needed

        Returns:
            List of sampled options
        """
        # Get existing options to check similarity against
        existing_options = [opt for opt in instance.options if opt]

        # Get all other instances
        other_instances = [
            inst for inst in dataset.instances
            if inst.id != instance.id
        ]

        if not other_instances:
            logger.warning(
                f"No other instances available to sample from for {instance.id}. "
                f"Using placeholder options."
            )
            self._expansion_stats['fallback_options_used'] += num_needed
            return [f"Option {i}" for i in range(num_needed)]

        # Shuffle to ensure randomness
        shuffled_instances = other_instances.copy()
        random.shuffle(shuffled_instances)

        # Extract options from sampled instances, checking for similarity
        additional_options = []
        candidates_checked = 0
        max_candidates = min(len(shuffled_instances), num_needed * 10)  # Check up to 10x needed

        for sampled in shuffled_instances[:max_candidates]:
            if len(additional_options) >= num_needed:
                break

            candidates_checked += 1

            # Take the first non-empty option from the sampled instance
            candidate_option = None
            if sampled.options:
                for opt in sampled.options:
                    if opt:
                        candidate_option = opt
                        break

            if not candidate_option:
                continue

            # Check similarity against existing and already-added options
            all_current_options = existing_options + additional_options
            is_duplicate = False

            for existing in all_current_options:
                similarity = _compute_string_similarity(candidate_option, existing)
                if similarity >= self.OPTION_SIMILARITY_THRESHOLD:
                    is_duplicate = True
                    self._expansion_stats['duplicate_options_avoided'] += 1
                    logger.debug(
                        f"Skipping similar option (similarity={similarity:.2f}): "
                        f"'{candidate_option[:50]}...' similar to '{existing[:50]}...'"
                    )
                    break

            if not is_duplicate:
                additional_options.append(candidate_option)
                self._expansion_stats['total_options_sampled'] += 1

        # If we don't have enough valid options, pad with placeholders
        while len(additional_options) < num_needed:
            placeholder = f"Alternative option {len(additional_options) + 1}"
            additional_options.append(placeholder)
            self._expansion_stats['fallback_options_used'] += 1
            logger.debug(f"Using placeholder option for {instance.id}: {placeholder}")

        return additional_options[:num_needed]
    
    def _normalize_to_four_options(self, options: List[str]) -> List[str]:
        """
        Normalize any option list to exactly 4 options.
        
        Args:
            options: List of options (any length)
            
        Returns:
            List of exactly 4 options
        """
        # Filter out empty strings
        valid_options = [opt for opt in options if opt]
        
        if len(valid_options) == 4:
            return valid_options
        elif len(valid_options) > 4:
            # Take first 4
            logger.warning(f"More than 4 options found, taking first 4")
            return valid_options[:4]
        else:
            # Pad with placeholders
            while len(valid_options) < 4:
                valid_options.append(f"Option {len(valid_options) + 1}")
            return valid_options
    
    def validate_processed_dataset(self, dataset: TaskDataset) -> bool:
        """
        Validate that all instances in the dataset are properly formatted.
        
        Args:
            dataset: Processed TaskDataset
            
        Returns:
            True if valid, False otherwise
        """
        logger.info(f"Validating {dataset.task_name} dataset...")
        
        is_valid = True
        errors = []
        
        for instance in dataset.instances:
            # Check number of options
            if len(instance.options) != 6:
                errors.append(
                    f"Instance {instance.id} has {len(instance.options)} options, "
                    f"expected 6"
                )
                is_valid = False
            
            # Check that options E and F are correct
            if len(instance.options) >= 6:
                if instance.options[4] != self.option_e:
                    errors.append(
                        f"Instance {instance.id} has incorrect option E: "
                        f"{instance.options[4]}"
                    )
                    is_valid = False
                
                if instance.options[5] != self.option_f:
                    errors.append(
                        f"Instance {instance.id} has incorrect option F: "
                        f"{instance.options[5]}"
                    )
                    is_valid = False
            
            # Check that answer is valid (A, B, C, or D only)
            if instance.answer not in ['A', 'B', 'C', 'D']:
                errors.append(
                    f"Instance {instance.id} has invalid answer: {instance.answer}"
                )
                is_valid = False
            
            # Check that correct answer exists in options
            answer_idx = ord(instance.answer) - 65
            if answer_idx >= len(instance.options):
                errors.append(
                    f"Instance {instance.id} answer index {answer_idx} "
                    f"out of range"
                )
                is_valid = False
        
        if errors:
            logger.error(f"Validation failed with {len(errors)} errors:")
            for error in errors[:10]:  # Show first 10 errors
                logger.error(f"  - {error}")
            if len(errors) > 10:
                logger.error(f"  ... and {len(errors) - 10} more errors")
        else:
            logger.info(f"Validation passed for {dataset.task_name}")
        
        return is_valid
    
    def get_statistics(self, dataset: TaskDataset) -> Dict:
        """
        Get statistics about the processed dataset.
        
        Args:
            dataset: TaskDataset to analyze
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            'task_name': dataset.task_name,
            'total_instances': len(dataset.instances),
            'answer_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0},
            'has_context': 0,
            'has_dialogue': 0,
            'has_document': 0,
            'avg_question_length': 0,
            'avg_option_length': 0,
        }
        
        total_question_length = 0
        total_option_length = 0
        
        for instance in dataset.instances:
            # Answer distribution
            if instance.answer in stats['answer_distribution']:
                stats['answer_distribution'][instance.answer] += 1
            
            # Context types
            if instance.context:
                stats['has_context'] += 1
            if instance.dialogue:
                stats['has_dialogue'] += 1
            if instance.document:
                stats['has_document'] += 1
            
            # Lengths
            total_question_length += len(instance.question)
            total_option_length += sum(len(opt) for opt in instance.options[:4])
        
        stats['avg_question_length'] = total_question_length / len(dataset.instances)
        stats['avg_option_length'] = total_option_length / (len(dataset.instances) * 4)
        
        return stats


def process_all_datasets(
    datasets: Dict[str, TaskDataset],
    seed: int = 42,
    validate: bool = True
) -> Dict[str, TaskDataset]:
    """
    Process multiple datasets to 6-option format.
    
    Args:
        datasets: Dictionary of TaskDataset objects
        seed: Random seed for reproducibility
        validate: Whether to validate processed datasets
        
    Returns:
        Dictionary of processed TaskDataset objects
    """
    processor = DatasetProcessor(seed=seed)
    processed_datasets = {}
    
    for task_name, dataset in datasets.items():
        logger.info(f"\nProcessing {task_name} dataset...")
        
        # Process dataset
        processed = processor.process_dataset(dataset)
        
        # Validate if requested
        if validate:
            is_valid = processor.validate_processed_dataset(processed)
            if not is_valid:
                raise ValueError(f"Validation failed for {task_name} dataset")
        
        # Get and log statistics
        stats = processor.get_statistics(processed)
        logger.info(f"Statistics for {task_name}:")
        logger.info(f"  Total instances: {stats['total_instances']}")
        logger.info(f"  Answer distribution: {stats['answer_distribution']}")
        logger.info(f"  Avg question length: {stats['avg_question_length']:.1f}")
        logger.info(f"  Avg option length: {stats['avg_option_length']:.1f}")
        
        processed_datasets[task_name] = processed
    
    return processed_datasets


def save_processed_dataset(
    dataset: TaskDataset,
    output_path: str,
    format: str = 'json'
) -> None:
    """
    Save processed dataset to file.
    
    Args:
        dataset: TaskDataset to save
        output_path: Path to save file
        format: Output format ('json' or 'jsonl')
    """
    import json
    from pathlib import Path
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert instances to dictionaries
    data = {
        'task_name': dataset.task_name,
        'task_type': dataset.task_type,
        'num_instances': len(dataset.instances),
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
    
    if format == 'json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    elif format == 'jsonl':
        with open(output_path, 'w', encoding='utf-8') as f:
            for instance in data['instances']:
                f.write(json.dumps(instance, ensure_ascii=False) + '\n')
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Saved {dataset.task_name} dataset to {output_path}")


def load_processed_dataset(input_path: str) -> TaskDataset:
    """
    Load processed dataset from file.
    
    Args:
        input_path: Path to load from
        
    Returns:
        TaskDataset object
    """
    import json
    from pathlib import Path
    
    input_path = Path(input_path)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert dictionaries to DataInstance objects
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
    
    dataset = TaskDataset(
        task_name=data['task_name'],
        instances=instances,
        task_type=data['task_type'],
        num_original_options=6
    )
    
    logger.info(f"Loaded {dataset.task_name} dataset from {input_path}")
    return dataset


# Example usage
if __name__ == "__main__":
    import logging
    from src.data.dataset_loader import load_all_datasets
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load datasets
    logger.info("Loading datasets...")
    datasets = load_all_datasets(
        tasks=["qa", "rc", "ci", "drs", "ds"],
        num_samples=100,  # Small sample for testing
        seed=42
    )
    
    # Process datasets
    logger.info("\nProcessing datasets...")
    processed_datasets = process_all_datasets(
        datasets=datasets,
        seed=42,
        validate=True
    )
    
    # Save processed datasets
    logger.info("\nSaving processed datasets...")
    for task_name, dataset in processed_datasets.items():
        output_path = f"./data/processed/{task_name}_processed.json"
        save_processed_dataset(dataset, output_path)
    
    # Print sample
    sample_task = list(processed_datasets.keys())[0]
    sample_dataset = processed_datasets[sample_task]
    sample_instance = sample_dataset.instances[0]
    
    print(f"\n{'='*80}")
    print(f"Sample from {sample_task.upper()} dataset:")
    print(f"{'='*80}")
    print(f"Question: {sample_instance.question}")
    print(f"\nOptions:")
    for i, opt in enumerate(sample_instance.options):
        letter = chr(65 + i)
        marker = " ← CORRECT" if letter == sample_instance.answer else ""
        print(f"  {letter}. {opt[:100]}...{marker}" if len(opt) > 100 else f"  {letter}. {opt}{marker}")
    print(f"\nCorrect Answer: {sample_instance.answer}")