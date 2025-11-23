"""
Demonstration Manager Module
Manages selection and formatting of demonstration examples for in-context learning.
"""

import logging
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from src.data.dataset_loader import DataInstance, TaskDataset

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class DemonstrationConfig:
    """Configuration for demonstration selection."""
    # Number of demonstrations per task
    num_demonstrations: int = 5
    
    # Selection strategy
    selection_strategy: str = "random"  # 'random', 'balanced', 'diverse', 'fixed'
    
    # Seed for reproducibility
    seed: int = 42
    
    # Constraints
    max_total_length: Optional[int] = None
    balance_by_answer: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'num_demonstrations': self.num_demonstrations,
            'selection_strategy': self.selection_strategy,
            'seed': self.seed,
            'max_total_length': self.max_total_length,
            'balance_by_answer': self.balance_by_answer
        }


class DemonstrationSelector:
    """Selects demonstration examples from a dataset."""
    
    def __init__(self, config: Optional[DemonstrationConfig] = None):
        """
        Initialize demonstration selector.
        
        Args:
            config: Configuration for demonstration selection
        """
        self.config = config or DemonstrationConfig()
        random.seed(self.config.seed)
        
        logger.info("Initialized DemonstrationSelector")
        logger.info(f"  Strategy: {self.config.selection_strategy}")
        logger.info(f"  Num demonstrations: {self.config.num_demonstrations}")
    
    def select_demonstrations(
        self,
        dataset: TaskDataset,
        num_demonstrations: Optional[int] = None
    ) -> List[DataInstance]:
        """
        Select demonstration examples from dataset.
        
        Args:
            dataset: TaskDataset to select from
            num_demonstrations: Number to select (overrides config)
            
        Returns:
            List of selected DataInstance objects
        """
        n = num_demonstrations or self.config.num_demonstrations
        
        if n <= 0:
            return []
        
        if n > len(dataset.instances):
            logger.warning(
                f"Requested {n} demonstrations but only {len(dataset.instances)} available. "
                f"Using all available."
            )
            n = len(dataset.instances)
        
        logger.info(f"Selecting {n} demonstrations using '{self.config.selection_strategy}' strategy")
        
        if self.config.selection_strategy == 'random':
            return self._select_random(dataset.instances, n)
        elif self.config.selection_strategy == 'balanced':
            return self._select_balanced(dataset.instances, n)
        elif self.config.selection_strategy == 'diverse':
            return self._select_diverse(dataset.instances, n)
        elif self.config.selection_strategy == 'fixed':
            return self._select_fixed(dataset.instances, n)
        else:
            raise ValueError(f"Unknown strategy: {self.config.selection_strategy}")
    
    def _select_random(
        self,
        instances: List[DataInstance],
        n: int
    ) -> List[DataInstance]:
        """Select n random instances."""
        return random.sample(instances, n)
    
    def _select_balanced(
        self,
        instances: List[DataInstance],
        n: int
    ) -> List[DataInstance]:
        """
        Select instances balanced by answer.
        
        Ensures roughly equal representation of A, B, C, D answers.
        """
        # Group by answer
        answer_groups = {'A': [], 'B': [], 'C': [], 'D': []}
        
        for instance in instances:
            if instance.answer in answer_groups:
                answer_groups[instance.answer].append(instance)
        
        # Calculate how many from each group
        per_group = n // 4
        remainder = n % 4
        
        selected = []
        
        for i, (answer, group) in enumerate(sorted(answer_groups.items())):
            if not group:
                continue
            
            # Add one more to first 'remainder' groups
            num_from_group = per_group + (1 if i < remainder else 0)
            num_from_group = min(num_from_group, len(group))
            
            selected.extend(random.sample(group, num_from_group))
        
        # If we don't have enough, sample more randomly
        if len(selected) < n:
            remaining = [inst for inst in instances if inst not in selected]
            needed = n - len(selected)
            if remaining:
                selected.extend(random.sample(remaining, min(needed, len(remaining))))
        
        # Shuffle to mix answers
        random.shuffle(selected)
        
        logger.info(f"Selected {len(selected)} balanced demonstrations")
        return selected[:n]
    
    def _select_diverse(
        self,
        instances: List[DataInstance],
        n: int
    ) -> List[DataInstance]:
        """
        Select diverse instances (different question types if available).
        
        For now, uses balanced selection + random sampling.
        Could be enhanced with embeddings for true diversity.
        """
        # Start with balanced selection
        selected = self._select_balanced(instances, n)
        
        logger.info(f"Selected {len(selected)} diverse demonstrations")
        return selected
    
    def _select_fixed(
        self,
        instances: List[DataInstance],
        n: int
    ) -> List[DataInstance]:
        """
        Select first n instances (fixed set for reproducibility).
        """
        selected = instances[:n]
        logger.info(f"Selected first {len(selected)} instances (fixed)")
        return selected
    
    def validate_demonstrations(
        self,
        demonstrations: List[DataInstance]
    ) -> bool:
        """
        Validate that demonstrations are properly formatted.
        
        Args:
            demonstrations: List of demonstration instances
            
        Returns:
            True if valid, False otherwise
        """
        if not demonstrations:
            return True
        
        is_valid = True
        
        for i, demo in enumerate(demonstrations):
            # Check that answer is valid
            if demo.answer not in ['A', 'B', 'C', 'D']:
                logger.error(
                    f"Demonstration {i} has invalid answer: {demo.answer}"
                )
                is_valid = False
            
            # Check that options exist
            if not demo.options or len(demo.options) != 6:
                logger.error(
                    f"Demonstration {i} has invalid options: {len(demo.options) if demo.options else 0}"
                )
                is_valid = False
            
            # Check that question exists
            if not demo.question:
                logger.error(f"Demonstration {i} has empty question")
                is_valid = False
        
        return is_valid


class DemonstrationManager:
    """Manages demonstrations for all tasks and strategies."""
    
    def __init__(self, config: Optional[DemonstrationConfig] = None):
        """
        Initialize demonstration manager.
        
        Args:
            config: Configuration for demonstration selection
        """
        self.config = config or DemonstrationConfig()
        self.selector = DemonstrationSelector(self.config)
        
        # Cache for selected demonstrations
        self.demonstrations_cache: Dict[str, List[DataInstance]] = {}
        
        logger.info("Initialized DemonstrationManager")
    
    def get_demonstrations(
        self,
        task_name: str,
        dataset: TaskDataset,
        num_demonstrations: Optional[int] = None,
        force_reselect: bool = False
    ) -> List[DataInstance]:
        """
        Get demonstrations for a task.
        
        Args:
            task_name: Name of the task
            dataset: TaskDataset to select from
            num_demonstrations: Number to select (overrides config)
            force_reselect: Force reselection even if cached
            
        Returns:
            List of demonstration instances
        """
        cache_key = f"{task_name}_{num_demonstrations or self.config.num_demonstrations}"
        
        # Check cache
        if cache_key in self.demonstrations_cache and not force_reselect:
            logger.info(f"Using cached demonstrations for {task_name}")
            return self.demonstrations_cache[cache_key]
        
        # Select demonstrations
        demonstrations = self.selector.select_demonstrations(
            dataset=dataset,
            num_demonstrations=num_demonstrations
        )
        
        # Validate
        is_valid = self.selector.validate_demonstrations(demonstrations)
        if not is_valid:
            logger.error(f"Invalid demonstrations selected for {task_name}")
        
        # Cache
        self.demonstrations_cache[cache_key] = demonstrations
        
        logger.info(f"Selected and cached {len(demonstrations)} demonstrations for {task_name}")
        
        return demonstrations
    
    def get_demonstrations_for_all_tasks(
        self,
        datasets: Dict[str, TaskDataset],
        num_demonstrations_per_task: Optional[Dict[str, int]] = None
    ) -> Dict[str, List[DataInstance]]:
        """
        Get demonstrations for all tasks.
        
        Args:
            datasets: Dictionary of TaskDataset objects
            num_demonstrations_per_task: Optional dict specifying num demos per task
            
        Returns:
            Dictionary mapping task names to demonstration lists
        """
        all_demonstrations = {}
        
        for task_name, dataset in datasets.items():
            num_demos = None
            if num_demonstrations_per_task:
                num_demos = num_demonstrations_per_task.get(task_name)
            
            demonstrations = self.get_demonstrations(
                task_name=task_name,
                dataset=dataset,
                num_demonstrations=num_demos
            )
            
            all_demonstrations[task_name] = demonstrations
        
        return all_demonstrations
    
    def clear_cache(self) -> None:
        """Clear the demonstrations cache."""
        self.demonstrations_cache.clear()
        logger.info("Cleared demonstrations cache")
    
    def get_demonstration_statistics(
        self,
        demonstrations: List[DataInstance]
    ) -> Dict[str, Any]:
        """
        Get statistics about demonstrations.
        
        Args:
            demonstrations: List of demonstration instances
            
        Returns:
            Statistics dictionary
        """
        if not demonstrations:
            return {'count': 0}
        
        # Answer distribution
        answer_dist = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        for demo in demonstrations:
            if demo.answer in answer_dist:
                answer_dist[demo.answer] += 1
        
        # Length statistics
        question_lengths = [len(demo.question) for demo in demonstrations]
        
        # Context presence
        has_context = sum(1 for demo in demonstrations if demo.context)
        has_dialogue = sum(1 for demo in demonstrations if demo.dialogue)
        has_document = sum(1 for demo in demonstrations if demo.document)
        
        return {
            'count': len(demonstrations),
            'answer_distribution': answer_dist,
            'avg_question_length': sum(question_lengths) / len(question_lengths),
            'min_question_length': min(question_lengths),
            'max_question_length': max(question_lengths),
            'has_context_count': has_context,
            'has_dialogue_count': has_dialogue,
            'has_document_count': has_document
        }


# Example usage
if __name__ == "__main__":
    import logging
    from src.data.dataset_loader import DataInstance, TaskDataset
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*80)
    print("Demonstration Manager Test")
    print("="*80)
    
    # Create sample dataset
    instances = []
    for i in range(20):
        instance = DataInstance(
            id=f"sample_{i}",
            question=f"Sample question {i}?",
            options=[
                f"Option A{i}",
                f"Option B{i}",
                f"Option C{i}",
                f"Option D{i}",
                "I don't know",
                "None of the above"
            ],
            answer=random.choice(['A', 'B', 'C', 'D'])
        )
        instances.append(instance)
    
    dataset = TaskDataset(
        task_name="sample_task",
        instances=instances,
        task_type="qa"
    )
    
    # Test different selection strategies
    strategies = ['random', 'balanced', 'diverse', 'fixed']
    
    for strategy in strategies:
        print(f"\n{'-'*80}")
        print(f"Testing '{strategy}' strategy")
        print(f"{'-'*80}")
        
        config = DemonstrationConfig(
            num_demonstrations=5,
            selection_strategy=strategy,
            seed=42
        )
        
        manager = DemonstrationManager(config)
        
        demonstrations = manager.get_demonstrations(
            task_name="sample_task",
            dataset=dataset
        )
        
        print(f"\nSelected {len(demonstrations)} demonstrations")
        
        stats = manager.get_demonstration_statistics(demonstrations)
        print(f"Answer distribution: {stats['answer_distribution']}")
        print(f"Avg question length: {stats['avg_question_length']:.1f} chars")
    
    # Test caching
    print("\n" + "="*80)
    print("Testing caching...")
    print("="*80)
    
    manager = DemonstrationManager()
    
    # First call - should select
    demos1 = manager.get_demonstrations("test", dataset, num_demonstrations=3)
    
    # Second call - should use cache
    demos2 = manager.get_demonstrations("test", dataset, num_demonstrations=3)
    
    print(f"Same demonstrations returned: {demos1[0].id == demos2[0].id}")
    
    # Force reselect
    demos3 = manager.get_demonstrations("test", dataset, num_demonstrations=3, force_reselect=True)
    print(f"After force reselect, different: {demos1[0].id != demos3[0].id}")
    
    # Test chat formatting
    print("\n" + "="*80)
    print("Testing chat format conversion...")
    print("="*80)
    
    from src.prompting.prompt_templates import ChatPromptFormatter
    from src.prompting.prompt_builder import PromptBuilder
    
    # Build a base prompt
    builder = PromptBuilder(task_type='qa')
    base_prompt = builder.build_prompt(
        instance=instances[0],
        strategy='base',
        demonstrations=demonstrations[:2]
    )
    
    print("\nBase prompt:")
    print(base_prompt[:200] + "...")
    
    # Convert to chat formats
    formatter = ChatPromptFormatter()
    
    print("\n" + "-"*80)
    print("Llama-2-Chat format:")
    chat_prompt = formatter.format_for_llama2_chat(base_prompt)
    print(chat_prompt[:200] + "...")