"""
Dataset Configuration Module
Centralized configuration for all dataset-related settings and metadata.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import json

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class TaskConfig:
    """Configuration for a single task."""
    task_name: str
    task_type: str  # 'qa', 'rc', 'ci', 'drs', 'ds'
    dataset_name: str
    num_samples: int = 10000
    num_demonstrations: int = 5
    max_input_length: int = 2048
    has_context: bool = False
    context_key: Optional[str] = None  # 'context', 'dialogue', 'document'
    description: str = ""
    
    # Dataset-specific settings
    dataset_path: Optional[str] = None
    dataset_config: Optional[str] = None
    splits_to_use: List[str] = field(default_factory=lambda: ["train", "validation"])
    
    # Sampling settings
    sample_strategy: str = "random"  # 'random', 'stratified', 'balanced'
    category_key: Optional[str] = None  # For stratified sampling
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'task_name': self.task_name,
            'task_type': self.task_type,
            'dataset_name': self.dataset_name,
            'num_samples': self.num_samples,
            'num_demonstrations': self.num_demonstrations,
            'max_input_length': self.max_input_length,
            'has_context': self.has_context,
            'context_key': self.context_key,
            'description': self.description,
            'dataset_path': self.dataset_path,
            'dataset_config': self.dataset_config,
            'splits_to_use': self.splits_to_use,
            'sample_strategy': self.sample_strategy,
            'category_key': self.category_key
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class DatasetPipelineConfig:
    """Configuration for the entire data pipeline."""
    # General settings
    seed: int = 42
    cache_dir: str = "./data/cache"
    output_dir: str = "./data/processed"
    
    # Task configurations
    tasks: Dict[str, TaskConfig] = field(default_factory=dict)
    
    # Splitting settings
    calibration_ratio: float = 0.5
    stratify_split: bool = True
    
    # Processing settings
    add_idk_option: bool = True  # Add "I don't know"
    add_nota_option: bool = True  # Add "None of the above"
    idk_text: str = "I don't know"
    nota_text: str = "None of the above"
    
    # Validation settings
    validate_after_processing: bool = True
    validate_after_splitting: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'seed': self.seed,
            'cache_dir': self.cache_dir,
            'output_dir': self.output_dir,
            'tasks': {name: config.to_dict() for name, config in self.tasks.items()},
            'calibration_ratio': self.calibration_ratio,
            'stratify_split': self.stratify_split,
            'add_idk_option': self.add_idk_option,
            'add_nota_option': self.add_nota_option,
            'idk_text': self.idk_text,
            'nota_text': self.nota_text,
            'validate_after_processing': self.validate_after_processing,
            'validate_after_splitting': self.validate_after_splitting
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetPipelineConfig':
        """Create from dictionary."""
        tasks = {}
        if 'tasks' in data:
            tasks = {
                name: TaskConfig.from_dict(config)
                for name, config in data['tasks'].items()
            }
        
        config_copy = data.copy()
        config_copy['tasks'] = tasks
        return cls(**config_copy)
    
    def save_to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved dataset pipeline config to {path}")
    
    def save_to_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Saved dataset pipeline config to {path}")
    
    @classmethod
    def load_from_yaml(cls, path: str) -> 'DatasetPipelineConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        logger.info(f"Loaded dataset pipeline config from {path}")
        return cls.from_dict(data)
    
    @classmethod
    def load_from_json(cls, path: str) -> 'DatasetPipelineConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded dataset pipeline config from {path}")
        return cls.from_dict(data)


class DefaultTaskConfigs:
    """Default configurations for all benchmark tasks."""
    
    @staticmethod
    def get_qa_config() -> TaskConfig:
        """Get configuration for Question Answering (MMLU) task."""
        return TaskConfig(
            task_name="qa",
            task_type="qa",
            dataset_name="mmlu",
            num_samples=10000,
            num_demonstrations=5,
            max_input_length=2048,
            has_context=False,
            context_key=None,
            description="Question answering task testing world knowledge across 57 subjects in 4 categories.",
            dataset_path="cais/mmlu",
            dataset_config="all",
            splits_to_use=["test", "validation", "dev"],
            sample_strategy="stratified",
            category_key="subject"
        )
    
    @staticmethod
    def get_rc_config() -> TaskConfig:
        """Get configuration for Reading Comprehension (CosmosQA) task."""
        return TaskConfig(
            task_name="rc",
            task_type="rc",
            dataset_name="cosmosqa",
            num_samples=10000,
            num_demonstrations=5,
            max_input_length=2048,
            has_context=True,
            context_key="context",
            description="Reading comprehension task requiring understanding of everyday narratives and reasoning beyond exact text spans.",
            dataset_path="cosmos_qa",
            dataset_config=None,
            splits_to_use=["train", "validation"],
            sample_strategy="random"
        )
    
    @staticmethod
    def get_ci_config() -> TaskConfig:
        """Get configuration for Commonsense Inference (HellaSwag) task."""
        return TaskConfig(
            task_name="ci",
            task_type="ci",
            dataset_name="hellaswag",
            num_samples=10000,
            num_demonstrations=5,
            max_input_length=2048,
            has_context=True,
            context_key="context",
            description="Commonsense natural language inference task selecting the most likely followup to an event description.",
            dataset_path="Rowan/hellaswag",
            dataset_config=None,
            splits_to_use=["train", "validation"],
            sample_strategy="random"
        )
    
    @staticmethod
    def get_drs_config() -> TaskConfig:
        """Get configuration for Dialogue Response Selection (HaluDial) task."""
        return TaskConfig(
            task_name="drs",
            task_type="drs",
            dataset_name="haludial",
            num_samples=10000,
            num_demonstrations=3,
            max_input_length=2048,
            has_context=True,
            context_key="dialogue",
            description="Dialogue response selection task choosing appropriate responses without hallucination or non-factual information.",
            dataset_path="pminervini/HaluEval",
            dataset_config="dialogue",
            splits_to_use=["data"],
            sample_strategy="random"
        )
    
    @staticmethod
    def get_ds_config() -> TaskConfig:
        """Get configuration for Document Summarization (HaluSum) task."""
        return TaskConfig(
            task_name="ds",
            task_type="ds",
            dataset_name="halusum",
            num_samples=10000,
            num_demonstrations=1,  # Only 1 due to length constraints
            max_input_length=2048,
            has_context=True,
            context_key="document",
            description="Document summarization task selecting accurate summaries without hallucination or non-factual information.",
            dataset_path="pminervini/HaluEval",
            dataset_config="summarization",
            splits_to_use=["data"],
            sample_strategy="random"
        )
    
    @classmethod
    def get_all_configs(cls) -> Dict[str, TaskConfig]:
        """Get all default task configurations."""
        return {
            "qa": cls.get_qa_config(),
            "rc": cls.get_rc_config(),
            "ci": cls.get_ci_config(),
            "drs": cls.get_drs_config(),
            "ds": cls.get_ds_config()
        }


class DatasetConfigManager:
    """Manager for dataset configurations."""
    
    def __init__(self, config: Optional[DatasetPipelineConfig] = None):
        """
        Initialize the config manager.
        
        Args:
            config: DatasetPipelineConfig to use. If None, uses defaults.
        """
        if config is None:
            self.config = self.create_default_config()
        else:
            self.config = config
    
    @staticmethod
    def create_default_config() -> DatasetPipelineConfig:
        """Create default pipeline configuration."""
        tasks = DefaultTaskConfigs.get_all_configs()
        
        return DatasetPipelineConfig(
            seed=42,
            cache_dir="./data/cache",
            output_dir="./data/processed",
            tasks=tasks,
            calibration_ratio=0.5,
            stratify_split=True,
            add_idk_option=True,
            add_nota_option=True,
            idk_text="I don't know",
            nota_text="None of the above",
            validate_after_processing=True,
            validate_after_splitting=True
        )
    
    def get_task_config(self, task_name: str) -> TaskConfig:
        """Get configuration for a specific task."""
        if task_name not in self.config.tasks:
            raise ValueError(
                f"Task '{task_name}' not found. "
                f"Available tasks: {list(self.config.tasks.keys())}"
            )
        return self.config.tasks[task_name]
    
    def add_task_config(self, task_config: TaskConfig) -> None:
        """Add or update a task configuration."""
        self.config.tasks[task_config.task_name] = task_config
        logger.info(f"Added/updated config for task: {task_config.task_name}")
    
    def remove_task_config(self, task_name: str) -> None:
        """Remove a task configuration."""
        if task_name in self.config.tasks:
            del self.config.tasks[task_name]
            logger.info(f"Removed config for task: {task_name}")
        else:
            logger.warning(f"Task '{task_name}' not found in configuration")
    
    def get_tasks_by_type(self, task_type: str) -> List[TaskConfig]:
        """Get all task configurations of a specific type."""
        return [
            config for config in self.config.tasks.values()
            if config.task_type == task_type
        ]
    
    def get_enabled_tasks(self) -> List[str]:
        """Get list of enabled task names."""
        return list(self.config.tasks.keys())
    
    def validate_config(self) -> bool:
        """Validate the configuration."""
        is_valid = True
        errors = []
        
        # Check calibration ratio
        if not 0 < self.config.calibration_ratio < 1:
            errors.append(
                f"calibration_ratio must be between 0 and 1, "
                f"got {self.config.calibration_ratio}"
            )
            is_valid = False
        
        # Check each task config
        for task_name, task_config in self.config.tasks.items():
            # Check num_samples
            if task_config.num_samples <= 0:
                errors.append(
                    f"Task '{task_name}': num_samples must be positive, "
                    f"got {task_config.num_samples}"
                )
                is_valid = False
            
            # Check num_demonstrations
            if task_config.num_demonstrations < 0:
                errors.append(
                    f"Task '{task_name}': num_demonstrations must be non-negative, "
                    f"got {task_config.num_demonstrations}"
                )
                is_valid = False
            
            # Check max_input_length
            if task_config.max_input_length <= 0:
                errors.append(
                    f"Task '{task_name}': max_input_length must be positive, "
                    f"got {task_config.max_input_length}"
                )
                is_valid = False
            
            # Check context settings
            if task_config.has_context and not task_config.context_key:
                errors.append(
                    f"Task '{task_name}': has_context is True but context_key is None"
                )
                is_valid = False
        
        if not is_valid:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
        else:
            logger.info("Configuration validation passed")
        
        return is_valid
    
    def print_summary(self) -> None:
        """Print a summary of the configuration."""
        print("\n" + "="*80)
        print("DATASET PIPELINE CONFIGURATION SUMMARY")
        print("="*80)
        
        print(f"\nGeneral Settings:")
        print(f"  Seed: {self.config.seed}")
        print(f"  Cache directory: {self.config.cache_dir}")
        print(f"  Output directory: {self.config.output_dir}")
        print(f"  Calibration ratio: {self.config.calibration_ratio:.0%}")
        print(f"  Stratify split: {self.config.stratify_split}")
        
        print(f"\nTasks: {len(self.config.tasks)}")
        for task_name, task_config in self.config.tasks.items():
            print(f"\n  [{task_name.upper()}] {task_config.dataset_name}")
            print(f"    Type: {task_config.task_type}")
            print(f"    Samples: {task_config.num_samples:,}")
            print(f"    Demonstrations: {task_config.num_demonstrations}")
            print(f"    Max length: {task_config.max_input_length}")
            print(f"    Has context: {task_config.has_context}")
            if task_config.has_context:
                print(f"    Context key: {task_config.context_key}")
            print(f"    Description: {task_config.description[:80]}...")
        
        print("\n" + "="*80 + "\n")
    
    def save(self, path: str, format: str = 'yaml') -> None:
        """Save configuration to file."""
        if format == 'yaml':
            self.config.save_to_yaml(path)
        elif format == 'json':
            self.config.save_to_json(path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def load(cls, path: str, format: str = 'yaml') -> 'DatasetConfigManager':
        """Load configuration from file."""
        if format == 'yaml':
            config = DatasetPipelineConfig.load_from_yaml(path)
        elif format == 'json':
            config = DatasetPipelineConfig.load_from_json(path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return cls(config)


def create_experiment_configs() -> Dict[str, DatasetPipelineConfig]:
    """
    Create configurations for different experimental setups.
    
    Returns:
        Dictionary of experiment names to configs
    """
    experiments = {}
    
    # Standard configuration
    experiments['standard'] = DatasetConfigManager.create_default_config()
    
    # Quick test configuration (smaller samples)
    quick_config = DatasetConfigManager.create_default_config()
    for task_config in quick_config.tasks.values():
        task_config.num_samples = 1000
        task_config.num_demonstrations = 2
    experiments['quick_test'] = quick_config
    
    # No stratification configuration
    no_strat_config = DatasetConfigManager.create_default_config()
    no_strat_config.stratify_split = False
    experiments['no_stratification'] = no_strat_config
    
    # Different calibration ratios
    for ratio in [0.1, 0.2, 0.3, 0.4]:
        cal_config = DatasetConfigManager.create_default_config()
        cal_config.calibration_ratio = ratio
        experiments[f'cal_ratio_{int(ratio*100)}'] = cal_config
    
    return experiments


# Example usage
if __name__ == "__main__":
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create default config manager
    manager = DatasetConfigManager()
    
    # Print summary
    manager.print_summary()
    
    # Validate configuration
    is_valid = manager.validate_config()
    print(f"\nConfiguration valid: {is_valid}")
    
    # Save configuration
    output_dir = Path("./configs")
    output_dir.mkdir(exist_ok=True)
    
    manager.save(output_dir / "dataset_config.yaml", format='yaml')
    manager.save(output_dir / "dataset_config.json", format='json')
    
    # Test loading
    loaded_manager = DatasetConfigManager.load(
        output_dir / "dataset_config.yaml",
        format='yaml'
    )
    print("\nSuccessfully loaded config from file")
    
    # Get specific task config
    qa_config = manager.get_task_config("qa")
    print(f"\nQA Task Configuration:")
    print(f"  Dataset: {qa_config.dataset_name}")
    print(f"  Samples: {qa_config.num_samples:,}")
    print(f"  Demonstrations: {qa_config.num_demonstrations}")
    
    # Create experimental configs
    print("\n" + "="*80)
    print("EXPERIMENTAL CONFIGURATIONS")
    print("="*80)
    
    exp_configs = create_experiment_configs()
    for exp_name, exp_config in exp_configs.items():
        print(f"\n{exp_name}:")
        print(f"  Tasks: {len(exp_config.tasks)}")
        print(f"  Calibration ratio: {exp_config.calibration_ratio:.0%}")
        print(f"  Stratify: {exp_config.stratify_split}")
        
        # Save experimental configs
        exp_path = output_dir / f"experiment_{exp_name}.yaml"
        exp_config.save_to_yaml(str(exp_path))
    
    print(f"\nSaved {len(exp_configs)} experimental configurations to {output_dir}")