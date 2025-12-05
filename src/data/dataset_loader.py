"""
Dataset Loader Module
Handles loading and initial processing of benchmark datasets.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import threading

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm

try:
    import datasets
    DATASETS_VERSION = datasets.__version__
except ImportError:
    DATASETS_VERSION = "unknown"

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class DataInstance:
    """Represents a single data instance across all tasks."""
    id: str
    question: str
    options: List[str]  # Originally 4 options [A, B, C, D]
    answer: str  # Correct answer letter (A, B, C, or D)
    context: Optional[str] = None  # For RC, CI tasks
    dialogue: Optional[str] = None  # For DRS task
    document: Optional[str] = None  # For DS task
    metadata: Optional[Dict] = None


@dataclass
class TaskDataset:
    """Container for a complete task dataset."""
    task_name: str
    instances: List[DataInstance]
    task_type: str  # 'qa', 'rc', 'ci', 'drs', 'ds'
    num_original_options: int = 4


class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders."""
    
    def __init__(self, cache_dir: Optional[str] = None, seed: int = 42):
        self.cache_dir = cache_dir or "./data/cache"
        self.seed = seed
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _load_dataset_with_timeout(self, dataset_name: str, config: Optional[str] = None, 
                                   timeout: int = 300, **kwargs):
        """
        Load dataset with timeout to prevent hanging on file locks.
        
        Args:
            dataset_name: Name of the dataset
            config: Optional dataset configuration
            timeout: Timeout in seconds (default: 5 minutes)
            **kwargs: Additional arguments for load_dataset
            
        Returns:
            Loaded dataset
            
        Raises:
            TimeoutError: If loading takes longer than timeout
            RuntimeError: If there are lock issues with helpful error message
        """
        result = [None]
        exception = [None]
        
        def load_worker():
            try:
                if config:
                    result[0] = load_dataset(dataset_name, config, cache_dir=self.cache_dir, **kwargs)
                else:
                    result[0] = load_dataset(dataset_name, cache_dir=self.cache_dir, **kwargs)
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=load_worker, daemon=True)
        thread.start()
        thread.join(timeout=timeout)
        
        if thread.is_alive():
            error_msg = (
                f"Dataset loading timed out after {timeout} seconds. "
                "This often happens due to file lock issues.\n\n"
                "SOLUTIONS:\n"
                "1. Wait for any other dataset downloads to complete\n"
                "2. Clear stale lock files:\n"
                f"   - Check for lock files in: {self.cache_dir}\n"
                "   - Delete any *.lock files that are older than a few minutes\n"
                "3. Clear the dataset cache and retry:\n"
                f"   - Delete or rename: {self.cache_dir}\n"
                "4. Check network connection (first download may be slow)\n"
                f"5. Try loading manually: from datasets import load_dataset; load_dataset('{dataset_name}'{', ' + repr(config) if config else ''})"
            )
            logger.error(error_msg)
            raise TimeoutError(error_msg)
        
        if exception[0]:
            error_str = str(exception[0])
            if "FileLock" in error_str or "lock" in error_str.lower():
                error_msg = (
                    f"Dataset loading failed due to file lock: {error_str}\n\n"
                    "SOLUTIONS:\n"
                    "1. Wait for any other dataset downloads to complete\n"
                    "2. Clear stale lock files:\n"
                    f"   - Check for lock files in: {self.cache_dir}\n"
                    "   - Delete any *.lock files\n"
                    "3. Clear the dataset cache:\n"
                    f"   - Delete or rename: {self.cache_dir}\n"
                    "4. Restart the script"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg) from exception[0]
            else:
                raise exception[0]
        
        return result[0]
        
    @abstractmethod
    def load(self, num_samples: int = 10000) -> TaskDataset:
        """Load and return the dataset."""
        pass
    
    @abstractmethod
    def get_task_name(self) -> str:
        """Return the task name."""
        pass
    
    def _sample_dataset(self, dataset: Union[Dataset, List], num_samples: int) -> List:
        """Sample num_samples from dataset with fixed seed."""
        if isinstance(dataset, Dataset):
            dataset = dataset.shuffle(seed=self.seed)
            if len(dataset) > num_samples:
                dataset = dataset.select(range(num_samples))
            return list(dataset)
        else:
            # For list-based datasets
            import random
            random.seed(self.seed)
            if len(dataset) > num_samples:
                return random.sample(dataset, num_samples)
            return dataset


class MMLULoader(BaseDatasetLoader):
    """Loader for MMLU (Question Answering) dataset."""
    
    def get_task_name(self) -> str:
        return "qa"
    
    def load(self, num_samples: int = 10000) -> TaskDataset:
        """
        Load MMLU dataset.
        Samples evenly across 4 categories: humanities, social_sciences, stem, other.
        """
        logger.info(f"Loading MMLU dataset with {num_samples} samples...")
        
        # Load MMLU dataset with timeout to prevent hanging
        try:
            dataset = self._load_dataset_with_timeout("cais/mmlu", "all", timeout=600)
        except (TimeoutError, RuntimeError) as e:
            logger.error(f"Failed to load MMLU dataset: {e}")
            raise
        
        # Categories to sample from
        categories = {
            "humanities": [],
            "social_sciences": [],
            "stem": [],
            "other": []
        }
        
        # Subject to category mapping (simplified - you may need to expand this)
        subject_to_category = self._get_subject_category_mapping()
        
        # Collect samples by category
        for split in ["test", "validation", "dev"]:
            if split in dataset:
                for item in dataset[split]:
                    subject = item.get("subject", "other")
                    category = subject_to_category.get(subject, "other")
                    categories[category].append(item)
        
        # Sample evenly from each category
        samples_per_category = num_samples // 4
        all_instances = []
        
        for category, items in categories.items():
            sampled = self._sample_dataset(items, samples_per_category)
            
            for idx, item in enumerate(sampled):
                instance = DataInstance(
                    id=f"mmlu_{category}_{idx}",
                    question=item["question"],
                    options=[
                        item["choices"][0],
                        item["choices"][1],
                        item["choices"][2],
                        item["choices"][3]
                    ],
                    answer=chr(65 + item["answer"]),  # Convert 0,1,2,3 to A,B,C,D
                    context=None,
                    metadata={"subject": item.get("subject"), "category": category}
                )
                all_instances.append(instance)
        
        logger.info(f"Loaded {len(all_instances)} MMLU instances")
        return TaskDataset(
            task_name="qa",
            instances=all_instances,
            task_type="qa",
            num_original_options=4
        )
    
    def _get_subject_category_mapping(self) -> Dict[str, str]:
        """Map MMLU subjects to 4 main categories."""
        return {
            # Humanities
            "formal_logic": "humanities",
            "high_school_european_history": "humanities",
            "high_school_us_history": "humanities",
            "high_school_world_history": "humanities",
            "international_law": "humanities",
            "jurisprudence": "humanities",
            "logical_fallacies": "humanities",
            "moral_disputes": "humanities",
            "moral_scenarios": "humanities",
            "philosophy": "humanities",
            "prehistory": "humanities",
            "professional_law": "humanities",
            "world_religions": "humanities",
            
            # Social Sciences
            "econometrics": "social_sciences",
            "high_school_geography": "social_sciences",
            "high_school_government_and_politics": "social_sciences",
            "high_school_macroeconomics": "social_sciences",
            "high_school_microeconomics": "social_sciences",
            "high_school_psychology": "social_sciences",
            "human_sexuality": "social_sciences",
            "professional_psychology": "social_sciences",
            "public_relations": "social_sciences",
            "security_studies": "social_sciences",
            "sociology": "social_sciences",
            "us_foreign_policy": "social_sciences",
            
            # STEM
            "abstract_algebra": "stem",
            "anatomy": "stem",
            "astronomy": "stem",
            "college_biology": "stem",
            "college_chemistry": "stem",
            "college_computer_science": "stem",
            "college_mathematics": "stem",
            "college_physics": "stem",
            "computer_security": "stem",
            "conceptual_physics": "stem",
            "electrical_engineering": "stem",
            "elementary_mathematics": "stem",
            "high_school_biology": "stem",
            "high_school_chemistry": "stem",
            "high_school_computer_science": "stem",
            "high_school_mathematics": "stem",
            "high_school_physics": "stem",
            "high_school_statistics": "stem",
            "machine_learning": "stem",
            
            # Other (business, health, misc)
            "business_ethics": "other",
            "clinical_knowledge": "other",
            "college_medicine": "other",
            "global_facts": "other",
            "human_aging": "other",
            "management": "other",
            "marketing": "other",
            "medical_genetics": "other",
            "miscellaneous": "other",
            "nutrition": "other",
            "professional_accounting": "other",
            "professional_medicine": "other",
            "virology": "other",
        }


class CosmosQALoader(BaseDatasetLoader):
    """Loader for CosmosQA (Reading Comprehension) dataset."""
    
    def get_task_name(self) -> str:
        return "rc"
    
    def load(self, num_samples: int = 10000) -> TaskDataset:
        """Load CosmosQA dataset."""
        logger.info(f"Loading CosmosQA dataset with {num_samples} samples...")
        
        # Check datasets library version and provide guidance
        try:
            version_parts = DATASETS_VERSION.split(".")
            major = int(version_parts[0]) if len(version_parts) > 0 else 0
            minor = int(version_parts[1]) if len(version_parts) > 1 else 0
            if major > 2 or (major == 2 and minor >= 15):
                logger.warning(
                    f"Detected datasets library version {DATASETS_VERSION} (>= 2.15.0). "
                    "CosmosQA uses deprecated dataset scripts that are not supported in this version."
                )
        except (ValueError, AttributeError, IndexError):
            logger.debug(f"Could not parse datasets version: {DATASETS_VERSION}")
        
        # Try loading the dataset with fallback for newer datasets versions
        dataset = None

        # First attempt: standard loading
        try:
            dataset = load_dataset("cosmos_qa", cache_dir=self.cache_dir)
            logger.info("CosmosQA loaded successfully with standard method")
        except (RuntimeError, ValueError) as e:
            error_str = str(e)
            if "Dataset scripts are no longer supported" in error_str or "trust_remote_code" in error_str:
                logger.warning(
                    "Standard loading failed due to deprecated dataset scripts. "
                    "Attempting fallback with trust_remote_code=True..."
                )
                # Fallback: try with trust_remote_code=True
                try:
                    dataset = load_dataset("cosmos_qa", cache_dir=self.cache_dir, trust_remote_code=True)
                    logger.info("CosmosQA loaded successfully with trust_remote_code=True")
                except Exception as fallback_e:
                    logger.error(
                        "=" * 80 + "\n"
                        "CosmosQA dataset cannot be loaded.\n"
                        "Both standard and fallback methods failed.\n"
                        "\n"
                        "SOLUTIONS:\n"
                        "1. Downgrade datasets library to a compatible version:\n"
                        "   pip install 'datasets>=2.10.0,<2.15.0'\n"
                        "\n"
                        "2. Skip the 'rc' task when running the benchmark:\n"
                        "   python run_benchmark.py --quick-test --models tinyllama-1.1b --tasks qa ci drs ds\n"
                        "\n"
                        "3. Or exclude 'rc' from the task list in your configuration.\n"
                        "=" * 80
                    )
                    raise RuntimeError(
                        f"Failed to load CosmosQA dataset: {fallback_e}\n\n"
                        "The dataset uses deprecated script format.\n"
                        "Quick fix: pip install 'datasets>=2.10.0,<2.15.0'\n"
                        "Or skip the 'rc' task: python run_benchmark.py --tasks qa ci drs ds"
                    ) from fallback_e
            else:
                raise
        except Exception as e:
            logger.error(f"Failed to load CosmosQA dataset: {e}")
            raise RuntimeError(
                f"Failed to load CosmosQA dataset: {e}\n"
                "If you see 'Dataset scripts are no longer supported', try: pip install 'datasets>=2.10.0,<2.15.0'"
            ) from e
        
        # Combine train and validation
        all_data = []
        for split in ["train", "validation"]:
            if split in dataset:
                all_data.extend(list(dataset[split]))
        
        if not all_data:
            raise RuntimeError(
                "CosmosQA dataset loaded but contains no data. "
                "Please check the dataset availability or skip the 'rc' task."
            )
        
        # Sample
        sampled_data = self._sample_dataset(all_data, num_samples)
        
        instances = []
        for idx, item in enumerate(sampled_data):
            instance = DataInstance(
                id=f"cosmosqa_{idx}",
                question=item["question"],
                options=[
                    item["answer0"],
                    item["answer1"],
                    item["answer2"],
                    item["answer3"]
                ],
                answer=chr(65 + item["label"]),  # Convert to A,B,C,D
                context=item["context"],
                metadata={"split": "train_val"}
            )
            instances.append(instance)
        
        logger.info(f"Loaded {len(instances)} CosmosQA instances")
        return TaskDataset(
            task_name="rc",
            instances=instances,
            task_type="rc",
            num_original_options=4
        )


class HellaSwagLoader(BaseDatasetLoader):
    """Loader for HellaSwag (Commonsense Inference) dataset."""
    
    def get_task_name(self) -> str:
        return "ci"
    
    def load(self, num_samples: int = 10000) -> TaskDataset:
        """Load HellaSwag dataset."""
        logger.info(f"Loading HellaSwag dataset with {num_samples} samples...")
        
        # Load dataset (train + validation since test labels not available)
        dataset = load_dataset("Rowan/hellaswag", cache_dir=self.cache_dir)
        
        # Combine train and validation
        all_data = []
        for split in ["train", "validation"]:
            if split in dataset:
                all_data.extend(list(dataset[split]))
        
        # Sample
        sampled_data = self._sample_dataset(all_data, num_samples)
        
        instances = []
        for idx, item in enumerate(sampled_data):
            instance = DataInstance(
                id=f"hellaswag_{idx}",
                question="What happens next?",  # Implicit question
                options=[
                    item["endings"][0],
                    item["endings"][1],
                    item["endings"][2],
                    item["endings"][3]
                ],
                answer=chr(65 + int(item["label"])),  # Convert to A,B,C,D
                context=item["ctx"],
                metadata={
                    "activity_label": item.get("activity_label"),
                    "ctx_a": item.get("ctx_a"),
                    "ctx_b": item.get("ctx_b")
                }
            )
            instances.append(instance)
        
        logger.info(f"Loaded {len(instances)} HellaSwag instances")
        return TaskDataset(
            task_name="ci",
            instances=instances,
            task_type="ci",
            num_original_options=4
        )


class HaluDialLoader(BaseDatasetLoader):
    """Loader for HaluDial (Dialogue Response Selection) dataset."""
    
    def get_task_name(self) -> str:
        return "drs"
    
    def load(self, num_samples: int = 10000) -> TaskDataset:
        """Load HaluDial dataset from HaluEval benchmark."""
        logger.info(f"Loading HaluDial dataset with {num_samples} samples...")
        
        try:
            # Try loading from HuggingFace
            dataset = load_dataset("pminervini/HaluEval", "dialogue", cache_dir=self.cache_dir)
            all_data = list(dataset["data"])
        except:
            # Fallback: load from local file or alternative source
            logger.warning("Could not load HaluDial from HuggingFace, using fallback method")
            all_data = self._load_haludial_fallback()
        
        # Sample exactly 10000 if available
        if len(all_data) >= num_samples:
            sampled_data = self._sample_dataset(all_data, num_samples)
        else:
            sampled_data = all_data
            logger.warning(f"Only {len(all_data)} samples available, requested {num_samples}")
        
        instances = []
        for idx, item in enumerate(sampled_data):
            # HaluDial format: dialogue history + correct response + incorrect response
            # We need to create 4 options (2 from data, 2 additional)
            instance = self._parse_haludial_item(item, idx)
            instances.append(instance)
        
        logger.info(f"Loaded {len(instances)} HaluDial instances")
        return TaskDataset(
            task_name="drs",
            instances=instances,
            task_type="drs",
            num_original_options=2  # Original has 2, we'll expand to 4
        )
    
    def _parse_haludial_item(self, item: Dict, idx: int) -> DataInstance:
        """Parse a HaluDial item into DataInstance format."""
        # Extract dialogue and responses
        dialogue = item.get("dialogue", item.get("knowledge", ""))
        correct_response = item.get("right_response", item.get("response", ""))
        hallucinated_response = item.get("hallucinated_response", "")
        
        # Create 4 options (2 real, 2 placeholder for now)
        # In the actual paper, they sample from other instances
        options = [
            correct_response,
            hallucinated_response,
            "",  # Will be filled by dataset_processor
            ""   # Will be filled by dataset_processor
        ]
        
        return DataInstance(
            id=f"haludial_{idx}",
            question="Which response is most appropriate for this dialogue?",
            options=options,
            answer="A",  # Correct response is first
            dialogue=dialogue,
            metadata={
                "has_hallucination": item.get("hallucination", True),
                "original_options": 2
            }
        )
    
    def _load_haludial_fallback(self) -> List[Dict]:
        """Fallback method to load HaluDial data."""
        # Implement fallback loading logic
        # This could load from local JSON files, etc.
        logger.warning("Fallback loading not fully implemented")
        return []


class HaluSumLoader(BaseDatasetLoader):
    """Loader for HaluSum (Document Summarization) dataset."""
    
    def get_task_name(self) -> str:
        return "ds"
    
    def load(self, num_samples: int = 10000) -> TaskDataset:
        """Load HaluSum dataset from HaluEval benchmark."""
        logger.info(f"Loading HaluSum dataset with {num_samples} samples...")
        
        try:
            # Try loading from HuggingFace
            dataset = load_dataset("pminervini/HaluEval", "summarization", cache_dir=self.cache_dir)
            all_data = list(dataset["data"])
        except:
            # Fallback
            logger.warning("Could not load HaluSum from HuggingFace, using fallback method")
            all_data = self._load_halusum_fallback()
        
        # Sample
        if len(all_data) >= num_samples:
            sampled_data = self._sample_dataset(all_data, num_samples)
        else:
            sampled_data = all_data
            logger.warning(f"Only {len(all_data)} samples available, requested {num_samples}")
        
        instances = []
        for idx, item in enumerate(sampled_data):
            instance = self._parse_halusum_item(item, idx)
            instances.append(instance)
        
        logger.info(f"Loaded {len(instances)} HaluSum instances")
        return TaskDataset(
            task_name="ds",
            instances=instances,
            task_type="ds",
            num_original_options=2  # Original has 2, we'll expand to 4
        )
    
    def _parse_halusum_item(self, item: Dict, idx: int) -> DataInstance:
        """Parse a HaluSum item into DataInstance format."""
        document = item.get("document", "")
        correct_summary = item.get("right_summary", item.get("summary", ""))
        hallucinated_summary = item.get("hallucinated_summary", "")
        
        options = [
            correct_summary,
            hallucinated_summary,
            "",  # Will be filled by dataset_processor
            ""   # Will be filled by dataset_processor
        ]
        
        return DataInstance(
            id=f"halusum_{idx}",
            question="Which summary best represents the document?",
            options=options,
            answer="A",  # Correct summary is first
            document=document,
            metadata={
                "has_hallucination": item.get("hallucination", True),
                "original_options": 2
            }
        )
    
    def _load_halusum_fallback(self) -> List[Dict]:
        """Fallback method to load HaluSum data."""
        logger.warning("Fallback loading not fully implemented")
        return []


class DatasetLoaderFactory:
    """Factory class to create appropriate dataset loaders."""
    
    _loaders = {
        "qa": MMLULoader,
        "mmlu": MMLULoader,
        "rc": CosmosQALoader,
        "cosmosqa": CosmosQALoader,
        "ci": HellaSwagLoader,
        "hellaswag": HellaSwagLoader,
        "drs": HaluDialLoader,
        "haludial": HaluDialLoader,
        "ds": HaluSumLoader,
        "halusum": HaluSumLoader,
    }
    
    @classmethod
    def create_loader(
        cls,
        task_name: str,
        cache_dir: Optional[str] = None,
        seed: int = 42
    ) -> BaseDatasetLoader:
        """Create and return appropriate dataset loader."""
        task_name_lower = task_name.lower()
        
        if task_name_lower not in cls._loaders:
            raise ValueError(
                f"Unknown task: {task_name}. "
                f"Available tasks: {list(cls._loaders.keys())}"
            )
        
        loader_class = cls._loaders[task_name_lower]
        return loader_class(cache_dir=cache_dir, seed=seed)
    
    @classmethod
    def get_available_tasks(cls) -> List[str]:
        """Return list of available task names."""
        return list(set(cls._loaders.keys()))


def load_all_datasets(
    tasks: List[str],
    num_samples: int = 10000,
    cache_dir: Optional[str] = None,
    seed: int = 42
) -> Dict[str, TaskDataset]:
    """
    Load multiple datasets.
    
    Args:
        tasks: List of task names to load
        num_samples: Number of samples per task
        cache_dir: Directory to cache datasets
        seed: Random seed for sampling
        
    Returns:
        Dictionary mapping task names to TaskDataset objects
    """
    datasets = {}
    
    for task in tqdm(tasks, desc="Loading datasets"):
        try:
            loader = DatasetLoaderFactory.create_loader(
                task_name=task,
                cache_dir=cache_dir,
                seed=seed
            )
            dataset = loader.load(num_samples=num_samples)
            datasets[dataset.task_name] = dataset
            logger.info(f"Successfully loaded {task} dataset")
        except Exception as e:
            logger.error(f"Failed to load {task} dataset: {e}")
            raise
    
    return datasets


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load all datasets
    tasks = ["qa", "rc", "ci", "drs", "ds"]
    datasets = load_all_datasets(
        tasks=tasks,
        num_samples=10000,
        cache_dir="./data/cache",
        seed=42
    )
    
    # Print summary
    for task_name, dataset in datasets.items():
        print(f"\n{task_name.upper()} Dataset:")
        print(f"  Total instances: {len(dataset.instances)}")
        print(f"  Task type: {dataset.task_type}")
        print(f"  Sample instance: {dataset.instances[0].question[:100]}...")