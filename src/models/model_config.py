"""
Model Configuration Module
Centralized configuration for model-related settings and registry.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import json

from src.models.model_loader import ModelLoadConfig
from src.models.inference_engine import InferenceConfig
from src.models.probability_extractor import ProbabilityExtractionConfig

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class ModelPipelineConfig:
    """Complete configuration for a model's inference pipeline."""
    # Model identification
    name: str
    model_id: str
    
    # Loading configuration
    load_config: ModelLoadConfig
    
    # Inference configuration
    inference_config: InferenceConfig
    
    # Probability extraction configuration
    probability_config: ProbabilityExtractionConfig
    
    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    paper_reference: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'model_id': self.model_id,
            'load_config': self.load_config.to_dict(),
            'inference_config': self.inference_config.to_dict(),
            'probability_config': {
                'temperature': self.probability_config.temperature,
                'calibration_method': self.probability_config.calibration_method,
                'normalize': self.probability_config.normalize,
                'min_prob': self.probability_config.min_prob,
                'max_prob': self.probability_config.max_prob
            },
            'description': self.description,
            'tags': self.tags,
            'paper_reference': self.paper_reference
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelPipelineConfig':
        """Create from dictionary."""
        load_config = ModelLoadConfig(**data['load_config'])
        
        inference_config_data = data['inference_config']
        inference_config = InferenceConfig(**inference_config_data)
        
        probability_config = ProbabilityExtractionConfig(
            **data['probability_config']
        )
        
        return cls(
            name=data['name'],
            model_id=data['model_id'],
            load_config=load_config,
            inference_config=inference_config,
            probability_config=probability_config,
            description=data.get('description', ''),
            tags=data.get('tags', []),
            paper_reference=data.get('paper_reference')
        )


@dataclass
class ModelBenchmarkConfig:
    """Configuration for benchmarking multiple models."""
    # Models to benchmark
    models: Dict[str, ModelPipelineConfig] = field(default_factory=dict)
    
    # General settings
    cache_dir: str = "./models/cache"
    output_dir: str = "./results/models"
    
    # Execution settings
    run_sequentially: bool = True  # Run one model at a time
    save_intermediate: bool = True
    
    # Comparison settings
    compare_base_vs_instruct: bool = True
    compare_different_sizes: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'models': {
                name: config.to_dict() 
                for name, config in self.models.items()
            },
            'cache_dir': self.cache_dir,
            'output_dir': self.output_dir,
            'run_sequentially': self.run_sequentially,
            'save_intermediate': self.save_intermediate,
            'compare_base_vs_instruct': self.compare_base_vs_instruct,
            'compare_different_sizes': self.compare_different_sizes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelBenchmarkConfig':
        """Create from dictionary."""
        models = {}
        if 'models' in data:
            models = {
                name: ModelPipelineConfig.from_dict(config)
                for name, config in data['models'].items()
            }
        
        return cls(
            models=models,
            cache_dir=data.get('cache_dir', './models/cache'),
            output_dir=data.get('output_dir', './results/models'),
            run_sequentially=data.get('run_sequentially', True),
            save_intermediate=data.get('save_intermediate', True),
            compare_base_vs_instruct=data.get('compare_base_vs_instruct', True),
            compare_different_sizes=data.get('compare_different_sizes', True)
        )
    
    def save_to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved model benchmark config to {path}")
    
    def save_to_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Saved model benchmark config to {path}")
    
    @classmethod
    def load_from_yaml(cls, path: str) -> 'ModelBenchmarkConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        logger.info(f"Loaded model benchmark config from {path}")
        return cls.from_dict(data)
    
    @classmethod
    def load_from_json(cls, path: str) -> 'ModelBenchmarkConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded model benchmark config from {path}")
        return cls.from_dict(data)


class DefaultModelConfigs:
    """Default configurations for popular SLMs."""
    
    @staticmethod
    def create_base_configs() -> Dict[str, ModelLoadConfig]:
        """Create base loading configurations for models."""
        return {
            # Phi models (Microsoft)
            "phi-2": ModelLoadConfig(
                model_id="microsoft/phi-2",
                name="phi-2",
                dtype="float16",
                device="auto"
            ),
            "phi-1.5": ModelLoadConfig(
                model_id="microsoft/phi-1_5",
                name="phi-1.5",
                dtype="float16",
                device="auto"
            ),
            
            # StableLM models
            "stablelm-2-1.6b": ModelLoadConfig(
                model_id="stabilityai/stablelm-2-1_6b",
                name="stablelm-2-1.6b",
                dtype="float16",
                device="auto"
            ),
            "stablelm-2-zephyr-1.6b": ModelLoadConfig(
                model_id="stabilityai/stablelm-2-zephyr-1_6b",
                name="stablelm-2-zephyr-1.6b",
                dtype="float16",
                device="auto"
            ),
            
            # TinyLlama
            "tinyllama-1.1b": ModelLoadConfig(
                model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                name="tinyllama-1.1b",
                dtype="float16",
                device="auto"
            ),
            
            # Qwen models
            "qwen-1.8b": ModelLoadConfig(
                model_id="Qwen/Qwen-1_8B",
                name="qwen-1.8b",
                dtype="float16",
                device="auto"
            ),
            "qwen-1.8b-chat": ModelLoadConfig(
                model_id="Qwen/Qwen-1_8B-Chat",
                name="qwen-1.8b-chat",
                dtype="float16",
                device="auto"
            ),
            
            # Gemma models
            "gemma-2b": ModelLoadConfig(
                model_id="google/gemma-2b",
                name="gemma-2b",
                dtype="bfloat16",
                device="auto"
            ),
            "gemma-2b-it": ModelLoadConfig(
                model_id="google/gemma-2b-it",
                name="gemma-2b-it",
                dtype="bfloat16",
                device="auto"
            ),
            
            # SmolLM models
            "smollm-135m": ModelLoadConfig(
                model_id="HuggingFaceTB/SmolLM-135M",
                name="smollm-135m",
                dtype="float32",
                device="auto"
            ),
            "smollm-360m": ModelLoadConfig(
                model_id="HuggingFaceTB/SmolLM-360M",
                name="smollm-360m",
                dtype="float32",
                device="auto"
            ),
            "smollm-1.7b": ModelLoadConfig(
                model_id="HuggingFaceTB/SmolLM-1.7B",
                name="smollm-1.7b",
                dtype="float16",
                device="auto"
            ),
            
            # OpenELM models
            "openelm-270m": ModelLoadConfig(
                model_id="apple/OpenELM-270M",
                name="openelm-270m",
                dtype="float16",
                device="auto"
            ),
            "openelm-450m": ModelLoadConfig(
                model_id="apple/OpenELM-450M",
                name="openelm-450m",
                dtype="float16",
                device="auto"
            ),
            "openelm-1.1b": ModelLoadConfig(
                model_id="apple/OpenELM-1_1B",
                name="openelm-1.1b",
                dtype="float16",
                device="auto"
            ),
            
            # H2O-Danube models
            "h2o-danube-1.8b": ModelLoadConfig(
                model_id="h2oai/h2o-danube-1.8b-base",
                name="h2o-danube-1.8b",
                dtype="float16",
                device="auto"
            ),
            "h2o-danube-1.8b-chat": ModelLoadConfig(
                model_id="h2oai/h2o-danube-1.8b-chat",
                name="h2o-danube-1.8b-chat",
                dtype="float16",
                device="auto"
            ),
        }
    
    @staticmethod
    def create_pipeline_config(
        model_name: str,
        batch_size: int = 1,
        temperature: float = 1.0
    ) -> ModelPipelineConfig:
        """
        Create a complete pipeline configuration for a model.
        
        Args:
            model_name: Name of the model
            batch_size: Batch size for inference
            temperature: Temperature for probability extraction
            
        Returns:
            ModelPipelineConfig
        """
        base_configs = DefaultModelConfigs.create_base_configs()
        
        if model_name not in base_configs:
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available models: {list(base_configs.keys())}"
            )
        
        load_config = base_configs[model_name]
        
        inference_config = InferenceConfig(
            batch_size=batch_size,
            max_length=2048,
            temperature=temperature,
            use_fp16=True,
            use_cache=True
        )
        
        probability_config = ProbabilityExtractionConfig(
            temperature=temperature,
            normalize=True
        )
        
        # Add metadata
        description = f"Small Language Model: {model_name}"
        tags = ["slm"]
        
        # Categorize by model family
        if "phi" in model_name:
            tags.extend(["microsoft", "phi"])
        elif "stablelm" in model_name:
            tags.extend(["stability-ai", "stablelm"])
        elif "tinyllama" in model_name:
            tags.extend(["tinyllama"])
        elif "qwen" in model_name:
            tags.extend(["alibaba", "qwen"])
        elif "gemma" in model_name:
            tags.extend(["google", "gemma"])
        elif "smollm" in model_name:
            tags.extend(["huggingface", "smollm"])
        elif "openelm" in model_name:
            tags.extend(["apple", "openelm"])
        elif "danube" in model_name:
            tags.extend(["h2o", "danube"])
        
        # Tag instruct-tuned models
        if any(keyword in model_name for keyword in ["chat", "instruct", "it", "zephyr"]):
            tags.append("instruct-tuned")
        else:
            tags.append("base")
        
        return ModelPipelineConfig(
            name=model_name,
            model_id=load_config.model_id,
            load_config=load_config,
            inference_config=inference_config,
            probability_config=probability_config,
            description=description,
            tags=tags
        )
    
    @staticmethod
    def get_all_pipeline_configs(
        batch_size: int = 1,
        temperature: float = 1.0
    ) -> Dict[str, ModelPipelineConfig]:
        """Get pipeline configs for all available models."""
        base_configs = DefaultModelConfigs.create_base_configs()
        
        pipeline_configs = {}
        for model_name in base_configs.keys():
            pipeline_configs[model_name] = DefaultModelConfigs.create_pipeline_config(
                model_name=model_name,
                batch_size=batch_size,
                temperature=temperature
            )
        
        return pipeline_configs


class ModelConfigManager:
    """Manager for model configurations."""
    
    def __init__(self, config: Optional[ModelBenchmarkConfig] = None):
        """
        Initialize the config manager.
        
        Args:
            config: ModelBenchmarkConfig to use. If None, uses defaults.
        """
        if config is None:
            self.config = self.create_default_config()
        else:
            self.config = config
    
    @staticmethod
    def create_default_config() -> ModelBenchmarkConfig:
        """Create default benchmark configuration."""
        models = DefaultModelConfigs.get_all_pipeline_configs()
        
        return ModelBenchmarkConfig(
            models=models,
            cache_dir="./models/cache",
            output_dir="./results/models",
            run_sequentially=True,
            save_intermediate=True,
            compare_base_vs_instruct=True,
            compare_different_sizes=True
        )
    
    def get_model_config(self, model_name: str) -> ModelPipelineConfig:
        """Get configuration for a specific model."""
        if model_name not in self.config.models:
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available models: {list(self.config.models.keys())}"
            )
        return self.config.models[model_name]
    
    def add_model_config(self, config: ModelPipelineConfig) -> None:
        """Add or update a model configuration."""
        self.config.models[config.name] = config
        logger.info(f"Added/updated config for model: {config.name}")
    
    def remove_model_config(self, model_name: str) -> None:
        """Remove a model configuration."""
        if model_name in self.config.models:
            del self.config.models[model_name]
            logger.info(f"Removed config for model: {model_name}")
        else:
            logger.warning(f"Model '{model_name}' not found in configuration")
    
    def get_models_by_tag(self, tag: str) -> List[ModelPipelineConfig]:
        """Get all models with a specific tag."""
        return [
            config for config in self.config.models.values()
            if tag in config.tags
        ]
    
    def get_model_families(self) -> Dict[str, List[str]]:
        """Get models grouped by family."""
        families = {}
        
        for name, config in self.config.models.items():
            # Determine family from tags
            family = "other"
            for tag in config.tags:
                if tag in ["phi", "stablelm", "tinyllama", "qwen", "gemma", "smollm", "openelm", "danube"]:
                    family = tag
                    break
            
            if family not in families:
                families[family] = []
            families[family].append(name)
        
        return families
    
    def get_base_instruct_pairs(self) -> List[Tuple[str, str]]:
        """Get pairs of base and instruct-tuned models."""
        base_models = self.get_models_by_tag("base")
        instruct_models = self.get_models_by_tag("instruct-tuned")
        
        pairs = []
        for base_config in base_models:
            base_name = base_config.name
            # Look for corresponding instruct model
            for instruct_config in instruct_models:
                instruct_name = instruct_config.name
                # Simple heuristic: same model family
                base_family = base_name.split('-')[0]
                instruct_family = instruct_name.split('-')[0]
                
                if base_family == instruct_family:
                    pairs.append((base_name, instruct_name))
        
        return pairs
    
    def list_available_models(self) -> List[str]:
        """List all available model names."""
        return list(self.config.models.keys())
    
    def print_summary(self) -> None:
        """Print summary of the configuration."""
        print("\n" + "="*80)
        print("MODEL BENCHMARK CONFIGURATION SUMMARY")
        print("="*80)
        
        print(f"\nGeneral Settings:")
        print(f"  Cache directory: {self.config.cache_dir}")
        print(f"  Output directory: {self.config.output_dir}")
        print(f"  Run sequentially: {self.config.run_sequentially}")
        
        print(f"\nModels: {len(self.config.models)}")
        
        # Group by family
        families = self.get_model_families()
        for family, models in families.items():
            print(f"\n  {family.upper()} ({len(models)} models):")
            for model_name in models:
                config = self.config.models[model_name]
                print(f"    - {model_name}")
                print(f"        Model ID: {config.model_id}")
                print(f"        Tags: {', '.join(config.tags)}")
        
        # Base vs Instruct pairs
        pairs = self.get_base_instruct_pairs()
        if pairs:
            print(f"\n  Base-Instruct Pairs: {len(pairs)}")
            for base, instruct in pairs[:5]:  # Show first 5
                print(f"    - {base} <-> {instruct}")
            if len(pairs) > 5:
                print(f"    ... and {len(pairs) - 5} more")
        
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
    def load(cls, path: str, format: str = 'yaml') -> 'ModelConfigManager':
        """Load configuration from file."""
        if format == 'yaml':
            config = ModelBenchmarkConfig.load_from_yaml(path)
        elif format == 'json':
            config = ModelBenchmarkConfig.load_from_json(path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return cls(config)


# Example usage
if __name__ == "__main__":
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create default config manager
    manager = ModelConfigManager()
    
    # Print summary
    manager.print_summary()
    
    # Get specific model config
    phi2_config = manager.get_model_config("phi-2")
    print(f"\nPhi-2 Configuration:")
    print(f"  Model ID: {phi2_config.model_id}")
    print(f"  Tags: {phi2_config.tags}")
    print(f"  Batch size: {phi2_config.inference_config.batch_size}")
    print(f"  Temperature: {phi2_config.probability_config.temperature}")
    
    # Get models by tag
    base_models = manager.get_models_by_tag("base")
    print(f"\nBase models ({len(base_models)}):")
    for config in base_models[:5]:
        print(f"  - {config.name}")
    
    # Get model families
    families = manager.get_model_families()
    print(f"\nModel families:")
    for family, models in families.items():
        print(f"  {family}: {len(models)} models")
    
    # Save configuration
    output_dir = Path("./configs")
    output_dir.mkdir(exist_ok=True)
    
    manager.save(output_dir / "model_config.yaml", format='yaml')
    manager.save(output_dir / "model_config.json", format='json')
    
    # Test loading
    loaded_manager = ModelConfigManager.load(
        output_dir / "model_config.yaml",
        format='yaml'
    )
    print(f"\nSuccessfully loaded config with {len(loaded_manager.config.models)} models")
    
    # Create a minimal config with just a few models for testing
    print("\n" + "="*80)
    print("Creating minimal test configuration...")
    print("="*80)
    
    minimal_config = ModelBenchmarkConfig(
        models={
            "tinyllama-1.1b": DefaultModelConfigs.create_pipeline_config("tinyllama-1.1b"),
            "phi-2": DefaultModelConfigs.create_pipeline_config("phi-2"),
            "smollm-360m": DefaultModelConfigs.create_pipeline_config("smollm-360m")
        },
        cache_dir="./models/cache",
        output_dir="./results/test"
    )
    
    minimal_manager = ModelConfigManager(minimal_config)
    minimal_manager.print_summary()
    minimal_manager.save(output_dir / "model_config_minimal.yaml", format='yaml')