"""
Model Loader Module
Handles loading and initialization of Small Language Models (SLMs).
"""

import logging
import torch
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import json

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig
)

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    name: str
    model_id: str
    model_type: str  # 'causal', 'seq2seq'
    num_parameters: int
    vocab_size: int
    max_length: int
    device: str
    dtype: str
    quantization: Optional[str] = None  # '8bit', '4bit', None
    is_instruct_tuned: bool = False
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'model_id': self.model_id,
            'model_type': self.model_type,
            'num_parameters': self.num_parameters,
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'device': self.device,
            'dtype': self.dtype,
            'quantization': self.quantization,
            'is_instruct_tuned': self.is_instruct_tuned,
            'metadata': self.metadata or {}
        }
    
    def print_summary(self) -> None:
        """Print model summary."""
        print(f"\nModel: {self.name}")
        print(f"  ID: {self.model_id}")
        print(f"  Type: {self.model_type}")
        print(f"  Parameters: {self.num_parameters:,}")
        print(f"  Vocab size: {self.vocab_size:,}")
        print(f"  Max length: {self.max_length}")
        print(f"  Device: {self.device}")
        print(f"  Dtype: {self.dtype}")
        if self.quantization:
            print(f"  Quantization: {self.quantization}")
        print(f"  Instruct-tuned: {self.is_instruct_tuned}")


@dataclass
class ModelLoadConfig:
    """Configuration for loading a model."""
    model_id: str
    name: Optional[str] = None
    device: str = "auto"  # 'auto', 'cuda', 'cpu', 'cuda:0', etc.
    dtype: str = "auto"  # 'auto', 'float16', 'bfloat16', 'float32'
    quantization: Optional[str] = None  # '8bit', '4bit', None
    trust_remote_code: bool = True
    use_flash_attention: bool = False
    low_cpu_mem_usage: bool = True
    cache_dir: Optional[str] = None
    token: Optional[str] = None  # HuggingFace token for gated models
    
    # Tokenizer settings
    padding_side: str = "left"  # Important for batch inference
    truncation_side: str = "left"
    add_bos_token: bool = True
    add_eos_token: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_id': self.model_id,
            'name': self.name,
            'device': self.device,
            'dtype': self.dtype,
            'quantization': self.quantization,
            'trust_remote_code': self.trust_remote_code,
            'use_flash_attention': self.use_flash_attention,
            'low_cpu_mem_usage': self.low_cpu_mem_usage,
            'cache_dir': self.cache_dir,
            'padding_side': self.padding_side,
            'truncation_side': self.truncation_side,
            'add_bos_token': self.add_bos_token,
            'add_eos_token': self.add_eos_token
        }


class ModelLoader:
    """Loads and manages language models."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the model loader.
        
        Args:
            cache_dir: Directory to cache downloaded models
        """
        self.cache_dir = cache_dir or "./models/cache"
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Keep track of loaded models
        self.loaded_models: Dict[str, tuple] = {}  # name -> (model, tokenizer, info)
    
    def load_model(
        self,
        config: ModelLoadConfig
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer, ModelInfo]:
        """
        Load a model and tokenizer.
        
        Args:
            config: ModelLoadConfig with loading settings
            
        Returns:
            Tuple of (model, tokenizer, model_info)
        """
        model_name = config.name or config.model_id.split('/')[-1]
        
        # Check if already loaded
        if model_name in self.loaded_models:
            logger.info(f"Model '{model_name}' already loaded, returning cached version")
            return self.loaded_models[model_name]
        
        logger.info(f"Loading model: {model_name} ({config.model_id})")
        
        # Load tokenizer
        tokenizer = self._load_tokenizer(config)
        
        # Load model
        model = self._load_model(config)
        
        # Get model info
        model_info = self._extract_model_info(model, tokenizer, config)
        
        # Cache the loaded model
        self.loaded_models[model_name] = (model, tokenizer, model_info)
        
        # Print summary
        model_info.print_summary()
        
        return model, tokenizer, model_info
    
    def _load_tokenizer(self, config: ModelLoadConfig) -> PreTrainedTokenizer:
        """Load tokenizer with specified configuration."""
        logger.info(f"Loading tokenizer from {config.model_id}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_id,
                trust_remote_code=config.trust_remote_code,
                cache_dir=config.cache_dir or self.cache_dir,
                token=config.token,
                padding_side=config.padding_side,
                truncation_side=config.truncation_side
            )
            
            # Set special tokens if not present
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            # Apply token settings
            if config.add_bos_token and not hasattr(tokenizer, 'add_bos_token'):
                tokenizer.add_bos_token = config.add_bos_token
            if config.add_eos_token and not hasattr(tokenizer, 'add_eos_token'):
                tokenizer.add_eos_token = config.add_eos_token
            
            logger.info(f"Tokenizer loaded successfully")
            logger.info(f"  Vocab size: {len(tokenizer)}")
            logger.info(f"  Padding side: {tokenizer.padding_side}")
            logger.info(f"  PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
            
            return tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def _load_model(self, config: ModelLoadConfig) -> PreTrainedModel:
        """Load model with specified configuration."""
        logger.info(f"Loading model from {config.model_id}")
        
        try:
            # Prepare model loading kwargs
            model_kwargs = {
                'pretrained_model_name_or_path': config.model_id,
                'trust_remote_code': config.trust_remote_code,
                'cache_dir': config.cache_dir or self.cache_dir,
                'low_cpu_mem_usage': config.low_cpu_mem_usage,
                'token': config.token
            }
            
            # Set device map
            if config.device == "auto":
                model_kwargs['device_map'] = "auto"
            elif config.device != "cpu":
                model_kwargs['device_map'] = config.device
            
            # Set dtype
            if config.dtype == "float16":
                model_kwargs['torch_dtype'] = torch.float16
            elif config.dtype == "bfloat16":
                model_kwargs['torch_dtype'] = torch.bfloat16
            elif config.dtype == "float32":
                model_kwargs['torch_dtype'] = torch.float32
            elif config.dtype == "auto":
                model_kwargs['torch_dtype'] = "auto"
            
            # Set quantization
            if config.quantization == "8bit":
                model_kwargs['load_in_8bit'] = True
                model_kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
                logger.info("Using 8-bit quantization")
            elif config.quantization == "4bit":
                model_kwargs['load_in_4bit'] = True
                model_kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                logger.info("Using 4-bit quantization")
            
            # Flash attention (if supported)
            if config.use_flash_attention:
                model_kwargs['attn_implementation'] = "flash_attention_2"
                logger.info("Using Flash Attention 2")
            
            # Load the model
            model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
            
            # Move to device if not using device_map
            if config.device != "auto" and config.device != "cpu" and 'device_map' not in model_kwargs:
                model = model.to(config.device)
            
            # Set to evaluation mode
            model.eval()
            
            logger.info("Model loaded successfully")
            
            # Log memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"  GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _extract_model_info(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: ModelLoadConfig
    ) -> ModelInfo:
        """Extract information about the loaded model."""
        # Get number of parameters
        num_params = sum(p.numel() for p in model.parameters())
        
        # Get model config
        model_config = model.config
        
        # Determine model type
        model_type = "causal"  # Default for most SLMs
        if hasattr(model_config, 'model_type'):
            if 'seq2seq' in model_config.model_type or 't5' in model_config.model_type:
                model_type = "seq2seq"
        
        # Get max length
        max_length = 2048  # Default
        if hasattr(model_config, 'max_position_embeddings'):
            max_length = model_config.max_position_embeddings
        elif hasattr(model_config, 'n_positions'):
            max_length = model_config.n_positions
        elif hasattr(tokenizer, 'model_max_length'):
            max_length = tokenizer.model_max_length
        
        # Get device
        device = str(next(model.parameters()).device)
        
        # Get dtype
        dtype = str(next(model.parameters()).dtype)
        
        # Check if instruct-tuned (heuristic based on model name)
        model_id_lower = config.model_id.lower()
        is_instruct = any(
            keyword in model_id_lower 
            for keyword in ['instruct', 'chat', 'it', 'alpaca', 'vicuna']
        )
        
        # Create metadata
        metadata = {
            'model_config': model_config.to_dict() if hasattr(model_config, 'to_dict') else {},
            'architecture': model_config.architectures[0] if hasattr(model_config, 'architectures') else "unknown"
        }
        
        return ModelInfo(
            name=config.name or config.model_id.split('/')[-1],
            model_id=config.model_id,
            model_type=model_type,
            num_parameters=num_params,
            vocab_size=len(tokenizer),
            max_length=max_length,
            device=device,
            dtype=dtype,
            quantization=config.quantization,
            is_instruct_tuned=is_instruct,
            metadata=metadata
        )
    
    def unload_model(self, name: str) -> None:
        """
        Unload a model from memory.
        
        Args:
            name: Name of the model to unload
        """
        if name in self.loaded_models:
            model, _, _ = self.loaded_models[name]
            
            # Move to CPU and delete
            if torch.cuda.is_available():
                model.cpu()
                torch.cuda.empty_cache()
            
            del self.loaded_models[name]
            logger.info(f"Unloaded model: {name}")
        else:
            logger.warning(f"Model '{name}' not found in loaded models")
    
    def unload_all_models(self) -> None:
        """Unload all loaded models."""
        model_names = list(self.loaded_models.keys())
        for name in model_names:
            self.unload_model(name)
        
        logger.info("Unloaded all models")
    
    def get_loaded_models(self) -> List[str]:
        """Get list of loaded model names."""
        return list(self.loaded_models.keys())
    
    def get_model_info(self, name: str) -> Optional[ModelInfo]:
        """Get info for a loaded model."""
        if name in self.loaded_models:
            _, _, info = self.loaded_models[name]
            return info
        return None


class SLMModelRegistry:
    """Registry of popular Small Language Models."""
    
    @staticmethod
    def get_model_configs() -> Dict[str, ModelLoadConfig]:
        """Get pre-configured ModelLoadConfig for popular SLMs."""
        configs = {
            # Microsoft Phi models
            "phi-2": ModelLoadConfig(
                model_id="microsoft/phi-2",
                name="phi-2",
                dtype="float16"
            ),
            "phi-1.5": ModelLoadConfig(
                model_id="microsoft/phi-1_5",
                name="phi-1.5",
                dtype="float16"
            ),
            
            # StableLM models
            "stablelm-2-1.6b": ModelLoadConfig(
                model_id="stabilityai/stablelm-2-1_6b",
                name="stablelm-2-1.6b",
                dtype="float16"
            ),
            "stablelm-2-zephyr-1.6b": ModelLoadConfig(
                model_id="stabilityai/stablelm-2-zephyr-1_6b",
                name="stablelm-2-zephyr-1.6b",
                dtype="float16"
            ),
            
            # TinyLlama
            "tinyllama-1.1b": ModelLoadConfig(
                model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                name="tinyllama-1.1b",
                dtype="float16"
            ),
            
            # Qwen models
            "qwen-1.8b": ModelLoadConfig(
                model_id="Qwen/Qwen-1_8B",
                name="qwen-1.8b",
                dtype="float16"
            ),
            "qwen-1.8b-chat": ModelLoadConfig(
                model_id="Qwen/Qwen-1_8B-Chat",
                name="qwen-1.8b-chat",
                dtype="float16"
            ),
            
            # Gemma models (requires auth)
            "gemma-2b": ModelLoadConfig(
                model_id="google/gemma-2b",
                name="gemma-2b",
                dtype="bfloat16"
            ),
            "gemma-2b-it": ModelLoadConfig(
                model_id="google/gemma-2b-it",
                name="gemma-2b-it",
                dtype="bfloat16"
            ),
            
            # OpenELM models
            "openelm-1.1b": ModelLoadConfig(
                model_id="apple/OpenELM-1_1B",
                name="openelm-1.1b",
                dtype="float16"
            ),
            
            # SmolLM models
            "smollm-135m": ModelLoadConfig(
                model_id="HuggingFaceTB/SmolLM-135M",
                name="smollm-135m",
                dtype="float32"
            ),
            "smollm-360m": ModelLoadConfig(
                model_id="HuggingFaceTB/SmolLM-360M",
                name="smollm-360m",
                dtype="float32"
            ),
        }
        
        return configs
    
    @staticmethod
    def get_model_config(model_name: str) -> ModelLoadConfig:
        """Get configuration for a specific model."""
        configs = SLMModelRegistry.get_model_configs()
        if model_name not in configs:
            raise ValueError(
                f"Model '{model_name}' not found in registry. "
                f"Available models: {list(configs.keys())}"
            )
        return configs[model_name]
    
    @staticmethod
    def list_available_models() -> List[str]:
        """List all available models in the registry."""
        return list(SLMModelRegistry.get_model_configs().keys())


# Example usage
if __name__ == "__main__":
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # List available models
    print("\nAvailable models in registry:")
    for model_name in SLMModelRegistry.list_available_models():
        print(f"  - {model_name}")
    
    # Initialize loader
    loader = ModelLoader(cache_dir="./models/cache")
    
    # Load a small model for testing
    print("\n" + "="*80)
    print("Loading model...")
    print("="*80)
    
    config = ModelLoadConfig(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        name="tinyllama-test",
        dtype="float16",
        device="auto"
    )
    
    model, tokenizer, info = loader.load_model(config)
    
    # Test inference
    print("\n" + "="*80)
    print("Testing inference...")
    print("="*80)
    
    test_prompt = "Question: What is the capital of France?\nA. London\nB. Paris\nC. Berlin\nD. Madrid\nAnswer:"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Last token logits
    
    # Get logits for option tokens
    option_tokens = [tokenizer.encode(opt, add_special_tokens=False)[0] for opt in ['A', 'B', 'C', 'D']]
    option_logits = logits[option_tokens]
    probs = torch.softmax(option_logits, dim=0)
    
    print(f"\nOption probabilities:")
    for i, (opt, prob) in enumerate(zip(['A', 'B', 'C', 'D'], probs)):
        print(f"  {opt}: {prob.item():.4f}")
    
    # Unload model
    print("\n" + "="*80)
    loader.unload_model("tinyllama-test")
    print("Model unloaded successfully")