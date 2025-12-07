"""
Inference Engine Module
Handles model inference and logit extraction for multiple-choice questions.
"""

import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from tqdm import tqdm

from transformers import PreTrainedModel, PreTrainedTokenizer
from src.models.model_loader import ModelInfo

# Setup logger
logger = logging.getLogger(__name__)

# Memory management settings for A100 optimization
CACHE_CLEAR_INTERVAL = 50  # Clear GPU cache every N batches to prevent fragmentation


def estimate_optimal_batch_size(
    model_params_billions: float,
    max_length: int = 2048,
    gpu_memory_gb: float = 80.0
) -> int:
    """
    Estimate optimal batch size based on model size and GPU memory.

    This is a heuristic based on typical A100 80GB memory usage patterns.
    Actual optimal batch size may vary based on model architecture.

    Args:
        model_params_billions: Model size in billions of parameters
        max_length: Maximum sequence length
        gpu_memory_gb: Available GPU memory in GB

    Returns:
        Recommended batch size
    """
    # Base memory estimates (empirical, for FP16 with 2048 token context)
    # These are conservative estimates to avoid OOM
    if model_params_billions <= 0.5:
        # <500M params: very small models
        base_batch = 32
    elif model_params_billions <= 1.5:
        # 500M-1.5B params: small models (TinyLlama, SmolLM, etc.)
        base_batch = 16
    elif model_params_billions <= 3.0:
        # 1.5B-3B params: medium models (Phi-2, StableLM, Gemma-2B)
        base_batch = 8
    elif model_params_billions <= 7.0:
        # 3B-7B params: larger models (Mistral-7B)
        base_batch = 4
    else:
        # >7B params: large models
        base_batch = 2

    # Adjust for sequence length (longer sequences need more memory)
    if max_length > 2048:
        length_factor = 2048 / max_length
        base_batch = max(1, int(base_batch * length_factor))

    # Adjust for GPU memory (base estimates assume 80GB)
    memory_factor = gpu_memory_gb / 80.0
    adjusted_batch = max(1, int(base_batch * memory_factor))

    logger.debug(
        f"Estimated batch size for {model_params_billions}B params, "
        f"{max_length} tokens, {gpu_memory_gb}GB GPU: {adjusted_batch}"
    )

    return adjusted_batch


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    batch_size: int = 4  # Increased default for better GPU utilization
    max_length: int = 2048
    device: Optional[str] = None  # If None, use model's device
    use_fp16: bool = True
    use_cache: bool = True

    # Temperature for probability calibration (if needed)
    temperature: float = 1.0

    # Option extraction settings
    option_letters: List[str] = None
    extract_last_token_only: bool = True

    def __post_init__(self):
        if self.option_letters is None:
            self.option_letters = ['A', 'B', 'C', 'D', 'E', 'F']

    @classmethod
    def with_auto_batch_size(
        cls,
        model_params_billions: float,
        max_length: int = 2048,
        gpu_memory_gb: float = 80.0,
        **kwargs
    ) -> 'InferenceConfig':
        """
        Create config with automatically estimated batch size.

        Args:
            model_params_billions: Model size in billions of parameters
            max_length: Maximum sequence length
            gpu_memory_gb: Available GPU memory in GB
            **kwargs: Additional config parameters

        Returns:
            InferenceConfig with optimal batch size
        """
        batch_size = estimate_optimal_batch_size(
            model_params_billions, max_length, gpu_memory_gb
        )
        return cls(batch_size=batch_size, max_length=max_length, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'batch_size': self.batch_size,
            'max_length': self.max_length,
            'device': self.device,
            'use_fp16': self.use_fp16,
            'use_cache': self.use_cache,
            'temperature': self.temperature,
            'option_letters': self.option_letters,
            'extract_last_token_only': self.extract_last_token_only
        }


@dataclass
class InferenceResult:
    """Result from inference on a single instance."""
    instance_id: str
    prompt: str
    logits: np.ndarray  # Raw logits for option tokens
    probabilities: np.ndarray  # Softmax probabilities
    predicted_option: str  # Most likely option (A-F)
    option_tokens: List[int]  # Token IDs for each option
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'instance_id': self.instance_id,
            'prompt': self.prompt[:100] + '...' if len(self.prompt) > 100 else self.prompt,
            'logits': self.logits.tolist(),
            'probabilities': self.probabilities.tolist(),
            'predicted_option': self.predicted_option,
            'option_tokens': self.option_tokens,
            'metadata': self.metadata or {}
        }


@dataclass
class BatchInferenceResult:
    """Results from batch inference."""
    results: List[InferenceResult]
    total_time: float
    avg_time_per_instance: float
    
    def __len__(self) -> int:
        return len(self.results)
    
    def __getitem__(self, idx: int) -> InferenceResult:
        return self.results[idx]
    
    def get_predictions(self) -> List[str]:
        """Get predicted options for all instances."""
        return [r.predicted_option for r in self.results]
    
    def get_probabilities(self) -> np.ndarray:
        """Get probability matrix (num_instances x num_options)."""
        return np.array([r.probabilities for r in self.results])
    
    def get_logits(self) -> np.ndarray:
        """Get logit matrix (num_instances x num_options)."""
        return np.array([r.logits for r in self.results])


class InferenceEngine:
    """Engine for running inference on language models."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        model_info: ModelInfo,
        config: Optional[InferenceConfig] = None
    ):
        """
        Initialize the inference engine.
        
        Args:
            model: The language model
            tokenizer: The tokenizer
            model_info: Information about the model
            config: Inference configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_info = model_info
        self.config = config or InferenceConfig()
        
        # Set device
        if self.config.device is None:
            self.config.device = str(next(model.parameters()).device)
        
        # Precompute option token IDs
        self.option_token_ids = self._get_option_token_ids()
        
        logger.info(f"Initialized InferenceEngine for {model_info.name}")
        logger.info(f"  Device: {self.config.device}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Option tokens: {self.option_token_ids}")
    
    def _get_option_token_ids(self) -> Dict[str, int]:
        """
        Get token IDs for option letters (A, B, C, D, E, F).

        Prioritizes space-prefixed tokens (e.g., " A") since prompts end with
        "Answer: " (trailing space), so the model predicts " A" not "A".

        Returns:
            Dictionary mapping option letters to token IDs
        """
        option_token_ids = {}

        for letter in self.config.option_letters:
            # Try different encodings to find the single token
            # Prioritize space-prefixed versions since prompt ends with space
            encodings = [
                f" {letter}",      # " A" - most likely for SentencePiece models
                letter,            # "A" - fallback for some tokenizers
                f" {letter}.",     # " A." - some models
                f"{letter}.",      # "A." - rare
            ]

            token_id = None
            matched_encoding = None
            for encoding in encodings:
                tokens = self.tokenizer.encode(encoding, add_special_tokens=False)
                if len(tokens) == 1:
                    token_id = tokens[0]
                    matched_encoding = encoding
                    break

            if token_id is None:
                # Fallback: use first token from space-prefixed version
                tokens = self.tokenizer.encode(f" {letter}", add_special_tokens=False)
                if len(tokens) > 0:
                    token_id = tokens[0]
                    matched_encoding = f" {letter} (first token)"
                else:
                    # Last resort fallback
                    tokens = self.tokenizer.encode(letter, add_special_tokens=False)
                    token_id = tokens[0]
                    matched_encoding = f"{letter} (fallback)"
                logger.warning(
                    f"Could not find single token for option '{letter}', "
                    f"using token {token_id} from '{matched_encoding}'"
                )

            option_token_ids[letter] = token_id
            logger.debug(f"Option '{letter}' -> token {token_id} (from '{matched_encoding}')")

        return option_token_ids
    
    def infer_single(
        self,
        prompt: str,
        instance_id: str = "unknown"
    ) -> InferenceResult:
        """
        Run inference on a single prompt.
        
        Args:
            prompt: The input prompt
            instance_id: ID for tracking
            
        Returns:
            InferenceResult containing logits and probabilities
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
            padding=False
        ).to(self.config.device)
        
        # Run inference
        with torch.no_grad():
            if self.config.use_fp16 and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs = self.model(**inputs, use_cache=self.config.use_cache)
            else:
                outputs = self.model(**inputs, use_cache=self.config.use_cache)
        
        # Extract logits for last token
        if self.config.extract_last_token_only:
            last_token_logits = outputs.logits[0, -1, :]  # Shape: [vocab_size]
        else:
            # Average over all tokens (alternative approach)
            last_token_logits = outputs.logits[0, :, :].mean(dim=0)  # Shape: [vocab_size]
        
        # Extract logits for option tokens
        option_logits = []
        option_tokens = []
        for letter in self.config.option_letters:
            token_id = self.option_token_ids[letter]
            option_tokens.append(token_id)
            option_logits.append(last_token_logits[token_id].item())
        
        option_logits = np.array(option_logits)
        
        # Apply temperature and compute probabilities
        if self.config.temperature != 1.0:
            option_logits = option_logits / self.config.temperature
        
        # Compute softmax probabilities
        probabilities = self._softmax(option_logits)
        
        # Get predicted option
        predicted_idx = np.argmax(probabilities)
        predicted_option = self.config.option_letters[predicted_idx]
        
        return InferenceResult(
            instance_id=instance_id,
            prompt=prompt,
            logits=option_logits,
            probabilities=probabilities,
            predicted_option=predicted_option,
            option_tokens=option_tokens
        )
    
    def infer_batch(
        self,
        prompts: List[str],
        instance_ids: Optional[List[str]] = None,
        show_progress: bool = True
    ) -> BatchInferenceResult:
        """
        Run inference on a batch of prompts.
        
        Args:
            prompts: List of input prompts
            instance_ids: List of IDs for tracking
            show_progress: Whether to show progress bar
            
        Returns:
            BatchInferenceResult containing all results
        """
        if instance_ids is None:
            instance_ids = [f"instance_{i}" for i in range(len(prompts))]
        
        if len(prompts) != len(instance_ids):
            raise ValueError(
                f"Number of prompts ({len(prompts)}) must match "
                f"number of instance IDs ({len(instance_ids)})"
            )
        
        import time
        start_time = time.time()
        
        results = []
        
        # Process in batches
        num_batches = (len(prompts) + self.config.batch_size - 1) // self.config.batch_size

        iterator = range(0, len(prompts), self.config.batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=num_batches, desc="Running inference")

        batch_count = 0
        for i in iterator:
            batch_prompts = prompts[i:i + self.config.batch_size]
            batch_ids = instance_ids[i:i + self.config.batch_size]

            if self.config.batch_size == 1 or len(batch_prompts) == 1:
                # Single inference
                result = self.infer_single(batch_prompts[0], batch_ids[0])
                results.append(result)
            else:
                # Batched inference
                batch_results = self._infer_batch_internal(batch_prompts, batch_ids)
                results.extend(batch_results)

            batch_count += 1

            # Periodic cache clearing to prevent memory fragmentation on long runs
            if batch_count % CACHE_CLEAR_INTERVAL == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug(f"Cleared GPU cache after {batch_count} batches")
        
        total_time = time.time() - start_time
        avg_time = total_time / len(prompts)
        
        return BatchInferenceResult(
            results=results,
            total_time=total_time,
            avg_time_per_instance=avg_time
        )
    
    def _infer_batch_internal(
        self,
        prompts: List[str],
        instance_ids: List[str]
    ) -> List[InferenceResult]:
        """Internal method for batched inference."""
        # Tokenize batch
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
            padding=True
        ).to(self.config.device)

        # For batch inference, disable KV-cache to save memory (20-30% reduction)
        # KV-cache is only beneficial for autoregressive generation, not single-pass inference
        use_cache = False if len(prompts) > 1 else self.config.use_cache

        # Run inference
        with torch.no_grad():
            if self.config.use_fp16 and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs = self.model(**inputs, use_cache=use_cache)
            else:
                outputs = self.model(**inputs, use_cache=use_cache)
        
        # Extract results for each instance in batch
        results = []
        for idx, (prompt, instance_id) in enumerate(zip(prompts, instance_ids)):
            # Get last token logits for this instance
            # Find the last non-padding token
            attention_mask = inputs['attention_mask'][idx]
            last_token_pos = attention_mask.sum() - 1
            
            last_token_logits = outputs.logits[idx, last_token_pos, :]
            
            # Extract option logits
            option_logits = []
            option_tokens = []
            for letter in self.config.option_letters:
                token_id = self.option_token_ids[letter]
                option_tokens.append(token_id)
                option_logits.append(last_token_logits[token_id].item())
            
            option_logits = np.array(option_logits)
            
            # Apply temperature
            if self.config.temperature != 1.0:
                option_logits = option_logits / self.config.temperature
            
            # Compute probabilities
            probabilities = self._softmax(option_logits)
            
            # Get prediction
            predicted_idx = np.argmax(probabilities)
            predicted_option = self.config.option_letters[predicted_idx]
            
            result = InferenceResult(
                instance_id=instance_id,
                prompt=prompt,
                logits=option_logits,
                probabilities=probabilities,
                predicted_option=predicted_option,
                option_tokens=option_tokens
            )
            results.append(result)
        
        return results
    
    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        return exp_logits / exp_logits.sum()
    
    def set_temperature(self, temperature: float) -> None:
        """Update temperature for probability scaling."""
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        self.config.temperature = temperature
        logger.info(f"Set temperature to {temperature}")
    
    def update_config(self, **kwargs) -> None:
        """Update inference configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated {key} to {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")


class InferenceCache:
    """Cache for storing inference results to avoid recomputation."""
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of results to cache
        """
        self.cache: Dict[str, InferenceResult] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _get_cache_key(self, prompt: str, model_name: str) -> str:
        """Generate cache key from prompt and model name."""
        import hashlib
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        return f"{model_name}_{prompt_hash}"
    
    def get(self, prompt: str, model_name: str) -> Optional[InferenceResult]:
        """Get cached result if available."""
        key = self._get_cache_key(prompt, model_name)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        else:
            self.misses += 1
            return None
    
    def put(self, prompt: str, model_name: str, result: InferenceResult) -> None:
        """Store result in cache."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        key = self._get_cache_key(prompt, model_name)
        self.cache[key] = result
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }


# Example usage
if __name__ == "__main__":
    import logging
    from src.models.model_loader import ModelLoader, ModelLoadConfig
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load model
    print("\n" + "="*80)
    print("Loading model...")
    print("="*80)
    
    loader = ModelLoader()
    config = ModelLoadConfig(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        name="tinyllama",
        dtype="float16",
        device="auto"
    )
    model, tokenizer, info = loader.load_model(config)
    
    # Initialize inference engine
    print("\n" + "="*80)
    print("Initializing inference engine...")
    print("="*80)
    
    inference_config = InferenceConfig(
        batch_size=2,
        max_length=512,
        temperature=1.0
    )
    engine = InferenceEngine(model, tokenizer, info, inference_config)
    
    # Test single inference
    print("\n" + "="*80)
    print("Testing single inference...")
    print("="*80)
    
    test_prompt = """Question: What is the capital of France?
A. London
B. Paris
C. Berlin
D. Madrid
E. I don't know
F. None of the above
Answer:"""
    
    result = engine.infer_single(test_prompt, instance_id="test_1")
    
    print(f"\nPrompt: {test_prompt[:100]}...")
    print(f"\nPredicted option: {result.predicted_option}")
    print(f"\nProbabilities:")
    for letter, prob in zip(inference_config.option_letters, result.probabilities):
        print(f"  {letter}: {prob:.4f}")
    
    # Test batch inference
    print("\n" + "="*80)
    print("Testing batch inference...")
    print("="*80)
    
    test_prompts = [
        """Question: What is 2 + 2?
A. 3
B. 4
C. 5
D. 6
E. I don't know
F. None of the above
Answer:""",
        """Question: What color is the sky?
A. Red
B. Blue
C. Green
D. Yellow
E. I don't know
F. None of the above
Answer:""",
        """Question: How many days in a week?
A. 5
B. 6
C. 7
D. 8
E. I don't know
F. None of the above
Answer:"""
    ]
    
    batch_result = engine.infer_batch(test_prompts, show_progress=True)
    
    print(f"\nProcessed {len(batch_result)} instances")
    print(f"Total time: {batch_result.total_time:.2f}s")
    print(f"Avg time per instance: {batch_result.avg_time_per_instance:.3f}s")
    
    print(f"\nPredictions:")
    for i, result in enumerate(batch_result.results):
        print(f"  Instance {i+1}: {result.predicted_option}")
    
    # Test cache
    print("\n" + "="*80)
    print("Testing inference cache...")
    print("="*80)
    
    cache = InferenceCache(max_size=100)
    cache.put(test_prompt, "tinyllama", result)
    
    cached_result = cache.get(test_prompt, "tinyllama")
    print(f"Cache hit: {cached_result is not None}")
    print(f"Cache stats: {cache.get_stats()}")
    
    # Clean up
    loader.unload_model("tinyllama")
    print("\nModel unloaded")