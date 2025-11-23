"""
Prompt Formatter Module
Advanced formatting utilities for prompts including truncation, chat conversion, and validation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from src.data.dataset_loader import DataInstance

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class FormattingConfig:
    """Configuration for prompt formatting."""
    max_length: int = 2048
    truncation_strategy: str = "context_first"  # 'context_first', 'demos_first', 'proportional'
    
    # Tokenizer settings (if available)
    count_tokens: bool = False
    tokenizer: Any = None  # Optional tokenizer for accurate length counting
    
    # Chat formatting
    apply_chat_template: bool = False
    chat_template_type: Optional[str] = None  # 'llama2', 'qwen', 'phi2', 'generic'
    
    # Validation
    validate_format: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'max_length': self.max_length,
            'truncation_strategy': self.truncation_strategy,
            'count_tokens': self.count_tokens,
            'apply_chat_template': self.apply_chat_template,
            'chat_template_type': self.chat_template_type,
            'validate_format': self.validate_format
        }


class PromptTruncator:
    """Handles truncation of prompts that exceed max length."""
    
    @staticmethod
    def truncate_context(
        context: str,
        max_length: int,
        preserve_end: bool = False
    ) -> str:
        """
        Truncate context to max length.
        
        Args:
            context: Context text
            max_length: Maximum length
            preserve_end: If True, keep end of context; otherwise keep beginning
            
        Returns:
            Truncated context
        """
        if len(context) <= max_length:
            return context
        
        if preserve_end:
            # Keep the end (might be more relevant for some tasks)
            truncated = "..." + context[-(max_length - 3):]
        else:
            # Keep the beginning
            truncated = context[:max_length - 3] + "..."
        
        logger.debug(f"Truncated context from {len(context)} to {len(truncated)} chars")
        return truncated
    
    @staticmethod
    def truncate_demonstrations(
        demonstrations: List[DataInstance],
        max_num: int
    ) -> List[DataInstance]:
        """
        Truncate number of demonstrations.
        
        Args:
            demonstrations: List of demonstrations
            max_num: Maximum number to keep
            
        Returns:
            Truncated list
        """
        if len(demonstrations) <= max_num:
            return demonstrations
        
        logger.debug(f"Truncated demonstrations from {len(demonstrations)} to {max_num}")
        return demonstrations[:max_num]
    
    @staticmethod
    def smart_truncate_prompt(
        prompt_parts: Dict[str, str],
        max_length: int,
        strategy: str = "context_first"
    ) -> Dict[str, str]:
        """
        Intelligently truncate prompt parts to fit within max length.
        
        Args:
            prompt_parts: Dict with keys like 'instruction', 'demonstrations', 
                          'context', 'question', 'options', 'answer_prefix'
            max_length: Maximum total length
            strategy: Truncation strategy
            
        Returns:
            Truncated prompt parts
        """
        # Calculate current lengths
        current_length = sum(len(part) for part in prompt_parts.values() if part)
        
        if current_length <= max_length:
            return prompt_parts
        
        logger.info(f"Truncating prompt from {current_length} to {max_length} chars using '{strategy}'")
        
        # Copy parts to modify
        truncated_parts = prompt_parts.copy()
        
        if strategy == "context_first":
            # Truncate context first, then demonstrations
            reduction_needed = current_length - max_length
            
            # Try truncating context
            if 'context' in truncated_parts and truncated_parts['context']:
                context_len = len(truncated_parts['context'])
                new_context_len = max(100, context_len - reduction_needed)
                truncated_parts['context'] = PromptTruncator.truncate_context(
                    truncated_parts['context'],
                    new_context_len
                )
                reduction_needed -= (context_len - len(truncated_parts['context']))
            
            # If still too long, truncate demonstrations
            if reduction_needed > 0 and 'demonstrations' in truncated_parts:
                demo_len = len(truncated_parts['demonstrations'])
                new_demo_len = max(0, demo_len - reduction_needed)
                truncated_parts['demonstrations'] = truncated_parts['demonstrations'][:new_demo_len]
        
        elif strategy == "demos_first":
            # Truncate demonstrations first
            reduction_needed = current_length - max_length
            
            if 'demonstrations' in truncated_parts and truncated_parts['demonstrations']:
                demo_len = len(truncated_parts['demonstrations'])
                new_demo_len = max(0, demo_len - reduction_needed)
                truncated_parts['demonstrations'] = truncated_parts['demonstrations'][:new_demo_len]
                reduction_needed -= (demo_len - len(truncated_parts['demonstrations']))
            
            # If still too long, truncate context
            if reduction_needed > 0 and 'context' in truncated_parts:
                context_len = len(truncated_parts['context'])
                new_context_len = max(100, context_len - reduction_needed)
                truncated_parts['context'] = PromptTruncator.truncate_context(
                    truncated_parts['context'],
                    new_context_len
                )
        
        return truncated_parts


class ChatTemplateConverter:
    """Converts prompts to chat templates for instruction-tuned models."""
    
    @staticmethod
    def convert_to_chat(
        prompt: str,
        template_type: str,
        system_message: str = "You are a helpful assistant."
    ) -> str:
        """
        Convert prompt to chat format.
        
        Args:
            prompt: Base prompt
            template_type: Type of chat template
            system_message: System message
            
        Returns:
            Chat-formatted prompt
        """
        if template_type == 'llama2':
            return ChatTemplateConverter.llama2_format(prompt, system_message)
        elif template_type == 'qwen':
            return ChatTemplateConverter.qwen_format(prompt, system_message)
        elif template_type == 'phi2':
            return ChatTemplateConverter.phi2_format(prompt)
        elif template_type == 'zephyr':
            return ChatTemplateConverter.zephyr_format(prompt, system_message)
        elif template_type == 'generic':
            return ChatTemplateConverter.generic_format(prompt, system_message)
        else:
            logger.warning(f"Unknown template type '{template_type}', using generic")
            return ChatTemplateConverter.generic_format(prompt, system_message)
    
    @staticmethod
    def llama2_format(prompt: str, system_message: str) -> str:
        """Llama-2-Chat format."""
        return (
            f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n"
            f"{prompt} [/INST]"
        )
    
    @staticmethod
    def qwen_format(prompt: str, system_message: str) -> str:
        """Qwen-Chat format."""
        return (
            f"<|im_start|>system\n{system_message}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    
    @staticmethod
    def phi2_format(prompt: str) -> str:
        """Phi-2 format (simple instruct format)."""
        return f"Instruct: {prompt}\nOutput:"
    
    @staticmethod
    def zephyr_format(prompt: str, system_message: str) -> str:
        """Zephyr/StableLM format."""
        return (
            f"<|system|>\n{system_message}</s>\n"
            f"<|user|>\n{prompt}</s>\n"
            f"<|assistant|>"
        )
    
    @staticmethod
    def generic_format(prompt: str, system_message: str) -> str:
        """Generic chat format."""
        return (
            f"### System:\n{system_message}\n\n"
            f"### User:\n{prompt}\n\n"
            f"### Assistant:"
        )


class PromptValidator:
    """Validates prompt formatting and content."""
    
    @staticmethod
    def validate_prompt(prompt: str, expected_components: Optional[List[str]] = None) -> bool:
        """
        Validate that a prompt is properly formatted.
        
        Args:
            prompt: Prompt string to validate
            expected_components: Optional list of expected components
            
        Returns:
            True if valid, False otherwise
        """
        is_valid = True
        issues = []
        
        # Check not empty
        if not prompt or len(prompt.strip()) == 0:
            issues.append("Prompt is empty")
            is_valid = False
        
        # Check for "Answer:" prefix
        if "Answer:" not in prompt and "answer:" not in prompt.lower():
            issues.append("Missing 'Answer:' prefix")
            is_valid = False
        
        # Check for options (A-F)
        has_options = all(
            f"{letter}." in prompt or f"{letter} " in prompt
            for letter in ['A', 'B', 'C', 'D']
        )
        if not has_options:
            issues.append("Missing option labels (A, B, C, D)")
            is_valid = False
        
        # Check for question
        if "Question:" not in prompt and "question:" not in prompt.lower():
            issues.append("Missing 'Question:' label")
            is_valid = False
        
        # Check expected components
        if expected_components:
            for component in expected_components:
                if component.lower() not in prompt.lower():
                    issues.append(f"Missing expected component: {component}")
                    is_valid = False
        
        if not is_valid:
            logger.warning(f"Prompt validation failed:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        return is_valid
    
    @staticmethod
    def count_tokens(prompt: str, tokenizer: Any) -> int:
        """
        Count tokens in prompt using tokenizer.
        
        Args:
            prompt: Prompt string
            tokenizer: Tokenizer instance
            
        Returns:
            Number of tokens
        """
        if tokenizer is None:
            # Rough estimation: ~4 chars per token
            return len(prompt) // 4
        
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        return len(tokens)


class PromptFormatter:
    """Advanced prompt formatter with all features."""
    
    def __init__(self, config: Optional[FormattingConfig] = None):
        """
        Initialize prompt formatter.
        
        Args:
            config: Formatting configuration
        """
        self.config = config or FormattingConfig()
        self.truncator = PromptTruncator()
        self.converter = ChatTemplateConverter()
        self.validator = PromptValidator()
        
        logger.info("Initialized PromptFormatter")
        logger.info(f"  Max length: {self.config.max_length}")
        logger.info(f"  Apply chat template: {self.config.apply_chat_template}")
    
    def format(
        self,
        prompt: str,
        apply_truncation: bool = True,
        apply_chat_template: Optional[bool] = None,
        validate: Optional[bool] = None
    ) -> str:
        """
        Apply all formatting to a prompt.
        
        Args:
            prompt: Base prompt
            apply_truncation: Whether to truncate if too long
            apply_chat_template: Whether to apply chat template (overrides config)
            validate: Whether to validate (overrides config)
            
        Returns:
            Formatted prompt
        """
        formatted_prompt = prompt
        
        # Truncate if needed
        if apply_truncation and len(formatted_prompt) > self.config.max_length:
            # Simple character-based truncation
            # For smarter truncation, use truncate_smart with prompt_parts
            logger.debug(f"Truncating prompt from {len(formatted_prompt)} to {self.config.max_length}")
            formatted_prompt = formatted_prompt[:self.config.max_length]
        
        # Apply chat template if configured
        should_apply_chat = apply_chat_template if apply_chat_template is not None else self.config.apply_chat_template
        
        if should_apply_chat and self.config.chat_template_type:
            formatted_prompt = self.converter.convert_to_chat(
                formatted_prompt,
                self.config.chat_template_type
            )
        
        # Validate if configured
        should_validate = validate if validate is not None else self.config.validate_format
        
        if should_validate:
            is_valid = self.validator.validate_prompt(formatted_prompt)
            if not is_valid:
                logger.warning("Prompt validation failed")
        
        return formatted_prompt
    
    def format_batch(
        self,
        prompts: List[str],
        apply_truncation: bool = True,
        apply_chat_template: Optional[bool] = None
    ) -> List[str]:
        """
        Format a batch of prompts.
        
        Args:
            prompts: List of base prompts
            apply_truncation: Whether to truncate
            apply_chat_template: Whether to apply chat template
            
        Returns:
            List of formatted prompts
        """
        return [
            self.format(
                prompt,
                apply_truncation=apply_truncation,
                apply_chat_template=apply_chat_template,
                validate=False  # Skip validation for batch processing
            )
            for prompt in prompts
        ]
    
    def get_length(self, prompt: str) -> int:
        """
        Get length of prompt.
        
        Args:
            prompt: Prompt string
            
        Returns:
            Length (in tokens if tokenizer available, else characters)
        """
        if self.config.count_tokens and self.config.tokenizer:
            return self.validator.count_tokens(prompt, self.config.tokenizer)
        else:
            return len(prompt)


class PromptStatistics:
    """Compute statistics about prompts."""
    
    @staticmethod
    def analyze_prompts(
        prompts: List[str],
        tokenizer: Any = None
    ) -> Dict[str, Any]:
        """
        Analyze a collection of prompts.
        
        Args:
            prompts: List of prompts
            tokenizer: Optional tokenizer for token counting
            
        Returns:
            Statistics dictionary
        """
        if not prompts:
            return {'count': 0}
        
        # Character lengths
        char_lengths = [len(p) for p in prompts]
        
        stats = {
            'count': len(prompts),
            'char_length': {
                'mean': float(np.mean(char_lengths)),
                'median': float(np.median(char_lengths)),
                'min': int(min(char_lengths)),
                'max': int(max(char_lengths)),
                'std': float(np.std(char_lengths))
            }
        }
        
        # Token lengths if tokenizer available
        if tokenizer:
            token_lengths = []
            for prompt in prompts:
                tokens = tokenizer.encode(prompt, add_special_tokens=True)
                token_lengths.append(len(tokens))
            
            stats['token_length'] = {
                'mean': float(np.mean(token_lengths)),
                'median': float(np.median(token_lengths)),
                'min': int(min(token_lengths)),
                'max': int(max(token_lengths)),
                'std': float(np.std(token_lengths))
            }
        
        # Count prompts with specific components
        stats['components'] = {
            'has_context': sum(1 for p in prompts if 'Context:' in p or 'Dialogue:' in p or 'Document:' in p),
            'has_demonstrations': sum(1 for p in prompts if p.count('Answer:') > 1),  # More than one Answer: means demos
            'has_instruction': sum(1 for p in prompts if 'Below are' in p or 'Instruct:' in p)
        }
        
        return stats


# Example usage
if __name__ == "__main__":
    import logging
    import numpy as np
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*80)
    print("Prompt Formatter Test")
    print("="*80)
    
    # Create a sample long prompt
    long_prompt = """Below are some examples of multiple-choice questions.

Context: """ + "This is a very long context. " * 200 + """

Question: What is the main topic?
Choices:
A. Topic A
B. Topic B
C. Topic C
D. Topic D
E. I don't know
F. None of the above
Answer:"""
    
    print(f"\nOriginal prompt length: {len(long_prompt)} characters")
    
    # Test truncation
    print("\n" + "="*80)
    print("Testing truncation...")
    print("="*80)
    
    config = FormattingConfig(
        max_length=500,
        truncation_strategy='context_first'
    )
    
    formatter = PromptFormatter(config)
    truncated = formatter.format(long_prompt, apply_truncation=True)
    
    print(f"Truncated prompt length: {len(truncated)} characters")
    print(f"\nTruncated prompt preview:")
    print(truncated[:300] + "...")
    
    # Test chat template conversion
    print("\n" + "="*80)
    print("Testing chat template conversion...")
    print("="*80)
    
    short_prompt = """Question: What is 2+2?
Choices:
A. 3
B. 4
C. 5
D. 6
E. I don't know
F. None of the above
Answer:"""
    
    for template_type in ['llama2', 'qwen', 'phi2', 'zephyr', 'generic']:
        print(f"\n{'-'*80}")
        print(f"{template_type.upper()} format:")
        print(f"{'-'*80}")
        
        config = FormattingConfig(
            apply_chat_template=True,
            chat_template_type=template_type
        )
        
        formatter = PromptFormatter(config)
        chat_prompt = formatter.format(short_prompt)
        print(chat_prompt)
    
    # Test validation
    print("\n" + "="*80)
    print("Testing prompt validation...")
    print("="*80)
    
    validator = PromptValidator()
    
    valid_prompt = short_prompt
    invalid_prompt = "Just a question without proper formatting"
    
    print(f"\nValidating valid prompt: {validator.validate_prompt(valid_prompt)}")
    print(f"Validating invalid prompt: {validator.validate_prompt(invalid_prompt)}")
    
    # Test statistics
    print("\n" + "="*80)
    print("Testing prompt statistics...")
    print("="*80)
    
    sample_prompts = [short_prompt, long_prompt[:1000], long_prompt[:500]]
    
    stats = PromptStatistics.analyze_prompts(sample_prompts)
    
    print(f"\nPrompt Statistics:")
    print(f"  Count: {stats['count']}")
    print(f"  Character length:")
    print(f"    Mean: {stats['char_length']['mean']:.1f}")
    print(f"    Min: {stats['char_length']['min']}")
    print(f"    Max: {stats['char_length']['max']}")
    print(f"  Components:")
    for component, count in stats['components'].items():
        print(f"    {component}: {count}")