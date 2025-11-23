"""
Prompt Templates Module
Stores and manages prompt templates for different tasks and strategies.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class TaskPromptTemplates:
    """Templates for all strategies for a specific task."""
    task_name: str
    task_type: str
    
    # Instructions for each strategy
    base_instruction: Optional[str] = None
    shared_instruction: Optional[str] = None
    task_specific_instruction: Optional[str] = None
    
    # Context handling
    context_label: str = "Context"  # "Context", "Dialogue", or "Document"
    has_context: bool = False
    
    # Question prefix
    question_prefix: str = "Question:"
    answer_prefix: str = "Answer:"
    
    def get_instruction(self, strategy: str) -> Optional[str]:
        """Get instruction for a specific strategy."""
        if strategy == 'base':
            return self.base_instruction
        elif strategy == 'shared_instruction':
            return self.shared_instruction
        elif strategy == 'task_specific':
            return self.task_specific_instruction
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'task_name': self.task_name,
            'task_type': self.task_type,
            'base_instruction': self.base_instruction,
            'shared_instruction': self.shared_instruction,
            'task_specific_instruction': self.task_specific_instruction,
            'context_label': self.context_label,
            'has_context': self.has_context,
            'question_prefix': self.question_prefix,
            'answer_prefix': self.answer_prefix
        }


class PromptTemplateRegistry:
    """Registry of prompt templates for all tasks."""
    
    @staticmethod
    def get_qa_templates() -> TaskPromptTemplates:
        """Get templates for Question Answering (MMLU) task."""
        return TaskPromptTemplates(
            task_name="qa",
            task_type="qa",
            base_instruction=None,
            shared_instruction=(
                "Below are some examples of multiple-choice questions with six potential "
                "answers. For each question, only one option is correct."
            ),
            task_specific_instruction=(
                "Below are some examples of multiple-choice questions about question answering. "
                "Each question should be answered based on your world knowledge and problem solving ability."
            ),
            context_label="Context",
            has_context=False,
            question_prefix="Question:",
            answer_prefix="Answer:"
        )
    
    @staticmethod
    def get_rc_templates() -> TaskPromptTemplates:
        """Get templates for Reading Comprehension (CosmosQA) task."""
        return TaskPromptTemplates(
            task_name="rc",
            task_type="rc",
            base_instruction=None,
            shared_instruction=(
                "Below are some examples of multiple-choice questions with six potential "
                "answers. For each question, only one option is correct."
            ),
            task_specific_instruction=(
                "Below are some examples of multiple-choice questions about reading comprehension. "
                "Each question should be answered based on the given context and commonsense reasoning when necessary."
            ),
            context_label="Context",
            has_context=True,
            question_prefix="Question:",
            answer_prefix="Answer:"
        )
    
    @staticmethod
    def get_ci_templates() -> TaskPromptTemplates:
        """Get templates for Commonsense Inference (HellaSwag) task."""
        return TaskPromptTemplates(
            task_name="ci",
            task_type="ci",
            base_instruction=None,
            shared_instruction=(
                "Below are some examples of multiple-choice questions with six potential "
                "answers. For each question, only one option is correct."
            ),
            task_specific_instruction=(
                "Below are some examples of multiple-choice questions about commonsense natural language inference. "
                "For each question, there is a given context and the answer is the option that most likely follows the context."
            ),
            context_label="Context",
            has_context=True,
            question_prefix="Question:",
            answer_prefix="Answer:"
        )
    
    @staticmethod
    def get_drs_templates() -> TaskPromptTemplates:
        """Get templates for Dialogue Response Selection (HaluDial) task."""
        return TaskPromptTemplates(
            task_name="drs",
            task_type="drs",
            base_instruction=None,
            shared_instruction=(
                "Below are some examples of multiple-choice questions with six potential "
                "answers. For each question, only one option is correct."
            ),
            task_specific_instruction=(
                "Below are some examples of multiple-choice questions about dialogue response selection. "
                "For each question, the answer is the option that represents the most suitable response for the given "
                "dialogue history, without hallucination and non-factual information."
            ),
            context_label="Dialogue",
            has_context=True,
            question_prefix="Question:",
            answer_prefix="Answer:"
        )
    
    @staticmethod
    def get_ds_templates() -> TaskPromptTemplates:
        """Get templates for Document Summarization (HaluSum) task."""
        return TaskPromptTemplates(
            task_name="ds",
            task_type="ds",
            base_instruction=None,
            shared_instruction=(
                "Below are some examples of multiple-choice questions with six potential "
                "answers. For each question, only one option is correct."
            ),
            task_specific_instruction=(
                "Below are some examples of multiple-choice questions about document summarization. "
                "For each question, the answer is the option that accurately summarizes the given document without "
                "hallucination and non-factual information."
            ),
            context_label="Document",
            has_context=True,
            question_prefix="Question:",
            answer_prefix="Answer:"
        )
    
    @staticmethod
    def get_all_templates() -> Dict[str, TaskPromptTemplates]:
        """Get all task templates."""
        return {
            'qa': PromptTemplateRegistry.get_qa_templates(),
            'rc': PromptTemplateRegistry.get_rc_templates(),
            'ci': PromptTemplateRegistry.get_ci_templates(),
            'drs': PromptTemplateRegistry.get_drs_templates(),
            'ds': PromptTemplateRegistry.get_ds_templates()
        }
    
    @staticmethod
    def get_template(task_type: str) -> TaskPromptTemplates:
        """Get template for a specific task."""
        templates = PromptTemplateRegistry.get_all_templates()
        
        if task_type not in templates:
            raise ValueError(
                f"Unknown task type: {task_type}. "
                f"Available: {list(templates.keys())}"
            )
        
        return templates[task_type]


class PromptFormatter:
    """Utility for formatting prompt components."""
    
    @staticmethod
    def format_options(options: List[str], letters: Optional[List[str]] = None) -> str:
        """
        Format options with letter labels.
        
        Args:
            options: List of option texts
            letters: Optional custom letters (default: A-F)
            
        Returns:
            Formatted options string
        """
        if letters is None:
            letters = [chr(65 + i) for i in range(len(options))]  # A, B, C, ...
        
        formatted = "Choices:\n"
        for letter, option in zip(letters, options):
            formatted += f"{letter}. {option}\n"
        
        return formatted.rstrip()  # Remove trailing newline
    
    @staticmethod
    def format_context(
        context_text: str,
        context_label: str = "Context",
        max_length: Optional[int] = None
    ) -> str:
        """
        Format context with label.
        
        Args:
            context_text: The context text
            context_label: Label for context ("Context", "Dialogue", "Document")
            max_length: Optional max length for truncation
            
        Returns:
            Formatted context string
        """
        if max_length and len(context_text) > max_length:
            context_text = context_text[:max_length] + "..."
            logger.debug(f"Truncated context to {max_length} characters")
        
        return f"{context_label}: {context_text}"
    
    @staticmethod
    def format_question(question_text: str, prefix: str = "Question:") -> str:
        """Format question with prefix."""
        return f"{prefix} {question_text}"
    
    @staticmethod
    def format_answer_prefix(prefix: str = "Answer:") -> str:
        """Format answer prefix."""
        return prefix
    
    @staticmethod
    def join_prompt_parts(parts: List[str], separator: str = "\n") -> str:
        """
        Join prompt parts with separator.
        
        Args:
            parts: List of prompt components
            separator: Separator between parts
            
        Returns:
            Joined prompt string
        """
        # Filter out None and empty strings
        parts = [p for p in parts if p]
        return separator.join(parts)


class ChatPromptFormatter:
    """
    Formatter for instruction-tuned models using chat format.
    
    Converts prompts to chat format for models like Llama-2-Chat, Qwen-Chat, etc.
    """
    
    @staticmethod
    def format_for_llama2_chat(prompt: str) -> str:
        """
        Format prompt for Llama-2-Chat model.
        
        Args:
            prompt: Base prompt
            
        Returns:
            Chat-formatted prompt
        """
        # Llama-2-Chat format
        formatted = (
            "[INST] <<SYS>>\n"
            "You are a helpful assistant.\n"
            "<</SYS>>\n\n"
            f"{prompt} [/INST]"
        )
        return formatted
    
    @staticmethod
    def format_for_qwen_chat(prompt: str) -> str:
        """
        Format prompt for Qwen-Chat model.
        
        Args:
            prompt: Base prompt
            
        Returns:
            Chat-formatted prompt
        """
        # Qwen-Chat format
        formatted = (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return formatted
    
    @staticmethod
    def format_for_phi2(prompt: str) -> str:
        """
        Format prompt for Phi-2 model.
        
        Args:
            prompt: Base prompt
            
        Returns:
            Formatted prompt (Phi-2 doesn't use special chat tokens)
        """
        # Phi-2 uses simple "Instruct:" format
        formatted = f"Instruct: {prompt}\nOutput:"
        return formatted
    
    @staticmethod
    def format_for_generic_chat(
        prompt: str,
        system_message: str = "You are a helpful assistant."
    ) -> str:
        """
        Generic chat format.
        
        Args:
            prompt: Base prompt
            system_message: System message
            
        Returns:
            Chat-formatted prompt
        """
        formatted = (
            f"System: {system_message}\n\n"
            f"User: {prompt}\n\n"
            f"Assistant:"
        )
        return formatted


# Example usage
if __name__ == "__main__":
    import logging
    from src.data.dataset_loader import DataInstance
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*80)
    print("Prompt Templates Test")
    print("="*80)
    
    # Get all templates
    all_templates = PromptTemplateRegistry.get_all_templates()
    
    print(f"\nAvailable templates: {list(all_templates.keys())}")
    
    for task_name, template in all_templates.items():
        print(f"\n{task_name.upper()}:")
        print(f"  Has context: {template.has_context}")
        print(f"  Context label: {template.context_label}")
        print(f"  Task-specific instruction: {template.task_specific_instruction[:80]}...")
    
    # Test prompt formatter
    print("\n" + "="*80)
    print("Testing Prompt Formatter")
    print("="*80)
    
    formatter = PromptFormatter()
    
    options = ["Paris", "London", "Berlin", "Madrid", "I don't know", "None of the above"]
    formatted_options = formatter.format_options(options)
    print("\nFormatted options:")
    print(formatted_options)
    
    context = "This is a sample context with important information."
    formatted_context = formatter.format_context(context, "Context")
    print(f"\nFormatted context:\n{formatted_context}")
    
    # Test chat formatters
    print("\n" + "="*80)
    print("Testing Chat Format Converters")
    print("="*80)
    
    base_prompt = "What is the capital of France?\nA. Paris\nB. London\nAnswer:"
    
    chat_formatter = ChatPromptFormatter()
    
    print("\nLlama-2-Chat format:")
    print(chat_formatter.format_for_llama2_chat(base_prompt))
    
    print("\n" + "-"*80)
    print("\nQwen-Chat format:")
    print(chat_formatter.format_for_qwen_chat(base_prompt))
    
    print("\n" + "-"*80)
    print("\nPhi-2 format:")
    print(chat_formatter.format_for_phi2(base_prompt))
    
    print("\n" + "-"*80)
    print("\nGeneric chat format:")
    print(chat_formatter.format_for_generic_chat(base_prompt))