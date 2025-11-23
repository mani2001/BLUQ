"""
Prompt Builder Module
Builds prompts for different strategies to evaluate LLMs.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

from src.data.dataset_loader import DataInstance

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """Template for building prompts."""
    name: str
    instruction: Optional[str] = None  # Task instruction
    format_string: str = ""  # How to format the prompt
    include_context: bool = False
    include_demonstrations: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'instruction': self.instruction,
            'format_string': self.format_string,
            'include_context': self.include_context,
            'include_demonstrations': self.include_demonstrations
        }


class BasePromptStrategy(ABC):
    """Abstract base class for prompting strategies."""
    
    def __init__(self, task_type: str):
        """
        Initialize prompt strategy.
        
        Args:
            task_type: Type of task ('qa', 'rc', 'ci', 'drs', 'ds')
        """
        self.task_type = task_type
        self.template = self._get_template()
    
    @abstractmethod
    def _get_template(self) -> PromptTemplate:
        """Get the prompt template for this strategy."""
        pass
    
    @abstractmethod
    def build_prompt(
        self,
        instance: DataInstance,
        demonstrations: Optional[List[DataInstance]] = None
    ) -> str:
        """
        Build a prompt for an instance.
        
        Args:
            instance: DataInstance to create prompt for
            demonstrations: Optional demonstration examples
            
        Returns:
            Formatted prompt string
        """
        pass
    
    def _format_options(self, options: List[str]) -> str:
        """Format options list."""
        formatted = "Choices:\n"
        for i, option in enumerate(options):
            letter = chr(65 + i)  # A, B, C, D, E, F
            formatted += f"{letter}. {option}\n"
        return formatted
    
    def _format_context(
        self,
        instance: DataInstance,
        context_key: str
    ) -> Optional[str]:
        """Format context/dialogue/document."""
        if context_key == 'context':
            return instance.context
        elif context_key == 'dialogue':
            return instance.dialogue
        elif context_key == 'document':
            return instance.document
        return None
    
    def _format_demonstration(
        self,
        instance: DataInstance,
        include_answer: bool = True
    ) -> str:
        """Format a demonstration example."""
        # Build demonstration in same format as main question
        # but include the answer
        demo = self.build_prompt(instance, demonstrations=None)
        
        if include_answer:
            demo += f"{instance.answer}\n"
        
        return demo


class BasePromptBuilder(BasePromptStrategy):
    """
    Base prompting strategy.
    
    Directly combines question and options with minimal formatting.
    No task instruction, just question -> answer format.
    """
    
    def _get_template(self) -> PromptTemplate:
        """Get base prompt template."""
        return PromptTemplate(
            name="base",
            instruction=None,
            include_context=True,
            include_demonstrations=True
        )
    
    def build_prompt(
        self,
        instance: DataInstance,
        demonstrations: Optional[List[DataInstance]] = None
    ) -> str:
        """Build base prompt."""
        prompt_parts = []
        
        # Add demonstrations if provided
        if demonstrations:
            for demo in demonstrations:
                demo_prompt = self._format_demonstration(demo, include_answer=True)
                prompt_parts.append(demo_prompt)
        
        # Add context if present
        context = self._get_context(instance)
        if context:
            context_label = self._get_context_label(instance)
            prompt_parts.append(f"{context_label}: {context}")
        
        # Add question
        prompt_parts.append(f"Question: {instance.question}")
        
        # Add options
        prompt_parts.append(self._format_options(instance.options))
        
        # Add answer prefix
        prompt_parts.append("Answer:")
        
        return "\n".join(prompt_parts)
    
    def _get_context(self, instance: DataInstance) -> Optional[str]:
        """Get context from instance based on task type."""
        if instance.context:
            return instance.context
        elif instance.dialogue:
            return instance.dialogue
        elif instance.document:
            return instance.document
        return None
    
    def _get_context_label(self, instance: DataInstance) -> str:
        """Get label for context based on what's present."""
        if instance.context:
            return "Context"
        elif instance.dialogue:
            return "Dialogue"
        elif instance.document:
            return "Document"
        return "Context"


class SharedInstructionPromptBuilder(BasePromptStrategy):
    """
    Shared instruction prompting strategy.
    
    Adds a general task description that applies to all tasks.
    Informs the model that it's solving a multiple-choice question.
    """
    
    def _get_template(self) -> PromptTemplate:
        """Get shared instruction template."""
        instruction = (
            "Below are some examples of multiple-choice questions with six potential "
            "answers. For each question, only one option is correct."
        )
        
        return PromptTemplate(
            name="shared_instruction",
            instruction=instruction,
            include_context=True,
            include_demonstrations=True
        )
    
    def build_prompt(
        self,
        instance: DataInstance,
        demonstrations: Optional[List[DataInstance]] = None
    ) -> str:
        """Build prompt with shared instruction."""
        prompt_parts = []
        
        # Add shared instruction (only for the main question, not in demos)
        if not demonstrations or len(demonstrations) == 0:
            # This is the main question
            prompt_parts.append(self.template.instruction)
            prompt_parts.append("")
        
        # Add demonstrations if provided
        if demonstrations:
            for demo in demonstrations:
                # Build demo without instruction
                base_builder = BasePromptBuilder(self.task_type)
                demo_prompt = base_builder.build_prompt(demo, demonstrations=None)
                demo_prompt += f"{demo.answer}\n"
                prompt_parts.append(demo_prompt)
        
        # Add instruction for the actual question
        if demonstrations:
            prompt_parts.append(
                "Now make your best effort and select the correct answer for the "
                "following question. You only need to output the option."
            )
            prompt_parts.append("")
        
        # Add context if present
        context = self._get_context(instance)
        if context:
            context_label = self._get_context_label(instance)
            prompt_parts.append(f"{context_label}: {context}")
        
        # Add question
        prompt_parts.append(f"Question: {instance.question}")
        
        # Add options
        prompt_parts.append(self._format_options(instance.options))
        
        # Add answer prefix
        prompt_parts.append("Answer:")
        
        return "\n".join(prompt_parts)
    
    def _get_context(self, instance: DataInstance) -> Optional[str]:
        """Get context from instance."""
        if instance.context:
            return instance.context
        elif instance.dialogue:
            return instance.dialogue
        elif instance.document:
            return instance.document
        return None
    
    def _get_context_label(self, instance: DataInstance) -> str:
        """Get label for context."""
        if instance.context:
            return "Context"
        elif instance.dialogue:
            return "Dialogue"
        elif instance.document:
            return "Document"
        return "Context"


class TaskSpecificInstructionPromptBuilder(BasePromptStrategy):
    """
    Task-specific instruction prompting strategy.
    
    Provides tailored instructions for each task type.
    """
    
    def _get_template(self) -> PromptTemplate:
        """Get task-specific template."""
        instructions = {
            'qa': (
                "Below are some examples of multiple-choice questions about question answering. "
                "Each question should be answered based on your world knowledge and problem solving ability."
            ),
            'rc': (
                "Below are some examples of multiple-choice questions about reading comprehension. "
                "Each question should be answered based on the given context and commonsense reasoning when necessary."
            ),
            'ci': (
                "Below are some examples of multiple-choice questions about commonsense natural language inference. "
                "For each question, there is a given context and the answer is the option that most likely follows the context."
            ),
            'drs': (
                "Below are some examples of multiple-choice questions about dialogue response selection. "
                "For each question, the answer is the option that represents the most suitable response for the given "
                "dialogue history, without hallucination and non-factual information."
            ),
            'ds': (
                "Below are some examples of multiple-choice questions about document summarization. "
                "For each question, the answer is the option that accurately summarizes the given document without "
                "hallucination and non-factual information."
            )
        }
        
        return PromptTemplate(
            name="task_specific",
            instruction=instructions.get(self.task_type, instructions['qa']),
            include_context=True,
            include_demonstrations=True
        )
    
    def build_prompt(
        self,
        instance: DataInstance,
        demonstrations: Optional[List[DataInstance]] = None
    ) -> str:
        """Build prompt with task-specific instruction."""
        prompt_parts = []
        
        # Add task-specific instruction (only for main question)
        if not demonstrations or len(demonstrations) == 0:
            prompt_parts.append(self.template.instruction)
            prompt_parts.append("")
        
        # Add demonstrations
        if demonstrations:
            for demo in demonstrations:
                base_builder = BasePromptBuilder(self.task_type)
                demo_prompt = base_builder.build_prompt(demo, demonstrations=None)
                demo_prompt += f"{demo.answer}\n"
                prompt_parts.append(demo_prompt)
        
        # Add instruction for actual question
        if demonstrations:
            prompt_parts.append(
                "Now make your best effort and select the correct answer for the "
                "following question. You only need to output the option."
            )
            prompt_parts.append("")
        
        # Add context if present
        context = self._get_context(instance)
        if context:
            context_label = self._get_context_label(instance)
            prompt_parts.append(f"{context_label}: {context}")
        
        # Add question
        prompt_parts.append(f"Question: {instance.question}")
        
        # Add options
        prompt_parts.append(self._format_options(instance.options))
        
        # Add answer prefix
        prompt_parts.append("Answer:")
        
        return "\n".join(prompt_parts)
    
    def _get_context(self, instance: DataInstance) -> Optional[str]:
        """Get context from instance."""
        if instance.context:
            return instance.context
        elif instance.dialogue:
            return instance.dialogue
        elif instance.document:
            return instance.document
        return None
    
    def _get_context_label(self, instance: DataInstance) -> str:
        """Get appropriate label for context type."""
        if instance.context:
            return "Context"
        elif instance.dialogue:
            return "Dialogue"
        elif instance.document:
            return "Document"
        return "Context"


class PromptBuilder:
    """Main builder that supports multiple prompting strategies."""
    
    STRATEGIES = {
        'base': BasePromptBuilder,
        'shared_instruction': SharedInstructionPromptBuilder,
        'task_specific': TaskSpecificInstructionPromptBuilder
    }
    
    def __init__(self, task_type: str):
        """
        Initialize prompt builder.
        
        Args:
            task_type: Type of task ('qa', 'rc', 'ci', 'drs', 'ds')
        """
        self.task_type = task_type
        
        # Initialize all strategies
        self.strategies = {}
        for name, strategy_class in self.STRATEGIES.items():
            self.strategies[name] = strategy_class(task_type)
        
        logger.info(f"Initialized PromptBuilder for task: {task_type}")
        logger.info(f"  Available strategies: {list(self.strategies.keys())}")
    
    def build_prompt(
        self,
        instance: DataInstance,
        strategy: str,
        demonstrations: Optional[List[DataInstance]] = None
    ) -> str:
        """
        Build a prompt using specified strategy.
        
        Args:
            instance: DataInstance to create prompt for
            strategy: Strategy name ('base', 'shared_instruction', 'task_specific')
            demonstrations: Optional demonstration examples
            
        Returns:
            Formatted prompt string
        """
        if strategy not in self.strategies:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Available: {list(self.strategies.keys())}"
            )
        
        strategy_builder = self.strategies[strategy]
        prompt = strategy_builder.build_prompt(instance, demonstrations)
        
        return prompt
    
    def build_all_prompts(
        self,
        instances: List[DataInstance],
        strategy: str,
        demonstrations: Optional[List[DataInstance]] = None
    ) -> List[str]:
        """
        Build prompts for a list of instances.
        
        Args:
            instances: List of DataInstance objects
            strategy: Strategy name
            demonstrations: Demonstration examples
            
        Returns:
            List of formatted prompts
        """
        logger.info(
            f"Building {len(instances)} prompts using '{strategy}' strategy "
            f"with {len(demonstrations) if demonstrations else 0} demonstrations"
        )
        
        prompts = []
        for instance in instances:
            prompt = self.build_prompt(instance, strategy, demonstrations)
            prompts.append(prompt)
        
        return prompts
    
    def build_multi_strategy(
        self,
        instances: List[DataInstance],
        strategies: List[str],
        demonstrations_per_strategy: Optional[Dict[str, List[DataInstance]]] = None
    ) -> Dict[str, List[str]]:
        """
        Build prompts using multiple strategies.
        
        Args:
            instances: List of DataInstance objects
            strategies: List of strategy names
            demonstrations_per_strategy: Dict mapping strategy to demonstrations
            
        Returns:
            Dictionary mapping strategy names to lists of prompts
        """
        logger.info(f"Building prompts for {len(strategies)} strategies...")
        
        all_prompts = {}
        
        for strategy in strategies:
            demonstrations = None
            if demonstrations_per_strategy:
                demonstrations = demonstrations_per_strategy.get(strategy)
            
            prompts = self.build_all_prompts(instances, strategy, demonstrations)
            all_prompts[strategy] = prompts
            
            logger.info(f"  {strategy}: {len(prompts)} prompts")
        
        return all_prompts
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy names."""
        return list(self.strategies.keys())


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
    print("Prompt Builder Test")
    print("="*80)
    
    # Create sample instances
    qa_instance = DataInstance(
        id="qa_1",
        question="What is the capital of France?",
        options=["London", "Paris", "Berlin", "Madrid", "I don't know", "None of the above"],
        answer="B"
    )
    
    rc_instance = DataInstance(
        id="rc_1",
        question="What did John do after breakfast?",
        options=[
            "Went to work",
            "Stayed home",
            "Visited a friend",
            "Went shopping",
            "I don't know",
            "None of the above"
        ],
        answer="A",
        context="John woke up early and had a quick breakfast. He was excited about his new job and left the house at 8 AM."
    )
    
    demo_instance = DataInstance(
        id="demo_1",
        question="What color is the sky?",
        options=["Red", "Blue", "Green", "Yellow", "I don't know", "None of the above"],
        answer="B"
    )
    
    # Test QA prompts
    print("\n" + "="*80)
    print("Testing QA Prompts")
    print("="*80)
    
    qa_builder = PromptBuilder(task_type='qa')
    
    for strategy in ['base', 'shared_instruction', 'task_specific']:
        print(f"\n{'-'*80}")
        print(f"Strategy: {strategy}")
        print(f"{'-'*80}")
        
        prompt = qa_builder.build_prompt(
            qa_instance,
            strategy=strategy,
            demonstrations=[demo_instance]
        )
        
        print(prompt)
    
    # Test RC prompts
    print("\n" + "="*80)
    print("Testing RC Prompts (with context)")
    print("="*80)
    
    rc_builder = PromptBuilder(task_type='rc')
    
    prompt = rc_builder.build_prompt(
        rc_instance,
        strategy='task_specific',
        demonstrations=None
    )
    
    print(prompt)
    
    # Test batch building
    print("\n" + "="*80)
    print("Testing batch prompt building")
    print("="*80)
    
    instances = [qa_instance, rc_instance]
    
    multi_prompts = qa_builder.build_multi_strategy(
        instances=instances,
        strategies=['base', 'shared_instruction'],
        demonstrations_per_strategy={
            'base': [demo_instance],
            'shared_instruction': [demo_instance]
        }
    )
    
    print(f"\nGenerated prompts for {len(instances)} instances")
    print(f"Strategies: {list(multi_prompts.keys())}")
    for strategy, prompts in multi_prompts.items():
        print(f"  {strategy}: {len(prompts)} prompts")