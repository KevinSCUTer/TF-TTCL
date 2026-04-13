"""
Actors - Teacher/TA/Student Role System
Defines the configurations and behaviors of Teacher, TA, and Student models.

Architecture Overview:
- Teacher: Generates stable baseline answers (temp=0) with a carefully designed System Prompt.
- TA (Teaching Assistant): Generates question variants to increase diversity.
- Students: Generate exploratory answers based on TA-transformed questions, with no preset System Prompt.
"""

import re
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

from .utils.llm_client import LLMClient, LLMResponse
from .utils.evaluator import PPLEvaluator, PPLResult

logger = logging.getLogger(__name__)


@dataclass
class ActorConfig:
    """Actor Configuration"""
    role_id: str                          # Role ID
    role_type: str = "base"               # Role type: teacher, ta, student
    temperature: float = 0.0              # Sampling temperature
    top_p: Optional[float] = None         # Top-P sampling
    top_k: Optional[int] = None           # Top-K sampling
    max_tokens: int = 512                 # Maximum generation tokens
    system_prompt: str = ""               # System prompt (Student defaults to empty)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "role_id": self.role_id,
            "role_type": self.role_type,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt[:50] + "..." if len(self.system_prompt) > 50 else self.system_prompt
        }


@dataclass
class GenerationResult:
    """Generation Result"""
    role_id: str                          # Role ID
    content: str                          # Generated content
    ppl_result: PPLResult                 # PPL calculation result
    llm_response: LLMResponse             # Original LLM response
    config: ActorConfig                   # Configuration used
    question_variant: Optional[str] = None # Question variant used (Student only)
    system_prompt: Optional[str] = None    # System prompt used
    
    @property
    def ppl(self) -> float:
        """Shortcut to get PPL value"""
        return self.ppl_result.ppl


class BaseActor:
    """Actor Base Class"""
    
    def __init__(
        self,
        config: ActorConfig,
        llm_client: LLMClient,
        evaluator: PPLEvaluator
    ):
        """
        Initialize Actor
        
        Args:
            config: Actor configuration
            llm_client: LLM client
            evaluator: PPL evaluator
        """
        self.config = config
        self.llm_client = llm_client
        self.evaluator = evaluator
    
    def generate(
        self,
        question: str,
        system_prompt_override: Optional[str] = None
    ) -> GenerationResult:
        """
        Generate response
        
        Args:
            question: Question
            system_prompt_override: Optional system prompt override (for rule injection)
            
        Returns:
            GenerationResult object
        """
        # Use overridden system_prompt or default config
        system_prompt = system_prompt_override if system_prompt_override is not None else self.config.system_prompt
        
        # Build messages
        messages = self.llm_client.build_messages(
            system_prompt=system_prompt,
            user_content=question
        )
        
        # Call LLM
        response = self.llm_client.generate(
            messages=messages,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            max_tokens=self.config.max_tokens,
            logprobs=True
        )
        
        # Calculate PPL
        if not response.logprobs:
            logger.warning(f"[{self.config.role_id}] Warning: No logprobs returned from LLM. PPL will be inf.")
        
        ppl_result = self.evaluator.compute_ppl(response.logprobs)
        
        return GenerationResult(
            role_id=self.config.role_id,
            content=response.content,
            ppl_result=ppl_result,
            llm_response=response,
            config=self.config,
            question_variant=question,
            system_prompt=system_prompt
        )


class Teacher(BaseActor):
    """
    Teacher Model
    - Uses a carefully designed domain-specific System Prompt.
    - Configured with stable sampling parameters (Temperature=0).
    - Serves as a baseline anchor, pursuing stability.
    
    Permissions: [private: system_prompt][public: question][public: answer]
    """
    
    # Default system prompt for Teacher (targetted at mathematical reasoning)
    DEFAULT_SYSTEM_PROMPT = ""

    @classmethod
    def create(
        cls,
        llm_client: LLMClient,
        evaluator: PPLEvaluator,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512
    ) -> 'Teacher':
        """
        Create Teacher instance
        
        Args:
            llm_client: LLM client
            evaluator: PPL evaluator
            system_prompt: System prompt, uses default if None
            max_tokens: Maximum generation tokens
            
        Returns:
            Teacher instance
        """
        config = ActorConfig(
            role_id="teacher",
            role_type="teacher",
            temperature=0.0,  # Stable output
            top_p=None,
            top_k=None,
            max_tokens=max_tokens,
            system_prompt=system_prompt or cls.DEFAULT_SYSTEM_PROMPT
        )
        return cls(config, llm_client, evaluator)
    
    def generate_with_rules(
        self,
        question: str,
        injected_system_prompt: str
    ) -> GenerationResult:
        """
        Generate response using system_prompt with rules injected
        
        Args:
            question: Question
            injected_system_prompt: System prompt after rule injection
            
        Returns:
            GenerationResult object
        """
        return self.generate(question=question, system_prompt_override=injected_system_prompt)


class TeachingAssistant(BaseActor):
    """
    TA (Teaching Assistant) Model
    - Converts the original question into multiple phrase variations.
    - Increases question diversity to help Students improve response quality.
    - Applicable to various open-domain scenarios (law, finance, medicine, science, etc.).
    
    Permissions: [private: system_prompt][public: question][private: verbalized question]
    Note: TA will not be injected with rule context.
    """
    
    VARIANT_COUNT = 4

    # Default system prompt for TA - General open domain
    DEFAULT_SYSTEM_PROMPT = ""
    
    # XML parsing regex patterns
    _RESPONSE_PATTERN = re.compile(
        r'<response>\s*<text>(.*?)</text>\s*<probability>([\d.]+)</probability>\s*</response>',
        re.DOTALL | re.IGNORECASE
    )
    _ALT_RESPONSE_PATTERN = re.compile(
        r'<response>\s*<probability>([\d.]+)</probability>\s*<text>(.*?)</text>\s*</response>',
        re.DOTALL | re.IGNORECASE
    )
    
    def __init__(
        self,
        config: ActorConfig,
        llm_client: LLMClient,
        evaluator: PPLEvaluator,
        verbose: bool = False
    ):
        super().__init__(config, llm_client, evaluator)
        self.verbose = verbose
    
    @classmethod
    def create(
        cls,
        llm_client: LLMClient,
        evaluator: PPLEvaluator,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        verbose: bool = False
    ) -> 'TeachingAssistant':
        """
        Create TA instance
        
        Args:
            llm_client: LLM client
            evaluator: PPL evaluator
            temperature: Sampling temperature
            top_p: Top-P sampling parameter
            system_prompt: System prompt
            max_tokens: Maximum generation tokens
            verbose: Whether to print detailed logs
            
        Returns:
            TeachingAssistant instance
        """
        config = ActorConfig(
            role_id="ta",
            role_type="ta",
            temperature=temperature,
            top_p=top_p,
            top_k=None,
            max_tokens=max_tokens,
            system_prompt=system_prompt or cls.DEFAULT_SYSTEM_PROMPT
        )
        return cls(config, llm_client, evaluator, verbose=verbose)
    
    def generate_question_variants(
        self,
        original_question: str,
        max_variants: int = 4,
        verbose: Optional[bool] = None,
        max_tokens_override: Optional[int] = None,
    ) -> List[str]:
        """
        Generate question variants
        
        Simple and direct workflow:
        1. Call LLM to generate 10 question rewrites.
        2. Parse XML response to extract (text, probability) pairs.
        3. Sort by probability (descending) and select top max_variants.
        4. Fallback: If not enough, supplement with the original question.
        
        Args:
            original_question: Original question
            max_variants: Maximum number of variants (default 4)
            verbose: Whether to print detailed logs
            
        Returns:
            List of question variants
        """
        should_log = verbose if verbose is not None else self.verbose
        
        # Dynamically adjust System Prompt to explicitly request target variant count.
        # If not specified in system_prompt or if more are needed than default,
        # we can append instructions to user_content.
        
        instruction_suffix = f"\n\nPlease generate at least {max_variants + 2} different variations of the above question."
        
        # Build messages and call LLM
        messages = self.llm_client.build_messages(
            system_prompt=self.config.system_prompt,
            user_content=original_question + instruction_suffix
        )
        
        response = self.llm_client.generate(
            messages=messages,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            max_tokens=max_tokens_override if max_tokens_override is not None else self.config.max_tokens,
            logprobs=False
        )
        
        # Parse response
        parsed = self._parse_xml_responses(response.content)
        
        if should_log:
            logger.info(f"[TA] Original question: {original_question[:80]}...")
            logger.info(f"[TA] LLM generated {len(parsed)} variants")
            for i, (text, prob) in enumerate(parsed):
                logger.info(f"[TA] Variant {i+1}: prob={prob:.3f}, text={text[:60]}...")
        
        # Sort by probability descending
        parsed.sort(key=lambda x: x[1], reverse=True)
        
        # Select top max_variants, remove duplicates
        selected: List[str] = []
        seen: set = set()
        for text, prob in parsed:
            cleaned = text.strip()
            if cleaned and cleaned not in seen:
                selected.append(cleaned)
                seen.add(cleaned)
            if len(selected) >= max_variants:
                break
        
        # Fallback: if not enough, supplement with the original question
        original_cleaned = original_question
        selected.append(original_cleaned)
        if len(selected) < max_variants and original_cleaned not in seen:
            selected.append(original_cleaned)
        
        if should_log:
            logger.info(f"[TA] Finally dynamic returned {len(selected)} variants")
        
        # If parsing completely fails, return at least the original question
        if not selected:
            logger.warning(f"[TA] Parsing failed, returning original question")
            return [original_cleaned]
        
        return selected
    
    def _parse_xml_responses(self, content: str) -> List[Tuple[str, float]]:
        """
        Parse XML formatted response
        
        Supports two formats:
        - <response><text>...</text><probability>...</probability></response>
        - <response><probability>...</probability><text>...</text></response>
        
        Args:
            content: Original content generated by LLM
            
        Returns:
            List of [(text, probability), ...]
        """
        results: List[Tuple[str, float]] = []
        
        # Try primary format
        for match in self._RESPONSE_PATTERN.finditer(content):
            text = match.group(1).strip()
            # Safety check: remove boxed
            if '\\boxed' in text:
                logger.warning(f"[TA] Detected boxed answer in variant, removing boxed content: {text[:50]}...")
                text = re.sub(r'\\boxed\{[^}]*\}', '', text)
            
            try:
                prob = float(match.group(2))
                prob = max(0.0, min(1.0, prob))  # Clamped to [0, 1]
            except ValueError:
                prob = 0.5
            if text:
                results.append((text, prob))
        
        # Try alternate format
        for match in self._ALT_RESPONSE_PATTERN.finditer(content):
            text = match.group(2).strip()
            # Safety check: remove boxed
            if '\\boxed' in text:
                logger.warning(f"[TA] Detected boxed answer in variant, removing boxed content: {text[:50]}...")
                text = re.sub(r'\\boxed\{[^}]*\}', '', text)

            try:
                prob = float(match.group(1))
                prob = max(0.0, min(1.0, prob))
            except ValueError:
                prob = 0.5
            if text and text not in [r[0] for r in results]:
                results.append((text, prob))
        
        # Fallback: if XML parsing completely fails, try bulleted list extraction
        if not results:
            results = self._fallback_parse(content)
        
        return results
    
    def _fallback_parse(self, content: str) -> List[Tuple[str, float]]:
        """
        Fallback Parsing: Extract questions from numbered lists
        
        Matches formats like:
        - 1. Question text
        - 1) Question text
        - (1) Question text
        """
        results: List[Tuple[str, float]] = []
        
        # Match numbered lists
        pattern = re.compile(r'^\s*(?:\d+[\.\)]\s*|\(\d+\)\s*)(.+)$', re.MULTILINE)
        for i, match in enumerate(pattern.finditer(content)):
            text = match.group(1).strip()
            # Safety check: remove boxed
            if '\\boxed' in text:
                logger.warning(f"[TA] Detected boxed answer in variant, removing boxed content: {text[:50]}...")
                text = re.sub(r'\\boxed\{[^}]*\}', '', text)

            if text and len(text) > 10:
                # Provide a descending default probability
                prob = max(0.1, 1.0 - i * 0.1)
                results.append((text, prob))
        
        return results


class Student(BaseActor):
    """
    Student Model
    - **No preset System Prompt** (relies on rule context injected by Context Manager).
    - Generates answers based on question variants transformed by TA.
    - Configured with diverse sampling parameters to increase exploration.
    
    Permissions: [context-injected: system_prompt][public: question][public: answer]
    
    Key changes (per tech-plan.md):
    - Student has no preset System Prompt.
    - System Prompt is entirely determined by Context Manager rule injection.
    - Temperature consistently maintained at 0.7.
    """
    
    # Student has no default System Prompt
    DEFAULT_SYSTEM_PROMPT = ""
    
    @classmethod
    def create(
        cls,
        role_id: str,
        llm_client: LLMClient,
        evaluator: PPLEvaluator,
        temperature: float = 0.7,  # Per tech-plan: Temperature maintained at 0.7
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: int = 512
    ) -> 'Student':
        """
        Create Student instance
        
        Note: Student has no preset System Prompt
        
        Args:
            role_id: Role ID
            llm_client: LLM client
            evaluator: PPL evaluator
            temperature: Sampling temperature (default 0.7)
            top_p: Top-P sampling parameter
            top_k: Top-K sampling parameter
            max_tokens: Maximum generation tokens
            
        Returns:
            Student instance
        """
        config = ActorConfig(
            role_id=role_id,
            role_type="student",
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            system_prompt=""  # No preset System Prompt
        )
        return cls(config, llm_client, evaluator)
    
    def generate_with_variant(
        self,
        question_variant: str,
        original_question: str,
        injected_system_prompt: Optional[str] = None
    ) -> GenerationResult:
        """
        Generate answer using question variant
        
        Args:
            question_variant: Question variant generated by TA
            original_question: Original question (for logging)
            injected_system_prompt: System Prompt with rules injected by Context Manager
            
        Returns:
            GenerationResult object
        """
        result = self.generate(
            question=question_variant,
            system_prompt_override=injected_system_prompt or ""
        )
        result.question_variant = question_variant
        return result


class ActorFactory:
    """
    Actor Factory Class
    Used to batch create predefined Teacher, TA, and Students.
    
    Update per tech-plan.md:
    - Students consistently use relatively aggressive hyperparameter configurations.
    - Temperature maintained at 0.7.
    """
    
    # Predefined Student configurations (4 Students)
    # Per technical plan: Temperature at 0.7, using aggressive sampling parameters
    STUDENT_CONFIGS = [
        {"role_id": "student_1", "temperature": 0.7, "top_p": 0.9, "top_k": 50, "purpose": "Standard Exploration"},
        {"role_id": "student_2", "temperature": 0.7, "top_p": 0.9, "top_k": 50, "purpose": "Standard Exploration"},
        {"role_id": "student_3", "temperature": 0.7, "top_p": 0.9, "top_k": 50, "purpose": "Standard Exploration"},
        {"role_id": "student_4", "temperature": 0.7, "top_p": 0.9, "top_k": 50, "purpose": "Standard Exploration"},
    ]
    
    # TA configuration
    TA_CONFIG = {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 1024
    }
    
    @classmethod
    def create_teacher(
        cls,
        llm_client: LLMClient,
        evaluator: PPLEvaluator,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512
    ) -> Teacher:
        """Create Teacher"""
        return Teacher.create(
            llm_client=llm_client,
            evaluator=evaluator,
            system_prompt=system_prompt,
            max_tokens=max_tokens
        )
    
    @classmethod
    def create_ta(
        cls,
        llm_client: LLMClient,
        evaluator: PPLEvaluator,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        verbose: bool = False
    ) -> TeachingAssistant:
        """Create TA (Teaching Assistant)"""
        return TeachingAssistant.create(
            llm_client=llm_client,
            evaluator=evaluator,
            temperature=cls.TA_CONFIG["temperature"],
            top_p=cls.TA_CONFIG["top_p"],
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            verbose=verbose
        )
    
    @classmethod
    def create_students(
        cls,
        llm_client: LLMClient,
        evaluator: PPLEvaluator,
        max_tokens: int = 512,
        count: int = 4
    ) -> List[Student]:
        """
        Create a specified number of Students
        
        Note: Students have no preset System Prompt
        
        Args:
            llm_client: LLM client
            evaluator: PPL evaluator
            max_tokens: Maximum generation tokens
            count: Number of students (default 4)
            
        Returns:
            List of Student instances
        """
        students = []
        for i in range(1, count + 1):
            role_id = f"student_{i}"
            # Use consistent aggressive configuration
            student = Student.create(
                role_id=role_id,
                llm_client=llm_client,
                evaluator=evaluator,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                max_tokens=max_tokens
            )
            students.append(student)
        return students
    
    @classmethod
    def create_all(
        cls,
        llm_client: LLMClient,
        evaluator: PPLEvaluator,
        teacher_prompt: Optional[str] = None,
        ta_prompt: Optional[str] = None,
        max_tokens: int = 512,
        verbose: bool = False,
        student_count: int = 4,
        ta_max_tokens: Optional[int] = None,
    ) -> tuple:
        """
        Create Teacher, TA, and all Students
        
        Note: student_prompt parameter removed as Student has no preset System Prompt
        
        Returns:
            (teacher, ta, students) tuple
        """
        teacher = cls.create_teacher(
            llm_client=llm_client,
            evaluator=evaluator,
            system_prompt=teacher_prompt,
            max_tokens=max_tokens
        )
        ta = cls.create_ta(
            llm_client=llm_client,
            evaluator=evaluator,
            system_prompt=ta_prompt,
            max_tokens=ta_max_tokens if ta_max_tokens is not None else 2048,
            verbose=verbose
        )
        students = cls.create_students(
            llm_client=llm_client,
            evaluator=evaluator,
            max_tokens=max_tokens,
            count=student_count
        )
        return teacher, ta, students
