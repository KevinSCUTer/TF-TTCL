# Context Demo Core Module
"""
Core Module Exports

Contains:
- LLMClient: LLM API wrapper
- EmbeddingClient: Embedding API wrapper (semantic similarity calculation)
- PPLEvaluator: Perplexity calculation
- Teacher, TeachingAssistant, Student: Role system
- ContextManager: Context and rule management
- RAGKernel: RAG retrieval kernel
- SelectionKernel: Selection kernel
- SummaryModule: Rule summarization module
- RuleStore: Rule storage
- PromptManager: Prompt management
"""

# Utils
from .utils.llm_client import LLMClient, LLMResponse
from .utils.embedding_client import EmbeddingClient, EmbeddingResult
from .utils.evaluator import PPLEvaluator, PPLResult
from .utils.extract_utils import (
    extract_gsm8k_answer_number,
    is_equiv,
    extract_math_answer,
    extract_logiqa_option
)

# Actors
from .actors import (
    Teacher, 
    TeachingAssistant, 
    Student, 
    ActorConfig, 
    ActorFactory,
    GenerationResult,
    BaseActor
)

# Context
from .context.context_manager import (
    ContextManager,
    RuleLibrary,  # Backward compatibility alias
    QAPair
)
from .context.rules import (
    Rule,
    RuleType,
    RuleStore
)

# RAG
from .rag.rag_kernel import (
    RAGKernel,
    RuleEmbedding,
    RetrievalResult,
    create_rag_kernel
)

# Compare
from .compare.selection_kernel import (
    SelectionKernel,
    SelectionResult,
    SelectionCandidate
)

# Summary
from .summary.summary_module import SummaryModule


__all__ = [
    # LLM
    'LLMClient',
    'LLMResponse',
    
    # Embedding
    'EmbeddingClient',
    'EmbeddingResult',
    
    # Evaluator
    'PPLEvaluator',
    'PPLResult',
    
    # Actors
    'Teacher',
    'TeachingAssistant',
    'Student',
    'ActorConfig',
    'ActorFactory',
    'GenerationResult',
    'BaseActor',
    
    # Context Manager
    'ContextManager',
    'RuleLibrary',  # Backward compatibility
    'QAPair',
    
    # Rules
    'Rule',
    'RuleType',
    'RuleStore',
    
    # RAG
    'RAGKernel',
    'RuleEmbedding',
    'RetrievalResult',
    'create_rag_kernel',
    
    # Selection Kernel
    'SelectionKernel',
    'SelectionResult',
    'SelectionCandidate',
    
    # Summary Module
    'SummaryModule',
        
    # Extract Utils
    'extract_gsm8k_answer_number',
    'is_equiv',
    'extract_math_answer',
    'extract_logiqa_option',
]
