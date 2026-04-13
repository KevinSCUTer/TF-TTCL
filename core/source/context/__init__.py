"""
Context Module

Contains:
- ContextManager: Context manager
- Rule: Rule data structure
- RuleType: Rule type enum
- RuleStore: Rule storage
- QAPair: QA pair data structure
"""

from .context_manager import (
    ContextManager,
    RuleLibrary,  # Backward compatibility alias
    QAPair
)
from .rules import (
    Rule,
    RuleType,
    RuleStore
)
from .prompt_loader import (\
    PromptLoader,
    PromptTemplates
)
__all__ = [
    'ContextManager',
    'RuleLibrary',
    'QAPair',
    'Rule',
    'RuleType',
    'RuleStore',
    'PromptLoader',
    'PromptTemplates'
]

