# Compare Module
"""
Comparison and Selection Module

Contains:
- SelectionKernel: Selection kernel, selects Best/Worst based on consensus and PPL
"""

from .selection_kernel import (
    SelectionKernel,
    SelectionResult,
    SelectionCandidate
)

__all__ = [
    'SelectionKernel',
    'SelectionResult',
    'SelectionCandidate',
]

