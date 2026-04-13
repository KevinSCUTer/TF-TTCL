"""
Confuse Mode - Confusion Mode Module

Used to test system robustness and self-healing capability:
- Inject obviously incorrect negative and positive rules
- Track ranking changes of rules during retrieval
- Observe whether the system can "squeeze out" incorrect rules
- Monitor performance recovery from degradation
"""

from .rule_injector import RuleInjector, InjectedRule
from .rule_tracker import RuleTracker, RuleStatus, TrackingSnapshot

__all__ = [
    "RuleInjector",
    "InjectedRule",
    "RuleTracker",
    "RuleStatus",
    "TrackingSnapshot",
]
