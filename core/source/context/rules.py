"""
Rules - Rule Management Module
Manages positive/negative rules in the RAG system

Responsibilities:
- Store and manage rules (Rule)
- Integrate with RAG Kernel to support vector retrieval
- Rule persistence and loading
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import time


class RuleType(Enum):
    """Rule Type"""
    POSITIVE = "positive"  # Positive rule (extracted from Best Answer)
    NEGATIVE = "negative"  # Negative rule (extracted from Worst Answer)


@dataclass
class Rule:
    """
    Rule Data Structure
    
    Stores rules generated from the Summary Module
    """
    rule_id: str                    # Rule ID
    rule_type: RuleType             # Rule Type
    content: str                    # Rule Content
    question: str                   # Source Question
    answer: str                     # Source Answer
    source_role: str                # Role that generated the answer
    ppl: float                      # Corresponding PPL
    question_index: int             # Question Index
    similarity: Optional[float] = None  # Similarity during retrieval (RAG)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "rule_id": self.rule_id,
            "rule_type": self.rule_type.value,
            "content": self.content,
            "question": self.question,
            "answer": self.answer[:200] + "..." if len(self.answer) > 200 else self.answer,
            "source_role": self.source_role,
            "ppl": self.ppl,
            "question_index": self.question_index,
            "similarity": self.similarity,
            "timestamp": self.timestamp
        }
    
    def format_for_injection(self) -> str:
        """Format for injection"""
        prefix = "✓" if self.rule_type == RuleType.POSITIVE else "✗"
        return f"{prefix} {self.content}"
    
    def estimate_tokens(self) -> int:
        """Estimate token count"""
        text = self.format_for_injection()
        return max(1, len(text) // 3)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Rule':
        """Create rule from dictionary"""
        return cls(
            rule_id=data["rule_id"],
            rule_type=RuleType(data["rule_type"]),
            content=data["content"],
            question=data.get("question", ""),
            answer=data.get("answer", ""),
            source_role=data.get("source_role", "unknown"),
            ppl=data.get("ppl", 0.0),
            question_index=data.get("question_index", -1),
            similarity=data.get("similarity"),
            timestamp=data.get("timestamp", time.time())
        )


class RuleStore:
    """
    Rule Store
    
    Manages storage and retrieval of all rules
    Supports LIFO (Traditional Mode) and RAG (Semantic Retrieval Mode)
    """
    
    def __init__(self, max_rules: int = 100):
        """
        Initialize rule store
        
        Args:
            max_rules: Maximum number of rules
        """
        self.max_rules = max_rules
        self._rules: List[Rule] = []
        self._rule_counter = 0
    
    @property
    def positive_rules(self) -> List[Rule]:
        """Get all positive rules"""
        return [r for r in self._rules if r.rule_type == RuleType.POSITIVE]
    
    @property
    def negative_rules(self) -> List[Rule]:
        """Get all negative rules"""
        return [r for r in self._rules if r.rule_type == RuleType.NEGATIVE]
    
    def add_rule(self, rule: Rule) -> None:
        """
        Add a rule
        
        Args:
            rule: Rule to add
        """
        self._rules.append(rule)
        
        # Remove earliest rule if max count is exceeded
        if len(self._rules) > self.max_rules:
            self._rules.pop(0)
    
    def create_positive_rule(
        self,
        content: str,
        question: str,
        answer: str,
        source_role: str,
        ppl: float,
        question_index: int
    ) -> Rule:
        """Create and add a positive rule"""
        self._rule_counter += 1
        rule = Rule(
            rule_id=f"pos_{self._rule_counter}",
            rule_type=RuleType.POSITIVE,
            content=content,
            question=question,
            answer=answer,
            source_role=source_role,
            ppl=ppl,
            question_index=question_index
        )
        self.add_rule(rule)
        return rule
    
    def create_negative_rule(
        self,
        content: str,
        question: str,
        answer: str,
        source_role: str,
        ppl: float,
        question_index: int
    ) -> Rule:
        """Create and add a negative rule"""
        self._rule_counter += 1
        rule = Rule(
            rule_id=f"neg_{self._rule_counter}",
            rule_type=RuleType.NEGATIVE,
            content=content,
            question=question,
            answer=answer,
            source_role=source_role,
            ppl=ppl,
            question_index=question_index
        )
        self.add_rule(rule)
        return rule
    
    def get_rules_lifo(
        self,
        max_tokens: int,
        include_positive: bool = True,
        include_negative: bool = True
    ) -> Tuple[List[Rule], int]:
        """
        Get rules in LIFO order (backward compatibility)
        
        Start from the latest rules and backtrack until token limit is reached
        
        Args:
            max_tokens: Maximum token count
            include_positive: Whether to include positive rules
            include_negative: Whether to include negative rules
            
        Returns:
            (List of selected rules, used token count)
        """
        selected: List[Rule] = []
        total_tokens = 0
        
        # Traverse in reverse (LIFO)
        for rule in reversed(self._rules):
            if rule.rule_type == RuleType.POSITIVE and not include_positive:
                continue
            if rule.rule_type == RuleType.NEGATIVE and not include_negative:
                continue
            
            rule_tokens = rule.estimate_tokens()
            if total_tokens + rule_tokens > max_tokens:
                break
            
            selected.append(rule)
            total_tokens += rule_tokens
        
        # Reverse to maintain original order (latest last)
        selected.reverse()
        
        return selected, total_tokens
    
    def clear(self) -> None:
        """Clear rule store"""
        self._rules.clear()
        self._rule_counter = 0
    
    def export(self) -> List[Dict]:
        """Export all rules"""
        return [r.to_dict() for r in self._rules]
    
    def save_to_file(self, filepath: str) -> None:
        """Save to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.export(), f, ensure_ascii=False, indent=2)
    
    def load_from_file(self, filepath: str) -> None:
        """Load from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._rules = [Rule.from_dict(d) for d in data]
        if self._rules:
            # Restore counter
            max_id = max(
                int(r.rule_id.split('_')[1]) 
                for r in self._rules 
                if '_' in r.rule_id
            )
            self._rule_counter = max_id
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        role_distribution = {}
        for rule in self._rules:
            role = rule.source_role
            role_distribution[role] = role_distribution.get(role, 0) + 1
        
        return {
            "total_rules": len(self._rules),
            "positive_rules": len(self.positive_rules),
            "negative_rules": len(self.negative_rules),
            "role_distribution": role_distribution
        }
    
    def __len__(self) -> int:
        return len(self._rules)
