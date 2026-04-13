"""
Context Manager
Responsible for managing the rule base and context injection.

Core Features:
1. Manage Rule Store
2. Support LIFO (Traditional Mode) and RAG (Semantic Retrieval Mode) injection strategies
3. Coordinate RAG Kernel for semantic retrieval
4. Only the Teacher can access the rule base (TA does not inject rules)
"""

from typing import List, Dict, Optional, Tuple, Any, TYPE_CHECKING
from dataclasses import dataclass
import json
import logging

from .rules import Rule, RuleType, RuleStore

if TYPE_CHECKING:
    from ..rag.rag_kernel import RAGKernel

logger = logging.getLogger(__name__)

@dataclass
class QAPair:
    """QA Pair (Retained for history compatibility)"""
    question: str
    answer: str
    ppl: float
    role_id: str
    index: int
    
    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "ppl": self.ppl,
            "role_id": self.role_id,
            "index": self.index
        }
    
    def format_for_context(self) -> str:
        return f"Q: {self.question}\nA: {self.answer}"
    
    def estimate_tokens(self) -> int:
        text = self.format_for_context()
        return max(1, len(text) // 3)


class ContextManager:
    """
    Context Manager
    
    Responsibilities:
    1. Manage Rule Store (Rule Storage)
    2. Support two injection modes:
       - LIFO: Injects the most recent rules chronologically
       - RAG: Retrieves relevant rules based on semantic similarity
    3. Access control (Only Teacher and Student can access the rule base)
    4. Coordinate RAG Kernel
    """
    
    def __init__(
        self,
        max_tokens: int = 8192,
        system_prompt_tokens: int = 500,
        question_buffer_tokens: int = 300,
        max_rules: int = 100,
        rag_kernel: Optional["RAGKernel"] = None,
        use_rag: bool = False,
        base_student_prompt: str = ""
    ):
        """
        Initialize Context Manager
        
        Args:
            max_tokens: Total token limit
            system_prompt_tokens: Reserved tokens for base system prompt
            question_buffer_tokens: Reserved tokens for the current question
            max_rules: Maximum number of rules
            rag_kernel: RAG Kernel (optional, required if RAG mode is enabled)
            use_rag: Whether to use RAG mode
            base_student_prompt: Base student prompt
        """
        self.max_tokens = max_tokens
        self.system_prompt_tokens = system_prompt_tokens
        self.question_buffer_tokens = question_buffer_tokens
        self.base_student_prompt = base_student_prompt
        
        # Rule storage
        self.rule_store = RuleStore(max_rules=max_rules)
        
        # RAG Kernel
        self.rag_kernel = rag_kernel
        self.use_rag = use_rag and rag_kernel is not None
        
        # History QA pairs (retained for other purposes)
        self.history: List[QAPair] = []
        
        if self.use_rag:
            logger.info("ContextManager: RAG mode enabled")
        else:
            logger.info("ContextManager: LIFO mode")
    
    @property
    def available_rule_tokens(self) -> int:
        """Token count available for rule injection"""
        return self.max_tokens - self.system_prompt_tokens - self.question_buffer_tokens
    
    @property
    def rule_library(self) -> RuleStore:
        """Backward compatibility: returns rule store"""
        return self.rule_store
    
    # ========== Rule Management ==========
    
    def add_positive_rule(
        self,
        content: str,
        question: str,
        answer: str,
        source_role: str,
        ppl: float,
        question_index: int
    ) -> Rule:
        """
        Add a positive rule
        
        Adds to both RuleStore and RAG Kernel (if enabled)
        """
        # Add to rule store
        rule = self.rule_store.create_positive_rule(
            content=content,
            question=question,
            answer=answer,
            source_role=source_role,
            ppl=ppl,
            question_index=question_index
        )
        
        # Sync to RAG Kernel
        if self.rag_kernel:
            self.rag_kernel.add_positive_rule(
                content=content,
                metadata={
                    "question": question,
                    "answer": answer[:200],
                    "source_role": source_role,
                    "ppl": ppl,
                    "question_index": question_index
                }
            )
        
        logger.debug(f"Added positive rule: {rule.rule_id}")
        return rule
    
    def add_negative_rule(
        self,
        content: str,
        question: str,
        answer: str,
        source_role: str,
        ppl: float,
        question_index: int
    ) -> Rule:
        """
        Add a negative rule
        
        Adds to both RuleStore and RAG Kernel (if enabled)
        """
        # Add to rule store
        rule = self.rule_store.create_negative_rule(
            content=content,
            question=question,
            answer=answer,
            source_role=source_role,
            ppl=ppl,
            question_index=question_index
        )
        
        # Sync to RAG Kernel
        if self.rag_kernel:
            self.rag_kernel.add_negative_rule(
                content=content,
                metadata={
                    "question": question,
                    "answer": answer[:200],
                    "source_role": source_role,
                    "ppl": ppl,
                    "question_index": question_index
                }
            )
        
        logger.debug(f"Added negative rule: {rule.rule_id}")
        return rule
    
    # ========== System Prompt Construction ==========
    
    def build_system_prompt_with_rules(
        self,
        base_system_prompt: str,
        role: str = "teacher",
        max_rule_tokens: Optional[int] = None,
        current_question: Optional[str] = None
    ) -> Tuple[str, int, int]:
        """
        Build system_prompt including rules
        
        Args:
            base_system_prompt: Base system prompt
            role: Role type ("teacher", "student", "ta")
            max_rule_tokens: Optional custom token limit for rules
            current_question: Current question (used for retrieval in RAG mode)
            
        Returns:
            (enhanced_system_prompt, rule_count, estimated_tokens) tuple
        """
        # TA does not inject rules
        if role == "ta":
            return base_system_prompt, 0, 0
        
        # Student does not inject rules via this method (Student has its own prompt builder)
        if role == "student":
            return base_system_prompt, 0, 0
        
        # Check if rule store is empty
        if len(self.rule_store) == 0:
            return base_system_prompt, 0, 0
        
        max_tokens = max_rule_tokens or self.available_rule_tokens
        
        # Get rules (select based on mode)
        if self.use_rag and current_question:
            rules, total_tokens = self._get_rules_rag(
                current_question, max_tokens
            )
        else:
            rules, total_tokens = self.rule_store.get_rules_lifo(max_tokens)
        
        if not rules:
            return base_system_prompt, 0, 0
        
        # Build enhanced system_prompt
        rules_section = self._format_rules_section(rules)
        enhanced_prompt = f"{base_system_prompt}\n\n{rules_section}"
        
        return enhanced_prompt, len(rules), total_tokens
    
    def build_student_system_prompt(
        self,
        max_rule_tokens: Optional[int] = None,
        current_question: Optional[str] = None
    ) -> Tuple[str, int, int]:
        """
        Build Student's system_prompt
        
        Student has no preset System Prompt and relies entirely on injected rule context.
        
        Args:
            max_rule_tokens: Optional token limit for rules
            current_question: Current question (used for retrieval in RAG mode)
            
        Returns:
            (system_prompt, rule_count, estimated_tokens) tuple
        """
        if len(self.rule_store) == 0:
            return self.base_student_prompt, 0, 0
        
        max_tokens = max_rule_tokens or self.available_rule_tokens
        
        # Get rules
        if self.use_rag and current_question:
            rules, total_tokens = self._get_rules_rag(
                current_question, max_tokens
            )
        else:
            rules, total_tokens = self.rule_store.get_rules_lifo(max_tokens)
        
        if not rules:
            return self.base_student_prompt, 0, 0
        
        # Build pure rule system_prompt
        parts = [
            "You are a helpful assistant solving problems.\n",
            self.base_student_prompt,
            "Apply the following learned rules when solving problems:",
            ""
        ]
        
        positive_rules = [r for r in rules if r.rule_type == RuleType.POSITIVE]
        negative_rules = [r for r in rules if r.rule_type == RuleType.NEGATIVE]
        
        if positive_rules:
            parts.append("[DO - Best practices]")
            for rule in positive_rules:
                parts.append(f"✓ {rule.content}")
            parts.append("")
        
        if negative_rules:
            parts.append("[DON'T - Common mistakes to avoid]")
            for rule in negative_rules:
                parts.append(f"✗ {rule.content}")
        
        return "\n".join(parts), len(rules), total_tokens
    
    def _get_rules_rag(
        self,
        question: str,
        max_tokens: int
    ) -> Tuple[List[Rule], int]:
        """
        Retrieve rules using RAG
        
        Args:
            question: Current question
            max_tokens: Maximum tokens
            
        Returns:
            (List of rules, estimated token count)
        """
        if not self.rag_kernel:
            return self.rule_store.get_rules_lifo(max_tokens)
        
        # Retrieve using RAG Kernel
        positive_results, negative_results = self.rag_kernel.retrieve(question)
        
        # Convert to Rule objects and merge alternatingly
        rules = []
        total_tokens = 0
        
        pos_idx, neg_idx = 0, 0
        while (pos_idx < len(positive_results) or neg_idx < len(negative_results)):
            if total_tokens >= max_tokens:
                break
            
            # Add positive rule
            if pos_idx < len(positive_results):
                result = positive_results[pos_idx]
                rule = Rule(
                    rule_id=result.rule_id,
                    rule_type=RuleType.POSITIVE,
                    content=result.content,
                    question=result.metadata.get("question", ""),
                    answer=result.metadata.get("answer", ""),
                    source_role=result.metadata.get("source_role", ""),
                    ppl=result.metadata.get("ppl", 0.0),
                    question_index=result.metadata.get("question_index", -1),
                    similarity=result.similarity
                )
                
                rule_tokens = rule.estimate_tokens()
                if total_tokens + rule_tokens <= max_tokens:
                    rules.append(rule)
                    total_tokens += rule_tokens
                pos_idx += 1
            
            # Add negative rule
            if neg_idx < len(negative_results):
                result = negative_results[neg_idx]
                rule = Rule(
                    rule_id=result.rule_id,
                    rule_type=RuleType.NEGATIVE,
                    content=result.content,
                    question=result.metadata.get("question", ""),
                    answer=result.metadata.get("answer", ""),
                    source_role=result.metadata.get("source_role", ""),
                    ppl=result.metadata.get("ppl", 0.0),
                    question_index=result.metadata.get("question_index", -1),
                    similarity=result.similarity
                )
                
                rule_tokens = rule.estimate_tokens()
                if total_tokens + rule_tokens <= max_tokens:
                    rules.append(rule)
                    total_tokens += rule_tokens
                neg_idx += 1
        
        logger.debug(f"RAG retrieved rules: {len(rules)}, ~{total_tokens} tokens")
        return rules, total_tokens
    
    def _format_rules_section(self, rules: List[Rule]) -> str:
        """Format the rules section"""
        parts = ["<BEGIN_RULES>"]
        parts.append("LEARNED RULES (from previous problem-solving experience)")
        parts.append("Apply these rules when solving similar problems:")
        
        positive_rules = [r for r in rules if r.rule_type == RuleType.POSITIVE]
        negative_rules = [r for r in rules if r.rule_type == RuleType.NEGATIVE]
        
        if positive_rules:
            parts.append("\n[POSITIVE PATTERNS - What works well]")
            for i, rule in enumerate(positive_rules, 1):
                sim_info = f" (sim={rule.similarity:.3f})" if rule.similarity else ""
                parts.append(f"  {i}. {rule.content}{sim_info}")
        
        if negative_rules:
            parts.append("\n[NEGATIVE PATTERNS - What to avoid]")
            for i, rule in enumerate(negative_rules, 1):
                sim_info = f" (sim={rule.similarity:.3f})" if rule.similarity else ""
                parts.append(f"  {i}. {rule.content}{sim_info}")
        
        parts.append("\n" + "<END_RULES>")
        
        return "\n".join(parts)
    
    # ========== QA History Management (Retained) ==========
    
    def add_qa_pair(self, qa_pair: QAPair) -> None:
        """Add QA pair to history"""
        self.history.append(qa_pair)
    
    def get_context_for_ta(self, max_examples: int = 3) -> str:
        """Get context for TA (question style reference only)"""
        if not self.history:
            return ""
        
        recent_pairs = self.history[-max_examples:]
        
        parts = ["Previous problem-answer pairs for reference:"]
        for i, qa_pair in enumerate(recent_pairs, 1):
            parts.append(f"\n[{i}] {qa_pair.format_for_context()}")
        
        return "\n".join(parts)
    
    # ========== Statistics and Export ==========
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        stats = self.rule_store.get_stats()
        stats.update({
            "total_qa_pairs": len(self.history),
            "max_tokens": self.max_tokens,
            "available_rule_tokens": self.available_rule_tokens,
            "use_rag": self.use_rag
        })
        
        if self.rag_kernel:
            stats["rag_kernel_stats"] = self.rag_kernel.get_stats()
        
        return stats
    
    def get_role_distribution(self) -> Dict[str, int]:
        """Get distribution of rules by source role"""
        return self.rule_store.get_stats().get("role_distribution", {})
    
    def export_rules(self) -> List[Dict]:
        """Export rule base"""
        return self.rule_store.export()
    
    def export_history(self) -> List[Dict]:
        """Export historical QA pairs"""
        return [qa.to_dict() for qa in self.history]
    
    def clear(self) -> None:
        """Clear all data"""
        self.rule_store.clear()
        self.history.clear()
        if self.rag_kernel:
            self.rag_kernel.clear()
    
    def save_to_file(self, filepath: str) -> None:
        """Save to file"""
        data = {
            "rules": self.export_rules(),
            "history": self.export_history(),
            "stats": self.get_stats()
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ========== Backward Compatibility ==========

# Alias for keeping compatibility with existing code
RuleLibrary = RuleStore
