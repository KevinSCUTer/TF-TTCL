"""
Random Kernel
Responsible for random rule extraction and ranking.

Core Features:
1. When the configuration uses ablation: random_rules, Random Kernel disables similarity retrieval 
   and uses a pure extraction strategy, randomly picking rules until the limit is reached or rules are exhausted.
2. The purpose of copying random_kernel.py from rag_kernel.py is to avoid unnecessary embedding calculation 
   overhead while maintaining compatibility for mutual replacement via the application.yaml configuration.
3. Alternating selection strategy: Positive Top 1, Negative Top 1, ...
4. Supports maximum retrieval limits (max_pos, max_neg)
"""

import json
import random
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging

from ..utils.embedding_client import EmbeddingClient

logger = logging.getLogger(__name__)


@dataclass
class RuleEmbedding:
    """Rule Embedding Data Structure (Real embeddings are not used in Random Kernel)"""
    rule_id: str
    rule_type: str  # "positive" or "negative"
    content: str
    embedding: Optional[np.ndarray] = None  # Not used in Random Kernel
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary (without embedding)"""
        return {
            "rule_id": self.rule_id,
            "rule_type": self.rule_type,
            "content": self.content,
            "metadata": self.metadata
        }


@dataclass
class RetrievalResult:
    """Retrieval Result"""
    rule_id: str
    rule_type: str
    content: str
    similarity: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class RAGKernel:
    """
    Random Kernel - Random Retrieval Kernel
    
    Responsibilities:
    1. Manage rule storage (no embedding calculation).
    2. Select relevant rules based on random extraction.
    3. Select positive/negative rules using an alternating strategy.
    """
    
    def __init__(
        self,
        embedding_client: EmbeddingClient,
        cache_dir: Optional[str] = None,
        max_positive: int = 10,
        max_negative: int = 10,
        remove_crr: bool = False
    ):
        """
        Initialize Random Kernel
        
        Args:
            embedding_client: Embedding client (unused, kept for interface compatibility)
            cache_dir: Cache directory (unused)
            max_positive: Maximum number of positive rules
            max_negative: Maximum number of negative rules
            remove_crr: Whether to remove cosine similarity retrieval (unused in Random Kernel)
        """
        self.embedding_client = embedding_client  # Kept for compatibility, but unused
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_positive = max_positive
        self.max_negative = max_negative
        self.remove_crr = remove_crr  # Kept for compatibility, but unused in Random Kernel
        
        # Rule storage (Memory)
        self._positive_rules: List[RuleEmbedding] = []
        self._negative_rules: List[RuleEmbedding] = []
        
        # Rule counter
        self._rule_counter = 0
        
        logger.info("Random Kernel initialized (Random extraction mode, no embeddings used)")
    
    # ========== Rule Management ==========
    
    def add_positive_rule(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RuleEmbedding:
        """
        Add a positive rule (No embedding calculation in Random Kernel)
        
        Args:
            content: Rule content
            metadata: Metadata (question, answer, ppl, etc.)
            
        Returns:
            RuleEmbedding object
        """
        self._rule_counter += 1
        rule_id = f"pos_{self._rule_counter}"
        
        rule = RuleEmbedding(
            rule_id=rule_id,
            rule_type="positive",
            content=content,
            embedding=None,  # Random Kernel does not use embeddings
            metadata=metadata or {}
        )
        
        self._positive_rules.append(rule)
        logger.debug(f"Added positive rule: {rule_id}, content length: {len(content)}")
        
        return rule
    
    def add_negative_rule(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RuleEmbedding:
        """
        Add a negative rule (No embedding calculation in Random Kernel)
        
        Args:
            content: Rule content
            metadata: Metadata
            
        Returns:
            RuleEmbedding object
        """
        self._rule_counter += 1
        rule_id = f"neg_{self._rule_counter}"
        
        rule = RuleEmbedding(
            rule_id=rule_id,
            rule_type="negative",
            content=content,
            embedding=None,  # Not used in Random Kernel
            metadata=metadata or {}
        )
        
        self._negative_rules.append(rule)
        logger.debug(f"Added negative rule: {rule_id}, content length: {len(content)}")
        
        return rule
    
    # ========== Retrieval Logic ==========
    
    def retrieve(
        self,
        query: str,
        max_positive: Optional[int] = None,
        max_negative: Optional[int] = None,
        interleave: bool = True
    ) -> Tuple[List[RetrievalResult], List[RetrievalResult]]:
        """
        Retrieve rules based on random selection
        
        Args:
            query: Query text (unused in Random Kernel)
            max_positive: Maximum positive rules (overrides default)
            max_negative: Maximum negative rules (overrides default)
            interleave: Whether to use alternating strategy (kept for compatibility)
            
        Returns:
            (positive_results, negative_results) tuple
        """
        max_pos = max_positive or self.max_positive
        max_neg = max_negative or self.max_negative
        
        # If rule base is empty, return empty results
        if not self._positive_rules and not self._negative_rules:
            return [], []
        
        # Random Kernel: Randomly select rules
        positive_results = self._random_select_rules(
            self._positive_rules, 
            max_pos
        )
        
        negative_results = self._random_select_rules(
            self._negative_rules, 
            max_neg
        )
        
        logger.info(f"Random Retrieval Complete: {len(positive_results)} Positive, {len(negative_results)} Negative")
        
        return positive_results, negative_results
    
    def retrieve_interleaved(
        self,
        query: str,
        max_positive: Optional[int] = None,
        max_negative: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Alternating rule retrieval (1 pos, 1 neg...)
        
        Args:
            query: Query text
            max_positive: Maximum positive rules
            max_negative: Maximum negative rules
            
        Returns:
            List of interleaved retrieval results
        """
        positive_results, negative_results = self.retrieve(
            query, max_positive, max_negative
        )
        
        # Interleave merge
        interleaved = []
        pos_idx, neg_idx = 0, 0
        
        while pos_idx < len(positive_results) or neg_idx < len(negative_results):
            # Add a positive rule
            if pos_idx < len(positive_results):
                interleaved.append(positive_results[pos_idx])
                pos_idx += 1
            
            # Add a negative rule
            if neg_idx < len(negative_results):
                interleaved.append(negative_results[neg_idx])
                neg_idx += 1
        
        return interleaved
    
    def _random_select_rules(
        self,
        rules: List[RuleEmbedding],
        max_count: int
    ) -> List[RetrievalResult]:
        """
        Randomly select rules
        
        Args:
            rules: Rule list
            max_count: Maximum return count
            
        Returns:
            List of randomly selected retrieval results
        """
        if not rules:
            return []
        
        # Randomly sample rules
        num_to_select = min(max_count, len(rules))
        selected_rules = random.sample(rules, num_to_select)
        
        # Build results
        results = []
        for rule in selected_rules:
            results.append(RetrievalResult(
                rule_id=rule.rule_id,
                rule_type=rule.rule_type,
                content=rule.content,
                similarity=0.0,  # Random Kernel does not calculate similarity, use dummy value
                metadata=rule.metadata
            ))
        
        return results
    
    # ========== Embedding Related (Unused in Random Kernel) ==========
    
    def save_cache(self) -> None:
        """Save cache to file (Unused in Random Kernel, kept for compatibility)"""
        logger.debug("Random Kernel does not use caching")
    
    # ========== Rule Base Management ==========
    
    def get_all_rules(self) -> Tuple[List[RuleEmbedding], List[RuleEmbedding]]:
        """Get all rules"""
        return self._positive_rules.copy(), self._negative_rules.copy()
    
    def clear(self) -> None:
        """Clear rule base"""
        self._positive_rules.clear()
        self._negative_rules.clear()
        self._rule_counter = 0
        logger.info("Rule base cleared (Random Kernel).")
    
    def export_rules(self) -> Dict[str, List[Dict]]:
        """Export rules (without embeddings)"""
        return {
            "positive_rules": [r.to_dict() for r in self._positive_rules],
            "negative_rules": [r.to_dict() for r in self._negative_rules]
        }
    
    def save_rules(self, filepath: str) -> None:
        """Save rules to file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.export_rules(), f, ensure_ascii=False, indent=2)
    
    @property
    def positive_count(self) -> int:
        """Number of positive rules."""
        return len(self._positive_rules)
    
    @property
    def negative_count(self) -> int:
        """Number of negative rules."""
        return len(self._negative_rules)
    
    @property
    def total_count(self) -> int:
        """Total number of rules."""
        return len(self._positive_rules) + len(self._negative_rules)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            "total_rules": self.total_count,
            "positive_rules": self.positive_count,
            "negative_rules": self.negative_count,
            "max_positive": self.max_positive,
            "max_negative": self.max_negative,
            "retrieval_mode": "random"
        }


# ========== Convenience Function ==========

def create_rag_kernel(
    embedding_api_url: str = "http://localhost:10000/v1",
    embedding_api_key: Optional[str] = None,
    embedding_model: str = "Qwen3-Embedding-0.6B",
    cache_dir: Optional[str] = None,
    max_positive: int = 10,
    max_negative: int = 10
) -> RAGKernel:
    """
    Create Random Kernel instance (kept for interface compatibility).
    
    Args:
        embedding_api_url: Embedding API URL (Unused)
        embedding_api_key: Embedding API Key (Unused)
        embedding_model: Embedding model name (Unused)
        cache_dir: Cache directory (Unused)
        max_positive: Maximum positive rules
        max_negative: Maximum negative rules
        
    Returns:
        RAGKernel instance (Random Kernel)
    """
    # Create embedding_client for compatibility (Random Kernel will not use it)
    embedding_client = EmbeddingClient(
        base_url=embedding_api_url,
        api_key=embedding_api_key,
        model_name=embedding_model
    )
    
    return RAGKernel(
        embedding_client=embedding_client,
        cache_dir=cache_dir,
        max_positive=max_positive,
        max_negative=max_negative
    )