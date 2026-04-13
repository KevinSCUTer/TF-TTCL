"""
RAG Kernel
Responsible for rule embedding calculation, caching, retrieval, and ranking.

Core Features:
1. Calculates similarity between questions and the rule base using embedding models.
2. Ranks positive and negative rules by similarity separately.
3. Alternating selection strategy: Positive Top 1, Negative Top 1, ...
4. Supports maximum retrieval limits (max_pos, max_neg).
5. Rule vector caching mechanism to avoid redundant calculations.
"""

# Note: When the configuration uses ablation: random_rules, RAG Kernel will disable similarity retrieval 
# and switch to a pure extraction strategy, randomly picking rules until the limit is reached or rules are exhausted.

import json
import os
import hashlib
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging

from ..utils.embedding_client import EmbeddingClient
from .similarity_pruning_adapt_layer import SimilarityPruningLayer

logger = logging.getLogger(__name__)


@dataclass
class RuleEmbedding:
    """Rule Embedding Data Structure"""
    rule_id: str
    rule_type: str  # "positive" or "negative"
    content: str
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary (excluding embedding)"""
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
    RAG Retrieval Kernel
    
    Responsibilities:
    1. Manage embedding calculation and caching for rules.
    2. Retrieve relevant rules based on similarity.
    3. Select positive/negative rules using an alternating strategy.
    """
    
    def __init__(
        self,
        embedding_client: EmbeddingClient,
        cache_dir: Optional[str] = None,
        max_positive: int = 10,
        max_negative: int = 10,
        remove_crr: bool = False,
        max_repo_size: Optional[int] = 1000,
        pruning_strategy: str = "fifo",
        similarity_threshold: float = 0.95
    ):
        """
        Initialize RAG Kernel
        
        Args:
            embedding_client: Embedding client
            cache_dir: Vector cache directory
            max_positive: Maximum number of positive rules
            max_negative: Maximum number of negative rules
            remove_crr: Whether to remove cosine similarity retrieval (use LIFO instead)
            max_repo_size: Rule base capacity limit (default 1000, None = unlimited)
            pruning_strategy: Pruning strategy ("fifo" | "random" | "similarity" | "similarity_fifo")
            similarity_threshold: Similarity merge threshold (only effective when strategy=similarity)
        """
        self.embedding_client = embedding_client
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_positive = max_positive
        self.max_negative = max_negative
        self.remove_crr = remove_crr
        self.max_repo_size = max_repo_size
        self.pruning_strategy = pruning_strategy
        self.similarity_threshold = similarity_threshold
        
        # Rule storage (in-memory)
        self._positive_rules: List[RuleEmbedding] = []
        self._negative_rules: List[RuleEmbedding] = []

        # Embedding cache (content_hash -> embedding)
        self._embedding_cache: Dict[str, np.ndarray] = {}

        # Rule counter
        self._rule_counter = 0

        # Similarity Pruning Layer (only active when strategy=similarity)
        self._similarity_layer: Optional[SimilarityPruningLayer] = None
        if self.pruning_strategy == "similarity":
            self._similarity_layer = SimilarityPruningLayer(
                threshold=self.similarity_threshold
            )
        
        # Load cache
        if self.cache_dir:
            self._load_cache()
    
    # ========== Rule Management ==========
    
    def add_positive_rule(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RuleEmbedding:
        """
        Add a positive rule
        
        Args:
            content: Rule content
            metadata: Metadata (question, answer, ppl, etc.)
            
        Returns:
            RuleEmbedding object (newly added or merged existing rule)
        """
        meta = metadata or {}
        meta.setdefault("frequency", 1)
        
        # Compute Embedding
        embedding = self._get_or_compute_embedding(content)
        
        # Similarity Pruning: Try merging into existing rules
        if self._similarity_layer:
            merged = self._similarity_layer.try_merge(
                embedding, content, self._positive_rules
            )
            if merged is not None:
                return merged
        
        self._rule_counter += 1
        rule_id = f"pos_{self._rule_counter}"
        
        rule = RuleEmbedding(
            rule_id=rule_id,
            rule_type="positive",
            content=content,
            embedding=embedding,
            metadata=meta
        )

        if self._should_replace_by_similarity_fifo(self._positive_rules):
            replaced, replace_sim = self._replace_most_similar_rule(
                rule, self._positive_rules
            )
            if replaced is not None:
                logger.debug(
                    f"[Pruning/similarity_fifo] Positive replacement: "
                    f"{replaced.rule_id} -> {rule.rule_id}, sim={replace_sim:.4f}"
                )
                return rule
        
        self._positive_rules.append(rule)
        logger.debug(f"Added positive rule: {rule_id}, content length: {len(content)}")
        
        self._prune_if_needed()
        
        return rule
    
    def add_negative_rule(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RuleEmbedding:
        """
        Add a negative rule
        
        Args:
            content: Rule content
            metadata: Metadata
            
        Returns:
            RuleEmbedding object (newly added or merged existing rule)
        """
        meta = metadata or {}
        meta.setdefault("frequency", 1)
        
        # Compute Embedding
        embedding = self._get_or_compute_embedding(content)
        
        # Similarity Pruning: Try merging into existing rules
        if self._similarity_layer:
            merged = self._similarity_layer.try_merge(
                embedding, content, self._negative_rules
            )
            if merged is not None:
                return merged
        
        self._rule_counter += 1
        rule_id = f"neg_{self._rule_counter}"
        
        rule = RuleEmbedding(
            rule_id=rule_id,
            rule_type="negative",
            content=content,
            embedding=embedding,
            metadata=meta
        )

        if self._should_replace_by_similarity_fifo(self._negative_rules):
            replaced, replace_sim = self._replace_most_similar_rule(
                rule, self._negative_rules
            )
            if replaced is not None:
                logger.debug(
                    f"[Pruning/similarity_fifo] Negative replacement: "
                    f"{replaced.rule_id} -> {rule.rule_id}, sim={replace_sim:.4f}"
                )
                return rule
        
        self._negative_rules.append(rule)
        logger.debug(f"Added negative rule: {rule_id}, content length: {len(content)}")
        
        self._prune_if_needed()
        
        return rule
    
    # ========== Pruning Logic ==========
    
    def _prune_if_needed(self) -> None:
        """
        Prune according to strategy when total rules exceed max_repo_size.
        
        Strategies:
        - fifo: First-In-First-Out, deletes the earliest joined rules, maintaining balance between pos/neg.
        - random: Randomly deletes a rule, maintaining balance between pos/neg.
        - similarity_fifo: Implemented by add_positive_rule/add_negative_rule via similar replacement at full capacity;
          falls back to FIFO if overflow persists.
        """
        if self.max_repo_size is None:
            return
        
        import random as _random
        
        while self.total_count > self.max_repo_size:
            # Prioritize deletion from the side with more rules to maintain balance
            remove_from_positive = (
                len(self._positive_rules) > 0 and (
                    not self._negative_rules or
                    len(self._positive_rules) >= len(self._negative_rules)
                )
            )
            
            if self.pruning_strategy == "random":
                if remove_from_positive:
                    idx = _random.randrange(len(self._positive_rules))
                    removed = self._positive_rules.pop(idx)
                elif self._negative_rules:
                    idx = _random.randrange(len(self._negative_rules))
                    removed = self._negative_rules.pop(idx)
                else:
                    break
            else:
                # Default FIFO: Delete earliest joined (at the head of the list)
                if remove_from_positive:
                    removed = self._positive_rules.pop(0)
                elif self._negative_rules:
                    removed = self._negative_rules.pop(0)
                else:
                    break
            
            logger.debug(f"[Pruning/{self.pruning_strategy}] Deleted rule: {removed.rule_id}")

    def _should_replace_by_similarity_fifo(self, target_rules: List[RuleEmbedding]) -> bool:
        """Check if similarity_fifo replacement should be triggered."""
        return (
            self.pruning_strategy == "similarity_fifo"
            and self.max_repo_size is not None
            and self.total_count >= self.max_repo_size
            and len(target_rules) > 0
        )

    def _replace_most_similar_rule(
        self,
        new_rule: RuleEmbedding,
        target_rules: List[RuleEmbedding]
    ) -> Tuple[Optional[RuleEmbedding], float]:
        """Replace the most similar existing rule of the same type with the new rule."""
        if not target_rules:
            return None, 0.0

        rule_embeddings = np.stack([r.embedding for r in target_rules])

        new_norm = new_rule.embedding / (np.linalg.norm(new_rule.embedding) + 1e-8)
        rules_norm = rule_embeddings / (
            np.linalg.norm(rule_embeddings, axis=1, keepdims=True) + 1e-8
        )
        similarities = rules_norm @ new_norm

        replace_idx = int(np.argmax(similarities))
        replaced_rule = target_rules[replace_idx]
        target_rules[replace_idx] = new_rule
        return replaced_rule, float(similarities[replace_idx])
    
    # ========== Retrieval Logic ==========
    
    def retrieve(
        self,
        query: str,
        max_positive: Optional[int] = None,
        max_negative: Optional[int] = None,
        interleave: bool = True
    ) -> Tuple[List[RetrievalResult], List[RetrievalResult]]:
        """
        Retrieve rules based on similarity
        
        Args:
            query: Query text (current question)
            max_positive: Maximum positive rules (overrides default)
            max_negative: Maximum negative rules (overrides default)
            interleave: Whether to use alternating strategy
            
        Returns:
            (positive_results, negative_results) tuple
        """
        max_pos = max_positive or self.max_positive
        max_neg = max_negative or self.max_negative
        
        # If rule base is empty, return empty results
        if not self._positive_rules and not self._negative_rules:
            return [], []
        
        # [Ablation] remove_crr: LIFO retrieval
        if self.remove_crr:
            # Take the last max_count rules and reverse them (newest first)
            pos_rules = self._positive_rules[-max_pos:][::-1] if max_pos > 0 else []
            neg_rules = self._negative_rules[-max_neg:][::-1] if max_neg > 0 else []
            
            positive_results = [
                RetrievalResult(
                    rule_id=r.rule_id,
                    rule_type=r.rule_type,
                    content=r.content,
                    similarity=1.0,  # Dummy similarity
                    metadata=r.metadata
                ) for r in pos_rules
            ]
            
            negative_results = [
                RetrievalResult(
                    rule_id=r.rule_id,
                    rule_type=r.rule_type,
                    content=r.content,
                    similarity=1.0,  # Dummy similarity
                    metadata=r.metadata
                ) for r in neg_rules
            ]
            
            logger.info(f"RAG Retrieval Complete (LIFO): {len(positive_results)} Positive, {len(negative_results)} Negative")
            return positive_results, negative_results

        # Compute Query Embedding
        query_embedding = self._get_or_compute_embedding(query)
        
        # Rank positive rules
        positive_results = self._rank_rules(
            query_embedding, 
            self._positive_rules, 
            max_pos
        )
        
        # Rank negative rules
        negative_results = self._rank_rules(
            query_embedding, 
            self._negative_rules, 
            max_neg
        )
        
        logger.info(f"RAG Retrieval Complete: {len(positive_results)} Positive, {len(negative_results)} Negative")
        
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
    
    def _rank_rules(
        self,
        query_embedding: np.ndarray,
        rules: List[RuleEmbedding],
        max_count: int
    ) -> List[RetrievalResult]:
        """
        Rank rules based on similarity.
        
        When pruning_strategy == "similarity", rank scores are calculated as
        similarity * log(1 + frequency) to prioritize high-frequency rules.
        
        Args:
            query_embedding: Query Embedding
            rules: Rule list
            max_count: Maximum return count
            
        Returns:
            List of ranked retrieval results
        """
        if not rules:
            return []
        
        # Compute similarities
        rule_embeddings = np.stack([r.embedding for r in rules])
        
        # L2 normalization
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        rules_norm = rule_embeddings / (np.linalg.norm(rule_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Cosine similarity
        similarities = rules_norm @ query_norm
        
        # Similarity strategy: Weight by frequency
        if self.pruning_strategy == "similarity":
            frequencies = np.array([
                r.metadata.get("frequency", 1) for r in rules
            ], dtype=np.float64)
            scores = similarities * np.log1p(frequencies)
        else:
            scores = similarities
        
        # Sort (descending)
        sorted_indices = np.argsort(scores)[::-1]
        
        # Build results
        results = []
        for idx in sorted_indices[:max_count]:
            rule = rules[idx]
            results.append(RetrievalResult(
                rule_id=rule.rule_id,
                rule_type=rule.rule_type,
                content=rule.content,
                similarity=float(similarities[idx]),
                metadata=rule.metadata
            ))
        
        return results
    
    # ========== Embedding Calculation and Caching ==========
    
    def _get_or_compute_embedding(self, text: str) -> np.ndarray:
        """
        Get or compute text embedding.
        
        Retrieves from cache if available, otherwise computes it.
        
        Args:
            text: Text content
            
        Returns:
            Embedding vector
        """
        # Compute content hash
        content_hash = self._hash_content(text)
        
        # Check cache
        if content_hash in self._embedding_cache:
            return self._embedding_cache[content_hash]
        
        # Compute Embedding
        embedding = self.embedding_client.get_embeddings(text)[0]
        
        # Store in cache
        self._embedding_cache[content_hash] = embedding
        
        return embedding
    
    def _hash_content(self, content: str) -> str:
        """Compute content hash."""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    # ========== Cache Persistence ==========
    
    def _load_cache(self) -> None:
        """Load cache from file."""
        if not self.cache_dir:
            return
        
        cache_file = self.cache_dir / "embedding_cache.npz"
        index_file = self.cache_dir / "embedding_index.json"
        
        if not cache_file.exists() or not index_file.exists():
            return
        
        try:
            # Load index
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            
            # Load embeddings
            data = np.load(cache_file)
            embeddings = data['embeddings']
            hashes = index['hashes']
            
            for i, h in enumerate(hashes):
                self._embedding_cache[h] = embeddings[i]
            
            logger.info(f"Loaded Embedding cache: {len(hashes)} entries")
        except Exception as e:
            logger.warning(f"Failed to load Embedding cache: {e}")
    
    def save_cache(self) -> None:
        """Save cache to file."""
        if not self.cache_dir or not self._embedding_cache:
            return
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_file = self.cache_dir / "embedding_cache.npz"
        index_file = self.cache_dir / "embedding_index.json"
        
        try:
            hashes = list(self._embedding_cache.keys())
            embeddings = np.stack([self._embedding_cache[h] for h in hashes])
            
            # Save embeddings
            np.savez_compressed(cache_file, embeddings=embeddings)
            
            # Save index
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump({'hashes': hashes}, f)
            
            logger.info(f"Saved Embedding cache: {len(hashes)} entries")
        except Exception as e:
            logger.warning(f"Failed to save Embedding cache: {e}")
    
    # ========== Rule Base Management ==========
    
    def get_all_rules(self) -> Tuple[List[RuleEmbedding], List[RuleEmbedding]]:
        """Get all rules."""
        return self._positive_rules.copy(), self._negative_rules.copy()
    
    def clear(self) -> None:
        """Clear the rule base."""
        self._positive_rules.clear()
        self._negative_rules.clear()
        self._rule_counter = 0
        logger.info("RAG Kernel rule base cleared.")
    
    def export_rules(self) -> Dict[str, List[Dict]]:
        """Export rules (without embeddings)."""
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
        stats = {
            "total_rules": self.total_count,
            "positive_rules": self.positive_count,
            "negative_rules": self.negative_count,
            "max_positive": self.max_positive,
            "max_negative": self.max_negative,
            "cache_size": len(self._embedding_cache),
            "max_repo_size": self.max_repo_size,
            "pruning_strategy": self.pruning_strategy if self.max_repo_size is not None else "none"
        }
        
        if self._similarity_layer:
            all_rules = self._positive_rules + self._negative_rules
            freqs = [r.metadata.get("frequency", 1) for r in all_rules]
            stats["similarity_threshold"] = self.similarity_threshold
            stats["total_merges"] = self._similarity_layer.merge_count
            stats["avg_frequency"] = sum(freqs) / len(freqs) if freqs else 0.0
        
        return stats


# ========== Convenience Function ==========

def create_rag_kernel(
    embedding_api_url: str = "http://localhost:10000/v1",
    embedding_api_key: Optional[str] = None,
    embedding_model: str = "Qwen3-Embedding-0.6B",
    cache_dir: Optional[str] = None,
    max_positive: int = 10,
    max_negative: int = 10,
    max_repo_size: Optional[int] = 1000,
    pruning_strategy: str = "fifo",
    similarity_threshold: float = 0.95
) -> RAGKernel:
    """
    Create RAG Kernel instance.
    
    Args:
        embedding_api_url: Embedding API URL
        embedding_api_key: Embedding API Key
        embedding_model: Embedding model name
        cache_dir: Cache directory
        max_positive: Maximum positive rules
        max_negative: Maximum negative rules
        max_repo_size: Rule base capacity limit (default 1000, None = unlimited)
        pruning_strategy: Pruning strategy ("fifo" | "random" | "similarity" | "similarity_fifo")
        similarity_threshold: Similarity merge threshold (only effective when strategy=similarity)
        
    Returns:
        RAGKernel instance
    """
    embedding_client = EmbeddingClient(
        base_url=embedding_api_url,
        api_key=embedding_api_key,
        model_name=embedding_model
    )
    
    return RAGKernel(
        embedding_client=embedding_client,
        cache_dir=cache_dir,
        max_positive=max_positive,
        max_negative=max_negative,
        max_repo_size=max_repo_size,
        pruning_strategy=pruning_strategy,
        similarity_threshold=similarity_threshold
    )