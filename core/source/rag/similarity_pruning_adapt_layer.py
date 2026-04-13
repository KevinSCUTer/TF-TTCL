"""
Similarity-based Frequency Pruning Adapt Layer

When the cosine similarity between a new rule R_new and an existing rule exceeds a threshold,
no new entry is added. Instead, the frequency count of the existing rule is incremented.
This gives higher weight to high-frequency "golden rules," causing the rule base size 
to grow logarithmically and eventually saturate.
"""

import numpy as np
import logging
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .rag_kernel import RuleEmbedding

logger = logging.getLogger(__name__)


class SimilarityPruningLayer:
    """
    Similarity-based Frequency Pruning Adapt Layer

    Intercepts rules before they are added to the rule base: calculates the cosine similarity 
    between the new rule and existing rules of the same type. If the maximum similarity 
    exceeds the threshold, the rules are merged (frequency+1); otherwise, the new rule is added.
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.merge_count = 0

    def try_merge(
        self,
        new_embedding: np.ndarray,
        new_content: str,
        existing_rules: List["RuleEmbedding"],
    ) -> Optional["RuleEmbedding"]:
        """
        Attempt to merge the new rule into existing rules.

        Args:
            new_embedding: Embedding vector of the new rule
            new_content: Text content of the new rule
            existing_rules: List of existing rules of the same type (positive/negative)

        Returns:
            The merged existing rule (with updated frequency), or None if it should be added as new.
        """
        if not existing_rules:
            return None

        rule_embeddings = np.stack([r.embedding for r in existing_rules])

        new_norm = new_embedding / (np.linalg.norm(new_embedding) + 1e-8)
        rules_norm = rule_embeddings / (
            np.linalg.norm(rule_embeddings, axis=1, keepdims=True) + 1e-8
        )

        similarities = rules_norm @ new_norm
        max_idx = int(np.argmax(similarities))
        max_sim = float(similarities[max_idx])

        if max_sim >= self.threshold:
            merged_rule = existing_rules[max_idx]
            merged_rule.metadata.setdefault("frequency", 1)
            merged_rule.metadata["frequency"] += 1
            self.merge_count += 1

            logger.debug(
                f"[SimilarityPrune] Merged rules: sim={max_sim:.4f}, "
                f"rule={merged_rule.rule_id}, freq={merged_rule.metadata['frequency']}"
            )
            return merged_rule

        return None
