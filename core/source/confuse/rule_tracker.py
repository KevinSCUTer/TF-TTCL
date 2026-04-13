"""
Rule Tracker

Tracks the state changes of injected rules during retrieval:
- Changes in retrieval ranking
- Whether it is pushed out of Top-K
- Whether it is pruned
- Whether it is merged due to similarity

Used to evaluate the system's self-healing capabilities.
"""

from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import logging

logger = logging.getLogger(__name__)


class RuleStatus(Enum):
    """Rule Status"""

    ACTIVE = "active"  # Still in the rule base
    PRUNED = "pruned"  # Removed by pruning
    MERGED = "merged"  # Merged by similarity
    RETRIEVED = "retrieved"  # Recently retrieved
    EXCLUDED = "excluded"  # Pushed out of Top-K


@dataclass
class TrackingSnapshot:
    """Tracking Snapshot"""

    question_idx: int
    timestamp: float
    poison_rules_in_topk: int  # Number of poison rules in Top-K
    poison_rules_total: int  # Total poison rules remaining in rule base
    poison_positive_in_topk: int  # Number of poison positive rules in Top-K
    poison_negative_in_topk: int  # Number of poison negative rules in Top-K
    avg_similarity_in_topk: float  # Average similarity in Top-K
    retrieved_poison_ids: List[str]  # IDs of retrieved poison rules
    pruned_poison_ids: List[str]  # IDs of pruned poison rules
    merged_poison_ids: List[str]  # IDs of merged poison rules

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_idx": self.question_idx,
            "timestamp": self.timestamp,
            "poison_rules_in_topk": self.poison_rules_in_topk,
            "poison_rules_total": self.poison_rules_total,
            "poison_positive_in_topk": self.poison_positive_in_topk,
            "poison_negative_in_topk": self.poison_negative_in_topk,
            "avg_similarity_in_topk": self.avg_similarity_in_topk,
            "retrieved_poison_ids": self.retrieved_poison_ids,
            "pruned_poison_ids": self.pruned_poison_ids,
            "merged_poison_ids": self.merged_poison_ids,
        }


class RuleTracker:
    """
    Rule Tracker

    Tracks the lifecycle of injected rules.
    """

    def __init__(self, poison_rule_ids: List[str], track_top_k: int = 10):
        """
        Initialize Tracker

        Args:
            poison_rule_ids: List of poison rule IDs
            track_top_k: Number of Top-K results to track
        """
        self.poison_rule_ids: Set[str] = set(poison_rule_ids)
        self.track_top_k = track_top_k

        # State tracking
        self._active_poison_ids: Set[str] = set(poison_rule_ids)
        self._pruned_poison_ids: Set[str] = set()
        self._merged_poison_ids: Set[str] = set()

        # Historical snapshots
        self._snapshots: List[TrackingSnapshot] = []

        # Performance tracking
        self._ppl_history: List[float] = []

        # Retrieval ranking history
        self._rank_history: Dict[str, List[int]] = {
            rule_id: [] for rule_id in poison_rule_ids
        }

    def update_from_retrieval(
        self,
        question_idx: int,
        positive_results: List[Any],
        negative_results: List[Any],
    ) -> TrackingSnapshot:
        """
        Update tracking state from retrieval results

        Args:
            question_idx: Question index
            positive_results: Retrieval results for positive rules (List of RetrievalResult)
            negative_results: Retrieval results for negative rules

        Returns:
            Tracking snapshot
        """
        timestamp = time.time()

        # Collect retrieved poison rules
        retrieved_poison_ids: List[str] = []
        poison_positive_in_topk = 0
        poison_negative_in_topk = 0
        total_sim = 0.0
        total_count = 0

        # Check positive rules
        for rank, result in enumerate(positive_results[: self.track_top_k]):
            rule_id = result.rule_id
            sim = getattr(result, "similarity", 1.0)
            total_sim += sim
            total_count += 1

            if rule_id in self.poison_rule_ids:
                retrieved_poison_ids.append(rule_id)
                poison_positive_in_topk += 1
                self._rank_history[rule_id].append(rank)

        # Check negative rules
        for rank, result in enumerate(negative_results[: self.track_top_k]):
            rule_id = result.rule_id
            sim = getattr(result, "similarity", 1.0)
            total_sim += sim
            total_count += 1

            if rule_id in self.poison_rule_ids:
                retrieved_poison_ids.append(rule_id)
                poison_negative_in_topk += 1
                self._rank_history[rule_id].append(rank)

        # Calculate average similarity
        avg_sim = total_sim / total_count if total_count > 0 else 0.0

        # Create snapshot
        snapshot = TrackingSnapshot(
            question_idx=question_idx,
            timestamp=timestamp,
            poison_rules_in_topk=poison_positive_in_topk + poison_negative_in_topk,
            poison_rules_total=len(self._active_poison_ids),
            poison_positive_in_topk=poison_positive_in_topk,
            poison_negative_in_topk=poison_negative_in_topk,
            avg_similarity_in_topk=avg_sim,
            retrieved_poison_ids=retrieved_poison_ids,
            pruned_poison_ids=list(self._pruned_poison_ids),
            merged_poison_ids=list(self._merged_poison_ids),
        )

        self._snapshots.append(snapshot)

        logger.debug(
            f"[Tracker] Q{question_idx}: "
            f"Top-K poison={snapshot.poison_rules_in_topk}, "
            f"total poison={snapshot.poison_rules_total}"
        )

        return snapshot

    def record_pruning(self, pruned_rules: List[Any]) -> None:
        """
        Record pruning events

        Args:
            pruned_rules: List of pruned rules
        """
        for rule in pruned_rules:
            rule_id = getattr(rule, "rule_id", None)
            if rule_id and rule_id in self._active_poison_ids:
                self._active_poison_ids.remove(rule_id)
                self._pruned_poison_ids.add(rule_id)
                logger.info(f"[Tracker] Poison rule pruned: {rule_id}")

    def record_merge(self, merged_rule_id: str) -> None:
        """
        Record merge events

        Args:
            merged_rule_id: ID of the merged rule
        """
        if merged_rule_id in self._active_poison_ids:
            self._active_poison_ids.remove(merged_rule_id)
            self._merged_poison_ids.add(merged_rule_id)
            logger.info(f"[Tracker] Poison rule merged: {merged_rule_id}")

    def record_ppl(self, ppl: float) -> None:
        """Record PPL value"""
        self._ppl_history.append(ppl)

    def get_current_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            "total_poison_rules": len(self.poison_rule_ids),
            "active_poison_rules": len(self._active_poison_ids),
            "pruned_poison_rules": len(self._pruned_poison_ids),
            "merged_poison_rules": len(self._merged_poison_ids),
            "elimination_rate": 1.0
            - (len(self._active_poison_ids) / max(len(self.poison_rule_ids), 1)),
        }

    def get_rank_trend(self, rule_id: str) -> List[int]:
        """Get ranking trend of a rule"""
        return self._rank_history.get(rule_id, [])

    def get_ppl_trend(self) -> List[float]:
        """Get PPL trend"""
        return self._ppl_history.copy()

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate tracking report

        Returns:
            Detailed report dictionary
        """
        # Calculate trend statistics
        topk_poison_trend = [s.poison_rules_in_topk for s in self._snapshots]
        total_poison_trend = [s.poison_rules_total for s in self._snapshots]

        # Calculate ranking trend for each poison rule
        rank_stats = {}
        for rule_id in self.poison_rule_ids:
            ranks = self._rank_history.get(rule_id, [])
            if ranks:
                rank_stats[rule_id] = {
                    "times_retrieved": len(ranks),
                    "avg_rank": sum(ranks) / len(ranks),
                    "last_rank": ranks[-1] if ranks else None,
                    "rank_trend": ranks[-10:] if len(ranks) > 10 else ranks,
                }
            else:
                rank_stats[rule_id] = {
                    "times_retrieved": 0,
                    "avg_rank": None,
                    "last_rank": None,
                    "rank_trend": [],
                }

        # PPL statistics
        ppl_stats = {}
        if self._ppl_history:
            ppl_stats = {
                "min": min(self._ppl_history),
                "max": max(self._ppl_history),
                "avg": sum(self._ppl_history) / len(self._ppl_history),
                "trend": self._ppl_history[-20:]
                if len(self._ppl_history) > 20
                else self._ppl_history,
            }

        # Self-healing assessment
        recovery_score = self._calculate_recovery_score()

        return {
            "summary": {
                "total_poison_injected": len(self.poison_rule_ids),
                "poison_eliminated": len(self._pruned_poison_ids)
                + len(self._merged_poison_ids),
                "poison_remaining": len(self._active_poison_ids),
                "elimination_rate": 1.0
                - (len(self._active_poison_ids) / max(len(self.poison_rule_ids), 1)),
                "recovery_score": recovery_score,
            },
            "trends": {
                "topk_poison_trend": topk_poison_trend,
                "total_poison_trend": total_poison_trend,
                "topk_poison_reduction": topk_poison_trend[0] - topk_poison_trend[-1]
                if topk_poison_trend
                else 0,
            },
            "rule_rank_stats": rank_stats,
            "ppl_stats": ppl_stats,
            "snapshots": [s.to_dict() for s in self._snapshots[-50:]],  # Last 50 snapshots
            "pruned_ids": list(self._pruned_poison_ids),
            "merged_ids": list(self._merged_poison_ids),
        }

    def _calculate_recovery_score(self) -> float:
        """
        Calculate system self-healing score

        Based on:
        1. Elimination rate of poison rules
        2. Reduction trend of poison rules in Top-K
        3. Improvement trend of PPL
        """
        if not self._snapshots:
            return 0.0

        # Elimination rate score (0-40 points)
        elimination_rate = 1.0 - (
            len(self._active_poison_ids) / max(len(self.poison_rule_ids), 1)
        )
        elimination_score = elimination_rate * 40

        # Top-K reduction score (0-30 points)
        topk_trend = [s.poison_rules_in_topk for s in self._snapshots]
        if len(topk_trend) >= 2:
            initial_topk = topk_trend[0] if topk_trend[0] > 0 else 1
            final_topk = topk_trend[-1]
            reduction_rate = max(0, (initial_topk - final_topk) / initial_topk)
            topk_score = reduction_rate * 30
        else:
            topk_score = 0

        # PPL improvement score (0-30 points)
        if len(self._ppl_history) >= 10:
            early_ppl = sum(self._ppl_history[:5]) / 5
            late_ppl = sum(self._ppl_history[-5:]) / 5
            ppl_improvement = max(0, (early_ppl - late_ppl) / early_ppl)
            ppl_score = min(ppl_improvement * 60, 30)  # Magnification effect, cap at 30
        else:
            ppl_score = 0

        return elimination_score + topk_score + ppl_score

    def save_report(self, filepath: str) -> None:
        """Save report to file"""
        report = self.generate_report()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
