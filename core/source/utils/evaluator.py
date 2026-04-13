"""
Evaluator - PPL Calculation Module
Computes text Perplexity (PPL) using standard NLP algorithms.
"""

import math
import logging
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PPLResult:
    """PPL Calculation Result"""
    ppl: float                    # Perplexity value
    average_logprob: float        # Average logprob
    token_count: int              # Number of tokens
    logprobs: List[float]         # List of original logprobs


class PPLEvaluator:
    
    def __init__(self, epsilon: float = 1e-10):
        """
        Initialize evaluator.
        
        Args:
            epsilon: Small constant to prevent numerical issues.
        """
        self.epsilon = epsilon
    
    def compute_ppl(self, logprobs: List[float]) -> PPLResult:
        """
        Compute perplexity.
        
        Args:
            logprobs: List of token logprobs (returned from API).
            
        Returns:
            PPLResult object containing PPL value and statistics.
        """
        if not logprobs:
            # Return infinite PPL for empty lists.
            logger.warning("compute_ppl received empty logprobs list")
            return PPLResult(
                ppl=float('inf'),
                average_logprob=float('-inf'),
                token_count=0,
                logprobs=[]
            )
        
        N = len(logprobs)
        
        # Debug logging
        try:
            min_lp = min(logprobs)
            max_lp = max(logprobs)
            logger.info(f"PPL Debug: N={N}, min={min_lp}, max={max_lp}, first5={logprobs[:5]}")
        except Exception as e:
            logger.error(f"PPL Debug Error: {e}")

        # Handle -inf values by truncating to -100 (exp(-100) ≈ 3e-44).
        # This prevents PPL from becoming infinity in common edge cases.
        cleaned_logprobs = []
        for lp in logprobs:
            if lp == float('-inf') or lp < -100:
                cleaned_logprobs.append(-100.0)
            else:
                cleaned_logprobs.append(lp)
        
        # Compute average logprob.
        # Logprob is already the log-probability, so take the direct mean.
        average_logprob = sum(cleaned_logprobs) / N
        
        # Calculate PPL.
        # PPL = exp(-average_logprob)
        # Since logprob is negative (log of probability), take negative then exp.
        try:
            ppl = math.exp(-average_logprob)
        except OverflowError:
            ppl = float('inf')
        
        return PPLResult(
            ppl=ppl,
            average_logprob=average_logprob,
            token_count=N,
            logprobs=logprobs
        )
    
    def select_best(
        self,
        candidates: List[PPLResult],
        labels: Optional[List[str]] = None
    ) -> tuple:
        """
        Select the candidate with the lowest PPL.
        
        Args:
            candidates: List of PPLResult objects.
            labels: Optional list of labels to identify candidates.
            
        Returns:
            (best_index, best_result, best_label) tuple.
        """
        if not candidates:
            raise ValueError("Candidate list cannot be empty")
        
        # Find index with minimum PPL.
        best_idx = 0
        best_ppl = candidates[0].ppl
        
        for i, result in enumerate(candidates):
            if result.ppl < best_ppl:
                best_ppl = result.ppl
                best_idx = i
        
        best_result = candidates[best_idx]
        best_label = labels[best_idx] if labels else f"candidate_{best_idx}"
        
        return best_idx, best_result, best_label
    
    def compare_ppl(self, ppl_a: float, ppl_b: float) -> int:
        """
        Compare two PPL values.
        
        Args:
            ppl_a: First PPL value.
            ppl_b: Second PPL value.
            
        Returns:
            -1 if ppl_a < ppl_b (a is better)
             0 if ppl_a == ppl_b
             1 if ppl_a > ppl_b (b is better)
        """
        diff = ppl_a - ppl_b
        if abs(diff) < self.epsilon:
            return 0
        return -1 if diff < 0 else 1
    
    def format_ppl(self, ppl: float, precision: int = 4) -> str:
        """
        Format PPL value for display.
        
        Args:
            ppl: PPL value.
            precision: Decimal precision.
            
        Returns:
            Formatted string.
        """
        if ppl == float('inf'):
            return "inf"
        return f"{ppl:.{precision}f}"

