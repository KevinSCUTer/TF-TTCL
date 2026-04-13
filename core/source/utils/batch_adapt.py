"""
Batch Adapter
Controls batch-based rule generation to avoid generating experience (exp) for every single question.

Core Logic:
1. Processes N questions per batch.
2. After N questions are answered, collects N Best and N Worst candidates.
3. Condenses N Best QA pairs into 1 positive experience rule.
4. Condenses N Worst QA pairs into 1 negative experience rule.
5. Rules can be accumulated across batches.

Usage:
    batch_adapter = BatchAdapter(batch_size=50, llm_client=llm_client)
    
    for i, question_data in enumerate(data):
        # Process question, get selection_result
        ...
        
        # Add to batch collector
        batch_adapter.add_candidate(
            question=question,
            selection_result=selection_result
        )
        
        # Check if batch summary generation is needed
        if batch_adapter.should_summarize():
            rules = batch_adapter.generate_batch_summary()
            # Add rules to context_manager
            ...
"""

import logging
import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

from ..compare.selection_kernel import SelectionResult, SelectionCandidate
from ..context.prompt_loader import PromptLoader


logger = logging.getLogger(__name__)


@dataclass
class BatchCandidate:
    """Batch Candidate Item"""
    question: str                           # Original question
    question_idx: int                       # Question index
    best_candidate: SelectionCandidate      # Best candidate
    worst_candidate: Optional[SelectionCandidate] = None  # Worst candidate (may be None)
    skipped: bool = False                   # Whether to skip summary (Case 1)
    
    def get_best_qa(self) -> Tuple[str, str]:
        """Get Best QA pair"""
        return (self.question, self.best_candidate.content)
    
    def get_worst_qa(self) -> Optional[Tuple[str, str]]:
        """Get Worst QA pair"""
        if self.worst_candidate:
            return (self.question, self.worst_candidate.content)
        return None


@dataclass
class BatchSummaryResult:
    """Batch Summary Result"""
    batch_id: int                           # Batch ID
    batch_size: int                         # Batch size
    question_indices: List[int]             # Included question indices
    positive_exp: str                       # Positive experience rule
    negative_exp: Optional[str] = None      # Negative experience rule
    best_qa_count: int = 0                  # Number of Best QA pairs
    worst_qa_count: int = 0                 # Number of Worst QA pairs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "batch_id": self.batch_id,
            "batch_size": self.batch_size,
            "question_indices": self.question_indices,
            "positive_exp": self.positive_exp,
            "negative_exp": self.negative_exp,
            "best_qa_count": self.best_qa_count,
            "worst_qa_count": self.worst_qa_count
        }


class BatchAdapter:
    """
    Batch Adapter
    
    Controls rule generation batching to avoid per-question experience generation.
    """
    
    def __init__(
        self,
        batch_size: int = 50,
        llm_client: Optional[Any] = None,
        prompt_loader: Optional[PromptLoader] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        mode: str = "close"
    ):
        """
        Initialize Batch Adapter
        
        Args:
            batch_size: Number of questions per batch (N)
            llm_client: LLM client (used to generate batch summaries)
            prompt_loader: Prompt loader
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            mode: Running mode
        """
        self.batch_size = batch_size
        self.llm_client = llm_client
        self.prompt_loader = prompt_loader
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.mode = mode
        
        # Candidate collector for the current batch
        self._current_batch: List[BatchCandidate] = []
        
        # Count of completed batches
        self._completed_batches: int = 0
        
        # Summary results for all batches
        self._batch_results: List[BatchSummaryResult] = []
        
        logger.info(f"BatchAdapter initialized: batch_size={batch_size}")
    
    @property
    def current_batch_size(self) -> int:
        """Number of candidates collected in the current batch"""
        return len(self._current_batch)
    
    @property
    def completed_batches(self) -> int:
        """Number of completed batches"""
        return self._completed_batches
    
    @property
    def batch_results(self) -> List[BatchSummaryResult]:
        """Summary results for all batches"""
        return self._batch_results
    
    def add_candidate(
        self,
        question: str,
        question_idx: int,
        selection_result: SelectionResult
    ) -> bool:
        """
        Add candidate to current batch
        
        Args:
            question: Question text
            question_idx: Question index
            selection_result: Selection result (includes Best/Worst)
            
        Returns:
            Whether to trigger batch summary (True means generate_batch_summary should be called)
        """
        # Create batch candidate
        batch_candidate = BatchCandidate(
            question=question,
            question_idx=question_idx,
            best_candidate=selection_result.best_candidate,
            worst_candidate=selection_result.worst_candidate,
            skipped=selection_result.skip_summary
        )
        
        self._current_batch.append(batch_candidate)
        
        if selection_result.skip_summary:
            logger.debug(f"Added skipped candidate: question_idx={question_idx} (skip_summary=True), batch_size={self.current_batch_size}/{self.batch_size}")
        else:
            logger.debug(f"Added candidate: question_idx={question_idx}, batch_size={self.current_batch_size}/{self.batch_size}")
        
        return self.should_summarize()
    
    def should_summarize(self) -> bool:
        """Check if batch summary should be generated"""
        return self.current_batch_size >= self.batch_size
    
    def generate_batch_summary(self) -> BatchSummaryResult:
        """
        Generate experience summary for the current batch
        
        Condenses N Best QA pairs into 1 positive experience rule
        Condenses N Worst QA pairs into 1 negative experience rule
        
        Returns:
            Batch summary result
        """
        if not self._current_batch:
            logger.warning("Current batch is empty, cannot generate summary")
            raise ValueError("Current batch is empty")
        
        if self.llm_client is None:
            logger.warning("LLM client not configured, using fallback rules")
            return self._generate_fallback_summary()
        
        batch_id = self._completed_batches
        question_indices = [c.question_idx for c in self._current_batch]
        
        logger.info(f"Starting experience summary generation for batch {batch_id} (size={len(self._current_batch)})")
        
        # Collect Best QA pairs (filter out skipped)
        best_qa_pairs = [c.get_best_qa() for c in self._current_batch if not c.skipped]
        
        # Collect Worst QA pairs (filter out None and skipped)
        worst_qa_pairs = [c.get_worst_qa() for c in self._current_batch if not c.skipped and c.get_worst_qa()]
        
        # Generate positive/negative experience in parallel
        positive_exp = "No valid candidates in this batch."
        negative_exp = None
        
        with ThreadPoolExecutor(max_workers=2) as pool:
            future_pos = None
            future_neg = None
            
            if best_qa_pairs:
                future_pos = pool.submit(self._generate_positive_exp, best_qa_pairs)
            else:
                logger.warning("  [!] No valid Best candidates in this batch (all skipped)")
            
            if worst_qa_pairs:
                future_neg = pool.submit(self._generate_negative_exp, worst_qa_pairs)
            else:
                logger.info("  [-] No Worst candidates, skipping negative experience generation")
            
            if future_pos:
                positive_exp = future_pos.result()
                logger.info(f"  [+] Positive experience: {positive_exp[:80]}...")
            
            if future_neg:
                negative_exp = future_neg.result()
                logger.info(f"  [-] Negative experience: {negative_exp[:80]}...")
        
        # Create summary result
        result = BatchSummaryResult(
            batch_id=batch_id,
            batch_size=len(self._current_batch),
            question_indices=question_indices,
            positive_exp=positive_exp,
            negative_exp=negative_exp,
            best_qa_count=len(best_qa_pairs),
            worst_qa_count=len(worst_qa_pairs)
        )
        
        # Save result and clear current batch
        self._batch_results.append(result)
        self._completed_batches += 1
        self._current_batch = []
        
        logger.info(f"Batch {batch_id} summary completed: +1 positive_exp, {'+1 negative_exp' if negative_exp else 'no negative_exp'}")
        
        return result
    
    def flush(self) -> Optional[BatchSummaryResult]:
        """
        Force flush of current batch (process remaining candidates)
        
        Used to handle the final batch that does not reach batch_size
        
        Returns:
            Batch summary result (if candidates remain) or None
        """
        if not self._current_batch:
            logger.info("Current batch is empty, no need to flush")
            return None
        
        logger.info(f"Forced batch flush: {len(self._current_batch)} remaining candidates")
        return self.generate_batch_summary()
    
    def _generate_positive_exp(self, qa_pairs: List[Tuple[str, str]]) -> str:
        """
        Generate positive experience rule
        
        Args:
            qa_pairs: List of Best QA pairs [(question, answer), ...]
            
        Returns:
            Positive experience rule
        """
        # Format QA pairs
        qa_text = self._format_qa_pairs(qa_pairs)
        
        # Build Prompt
        if self.prompt_loader:
            template = self.prompt_loader.get_rule_extract_prompt(self.mode, "positive", "batch")
            prompt = template.format(
                n=len(qa_pairs),
                qa_pairs=qa_text
            )
        else:
            # Fallback if no prompt loader
            prompt = f"Summarize {len(qa_pairs)} QA pairs into a positive rule:\n{qa_text}"
        
        # Call LLM
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.generate(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            logprobs=False
        )
        
        return self._clean_exp(response.content)
    
    def _generate_negative_exp(self, qa_pairs: List[Tuple[str, str]]) -> str:
        """
        Generate negative experience rule
        
        Args:
            qa_pairs: List of Worst QA pairs [(question, answer), ...]
            
        Returns:
            Negative experience rule
        """
        # Format QA pairs
        qa_text = self._format_qa_pairs(qa_pairs)
        
        # Build Prompt
        if self.prompt_loader:
            template = self.prompt_loader.get_rule_extract_prompt(self.mode, "negative", "batch")
            prompt = template.format(
                n=len(qa_pairs),
                qa_pairs=qa_text
            )
        else:
            # Fallback
            prompt = f"Summarize {len(qa_pairs)} QA pairs into a negative rule:\n{qa_text}"
        
        # Call LLM
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.generate(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            logprobs=False
        )
        
        return self._clean_exp(response.content)
    
    def _format_qa_pairs(self, qa_pairs: List[Tuple[str, str]], max_pairs: int = 20) -> str:
        """Format QA pairs"""
        # Limit count
        if len(qa_pairs) > max_pairs:
            qa_pairs = qa_pairs[:max_pairs]
            
        formatted = []
        for i, (q, a) in enumerate(qa_pairs, 1):
            formatted.append(f"Example {i}:\nQuestion: {q}\nAnswer: {a}\n")
            
        return "\n".join(formatted)
    
    def _clean_exp(self, exp: str) -> str:
        """Clean experience rules"""
        exp = exp.strip()
        
        # Remove boxed answer (prevent answer leakage)
        if '\\boxed' in exp or 'oxed{' in exp:
            exp = re.sub(r'\\boxed\{[^}]*\}', '', exp)
            exp = re.sub(r'oxed\{[^}]*\}', '', exp)
        
        # Remove trailing 'b' or '**b**' (common hallucination)
        if exp.endswith('b'):
            exp = exp[:-1].strip()
        if exp.endswith('**b**'):
            exp = exp[:-5].strip()
            
        return exp.strip()
    
    def _generate_fallback_summary(self) -> BatchSummaryResult:
        """Generate default summary (when LLM is unavailable)"""
        return BatchSummaryResult(
            batch_id=self._completed_batches,
            batch_size=len(self._current_batch),
            question_indices=[c.question_idx for c in self._current_batch],
            positive_exp="Fallback positive rule",
            negative_exp="Fallback negative rule",
            best_qa_count=len(self._current_batch),
            worst_qa_count=0
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistical information"""
        return {
            "completed_batches": self._completed_batches,
            "current_batch_size": self.current_batch_size,
            "total_processed": self._completed_batches * self.batch_size + self.current_batch_size
        }
    
    def get_all_positive_exps(self) -> List[str]:
        """Get all positive experience rules"""
        return [r.positive_exp for r in self._batch_results]
    
    def get_all_negative_exps(self) -> List[str]:
        """Get all negative experience rules"""
        return [r.negative_exp for r in self._batch_results if r.negative_exp]
