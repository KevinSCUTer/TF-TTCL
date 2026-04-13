"""
Selection Kernel
Responsible for analyzing Teacher and Students answers, determining the Case type based on consensus and PPL, 
and outputting Best/Worst Candidates.
"""

from typing import List, Dict, Optional, Tuple, Any, TYPE_CHECKING
from collections import Counter
from dataclasses import dataclass, field
import logging

from ..actors import GenerationResult
from ..utils.evaluator import PPLResult
from ..utils.extract_utils import extract_gsm8k_answer_number, is_equiv

if TYPE_CHECKING:
    from ..utils.embedding_client import EmbeddingClient

logger = logging.getLogger(__name__)


@dataclass
class SelectionCandidate:
    """Selection Candidate"""
    result: GenerationResult           # Original generation result
    extracted_answer: Optional[str]    # Extracted numerical answer
    similarity: Optional[float] = None # Semantic similarity with Teacher answer (used in Open Domain)
    
    @property
    def ppl(self) -> float:
        return self.result.ppl_result.ppl
    
    @property
    def role_id(self) -> str:
        return self.result.role_id
    
    @property
    def content(self) -> str:
        return self.result.content


@dataclass
class SelectionResult:
    """Selection Result"""
    case_type: int                                           # 1, 2, or 3
    best_candidate: SelectionCandidate                       # Best candidate
    worst_candidate: Optional[SelectionCandidate] = None     # Worst candidate (None for Case 1/2)
    consensus_group: List[SelectionCandidate] = field(default_factory=list)      # Consensus group
    non_consensus_group: List[SelectionCandidate] = field(default_factory=list)  # Non-consensus group
    description: str = ""                                    # Result description
    skip_summary: bool = False                               # Whether to skip summary module
    
    def get_best_result(self) -> GenerationResult:
        """Get GenerationResult of the best candidate"""
        return self.best_candidate.result
    
    def get_worst_result(self) -> Optional[GenerationResult]:
        """Get GenerationResult of the worst candidate"""
        return self.worst_candidate.result if self.worst_candidate else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        result = {
            "case_type": self.case_type,
            "description": self.description,
            "skip_summary": self.skip_summary,
            "best": {
                "role_id": self.best_candidate.role_id,
                "extracted_answer": self.best_candidate.extracted_answer,
                "ppl": self.best_candidate.ppl,
                "similarity": self.best_candidate.similarity
            },
            "worst": {
                "role_id": self.worst_candidate.role_id,
                "extracted_answer": self.worst_candidate.extracted_answer,
                "ppl": self.worst_candidate.ppl,
                "similarity": self.worst_candidate.similarity
            } if self.worst_candidate else None,
            "consensus_count": len(self.consensus_group),
            "non_consensus_count": len(self.non_consensus_group)
        }
        
        # Add similarity ranking info in Open Domain mode
        if self.case_type == 4:
            # Collect similarities of all non-Teacher candidates
            all_candidates = self.consensus_group + self.non_consensus_group
            student_sims = [
                {"role_id": c.role_id, "similarity": c.similarity, "ppl": c.ppl}
                for c in all_candidates if c.role_id != "teacher"
            ]
            # Sort by similarity in descending order
            student_sims.sort(key=lambda x: x.get("similarity") or 0, reverse=True)
            result["similarity_ranking"] = student_sims
        
        return result


class SelectionKernel:
    """
    Selection Kernel
    
    Core Functions:
    1. Use extract_utils to extract mathematical answers
    2. Judge consensus based on extracted answers
    3. Choose Best/Worst Candidates combined with PPL
    4. Group using semantic similarity in Open Domain mode
    """
    
    def __init__(
        self, 
        answer_extractor: str = "gsm8k",
        embedding_client: Optional["EmbeddingClient"] = None,
        use_max_ppl_for_negative_selection: bool = False
    ):
        """
        Initialize Selection Kernel
        
        Args:
            answer_extractor: Type of answer extractor (e.g., "gsm8k")
            embedding_client: Embedding client (required for Open Domain mode)
            use_max_ppl_for_negative_selection: Whether to use max PPL for negative sample selection (default False, i.e., use min PPL)
        """
        self.answer_extractor = answer_extractor
        self.embedding_client = embedding_client
        self.use_max_ppl_for_negative_selection = use_max_ppl_for_negative_selection
        
        logger.info(f"SelectionKernel initialized: use_max_ppl_for_negative_selection={use_max_ppl_for_negative_selection}")
    
    def extract_answer(self, content: str) -> Optional[str]:
        """
        Extract numerical answer from generated content
        
        Args:
            content: Complete model generated content
            
        Returns:
            Extracted numerical answer string, or None if extraction fails
        """
        if self.answer_extractor == "gsm8k":
            return extract_gsm8k_answer_number(content)
        else:
            # Default returns None, indicating extraction not supported
            return None
    
    def answers_are_equivalent(self, ans1: Optional[str], ans2: Optional[str]) -> bool:
        """
        Determine if two answers are equivalent
        
        Args:
            ans1: First answer
            ans2: Second answer
            
        Returns:
            Whether they are equivalent
        """
        if ans1 is None or ans2 is None:
            return False
        return is_equiv(ans1, ans2)
    
    def _build_candidates(
        self, 
        teacher_result: GenerationResult, 
        student_results: List[GenerationResult]
    ) -> List[SelectionCandidate]:
        """
        Build candidate list, extracting answer for each result
        
        Args:
            teacher_result: Teacher's generation result
            student_results: List of Student's generation results
            
        Returns:
            List of SelectionCandidate
        """
        all_results = [teacher_result] + student_results
        candidates = []
        
        for result in all_results:
            extracted = self.extract_answer(result.content)
            candidates.append(SelectionCandidate(
                result=result,
                extracted_answer=extracted
            ))
        
        return candidates
    
    def _group_by_answer(
        self, 
        candidates: List[SelectionCandidate]
    ) -> Dict[str, List[SelectionCandidate]]:
        """
        Group by extracted answer
        
        Note: Use is_equiv for equivalence judgment, not exact string matching
        
        Args:
            candidates: Candidate list
            
        Returns:
            {Answer: [Candidate List]} dictionary
        """
        groups: Dict[str, List[SelectionCandidate]] = {}
        
        for cand in candidates:
            ans = cand.extracted_answer
            
            if ans is None:
                # Answer cannot be extracted, group independently
                key = f"__none_{id(cand)}__"
                groups[key] = [cand]
                continue
            
            # Check if equivalent to existing answers
            matched = False
            for existing_ans in list(groups.keys()):
                if existing_ans.startswith("__none_"):
                    continue
                if self.answers_are_equivalent(ans, existing_ans):
                    groups[existing_ans].append(cand)
                    matched = True
                    break
            
            if not matched:
                groups[ans] = [cand]
        
        return groups
        
    def select(
        self, 
        teacher_result: GenerationResult, 
        student_results: List[GenerationResult],
        mode: str = "close"
    ) -> SelectionResult:
        """
        Execute selection logic
        
        Args:
            teacher_result: Teacher's generation result
            student_results: List of Student's generation results
            mode: Selection mode ("close" or "open")
            
        Returns:
            SelectionResult object
        """
        if mode == "open":
            return self._select_open(teacher_result, student_results)
        else:
            return self._select_close(teacher_result, student_results)

    def _select_open(
        self,
        teacher_result: GenerationResult,
        student_results: List[GenerationResult]
    ) -> SelectionResult:
        """
        Open Domain selection logic
        
        Group and rank using Semantic Similarity.
        
        Returns:
            SelectionResult object
        """
        # Check if embedding_client is available
        if self.embedding_client is None:
            logger.warning("EmbeddingClient not configured, Open Domain mode will use PPL as the sole criterion")
            return self._select_open_fallback(teacher_result, student_results)
        
        # ========== Rule 1: Build Teacher candidate (always in Group A) ==========
        teacher_candidate = SelectionCandidate(
            result=teacher_result, 
            extracted_answer=None,  # Open Domain does not require numerical answer extraction
            similarity=1.0  # Teacher's similarity with itself is 1.0
        )
        
        # ========== Rule 2: Compute Semantic Similarity ==========
        # Get content of all student answers
        student_contents = [r.content for r in student_results]
        
        try:
            # Call Embedding API to compute semantic similarity
            similarities = self.embedding_client.compute_similarity(
                query=teacher_result.content,
                documents=student_contents
            )
            
            logger.info(f"Semantic similarity calculation complete: {similarities}")
        except Exception as e:
            logger.error(f"Semantic similarity calculation failed: {e}, using fallback mode")
            return self._select_open_fallback(teacher_result, student_results)
        
        # Build student candidates and assign similarity
        student_candidates = []
        for i, result in enumerate(student_results):
            sim = float(similarities[i]) if i < len(similarities) else 0.0
            candidate = SelectionCandidate(
                result=result,
                extracted_answer=None,
                similarity=sim
            )
            student_candidates.append(candidate)
        
        # Sort by similarity descending (sim_1, sim_2, sim_3, sim_4...)
        student_candidates.sort(key=lambda x: x.similarity or 0.0, reverse=True)
        
        # Log sorted similarity info
        for i, cand in enumerate(student_candidates):
            logger.debug(f"  sim_{i+1}: {cand.role_id} -> similarity={cand.similarity:.4f}, ppl={cand.ppl:.4f}")
        
        # Rule 3: Similarity Filtering
        # Group top 50% into Group A, others into Group B
        group_a = [teacher_candidate]
        group_b = []
        
        top_n = int(len(student_candidates) * 0.5)
        if top_n < 1 and len(student_candidates) > 0:
            top_n = 1
            
        for i, cand in enumerate(student_candidates):
            if i < top_n:
                group_a.append(cand)
                logger.debug(f"  {cand.role_id} -> Group A (Top {top_n} Similarity)")
            else:
                group_b.append(cand)
                logger.debug(f"  {cand.role_id} -> Group B (Not Top {top_n} Similarity)")
        
        # ========== Rule 4: Select Best/Worst ==========
        # Group A min_PPL as positive sample (Best Candidate)
        best_candidate = min(group_a, key=lambda x: x.ppl)
        
        # Group B candidate as negative sample (Worst Candidate)
        worst_candidate = None
        if group_b:
            if self.use_max_ppl_for_negative_selection:
                worst_candidate = max(group_b, key=lambda x: x.ppl)
            else:
                worst_candidate = min(group_b, key=lambda x: x.ppl)
        
        # Build description info
        similarity_info = ", ".join([
            f"{c.role_id}={c.similarity:.3f}" 
            for c in student_candidates[:4]
        ])
        description = (
            f"Open Domain: Group A={len(group_a)}, Group B={len(group_b)}, "
            f"Similarities=[{similarity_info}]"
        )
        
        logger.info(f"Open Domain selection complete: Best={best_candidate.role_id}, Worst={worst_candidate.role_id if worst_candidate else 'None'}")
        
        return SelectionResult(
            case_type=4,  # 4 represents Open Domain
            best_candidate=best_candidate,
            worst_candidate=worst_candidate,
            consensus_group=group_a,      # Reuse field: Group A (Good)
            non_consensus_group=group_b,  # Reuse field: Group B (Bad)
            description=description,
            skip_summary=False
        )
    
    def _select_open_fallback(
        self,
        teacher_result: GenerationResult,
        student_results: List[GenerationResult]
    ) -> SelectionResult:
        """
        Open Domain selection Fallback mode (used when Embedding Client is unavailable)
        
        Group solely by PPL, without semantic similarity calculation.
        """
        # Build candidates
        teacher_candidate = SelectionCandidate(
            result=teacher_result, 
            extracted_answer=None,
            similarity=None
        )
        student_candidates = [
            SelectionCandidate(result=r, extracted_answer=None, similarity=None) 
            for r in student_results
        ]
        
        # Group by PPL only
        group_a = [teacher_candidate]
        group_b = []
        
        teacher_ppl = teacher_result.ppl_result.ppl
        ppl_threshold = teacher_ppl * 1.1
        
        for cand in student_candidates:
            if cand.ppl <= ppl_threshold:
                group_a.append(cand)
            else:
                group_b.append(cand)
        
        # Select Best/Worst
        best_candidate = min(group_a, key=lambda x: x.ppl)
        worst_candidate = None
        if group_b:
            if self.use_max_ppl_for_negative_selection:
                worst_candidate = max(group_b, key=lambda x: x.ppl)
            else:
                worst_candidate = min(group_b, key=lambda x: x.ppl)
        
        return SelectionResult(
            case_type=4,
            best_candidate=best_candidate,
            worst_candidate=worst_candidate,
            consensus_group=group_a,
            non_consensus_group=group_b,
            description=f"Open Domain (Fallback, no similarity): Group A={len(group_a)}, Group B={len(group_b)}",
            skip_summary=False
        )

    def _select_close(
        self, 
        teacher_result: GenerationResult, 
        student_results: List[GenerationResult]
    ) -> SelectionResult:
        """
        Closed Domain selection logic
        
        Use extracted mathematical answers for consensus judgment.
        """
        # 1. Build candidate list and extract answers
        candidates = self._build_candidates(teacher_result, student_results)
        teacher_candidate = candidates[0]
        
        # 2. Group by answer
        answer_groups = self._group_by_answer(candidates)
        
        # 3. Count each answer
        group_counts = [(ans, len(group)) for ans, group in answer_groups.items()]
        group_counts.sort(key=lambda x: x[1], reverse=True)
        
        num_unique_answers = len(answer_groups)
        total_candidates = len(candidates)
        
        # 4. Determine Case Type
        
        # Case 2: Complete Consensus (All answers are equivalent)
        if num_unique_answers == 1:
            best = min(candidates, key=lambda x: x.ppl)
            return SelectionResult(
                case_type=2,
                best_candidate=best,
                worst_candidate=None,
                consensus_group=candidates,
                non_consensus_group=[],
                description="Case 2: Complete Consensus - All answers are equivalent",
                skip_summary=False  # Case 2 can generate positive rules
            )
        
        # Case 1: No Consensus (All answers differ)
        if num_unique_answers == total_candidates:
            return SelectionResult(
                case_type=1,
                best_candidate=teacher_candidate,
                worst_candidate=None,
                consensus_group=[],
                non_consensus_group=candidates,
                description="Case 1: No Consensus - All answers differ",
                skip_summary=True  # Case 1 skips summary module
            )
        
        # Case 3: Partial Consensus (Exists >=2 same answers)
        max_count = group_counts[0][1]
        
        # Find all majority answer groups
        top_groups = [(ans, answer_groups[ans]) for ans, count in group_counts if count == max_count]
        
        if len(top_groups) == 1:
            # Single majority group
            consensus_answer, consensus_candidates = top_groups[0]
        else:
            # Multiple majority groups (e.g., 2 vs 2)
            # Pick group with global min_PPL as the consensus group
            min_ppl_by_group = {}
            for ans, group in top_groups:
                min_ppl = min(c.ppl for c in group)
                min_ppl_by_group[ans] = (min_ppl, group)
            
            consensus_answer = min(min_ppl_by_group, key=lambda x: min_ppl_by_group[x][0])
            consensus_candidates = min_ppl_by_group[consensus_answer][1]
        
        # Determine Best Candidate (min PPL in consensus group)
        best_candidate = min(consensus_candidates, key=lambda x: x.ppl)
        
        # Determine non-consensus group
        consensus_set = set(id(c) for c in consensus_candidates)
        non_consensus_candidates = [c for c in candidates if id(c) not in consensus_set]
        
        # Determine Worst Candidate (min PPL in non-consensus group)
        worst_candidate = None
        if non_consensus_candidates:
            if self.use_max_ppl_for_negative_selection:
                worst_candidate = max(non_consensus_candidates, key=lambda x: x.ppl)
            else:
                worst_candidate = min(non_consensus_candidates, key=lambda x: x.ppl)
        
        return SelectionResult(
            case_type=3,
            best_candidate=best_candidate,
            worst_candidate=worst_candidate,
            consensus_group=consensus_candidates,
            non_consensus_group=non_consensus_candidates,
            description=f"Case 3: Partial Consensus (Max Count: {max_count}, Groups: {num_unique_answers})",
            skip_summary=False
        )
    
    def format_selection_log(self, result: SelectionResult, candidates: List[SelectionCandidate] = None) -> str:
        """
        Format selection log
        
        Args:
            result: Selection result
            candidates: Optional candidate list
            
        Returns:
            Formatted log string
        """
        lines = [
            f"=== Selection Kernel Result ===",
            f"Case Type: {result.case_type}",
            f"Description: {result.description}",
            f"Skip Summary: {result.skip_summary}",
            f"",
            f"Best Candidate:",
            f"  - Role: {result.best_candidate.role_id}",
            f"  - Answer: {result.best_candidate.extracted_answer}",
            f"  - PPL: {result.best_candidate.ppl:.4f}",
        ]
        
        # Open Domain shows similarity
        if result.best_candidate.similarity is not None:
            lines.append(f"  - Similarity: {result.best_candidate.similarity:.4f}")
        
        if result.worst_candidate:
            lines.extend([
                f"",
                f"Worst Candidate:",
                f"  - Role: {result.worst_candidate.role_id}",
                f"  - Answer: {result.worst_candidate.extracted_answer}",
                f"  - PPL: {result.worst_candidate.ppl:.4f}",
            ])
            if result.worst_candidate.similarity is not None:
                lines.append(f"  - Similarity: {result.worst_candidate.similarity:.4f}")
        
        # Open Domain shows grouping information
        if result.case_type == 4:
            lines.extend([
                f"",
                f"Group A (Good): {len(result.consensus_group)} candidates",
                f"Group B (Bad): {len(result.non_consensus_group)} candidates",
            ])
            
            # Show similarity ranking
            all_candidates = result.consensus_group + result.non_consensus_group
            student_candidates = [c for c in all_candidates if c.role_id != "teacher"]
            student_candidates.sort(key=lambda x: x.similarity or 0, reverse=True)
            
            if student_candidates:
                lines.append(f"")
                lines.append(f"Similarity Ranking:")
                for i, c in enumerate(student_candidates):
                    sim_str = f"{c.similarity:.4f}" if c.similarity is not None else "N/A"
                    group = "A" if c in result.consensus_group else "B"
                    lines.append(f"  {i+1}. {c.role_id}: sim={sim_str}, ppl={c.ppl:.4f}, group={group}")
        else:
            lines.extend([
                f"",
                f"Consensus Group: {len(result.consensus_group)} candidates",
                f"Non-Consensus Group: {len(result.non_consensus_group)} candidates",
            ])
        
        return "\n".join(lines)
