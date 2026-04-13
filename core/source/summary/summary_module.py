"""
Summary Module
Responsible for transforming selected Best/Worst Candidates into positive/negative rules.

Uses an LLM to generate concise, general rules suitable for inclusion in the System Prompt.
"""

import re
from typing import Optional, Dict, Tuple
from ..utils.llm_client import LLMClient
from ..actors import GenerationResult
from ..context.prompt_loader import PromptLoader
from ..compare.selection_kernel import SelectionCandidate


class SummaryModule:
    """
    Summary Module
    
    Responsibilities:
    1. Generate positive rules based on Best Candidates.
    2. Generate negative rules based on Worst Candidates.
    3. Ensure rules are concise and general for System Prompt injection.
    """
    
    def __init__(
        self, 
        llm_client: LLMClient,
        prompt_loader: PromptLoader,
        temperature: float = 0.7,
        max_tokens: int = 100,
        mode: str = "close"
    ):
        """
        Initialize Summary Module
        
        Args:
            llm_client: LLM client
            prompt_loader: Prompt loader
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            mode: Operating mode ("close" or "open")
        """
        self.llm_client = llm_client
        self.prompt_loader = prompt_loader
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.mode = mode
    
    def generate_positive_rule(
        self, 
        question: str, 
        best_candidate: SelectionCandidate
    ) -> str:
        """
        Generate a positive rule.
        
        Based on the best answer, summarize "why this answer is good/correct".
        
        Args:
            question: Original question
            best_candidate: Best candidate
            
        Returns:
            Positive rule string
        """
        template = self.prompt_loader.get_rule_extract_prompt(self.mode, "positive", "single")

        if not template:
            prompt = (
                "Summarize one concise positive solving rule based on the QA below.\n"
                f"Question: {question}\n"
                f"Answer: {best_candidate.content}"
            )
        else:
            prompt = self._safe_format_template(
                template,
                {
                    "question": question,
                    "answer": best_candidate.content,
                    "correct_answer": best_candidate.content,
                    "best_answer": best_candidate.content,
                    "n": 1,
                    "qa_pairs": f"Question: {question}\nAnswer: {best_candidate.content}",
                },
            )
        
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.generate(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            logprobs=False
        )
        
        rule = response.content.strip()
        
        # Clean rule format
        rule = self._clean_rule(rule, prefix="Positive Rule:")
        
        return rule
    
    def generate_negative_rule(
        self, 
        question: str, 
        best_candidate: SelectionCandidate,
        worst_candidate: SelectionCandidate
    ) -> str:
        """
        Generate a negative rule.
        
        Based on the worst answer, summarize "why this answer is poor/wrong" or "what should be avoided".
        
        Args:
            question: Original question
            best_candidate: Best candidate (correct answer reference)
            worst_candidate: Worst candidate
            
        Returns:
            Negative rule string
        """
        template = self.prompt_loader.get_rule_extract_prompt(self.mode, "negative", "single")

        if not template:
            prompt = (
                "Summarize one concise negative solving rule based on the QA below.\n"
                f"Question: {question}\n"
                f"Correct Answer: {best_candidate.content}\n"
                f"Incorrect Answer: {worst_candidate.content}"
            )
        else:
            prompt = self._safe_format_template(
                template,
                {
                    "question": question,
                    "positive_answer": best_candidate.content,
                    "negative_answer": worst_candidate.content,
                    "correct_answer": best_candidate.content,
                    "incorrect_answer": worst_candidate.content,
                    "best_answer": best_candidate.content,
                    "worst_answer": worst_candidate.content,
                    "n": 1,
                    "qa_pairs": (
                        f"Question: {question}\n"
                        f"Correct Answer: {best_candidate.content}\n"
                        f"Incorrect Answer: {worst_candidate.content}"
                    ),
                },
            )
        
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.generate(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            logprobs=False
        )
        
        rule = response.content.strip()
        
        # Clean rule format
        rule = self._clean_rule(rule, prefix="Negative Rule:")
        
        return rule
    
    def generate_rules(
        self,
        question: str,
        best_candidate: SelectionCandidate,
        worst_candidate: Optional[SelectionCandidate] = None
    ) -> Tuple[str, Optional[str]]:
        """
        Generate both positive and negative rules.
        
        Args:
            question: Original question
            best_candidate: Best candidate
            worst_candidate: Worst candidate (optional)
            
        Returns:
            (positive_rule, negative_rule) tuple
        """
        positive_rule = self.generate_positive_rule(question, best_candidate)
        
        negative_rule = None
        if worst_candidate:
            negative_rule = self.generate_negative_rule(
                question, best_candidate, worst_candidate
            )
        
        return positive_rule, negative_rule
    
    def _clean_rule(self, rule: str, prefix: str = "") -> str:
        """
        Clean rule format.
        
        Removes common prefixes to ensure rules are concise.
        
        Args:
            rule: Original rule
            prefix: Prefix to remove
            
        Returns:
            Cleaned rule
        """
        rule = rule.strip()
        
        # Remove specific prefix (case-insensitive)
        if prefix and rule.lower().startswith(prefix.lower()):
            rule = rule[len(prefix):].strip()
        
        # Use regex to remove common redundant prefixes
        # Matches "Positive Rule:", "Rule:", "**Rule:**", "1.", "1)", etc.
        # Also handles potential newlines
        rule = re.sub(r'^(?:\*\*|#+\s*)?(?:Positive|Negative)?\s*Rule\s*:?\s*(?:\*\*)?', '', rule, flags=re.IGNORECASE).strip()
        
        # Remove numbering (e.g., "1. ", "1) ")
        rule = re.sub(r'^\d+[\.\)]\s*', '', rule).strip()
        
        # Remove quotes
        if rule.startswith('"') and rule.endswith('"'):
            rule = rule[1:-1]
        if rule.startswith("'") and rule.endswith("'"):
            rule = rule[1:-1]
        
        # Remove boxed answers (to prevent leakage)
        # Matches \boxed{...} or oxed{...} (due to truncation)
        if '\\boxed' in rule or 'oxed{' in rule:
            rule = re.sub(r'\\boxed\{[^}]*\}', '', rule)
            rule = re.sub(r'oxed\{[^}]*\}', '', rule)
        
        # Remove trailing 'b' or '**b**' (common hallucination)
        if rule.endswith('b'):
            rule = rule[:-1].strip()
        if rule.endswith('**b**'):
            rule = rule[:-5].strip()
        
        return rule.strip()

    def _safe_format_template(self, template: str, values: Dict[str, object]) -> str:
        """Safely format template while leaving unknown placeholders untouched."""

        class _SafeDict(dict):
            def __missing__(self, key):
                return "{" + key + "}"

        return template.format_map(_SafeDict(values))
    
    def format_rule_log(
        self,
        question: str,
        positive_rule: str,
        negative_rule: Optional[str] = None
    ) -> str:
        """
        Format rule log.
        
        Args:
            question: Question
            positive_rule: Positive rule
            negative_rule: Negative rule
            
        Returns:
            Formatted log string
        """
        lines = [
            "=== Summary Module Output ===",
            f"Question: {question[:100]}...",
            "",
            f"✓ Positive Rule: {positive_rule}",
        ]
        
        if negative_rule:
            lines.append(f"✗ Negative Rule: {negative_rule}")
        
        return "\n".join(lines)
