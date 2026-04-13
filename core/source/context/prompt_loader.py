import os
from pathlib import Path
from typing import Optional, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[3]

class PromptLoader:
    """
    Prompt Loader
    Responsible for loading prompts from the file system, supporting paths specified in configuration files.
    """
    
    def __init__(self, prompt_paths: Optional[Dict[str, str]] = None, base_dir: Optional[str] = None):
        """
        Initialize PromptLoader
        
        Args:
            prompt_paths: Dictionary of prompt paths, including:
                - teacher: Teacher System Prompt path
                - ta: TA System Prompt path
                - student: Student System Prompt path
                - positive_batch: Positive rule extraction Prompt path
                - negative_batch: Negative rule extraction Prompt path
        """
        self.prompt_paths = prompt_paths or {}
        self.base_dir = Path(base_dir).resolve() if base_dir else None
        self._cache: Dict[str, str] = {}
        
    def _resolve_path(self, path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path

        if self.base_dir:
            candidate = (self.base_dir / path).resolve()
            if candidate.exists():
                return candidate

        candidate = (PROJECT_ROOT / path).resolve()
        if candidate.exists():
            return candidate

        return path.resolve()

    def _read_file(self, path_str: str) -> str:
        if not path_str:
            return ""
            
        path = self._resolve_path(path_str)
        if not path.exists():
            print(f"Warning: Prompt file not found: {path}")
            return ""
            
        return path.read_text(encoding='utf-8').strip()

    def get_prompt_content(self, key: str) -> str:
        """Get the content of a specific prompt key"""
        if key in self._cache:
            return self._cache[key]
            
        path_str = self.prompt_paths.get(key)
        content = self._read_file(path_str)
        
        if content:
            self._cache[key] = content
            
        return content

    def get_teacher_prompt(self, mode: str = None, domain: str = None) -> str:
        """Get Teacher System Prompt"""
        return self.get_prompt_content('teacher') or self.get_prompt_content('teacher_prompt_file_path')

    def get_ta_prompt(self, mode: str = None, domain: str = None) -> str:
        """Get TA System Prompt"""
        return self.get_prompt_content('ta') or self.get_prompt_content('ta_prompt_file_path')

    def get_student_prompt(self) -> str:
        """Get Student System Prompt"""
        return self.get_prompt_content('student') or self.get_prompt_content('student_prompt_file_path')

    def get_rule_extract_prompt(self, mode: str, rule_type: str, method: str = "batch") -> str:
        """
        Get Rule Extraction Prompt
        Args:
            mode: ignored (kept for compatibility)
            rule_type: positive / negative
            method: ignored (kept for compatibility)
        """
        candidates = []

        if method == "single":
            candidates.extend(
                [
                    f"{rule_type}_single",
                    f"{rule_type}_rule",
                    f"{rule_type}_summary_single_file_path",
                ]
            )
        elif method == "batch":
            candidates.extend(
                [
                    f"{rule_type}_batch",
                    f"{rule_type}_summary_file_path",
                ]
            )

        # Backward-compatible fallback keys
        candidates.extend(
            [
                f"{rule_type}_summary_file_path",
                f"{rule_type}_batch",
                f"{rule_type}_prompt_file_path",
            ]
        )

        for key in candidates:
            content = self.get_prompt_content(key)
            if content:
                return content

        return ""
    
    def clear_cache(self) -> None:
        """Clear cache"""
        self._cache.clear()


class PromptTemplates:
    """
    Prompt Template Collection (Retained for rule injection formatting)
    """
    
    # Rule injection header template
    RULES_HEADER = """
<BEGIN_RULES>
LEARNED RULES (from previous problem-solving experience)
Apply these rules when solving similar problems:
"""

    # Positive rules section template
    POSITIVE_RULES_SECTION = """
[POSITIVE PATTERNS - What works well]
{rules}"""

    # Negative rules section template
    NEGATIVE_RULES_SECTION = """
[NEGATIVE PATTERNS - What to avoid]
{rules}"""

    # Rule injection footer template
    RULES_FOOTER = """
<END_RULES>
"""

    @classmethod
    def format_rules_section(
        cls,
        positive_rules: list,
        negative_rules: list,
        loader: Optional[PromptLoader] = None
    ) -> str:
        """
        Format the rules section
        
        Args:
            positive_rules: List of positive rules
            negative_rules: List of negative rules
            loader: PromptLoader instance (optional, for loading custom header, etc.)
            
        Returns:
            Formatted rules string
        """
        if not positive_rules and not negative_rules:
            return ""
        
        # Try to load header from loader
        header = cls.RULES_HEADER
        if loader:
            loaded_header = loader.get_prompt("default/rule/header.md")
            if loaded_header:
                header = loaded_header
        
        parts = [header]
        
        if positive_rules:
            formatted_positive = "\n".join(f"  {i}. ✓ {rule}" for i, rule in enumerate(positive_rules, 1))
            parts.append(cls.POSITIVE_RULES_SECTION.format(rules=formatted_positive))
        
        if negative_rules:
            formatted_negative = "\n".join(f"  {i}. ✗ {rule}" for i, rule in enumerate(negative_rules, 1))
            parts.append(cls.NEGATIVE_RULES_SECTION.format(rules=formatted_negative))
        
        parts.append(cls.RULES_FOOTER)
        
        return "\n".join(parts)
