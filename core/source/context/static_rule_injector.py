"""
Static Rule Injector

Used to inject manually curated rule files (rules.json format) into the ContextManager at once.
Supports:
1. Group filtering (Positive only / Negative only / All)
2. Control over the maximum number of positive and negative rules
3. Compatibility with existing ContextManager / RAGKernel
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StaticRuleConfig:
	"""Static Rule Injection Configuration"""
	enabled: bool = False
	path: str = ""
	group: str = "all"  # A/positive, B/negative, C/all
	max_positive: Optional[int] = None
	max_negative: Optional[int] = None
	strict: bool = True


class StaticRuleInjector:
	"""Static Rule Injector"""

	def __init__(self, context_manager, log: Optional[logging.Logger] = None):
		self.context_manager = context_manager
		self.logger = log or logger

	@staticmethod
	def _normalize_group(group: str) -> str:
		value = (group or "all").strip().lower()
		if value in {"a", "aa", "pos", "positive"}:
			return "positive"
		if value in {"b", "bb", "neg", "negative"}:
			return "negative"
		return "all"

	@staticmethod
	def _to_int(value: Any, default: int = -1) -> int:
		try:
			return int(value)
		except (TypeError, ValueError):
			return default

	@staticmethod
	def _to_float(value: Any, default: float = 0.0) -> float:
		try:
			return float(value)
		except (TypeError, ValueError):
			return default

	def _load_raw_rules(self, rules_file: Path) -> List[Dict[str, Any]]:
		with open(rules_file, "r", encoding="utf-8") as f:
			data = json.load(f)

		if isinstance(data, dict) and "rules" in data and isinstance(data["rules"], list):
			return data["rules"]

		if isinstance(data, list):
			return data

		raise ValueError(f"Unsupported rule file format: {rules_file}")

	def inject_from_file(self, config: StaticRuleConfig) -> Dict[str, int]:
		"""Load and inject static rules from file"""
		if not config.enabled:
			return {
				"loaded_total": 0,
				"loaded_positive": 0,
				"loaded_negative": 0,
				"skipped": 0
			}

		rules_file = Path(config.path)
		if not rules_file.exists():
			msg = f"Static rules file does not exist: {rules_file}"
			if config.strict:
				raise FileNotFoundError(msg)
			self.logger.warning(msg)
			return {
				"loaded_total": 0,
				"loaded_positive": 0,
				"loaded_negative": 0,
				"skipped": 0
			}

		group = self._normalize_group(config.group)
		raw_rules = self._load_raw_rules(rules_file)

		max_positive = config.max_positive if (config.max_positive is None or config.max_positive >= 0) else None
		max_negative = config.max_negative if (config.max_negative is None or config.max_negative >= 0) else None

		loaded_positive = 0
		loaded_negative = 0
		skipped = 0

		for item in raw_rules:
			rule_type = str(item.get("rule_type", "")).strip().lower()
			content = str(item.get("content", "")).strip()

			if not content or rule_type not in {"positive", "negative"}:
				skipped += 1
				continue

			if group == "positive" and rule_type != "positive":
				skipped += 1
				continue
			if group == "negative" and rule_type != "negative":
				skipped += 1
				continue

			if rule_type == "positive" and max_positive is not None and loaded_positive >= max_positive:
				skipped += 1
				continue
			if rule_type == "negative" and max_negative is not None and loaded_negative >= max_negative:
				skipped += 1
				continue

			question = str(item.get("question", "[StaticRules]"))
			answer = str(item.get("answer", ""))
			source_role = str(item.get("source_role", "static_rule_injector"))
			ppl = self._to_float(item.get("ppl", 0.0), 0.0)
			question_index = self._to_int(item.get("question_index", -1), -1)

			if rule_type == "positive":
				self.context_manager.add_positive_rule(
					content=content,
					question=question,
					answer=answer,
					source_role=source_role,
					ppl=ppl,
					question_index=question_index
				)
				loaded_positive += 1
			else:
				self.context_manager.add_negative_rule(
					content=content,
					question=question,
					answer=answer,
					source_role=source_role,
					ppl=ppl,
					question_index=question_index
				)
				loaded_negative += 1

		loaded_total = loaded_positive + loaded_negative
		if loaded_total == 0 and config.strict:
			raise ValueError(
				f"Zero rules after static rule injection. Please check group/max config and file content: {rules_file}"
			)

		self.logger.info(
			"Static rule injection completed: total=%d, pos=%d, neg=%d, skipped=%d, group=%s",
			loaded_total,
			loaded_positive,
			loaded_negative,
			skipped,
			group,
		)

		return {
			"loaded_total": loaded_total,
			"loaded_positive": loaded_positive,
			"loaded_negative": loaded_negative,
			"skipped": skipped
		}

