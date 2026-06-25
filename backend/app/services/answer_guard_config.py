import json
from dataclasses import dataclass
from functools import lru_cache

from ..core.config_profiles import resolve_config_path


@dataclass(frozen=True)
class QualifierGuardRule:
    name: str
    query_terms: tuple[str, ...]
    evidence_terms: tuple[str, ...]


@dataclass(frozen=True)
class AnswerGuardConfig:
    qualifier_rules: tuple[QualifierGuardRule, ...]


def _config_path():
    return resolve_config_path("answer_guard_rules.json")


@lru_cache
def load_answer_guard_config() -> AnswerGuardConfig:
    payload = json.loads(_config_path().read_text(encoding="utf-8"))
    return AnswerGuardConfig(
        qualifier_rules=tuple(
            QualifierGuardRule(
                name=str(item.get("name") or ""),
                query_terms=tuple(str(term) for term in list(item.get("query_terms") or []) if str(term).strip()),
                evidence_terms=tuple(str(term) for term in list(item.get("evidence_terms") or []) if str(term).strip()),
            )
            for item in list(payload.get("qualifier_rules") or [])
        )
    )
