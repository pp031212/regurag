import json
from dataclasses import dataclass
from functools import lru_cache

from ..core.config_profiles import resolve_config_path


@dataclass(frozen=True)
class CrossDomainGuardConfig:
    connectors: tuple[str, ...]
    domain_labels: dict[str, str]
    split_question_examples: dict[str, str]
    clarification_prefix: str
    clarification_question: str
    clarification_suffix: str


def _config_path():
    return resolve_config_path("cross_domain_guard_rules.json")


@lru_cache
def load_cross_domain_guard_config() -> CrossDomainGuardConfig:
    payload = json.loads(_config_path().read_text(encoding="utf-8"))
    return CrossDomainGuardConfig(
        connectors=tuple(str(item) for item in list(payload.get("connectors") or []) if str(item).strip()),
        domain_labels={
            str(name): str(label)
            for name, label in dict(payload.get("domain_labels") or {}).items()
            if str(name).strip() and str(label).strip()
        },
        split_question_examples={
            str(name): str(example)
            for name, example in dict(payload.get("split_question_examples") or {}).items()
            if str(name).strip() and str(example).strip()
        },
        clarification_prefix=str(payload.get("clarification_prefix") or ""),
        clarification_question=str(payload.get("clarification_question") or ""),
        clarification_suffix=str(payload.get("clarification_suffix") or ""),
    )
