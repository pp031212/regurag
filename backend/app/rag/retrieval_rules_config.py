import json
from dataclasses import dataclass
from functools import lru_cache

from ..core.config_profiles import resolve_config_path


@dataclass(frozen=True)
class RetrievalRulesConfig:
    policy_trigger_words: tuple[str, ...]
    policy_trigger_phrases: tuple[str, ...]
    behavior_keywords: tuple[str, ...]
    must_keep_child_keywords: tuple[str, ...]
    low_value_parent_keywords: tuple[str, ...]
    rule_signal_keywords: tuple[str, ...]
    overview_query_expansions: dict[str, tuple[str, ...]]


def _config_path():
    return resolve_config_path("retrieval_rules.json")


def _read_list(payload: dict[str, object], key: str) -> tuple[str, ...]:
    return tuple(str(item) for item in list(payload.get(key) or []) if str(item).strip())


def _read_mapping(payload: dict[str, object], key: str) -> dict[str, tuple[str, ...]]:
    raw_mapping = payload.get(key) or {}
    if not isinstance(raw_mapping, dict):
        return {}
    parsed: dict[str, tuple[str, ...]] = {}
    for raw_key, raw_values in raw_mapping.items():
        key_text = str(raw_key).strip()
        if not key_text:
            continue
        values = tuple(str(item).strip() for item in list(raw_values or []) if str(item).strip())
        if values:
            parsed[key_text] = values
    return parsed


@lru_cache
def load_retrieval_rules_config() -> RetrievalRulesConfig:
    payload = json.loads(_config_path().read_text(encoding="utf-8"))
    return RetrievalRulesConfig(
        policy_trigger_words=_read_list(payload, "policy_trigger_words"),
        policy_trigger_phrases=_read_list(payload, "policy_trigger_phrases"),
        behavior_keywords=_read_list(payload, "behavior_keywords"),
        must_keep_child_keywords=_read_list(payload, "must_keep_child_keywords"),
        low_value_parent_keywords=_read_list(payload, "low_value_parent_keywords"),
        rule_signal_keywords=_read_list(payload, "rule_signal_keywords"),
        overview_query_expansions=_read_mapping(payload, "overview_query_expansions"),
    )
