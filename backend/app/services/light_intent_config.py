import json
import re
from dataclasses import dataclass
from functools import lru_cache

from ..core.config_profiles import resolve_config_path


@dataclass(frozen=True)
class LightIntentRule:
    name: str
    patterns: tuple[str, ...]
    response_key: str
    finish_reason: str


@dataclass(frozen=True)
class LightIntentConfig:
    policy_keywords: tuple[str, ...]
    off_topic_patterns: tuple[str, ...]
    intent_rules: tuple[LightIntentRule, ...]
    responses: dict[str, str]


_LIGHT_INTENT_CLAUSE_SPLIT_RE = re.compile(r"[\s!！?？,，.。~～、:：;；]+")


@lru_cache
def load_light_intent_config() -> LightIntentConfig:
    rules_payload = json.loads(resolve_config_path("light_intents.json").read_text(encoding="utf-8"))
    responses_payload = json.loads(resolve_config_path("light_intent_responses.json").read_text(encoding="utf-8"))
    return LightIntentConfig(
        policy_keywords=tuple(str(item) for item in list(rules_payload.get("policy_keywords") or []) if str(item).strip()),
        off_topic_patterns=tuple(
            str(item) for item in list(rules_payload.get("off_topic_patterns") or []) if str(item).strip()
        ),
        intent_rules=tuple(
            LightIntentRule(
                name=str(rule["name"]),
                patterns=tuple(str(pattern) for pattern in list(rule.get("patterns") or []) if str(pattern).strip()),
                response_key=str(rule["response_key"]),
                finish_reason=str(rule["finish_reason"]),
            )
            for rule in list(rules_payload.get("intent_rules") or [])
        ),
        responses={str(key): str(value) for key, value in dict(responses_payload).items()},
    )


def normalize_query(text: str) -> str:
    normalized = re.sub(r"\s+", "", text.strip().lower())
    return re.sub(r"[!！?？,，.。~～、:：;；'\"“”‘’（）()\-]+", "", normalized)


def split_light_intent_clauses(text: str) -> tuple[str, ...]:
    return tuple(
        normalized
        for normalized in (normalize_query(part) for part in _LIGHT_INTENT_CLAUSE_SPLIT_RE.split(text))
        if normalized
    )


def format_light_intent_response(template: str, subject: str) -> str:
    examples = "相关制度、流程、要求、范围或处理规则"
    return template.format(
        subject=subject,
        examples=examples,
        example_query_1="相关流程是什么",
        example_query_2="这类情况怎么处理",
        example_query_3="有哪些具体要求",
    )
