import json
from dataclasses import dataclass
from functools import lru_cache

from ..core.config_profiles import resolve_config_path


@dataclass(frozen=True)
class FAQShortcutRule:
    name: str
    patterns: tuple[str, ...]
    answer_template: str
    finish_reason: str
    topic: str | None
    suggested_queries: tuple[str, ...]


@dataclass(frozen=True)
class FAQShortcutConfig:
    rules: tuple[FAQShortcutRule, ...]


@lru_cache
def load_faq_shortcut_config() -> FAQShortcutConfig:
    payload = json.loads(resolve_config_path("faq_shortcuts.json").read_text(encoding="utf-8"))
    return FAQShortcutConfig(
        rules=tuple(
            FAQShortcutRule(
                name=str(rule["name"]),
                patterns=tuple(str(pattern) for pattern in list(rule.get("patterns") or []) if str(pattern).strip()),
                answer_template=str(rule["answer_template"]),
                finish_reason=str(rule.get("finish_reason") or "faq_short_circuit"),
                topic=str(rule["topic"]).strip() if str(rule.get("topic") or "").strip() else None,
                suggested_queries=tuple(
                    str(item) for item in list(rule.get("suggested_queries") or []) if str(item).strip()
                ),
            )
            for rule in list(payload.get("rules") or [])
        )
    )


def format_faq_shortcut_response(
    template: str,
    *,
    subject: str,
    topic: str | None,
    suggested_queries: tuple[str, ...],
) -> str:
    examples = list(suggested_queries[:3])
    while len(examples) < 3:
        examples.append("补充更具体的场景或规则点")
    return template.format(
        subject=subject,
        topic=topic or subject,
        example_query_1=examples[0],
        example_query_2=examples[1],
        example_query_3=examples[2],
    )
