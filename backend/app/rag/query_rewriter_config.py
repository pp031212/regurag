import json
from dataclasses import dataclass
from functools import lru_cache

from ..core.config_profiles import resolve_config_path


@dataclass(frozen=True)
class QueryRewriteRule:
    match_all: tuple[str, ...]
    rewritten_query: str


@dataclass(frozen=True)
class QueryRewriterPromptConfig:
    rewrite_prompt_template: str
    history_rewrite_prompt_template: str
    subject_hint: str
    keyword_examples: str
    behavior_examples: str
    policy_examples: str
    rewrite_rules: tuple[QueryRewriteRule, ...]


def _config_path():
    return resolve_config_path("query_rewriter_prompts.json")


@lru_cache
def load_query_rewriter_prompt_config() -> QueryRewriterPromptConfig:
    payload = json.loads(_config_path().read_text(encoding="utf-8"))
    return QueryRewriterPromptConfig(
        rewrite_prompt_template=str(payload["rewrite_prompt_template"]),
        history_rewrite_prompt_template=str(payload["history_rewrite_prompt_template"]),
        subject_hint=str(payload["subject_hint"]),
        keyword_examples=str(payload["keyword_examples"]),
        behavior_examples=str(payload["behavior_examples"]),
        policy_examples=str(payload["policy_examples"]),
        rewrite_rules=tuple(
            QueryRewriteRule(
                match_all=tuple(str(item) for item in list(rule.get("match_all") or []) if str(item).strip()),
                rewritten_query=str(rule.get("rewritten_query") or "").strip(),
            )
            for rule in list(payload.get("rewrite_rules") or [])
            if str(rule.get("rewritten_query") or "").strip()
        ),
    )
