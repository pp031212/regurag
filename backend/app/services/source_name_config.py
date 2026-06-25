import json
import re
from dataclasses import dataclass
from functools import lru_cache

from ..core.config_profiles import resolve_config_path


@dataclass(frozen=True)
class SourceNameRule:
    match_any: tuple[str, ...]
    source_name: str


@dataclass(frozen=True)
class SourceNameConfig:
    rules: tuple[SourceNameRule, ...]


def _config_path():
    return resolve_config_path("source_name_rules.json")


@lru_cache
def load_source_name_config() -> SourceNameConfig:
    payload = json.loads(_config_path().read_text(encoding="utf-8"))
    return SourceNameConfig(
        rules=tuple(
            SourceNameRule(
                match_any=tuple(str(item) for item in list(rule.get("match_any") or []) if str(item).strip()),
                source_name=str(rule["source_name"]),
            )
            for rule in list(payload.get("rules") or [])
        )
    )


def resolve_source_name(filename: str, config: SourceNameConfig | None = None) -> str:
    normalized = filename.strip()
    if not normalized:
        return normalized

    effective_config = config or load_source_name_config()
    stem = re.sub(r"\.[^.]+$", "", normalized).strip()
    for rule in effective_config.rules:
        if any(keyword and keyword in normalized for keyword in rule.match_any):
            return rule.source_name

    return f"《{stem}》" if stem else normalized
