import json
from dataclasses import dataclass
from functools import lru_cache

from ..core.config_profiles import resolve_config_path


@dataclass(frozen=True)
class ConditionalAliasRule:
    match_all: tuple[tuple[str, ...], ...]
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class QueryAliasConfig:
    term_aliases: dict[str, tuple[str, ...]]
    conditional_alias_rules: tuple[ConditionalAliasRule, ...]


def _config_path():
    return resolve_config_path("query_aliases.json")


@lru_cache
def load_query_alias_config() -> QueryAliasConfig:
    payload = json.loads(_config_path().read_text(encoding="utf-8"))
    term_aliases = {
        str(term): tuple(str(alias) for alias in aliases if str(alias).strip())
        for term, aliases in dict(payload.get("term_aliases") or {}).items()
    }
    conditional_alias_rules = tuple(
        ConditionalAliasRule(
            match_all=tuple(
                tuple(str(keyword) for keyword in group if str(keyword).strip())
                for group in list(rule.get("match_all") or [])
            ),
            aliases=tuple(str(alias) for alias in list(rule.get("aliases") or []) if str(alias).strip()),
        )
        for rule in list(payload.get("conditional_alias_rules") or [])
    )
    return QueryAliasConfig(
        term_aliases=term_aliases,
        conditional_alias_rules=conditional_alias_rules,
    )


class QueryAliasExpander:
    def __init__(self, config: QueryAliasConfig | None = None) -> None:
        self.config = config or load_query_alias_config()

    @staticmethod
    def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword in text for keyword in keywords)

    def expand(self, query: str) -> str:
        aliases: list[str] = []

        for term, mapped_aliases in self.config.term_aliases.items():
            if term in query:
                aliases.extend(mapped_aliases)

        for rule in self.config.conditional_alias_rules:
            if all(self._contains_any(query, keyword_group) for keyword_group in rule.match_all):
                aliases.extend(rule.aliases)

        deduped_aliases: list[str] = []
        seen: set[str] = set()
        for alias in aliases:
            if alias and alias not in seen:
                seen.add(alias)
                deduped_aliases.append(alias)

        return " ".join(deduped_aliases)
