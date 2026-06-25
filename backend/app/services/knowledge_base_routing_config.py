import json
from dataclasses import dataclass, field
from functools import lru_cache

from ..core.config_profiles import resolve_config_path


@dataclass(frozen=True)
class RoutingScoreConfig:
    keyword_match: int
    metadata_term_min: int
    metadata_term_max: int
    preferred_knowledge_base: int
    category_match: int
    keep_requested_margin: int
    knowledge_base_alias_match: int
    knowledge_base_priority_match: int
    knowledge_base_exclude_penalty: int


@dataclass(frozen=True)
class RoutingEmbeddingConfig:
    enabled: bool = True
    min_similarity: float = 0.45
    min_margin: float = 0.03
    profile_document_limit: int = 8


@dataclass(frozen=True)
class RoutingDomainConfig:
    keywords: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class KnowledgeBaseQueryRuleConfig:
    name: str
    all_of: tuple[str, ...]
    any_of: tuple[str, ...]
    none_of: tuple[str, ...]
    boost: int


@dataclass(frozen=True)
class KnowledgeBaseRuleConfig:
    name: str
    match_any: tuple[str, ...]
    category: str | None
    aliases: tuple[str, ...]
    priority_terms: tuple[str, ...]
    exclude_terms: tuple[str, ...]
    query_rules: tuple[KnowledgeBaseQueryRuleConfig, ...]


@dataclass(frozen=True)
class KnowledgeBaseRoutingConfig:
    category_keywords: dict[str, tuple[str, ...]]
    knowledge_base_rules: tuple[KnowledgeBaseRuleConfig, ...]
    scores: RoutingScoreConfig
    embedding: RoutingEmbeddingConfig = field(default_factory=RoutingEmbeddingConfig)
    domains: RoutingDomainConfig = field(default_factory=lambda: RoutingDomainConfig(keywords={}))


def _config_path():
    return resolve_config_path("knowledge_base_routing.json")


def _read_keywords(payload: dict[str, object]) -> dict[str, tuple[str, ...]]:
    categories = dict(payload.get("categories") or {})
    return {
        str(name): tuple(str(keyword) for keyword in list(keywords or []) if str(keyword).strip())
        for name, keywords in categories.items()
    }


def _read_knowledge_base_rules(payload: dict[str, object]) -> tuple[KnowledgeBaseRuleConfig, ...]:
    return tuple(
        KnowledgeBaseRuleConfig(
            name=str(rule.get("name") or ""),
            match_any=tuple(str(item) for item in list(rule.get("match_any") or []) if str(item).strip()),
            category=str(rule.get("category")).strip() or None if rule.get("category") is not None else None,
            aliases=tuple(str(item) for item in list(rule.get("aliases") or []) if str(item).strip()),
            priority_terms=tuple(str(item) for item in list(rule.get("priority_terms") or []) if str(item).strip()),
            exclude_terms=tuple(str(item) for item in list(rule.get("exclude_terms") or []) if str(item).strip()),
            query_rules=tuple(
                KnowledgeBaseQueryRuleConfig(
                    name=str(item.get("name") or str(rule.get("name") or "")),
                    all_of=tuple(str(term) for term in list(item.get("all_of") or []) if str(term).strip()),
                    any_of=tuple(str(term) for term in list(item.get("any_of") or []) if str(term).strip()),
                    none_of=tuple(str(term) for term in list(item.get("none_of") or []) if str(term).strip()),
                    boost=int(item.get("boost", 0)),
                )
                for item in list(rule.get("query_rules") or [])
            ),
        )
        for rule in list(payload.get("knowledge_base_rules") or [])
    )


@lru_cache
def load_knowledge_base_routing_config() -> KnowledgeBaseRoutingConfig:
    payload = json.loads(_config_path().read_text(encoding="utf-8"))
    scores = dict(payload.get("scores") or {})
    return KnowledgeBaseRoutingConfig(
        category_keywords=_read_keywords(payload),
        knowledge_base_rules=_read_knowledge_base_rules(payload),
        scores=RoutingScoreConfig(
            keyword_match=int(scores.get("keyword_match", 8)),
            metadata_term_min=int(scores.get("metadata_term_min", 2)),
            metadata_term_max=int(scores.get("metadata_term_max", 6)),
            preferred_knowledge_base=int(scores.get("preferred_knowledge_base", 2)),
            category_match=int(scores.get("category_match", 10)),
            keep_requested_margin=int(scores.get("keep_requested_margin", 3)),
            knowledge_base_alias_match=int(scores.get("knowledge_base_alias_match", 12)),
            knowledge_base_priority_match=int(scores.get("knowledge_base_priority_match", 6)),
            knowledge_base_exclude_penalty=int(scores.get("knowledge_base_exclude_penalty", 12)),
        ),
        embedding=RoutingEmbeddingConfig(
            enabled=bool(dict(payload.get("embedding") or {}).get("enabled", True)),
            min_similarity=float(dict(payload.get("embedding") or {}).get("min_similarity", 0.45)),
            min_margin=float(dict(payload.get("embedding") or {}).get("min_margin", 0.03)),
            profile_document_limit=int(dict(payload.get("embedding") or {}).get("profile_document_limit", 8)),
        ),
        domains=RoutingDomainConfig(
            keywords={
                str(name): tuple(str(keyword) for keyword in list(keywords or []) if str(keyword).strip())
                for name, keywords in dict(payload.get("domains") or {}).items()
            }
        ),
    )
