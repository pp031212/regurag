import re
from dataclasses import dataclass

from ..core.config import get_settings
from .light_intent_config import normalize_query
from .knowledge_base_route_embedder import KnowledgeBaseRouteEmbedder, cosine_similarity
from .knowledge_base_routing_config import (
    KnowledgeBaseRoutingConfig,
    KnowledgeBaseQueryRuleConfig,
    KnowledgeBaseRuleConfig,
    load_knowledge_base_routing_config,
)


@dataclass(frozen=True)
class RoutedKnowledgeBase:
    knowledge_base_id: str
    knowledge_base_name: str
    subject: str
    auto_routed: bool
    requested_knowledge_base_id: str | None
    score: int
    reason: str


@dataclass(frozen=True)
class _RouteCandidate:
    knowledge_base: dict
    score: int
    reasons: list[str]
    profile_text: str
    embedding_similarity: float | None = None


def _extract_terms(*parts: str) -> set[str]:
    text = " ".join(part for part in parts if part).strip()
    if not text:
        return set()

    terms: set[str] = set()
    for match in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", text):
        value = match.strip().lower()
        if len(value) >= 2:
            terms.add(value)
    return terms


def _build_profile_text(
    knowledge_base: dict,
    documents: list[dict],
    *,
    knowledge_base_rule: KnowledgeBaseRuleConfig | None,
    document_limit: int,
) -> str:
    document_names = [str(item.get("filename") or "") for item in documents[:document_limit]]
    rule_terms: list[str] = []
    if knowledge_base_rule is not None:
        rule_terms.extend(knowledge_base_rule.aliases)
        rule_terms.extend(knowledge_base_rule.priority_terms)
        for query_rule in knowledge_base_rule.query_rules:
            rule_terms.extend(query_rule.all_of)
            rule_terms.extend(query_rule.any_of)
    return normalize_query(
        " ".join(
            [
                str(knowledge_base.get("name") or ""),
                str(knowledge_base.get("subject") or ""),
                str(knowledge_base.get("description") or ""),
                " ".join(document_names),
                " ".join(term for term in rule_terms if term),
            ]
        )
    )


def _score_query_against_profile(
    *,
    query: str,
    profile_text: str,
    query_terms: set[str],
    category_keywords: tuple[str, ...],
    preferred: bool,
    config: KnowledgeBaseRoutingConfig,
) -> tuple[int, list[str]]:
    normalized_query = normalize_query(query)
    score = 0
    reasons: list[str] = []

    for keyword in category_keywords:
        normalized_keyword = normalize_query(keyword)
        if normalized_keyword and normalized_keyword in normalized_query:
            if normalized_keyword in profile_text:
                score += config.scores.keyword_match
                reasons.append(f"命中关键词:{keyword}")

    for term in query_terms:
        normalized_term = normalize_query(term)
        if len(normalized_term) < 2:
            continue
        if normalized_term in profile_text:
            score += max(
                config.scores.metadata_term_min,
                min(len(normalized_term), config.scores.metadata_term_max),
            )
            reasons.append(f"命中元数据:{term}")

    if preferred:
        score += config.scores.preferred_knowledge_base
        reasons.append("当前选中知识库偏置")

    return score, reasons


def _match_knowledge_base_rule(
    profile_text: str,
    knowledge_base_id: str,
    config: KnowledgeBaseRoutingConfig,
) -> KnowledgeBaseRuleConfig | None:
    best_rule: KnowledgeBaseRuleConfig | None = None
    best_hits = 0
    normalized_id = normalize_query(knowledge_base_id)

    for rule in config.knowledge_base_rules:
        hits = 0
        for term in rule.match_any:
            normalized_term = normalize_query(term)
            if not normalized_term:
                continue
            if normalized_term == normalized_id or normalized_term in profile_text:
                hits += 1
        if hits > best_hits:
            best_hits = hits
            best_rule = rule

    return best_rule if best_hits > 0 else None


def _matches_query_rule(normalized_query: str, rule: KnowledgeBaseQueryRuleConfig) -> bool:
    has_boolean_clause = bool(rule.all_of or rule.any_of or rule.none_of)
    if not has_boolean_clause:
        return False
    all_of_matched = all(normalize_query(term) in normalized_query for term in rule.all_of if normalize_query(term))
    any_of_matched = True
    if rule.any_of:
        any_of_matched = any(normalize_query(term) in normalized_query for term in rule.any_of if normalize_query(term))
    none_of_matched = not any(normalize_query(term) in normalized_query for term in rule.none_of if normalize_query(term))
    return all_of_matched and any_of_matched and none_of_matched


def _score_query_against_knowledge_base_rule(
    query: str,
    rule: KnowledgeBaseRuleConfig | None,
    config: KnowledgeBaseRoutingConfig,
) -> tuple[int, list[str]]:
    if rule is None:
        return 0, []

    normalized_query = normalize_query(query)
    score = 0
    reasons: list[str] = []

    for alias in rule.aliases:
        normalized_alias = normalize_query(alias)
        if normalized_alias and normalized_alias in normalized_query:
            score += config.scores.knowledge_base_alias_match
            reasons.append(f"命中知识库别名:{alias}")

    for term in rule.priority_terms:
        normalized_term = normalize_query(term)
        if normalized_term and normalized_term in normalized_query:
            score += config.scores.knowledge_base_priority_match
            reasons.append(f"命中知识库优先词:{term}")

    for term in rule.exclude_terms:
        normalized_term = normalize_query(term)
        if normalized_term and normalized_term in normalized_query:
            score -= config.scores.knowledge_base_exclude_penalty
            reasons.append(f"命中知识库排除词:{term}")

    for query_rule in rule.query_rules:
        if _matches_query_rule(normalized_query, query_rule):
            score += query_rule.boost
            reasons.append(f"命中布尔规则:{query_rule.name}")

    return score, reasons


def _detect_category(text: str, category_keywords: dict[str, tuple[str, ...]]) -> str | None:
    best_category: str | None = None
    best_hits = 0
    for name, keywords in category_keywords.items():
        hits = sum(1 for keyword in keywords if normalize_query(keyword) in text)
        if hits > best_hits:
            best_category = name
            best_hits = hits
    return best_category if best_hits > 0 else None


def _category_keywords_for_profile(
    profile_text: str,
    category_keywords: dict[str, tuple[str, ...]],
) -> tuple[tuple[str, ...], str | None]:
    category = _detect_category(profile_text, category_keywords)
    return category_keywords.get(category or "", ()), category


def _query_category(query: str, category_keywords: dict[str, tuple[str, ...]]) -> str | None:
    normalized_query = normalize_query(query)
    return _detect_category(normalized_query, category_keywords)


def _query_domain(query: str, domain_keywords: dict[str, tuple[str, ...]]) -> str | None:
    normalized_query = normalize_query(query)
    return _detect_category(normalized_query, domain_keywords)


class KnowledgeBaseRouter:
    def __init__(self, repository, config: KnowledgeBaseRoutingConfig | None = None, embedder=None) -> None:
        self.repository = repository
        self.config = config or load_knowledge_base_routing_config()
        self.embedder = embedder
        self._profile_embedding_cache: dict[tuple[str, str], list[float]] = {}

    @staticmethod
    def _default_embedder():
        return KnowledgeBaseRouteEmbedder(get_settings().embedding_model_name)

    def _get_embedder(self):
        if self.embedder is None:
            self.embedder = self._default_embedder()
        return self.embedder

    def _encode_profile(self, knowledge_base_id: str, profile_text: str) -> list[float]:
        cache_key = (knowledge_base_id, profile_text)
        if cache_key not in self._profile_embedding_cache:
            self._profile_embedding_cache[cache_key] = self._get_embedder().encode(profile_text, is_query=False)
        return self._profile_embedding_cache[cache_key]

    def _select_embedding_candidate(
        self,
        query: str,
        candidates: list[_RouteCandidate],
    ) -> _RouteCandidate | None:
        if not self.config.embedding.enabled or not candidates:
            return None

        query_embedding = self._get_embedder().encode(query, is_query=True)
        embedded_candidates: list[_RouteCandidate] = []
        for candidate in candidates:
            knowledge_base_id = str(candidate.knowledge_base.get("id") or "")
            profile_embedding = self._encode_profile(knowledge_base_id, candidate.profile_text)
            similarity = cosine_similarity(query_embedding, profile_embedding)
            embedded_candidates.append(
                _RouteCandidate(
                    knowledge_base=candidate.knowledge_base,
                    score=candidate.score,
                    reasons=candidate.reasons,
                    profile_text=candidate.profile_text,
                    embedding_similarity=similarity,
                )
            )

        embedded_candidates.sort(key=lambda item: item.embedding_similarity or 0.0, reverse=True)
        best = embedded_candidates[0]
        second = embedded_candidates[1] if len(embedded_candidates) > 1 else None
        best_similarity = best.embedding_similarity or 0.0
        margin = best_similarity - (second.embedding_similarity or 0.0) if second else best_similarity
        if best_similarity < self.config.embedding.min_similarity:
            return None
        if second is not None and margin < self.config.embedding.min_margin:
            return None
        return _RouteCandidate(
            knowledge_base=best.knowledge_base,
            score=int(round(best_similarity * 1000)),
            reasons=best.reasons + [f"命中向量路由:{best_similarity:.3f}", f"向量路由差值:{margin:.3f}"],
            profile_text=best.profile_text,
            embedding_similarity=best_similarity,
        )

    def route(self, *, query: str, requested_knowledge_base_id: str | None = None) -> RoutedKnowledgeBase | None:
        knowledge_bases = [
            item
            for item in self.repository.list_knowledge_bases()
            if item.get("status") in {"ready", "indexing"}
        ]
        if not knowledge_bases:
            return None
        requested_knowledge_base = next(
            (item for item in knowledge_bases if item.get("id") == requested_knowledge_base_id),
            None,
        )
        query_domain = _query_domain(query, self.config.domains.keywords)
        if query_domain:
            domain_filtered = [item for item in knowledge_bases if str(item.get("domain") or "general") == query_domain]
            if domain_filtered:
                knowledge_bases = domain_filtered

        query_terms = _extract_terms(query)
        if not query_terms:
            query_terms = {normalize_query(query)}
        query_category = _query_category(query, self.config.category_keywords)

        candidates: list[_RouteCandidate] = []
        for knowledge_base in knowledge_bases:
            knowledge_base_id = str(knowledge_base.get("id") or "")
            documents = self.repository.list_documents(knowledge_base_id)
            base_profile_text = normalize_query(
                " ".join(
                    [
                        str(knowledge_base.get("name") or ""),
                        str(knowledge_base.get("subject") or ""),
                        str(knowledge_base.get("description") or ""),
                        " ".join(
                            str(item.get("filename") or "")
                            for item in documents[: self.config.embedding.profile_document_limit]
                        ),
                    ]
                )
            )
            knowledge_base_rule = _match_knowledge_base_rule(base_profile_text, knowledge_base_id, self.config)
            profile_text = _build_profile_text(
                knowledge_base,
                documents,
                knowledge_base_rule=knowledge_base_rule,
                document_limit=self.config.embedding.profile_document_limit,
            )
            category_keywords, profile_category = _category_keywords_for_profile(
                profile_text,
                self.config.category_keywords,
            )
            if knowledge_base_rule and knowledge_base_rule.category:
                profile_category = knowledge_base_rule.category
                category_keywords = self.config.category_keywords.get(profile_category, category_keywords)
            score, reasons = _score_query_against_profile(
                query=query,
                profile_text=profile_text,
                query_terms=query_terms,
                category_keywords=category_keywords,
                preferred=knowledge_base_id == requested_knowledge_base_id,
                config=self.config,
            )
            rule_score, rule_reasons = _score_query_against_knowledge_base_rule(
                query,
                knowledge_base_rule,
                self.config,
            )
            score += rule_score
            reasons.extend(rule_reasons)
            if query_category and profile_category == query_category:
                score += self.config.scores.category_match
                reasons.append(f"命中领域:{query_category}")
            if query_domain and str(knowledge_base.get("domain") or "general") == query_domain:
                reasons.append(f"命中业务域:{query_domain}")
            candidates.append(
                _RouteCandidate(
                    knowledge_base=knowledge_base,
                    score=score,
                    reasons=reasons,
                    profile_text=profile_text,
                )
            )

        candidates.sort(key=lambda item: item.score, reverse=True)
        rule_best = candidates[0]
        selected = self._select_embedding_candidate(query, candidates) or rule_best
        best = selected.knowledge_base
        best_score = selected.score
        reasons = selected.reasons

        if requested_knowledge_base_id:
            requested = next((item for item in knowledge_bases if item.get("id") == requested_knowledge_base_id), None)
            if requested is not None:
                requested_score = next(
                    (candidate.score for candidate in candidates if candidate.knowledge_base.get("id") == requested_knowledge_base_id),
                    0,
                )
                if (
                    best.get("id") != requested_knowledge_base_id
                    and best_score < requested_score + self.config.scores.keep_requested_margin
                ):
                    return RoutedKnowledgeBase(
                        knowledge_base_id=str(requested.get("id") or ""),
                        knowledge_base_name=str(requested.get("name") or ""),
                        subject=str(requested.get("subject") or ""),
                        auto_routed=False,
                        requested_knowledge_base_id=requested_knowledge_base_id,
                        score=requested_score,
                        reason="保留当前知识库",
                    )

        auto_routed = best.get("id") != requested_knowledge_base_id
        return RoutedKnowledgeBase(
            knowledge_base_id=str(best.get("id") or ""),
            knowledge_base_name=str(best.get("name") or ""),
            subject=str(best.get("subject") or ""),
            auto_routed=auto_routed,
            requested_knowledge_base_id=requested_knowledge_base_id,
            score=best_score,
            reason="; ".join(dict.fromkeys(reasons)) or ("自动路由" if auto_routed else "沿用当前知识库"),
        )
