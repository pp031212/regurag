"""知识库问答的高层编排服务。

本模块位于底层 RAG pipeline 之上，负责解析知识库与会话、在检索前执行低成本
保护逻辑、持久化消息，并在灰度阶段可选地对比 legacy pipeline 和 shadow graph。
"""

import hashlib
import json
import logging
import random
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import re
from functools import lru_cache
from time import perf_counter
from typing import Generator

from ..core.config import get_settings
from ..core.exceptions import ConversationNotFoundError, KnowledgeBaseNotFoundError, KnowledgeBaseNotReadyError
from ..core.file_lock import advisory_file_lock
from ..rag.pipeline import RAGPipeline
from ..repositories.metadata_repository import MetadataRepository, get_metadata_repository
from ..schemas.chat import ChatQueryRequest
from .answer_guard_config import load_answer_guard_config
from .conversation_context_resolver import ConversationContextResolver
from .cross_domain_guard_config import load_cross_domain_guard_config
from .faq_shortcut_config import format_faq_shortcut_response, load_faq_shortcut_config
from .intent_router import IntentDecision, IntentRouter, IntentType
from .knowledge_base_router import KnowledgeBaseRouter
from .knowledge_base_routing_config import load_knowledge_base_routing_config
from .light_intent_config import (
    format_light_intent_response,
    load_light_intent_config,
    normalize_query,
    split_light_intent_clauses,
)
from .source_name_config import resolve_source_name


logger = logging.getLogger(__name__)


@lru_cache
def _settings():
    return get_settings()



def _collection_name(base_name: str, knowledge_base_id: str) -> str:
    suffix = re.sub(r"[^a-zA-Z0-9_]+", "_", knowledge_base_id)
    return f"{base_name}_{suffix}"


class RAGPipelineRegistry:
    """按知识库懒加载并缓存 RAGPipeline。

    Pipeline 初始化会加载 embedding/reranker 资源并打开向量库 collection，
    每次请求都创建成本很高。这里同时使用进程内锁和文件锁，避免多个 Docker
    worker 同时初始化同一个知识库索引。
    """

    def __init__(self) -> None:
        self._pipelines: dict[str, RAGPipeline] = {}
        self._pipeline_locks: dict[str, threading.Lock] = {}
        self._pipeline_locks_guard = threading.Lock()

    @property
    def pipelines(self) -> dict[str, RAGPipeline]:
        return self._pipelines

    def _get_local_pipeline_lock(self, knowledge_base_id: str) -> threading.Lock:
        with self._pipeline_locks_guard:
            lock = self._pipeline_locks.get(knowledge_base_id)
            if lock is None:
                lock = threading.Lock()
                self._pipeline_locks[knowledge_base_id] = lock
            return lock

    @staticmethod
    def _bootstrap_lock_path(knowledge_base_id: str) -> str:
        settings = _settings()
        digest = hashlib.sha256(knowledge_base_id.encode("utf-8")).hexdigest()[:16]
        return str(settings.resolved_chroma_path / ".pipeline-bootstrap-locks" / f"pipeline_{digest}.lock")

    def get(self, knowledge_base_id: str, subject: str) -> RAGPipeline:
        pipeline = self._pipelines.get(knowledge_base_id)
        if pipeline is not None:
            return pipeline

        settings = _settings()
        local_lock = self._get_local_pipeline_lock(knowledge_base_id)
        with local_lock:
            # 拿到进程内锁后再检查一次，避免等待期间同进程的其他请求已完成初始化。
            pipeline = self._pipelines.get(knowledge_base_id)
            if pipeline is not None:
                return pipeline

            lock_path = self._bootstrap_lock_path(knowledge_base_id)
            wait_started = perf_counter()
            logger.info(
                "rag_pipeline_bootstrap_waiting knowledge_base_id=%s lock_path=%s",
                knowledge_base_id,
                lock_path,
            )
            with advisory_file_lock(
                Path(lock_path),
                timeout_seconds=float(settings.pipeline_bootstrap_lock_timeout_seconds),
            ):
                # 文件锁用于保护 API/worker 容器共享挂载的 Chroma/Milvus 资源。
                pipeline = self._pipelines.get(knowledge_base_id)
                if pipeline is not None:
                    logger.info(
                        "rag_pipeline_bootstrap_reused_after_lock knowledge_base_id=%s wait_ms=%s",
                        knowledge_base_id,
                        int((perf_counter() - wait_started) * 1000),
                    )
                    return pipeline

                logger.info(
                    "rag_pipeline_bootstrap_started knowledge_base_id=%s wait_ms=%s",
                    knowledge_base_id,
                    int((perf_counter() - wait_started) * 1000),
                )
                pipeline = RAGPipeline(
                    settings=settings,
                    collection_name=_collection_name(settings.chroma_collection_name, knowledge_base_id),
                    subject=subject,
                )
                self._pipelines[knowledge_base_id] = pipeline
                logger.info(
                    "rag_pipeline_bootstrap_completed knowledge_base_id=%s total_ms=%s",
                    knowledge_base_id,
                    int((perf_counter() - wait_started) * 1000),
                )
        return pipeline

    def reset_index(self, knowledge_base_id: str, subject: str) -> None:
        self.get(knowledge_base_id, subject).reset_index()

    def delete_knowledge_base_index(self, knowledge_base_id: str, subject: str) -> None:
        pipeline = self._pipelines.pop(knowledge_base_id, None)
        if pipeline is None:
            pipeline = self.get(knowledge_base_id, subject)
        pipeline.delete_index()

    def delete_document_index(self, knowledge_base_id: str, subject: str, document_id: str) -> None:
        self.get(knowledge_base_id, subject).delete_document(document_id)

    def clear(self) -> None:
        self._pipelines.clear()
        with self._pipeline_locks_guard:
            self._pipeline_locks.clear()


_DEFAULT_PIPELINE_REGISTRY = RAGPipelineRegistry()
_PIPELINES = _DEFAULT_PIPELINE_REGISTRY.pipelines


def get_default_pipeline_registry() -> RAGPipelineRegistry:
    return _DEFAULT_PIPELINE_REGISTRY


def _official_source_name(filename: str) -> str:
    return resolve_source_name(filename)


def _build_source_name_map(documents: list[dict]) -> dict[str, str]:
    return {
        str(item["id"]): _official_source_name(str(item.get("filename") or ""))
        for item in documents
        if item.get("id")
    }


def _stable_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _match_light_intent_rule(normalized: str):
    config = load_light_intent_config()
    for rule in config.intent_rules:
        if any(re.fullmatch(pattern, normalized) for pattern in rule.patterns):
            return rule
    return None


def _match_light_intent(query: str) -> dict | None:
    normalized = normalize_query(query)
    if not normalized:
        return None
    rule = _match_light_intent_rule(normalized)
    if rule is None:
        clauses = split_light_intent_clauses(query)
        if len(clauses) > 1:
            matched_rules = []
            for clause in clauses:
                clause_rule = _match_light_intent_rule(clause)
                if clause_rule is None:
                    matched_rules = []
                    break
                matched_rules.append(clause_rule)
            if matched_rules:
                # 复合轻意图优先取靠后的非问候子句，例如“你好，你是？”应落到身份介绍。
                rule = next((item for item in reversed(matched_rules) if item.name != "greeting"), matched_rules[-1])
    if rule is not None:
        return {
            "name": rule.name,
            "response_key": rule.response_key,
            "finish_reason": rule.finish_reason,
        }
    return None



def _is_policy_related_query(query: str) -> bool:
    config = load_light_intent_config()
    return any(keyword in query for keyword in config.policy_keywords)



def _should_fallback_off_topic(query: str) -> bool:
    config = load_light_intent_config()
    normalized = normalize_query(query)
    if not normalized or _is_policy_related_query(query):
        return False
    return any(re.fullmatch(pattern, normalized) for pattern in config.off_topic_patterns)



def _build_light_intent_response(query: str, rule: dict, subject: str) -> dict:
    config = load_light_intent_config()
    response_key = rule["response_key"] if isinstance(rule, dict) else rule.response_key
    finish_reason = rule["finish_reason"] if isinstance(rule, dict) else rule.finish_reason
    answer = format_light_intent_response(config.responses[response_key], subject)
    return {
        "answer": answer,
        "conversation_id": "",
        "citations": [],
        "debug": {
            "rewritten_query": query,
            "retrieved_count": 0,
            "reranked_count": 0,
            "rewrite_ms": 0,
            "retrieve_ms": 0,
            "rerank_ms": 0,
            "generate_ms": 0,
            "latency_ms": 0,
            "llm_finish_reason": finish_reason,
            "llm_model": None,
            "llm_prompt_tokens": None,
            "llm_completion_tokens": None,
            "llm_total_tokens": None,
        },
    }



def _match_faq_shortcut(query: str) -> dict | None:
    config = load_faq_shortcut_config()
    normalized = normalize_query(query)
    if not normalized:
        return None
    for rule in config.rules:
        for pattern in rule.patterns:
            if re.fullmatch(pattern, normalized):
                return {
                    "name": rule.name,
                    "topic": rule.topic,
                    "matched_pattern": pattern,
                    "answer_template": rule.answer_template,
                    "suggested_queries": rule.suggested_queries,
                    "finish_reason": rule.finish_reason,
                }
    return None


def _build_faq_shortcut_response(query: str, rule: dict, subject: str) -> dict:
    answer_template = rule["answer_template"] if isinstance(rule, dict) else rule.answer_template
    topic = rule.get("topic") if isinstance(rule, dict) else rule.topic
    suggested_queries = tuple(rule.get("suggested_queries") or ()) if isinstance(rule, dict) else rule.suggested_queries
    finish_reason = rule["finish_reason"] if isinstance(rule, dict) else rule.finish_reason
    answer = format_faq_shortcut_response(
        answer_template,
        subject=subject,
        topic=topic,
        suggested_queries=suggested_queries,
    )
    return {
        "answer": answer,
        "answer_source": "faq_short_circuit",
        "conversation_id": "",
        "citations": [],
        "debug": {
            "rewritten_query": query,
            "retrieved_count": 0,
            "reranked_count": 0,
            "rewrite_ms": 0,
            "retrieve_ms": 0,
            "rerank_ms": 0,
            "generate_ms": 0,
            "latency_ms": 0,
            "llm_finish_reason": finish_reason,
            "llm_model": None,
            "llm_prompt_tokens": None,
            "llm_completion_tokens": None,
            "llm_total_tokens": None,
        },
    }


def _build_off_topic_response(query: str, subject: str, *, finish_reason: str = "off_topic_short_circuit") -> dict:
    config = load_light_intent_config()
    return {
        "answer": format_light_intent_response(config.responses["off_topic"], subject),
        "conversation_id": "",
        "citations": [],
        "debug": {
            "rewritten_query": query,
            "retrieved_count": 0,
            "reranked_count": 0,
            "rewrite_ms": 0,
            "retrieve_ms": 0,
            "rerank_ms": 0,
            "generate_ms": 0,
            "latency_ms": 0,
            "llm_finish_reason": finish_reason,
            "llm_model": None,
            "llm_prompt_tokens": None,
            "llm_completion_tokens": None,
            "llm_total_tokens": None,
        },
    }


def _friendly_domain_label(domain: str) -> str:
    return load_cross_domain_guard_config().domain_labels.get(domain, domain)


def _detect_cross_domain_compound_query(query: str) -> list[str]:
    normalized_query = normalize_query(query)
    if not normalized_query:
        return []
    guard_config = load_cross_domain_guard_config()
    if not any(normalize_query(connector) in normalized_query for connector in guard_config.connectors):
        return []

    matched_domains: list[str] = []
    for domain, keywords in load_knowledge_base_routing_config().domains.keywords.items():
        if any(normalize_query(keyword) in normalized_query for keyword in keywords):
            matched_domains.append(domain)
    return matched_domains if len(matched_domains) >= 2 else []


def _build_cross_domain_guard_response(domains: list[str]) -> str:
    guard_config = load_cross_domain_guard_config()
    labels = [_friendly_domain_label(domain) for domain in domains]
    joined = "、".join(f"“{label}”" for label in labels)
    sample_questions = [
        guard_config.split_question_examples.get(domain)
        for domain in domains
        if guard_config.split_question_examples.get(domain)
    ]
    clarification_prefix = guard_config.clarification_prefix.format(domains=joined)
    clarification_question = guard_config.clarification_question.format(domains=joined)
    if sample_questions:
        clarification_suffix = guard_config.clarification_suffix.format(examples="；".join(sample_questions))
    else:
        clarification_suffix = "你也可以按不同业务域拆开提问。"
    return f"{clarification_prefix}{clarification_question}{clarification_suffix}"


def _find_unsupported_qualifiers(query: str, final_context_chunks: list[dict[str, object]]) -> list[str]:
    if not final_context_chunks:
        return []

    normalized_query = normalize_query(query)
    context_text = normalize_query(
        "\n".join(
            f"{chunk.get('parent_text') or ''}\n{chunk.get('source_name') or ''}"
            for chunk in final_context_chunks
        )
    )
    unsupported: list[str] = []
    for rule in load_answer_guard_config().qualifier_rules:
        if not any(normalize_query(term) in normalized_query for term in rule.query_terms):
            continue
        if any(normalize_query(term) in context_text for term in rule.evidence_terms):
            continue
        unsupported.append(rule.name)
    return unsupported


def _build_answer_guard_response(unsupported_qualifiers: list[str]) -> str:
    joined = "、".join(f"“{item}”" for item in unsupported_qualifiers)
    return (
        f"现有资料中没有明确覆盖{joined}这一限定场景，不能直接把已检索到的一般条款套用到该问题上。"
        "如果你想问的是一般规则，请直接说明一般场景；如果你问的是特定机构或特定阶段的管理规定，请补充更明确的主体和制度范围。"
    )


def _looks_like_no_answer_response(answer: str) -> bool:
    if "《" in answer and "》" in answer and any(marker in answer for marker in ("规定", "明确", "条款", "依据")):
        return False

    # 区间类回答可能合理覆盖多个档位，同时承认某个档位缺失；这种情况不算无答案。
    if len(re.findall(r"\d+\s*[—-]\s*\d+\s*分", answer)) >= 2:
        return False

    normalized = normalize_query(answer)
    if not normalized:
        return False

    positive_markers = (
        "需要区分具体情形",
        "按具体情形处理",
        "按对应规则处理",
        "按分段处理",
        "按分数段处理",
        "按具体行为",
    )
    if any(marker in normalized for marker in positive_markers):
        return False

    markers = (
        "根据现有参考资料无法确定",
        "根据现有资料无法确定",
        "参考资料中未提供关于",
        "资料未提供关于",
        "参考资料未明确",
        "没有直接对应条款",
        "无法依据资料判断",
    )
    return any(marker in normalized for marker in markers)


def _clear_final_context_for_no_answer(result: dict) -> None:
    result["citations"] = []
    debug_payload = result.get("debug") or {}
    debug_payload["final_context_chunks"] = []
    debug_payload["reranked_chunks"] = []
    debug_payload["reranked_count"] = 0


def _apply_route_debug_fields(
    result: dict,
    *,
    payload: ChatQueryRequest,
    routed,
    resolved_knowledge_base_id: str,
    knowledge_base_name: str | None,
) -> None:
    result["knowledge_base_id"] = resolved_knowledge_base_id
    result["knowledge_base_name"] = knowledge_base_name
    result["auto_routed"] = bool(routed and routed.auto_routed)
    result["debug"]["requested_knowledge_base_id"] = payload.knowledge_base_id
    result["debug"]["resolved_knowledge_base_id"] = resolved_knowledge_base_id
    result["debug"]["resolved_knowledge_base_name"] = knowledge_base_name
    result["debug"]["auto_routed"] = bool(routed and routed.auto_routed)
    result["debug"]["routing_reason"] = routed.reason if routed else "未启用自动路由"


def _set_guard_debug_defaults(result: dict) -> None:
    result["debug"]["cross_domain_guard_applied"] = False
    result["debug"]["detected_domains"] = []
    result["debug"]["cross_domain_guard_reason"] = None
    result["debug"]["answer_guard_applied"] = False
    result["debug"]["unsupported_qualifiers"] = []
    result["debug"]["answer_guard_reason"] = None
    result["debug"]["faq_short_circuit_applied"] = False
    result["debug"]["faq_rule_name"] = None
    result["debug"]["faq_topic"] = None


def _apply_intent_debug_fields(result: dict, decision: IntentDecision) -> None:
    result["debug"]["intent_name"] = decision.intent.value
    result["debug"]["intent_source"] = decision.source
    result["debug"]["intent_classifier_source"] = decision.classifier_source
    result["debug"]["intent_classifier_mode"] = decision.classifier_mode
    result["debug"]["intent_classifier_score"] = decision.classifier_score
    result["debug"]["intent_classifier_margin"] = decision.classifier_margin


def _update_stage_timing_debug_fields(result: dict) -> None:
    debug = result.get("debug")
    if not isinstance(debug, dict):
        return

    stage_timings = dict(debug.get("stage_timings_ms") or {})
    latency_ms = int(debug.get("latency_ms") or 0)
    history_rewrite_ms = int(debug.get("history_rewrite_ms") or 0)
    rewrite_ms = int(debug.get("rewrite_ms") or 0)
    retrieve_ms = int(debug.get("retrieve_ms") or 0)
    rerank_ms = int(debug.get("rerank_ms") or 0)
    context_build_ms = int(debug.get("context_build_ms") or 0)
    generate_ms = int(debug.get("generate_ms") or 0)
    llm_first_token_ms = debug.get("llm_first_token_ms")

    llm_after_first_token_ms: int | None = None
    if isinstance(llm_first_token_ms, int):
        llm_after_first_token_ms = max(generate_ms - llm_first_token_ms, 0)

    accounted_ms = history_rewrite_ms + rewrite_ms + retrieve_ms + rerank_ms + context_build_ms + generate_ms
    stage_timings.update(
        {
            "history_rewrite_ms": history_rewrite_ms,
            "rewrite_ms": rewrite_ms,
            "retrieve_ms": retrieve_ms,
            "rerank_ms": rerank_ms,
            "context_build_ms": context_build_ms,
            "generate_ms": generate_ms,
            "llm_first_token_ms": llm_first_token_ms,
            "llm_after_first_token_ms": llm_after_first_token_ms,
            "service_overhead_ms": max(latency_ms - accounted_ms, 0),
        }
    )
    debug["stage_timings_ms"] = stage_timings


def _compare_pipeline_and_shadow_graph(
    legacy_result: dict,
    shadow_result: dict,
    *,
    shadow_latency_ms: int,
) -> dict[str, object]:
    legacy_debug = dict(legacy_result.get("debug") or {})
    shadow_debug = dict(shadow_result.get("debug") or {})

    legacy_citations = list(legacy_result.get("citations") or [])
    shadow_citations = list(shadow_result.get("citations") or [])
    legacy_final_context = list(legacy_debug.get("final_context_chunks") or [])
    shadow_final_context = list(shadow_debug.get("final_context_chunks") or [])

    legacy_citation_ids = [str(item.get("chunk_id") or "") for item in legacy_citations]
    shadow_citation_ids = [str(item.get("chunk_id") or "") for item in shadow_citations]
    legacy_final_context_ids = [str(item.get("chunk_id") or "") for item in legacy_final_context]
    shadow_final_context_ids = [str(item.get("chunk_id") or "") for item in shadow_final_context]

    answer_match = str(legacy_result.get("answer") or "") == str(shadow_result.get("answer") or "")
    citation_count_match = len(legacy_citations) == len(shadow_citations)
    citation_ids_match = legacy_citation_ids == shadow_citation_ids
    final_context_ids_match = legacy_final_context_ids == shadow_final_context_ids
    rewritten_query_match = str(legacy_debug.get("rewritten_query") or "") == str(shadow_debug.get("rewritten_query") or "")

    mismatch_fields: list[str] = []
    if not answer_match:
        mismatch_fields.append("answer")
    if not citation_count_match:
        mismatch_fields.append("citation_count")
    if not citation_ids_match:
        mismatch_fields.append("citation_ids")
    if not final_context_ids_match:
        mismatch_fields.append("final_context_ids")
    if not rewritten_query_match:
        mismatch_fields.append("rewritten_query")

    return {
        "status": "match" if not mismatch_fields else "mismatch",
        "compared_stage": "pipeline_core",
        "answer_match": answer_match,
        "citation_count_match": citation_count_match,
        "citation_ids_match": citation_ids_match,
        "final_context_ids_match": final_context_ids_match,
        "rewritten_query_match": rewritten_query_match,
        "mismatch_fields": mismatch_fields,
        "legacy_answer_hash": _stable_hash(str(legacy_result.get("answer") or "")),
        "shadow_answer_hash": _stable_hash(str(shadow_result.get("answer") or "")),
        "legacy_citation_count": len(legacy_citations),
        "shadow_citation_count": len(shadow_citations),
        "legacy_final_context_count": len(legacy_final_context),
        "shadow_final_context_count": len(shadow_final_context),
        "shadow_latency_ms": shadow_latency_ms,
        "error": None,
    }


def _run_shadow_graph_compare(
    pipeline: RAGPipeline,
    *,
    query: str,
    top_k_retrieve: int,
    top_k_rerank: int,
    knowledge_base_id: str,
    source_name_by_document_id: dict[str, str] | None,
    enable_rewrite: bool,
    enable_rerank: bool,
    answer_style: str,
    query_prep: dict[str, object] | None,
    history_rewritten_query: str | None,
    history_message_count: int,
    history_rewrite_ms: int,
    legacy_result: dict,
) -> dict[str, object]:
    started_at = perf_counter()
    try:
        shadow_result = pipeline.run_shadow_graph(
            query=query,
            top_k_retrieve=top_k_retrieve,
            top_k_rerank=top_k_rerank,
            knowledge_base_id=knowledge_base_id,
            source_name_by_document_id=source_name_by_document_id,
            enable_rewrite=enable_rewrite,
            enable_rerank=enable_rerank,
            answer_style=answer_style,
            query_prep=query_prep,
            history_rewritten_query=history_rewritten_query,
            history_message_count=history_message_count,
            history_rewrite_ms=history_rewrite_ms,
        )
    except Exception as exc:
        return {
            "status": "error",
            "compared_stage": "pipeline_core",
            "answer_match": False,
            "citation_count_match": False,
            "citation_ids_match": False,
            "final_context_ids_match": False,
            "rewritten_query_match": False,
            "mismatch_fields": ["shadow_graph_error"],
            "legacy_answer_hash": _stable_hash(str(legacy_result.get("answer") or "")),
            "shadow_answer_hash": None,
            "legacy_citation_count": len(list(legacy_result.get("citations") or [])),
            "shadow_citation_count": 0,
            "legacy_final_context_count": len(list((legacy_result.get("debug") or {}).get("final_context_chunks") or [])),
            "shadow_final_context_count": 0,
            "shadow_latency_ms": int((perf_counter() - started_at) * 1000),
            "error": str(exc),
        }

    return _compare_pipeline_and_shadow_graph(
        legacy_result,
        shadow_result,
        shadow_latency_ms=int((perf_counter() - started_at) * 1000),
    )


def _run_shadow_graph_result(
    pipeline: RAGPipeline,
    *,
    query: str,
    top_k_retrieve: int,
    top_k_rerank: int,
    knowledge_base_id: str,
    source_name_by_document_id: dict[str, str] | None,
    enable_rewrite: bool,
    enable_rerank: bool,
    answer_style: str,
    query_prep: dict[str, object] | None,
    history_rewritten_query: str | None,
    history_message_count: int,
    history_rewrite_ms: int,
) -> tuple[dict[str, object], int]:
    started_at = perf_counter()
    shadow_result = pipeline.run_shadow_graph(
        query=query,
        top_k_retrieve=top_k_retrieve,
        top_k_rerank=top_k_rerank,
        knowledge_base_id=knowledge_base_id,
        source_name_by_document_id=source_name_by_document_id,
        enable_rewrite=enable_rewrite,
        enable_rerank=enable_rerank,
        answer_style=answer_style,
        query_prep=query_prep,
        history_rewritten_query=history_rewritten_query,
        history_message_count=history_message_count,
        history_rewrite_ms=history_rewrite_ms,
    )
    return shadow_result, int((perf_counter() - started_at) * 1000)


def _should_run_shadow_compare(payload: ChatQueryRequest) -> bool:
    if payload.debug_shadow_compare:
        return True
    settings = _settings()
    if not settings.chat_shadow_compare_enabled:
        return False
    sample_rate = settings.chat_shadow_compare_sample_rate
    if sample_rate <= 0:
        return False
    return random.random() < sample_rate


def _log_shadow_compare_result(compare_summary: dict[str, object], *, knowledge_base_id: str, query: str) -> None:
    status = str(compare_summary.get("status") or "unknown")
    if status == "match":
        return
    logger.warning(
        "shadow_compare_%s kb=%s query_hash=%s mismatch_fields=%s shadow_latency_ms=%s error=%s",
        status,
        knowledge_base_id,
        _stable_hash(query),
        compare_summary.get("mismatch_fields"),
        compare_summary.get("shadow_latency_ms"),
        compare_summary.get("error"),
    )


def _resolve_answer_style(payload: ChatQueryRequest) -> str:
    if payload.debug_answer_style is not None:
        return payload.debug_answer_style
    return "structured" if payload.debug else "concise"


def _prepare_pipeline_query_inputs(
    pipeline: RAGPipeline,
    *,
    query: str,
    enable_rewrite: bool,
    history_rewritten_query: str | None,
) -> dict[str, object] | None:
    if not hasattr(pipeline, "prepare_query_inputs"):
        return None
    return pipeline.prepare_query_inputs(
        query=query,
        enable_rewrite=enable_rewrite,
        history_rewritten_query=history_rewritten_query,
    )


def _persist_short_circuit_result(
    repository,
    *,
    conversation_id: str,
    knowledge_base_id: str,
    query: str,
    result: dict,
    include_debug: bool,
) -> None:
    user_message = repository.create_message(
        conversation_id=conversation_id,
        role="user",
        content=query,
    )
    repository.create_message_context(
        message_id=user_message["id"],
        knowledge_base_id=knowledge_base_id,
    )
    assistant_message = repository.create_message(
        conversation_id=conversation_id,
        role="assistant",
        content=result["answer"],
    )
    repository.create_message_context(
        message_id=assistant_message["id"],
        knowledge_base_id=knowledge_base_id,
        citations=[],
        debug=result["debug"] if include_debug else None,
    )


def _persist_answer_result(
    repository,
    *,
    conversation_id: str,
    knowledge_base_id: str,
    query: str,
    result: dict,
    include_debug: bool,
) -> None:
    user_message = repository.create_message(
        conversation_id=conversation_id,
        role="user",
        content=query,
    )
    repository.create_message_context(
        message_id=user_message["id"],
        knowledge_base_id=knowledge_base_id,
    )
    assistant_message = repository.create_message(
        conversation_id=conversation_id,
        role="assistant",
        content=result["answer"],
    )
    repository.create_message_context(
        message_id=assistant_message["id"],
        knowledge_base_id=knowledge_base_id,
        citations=result["citations"],
        debug=result["debug"] if include_debug else None,
    )


def _build_persistable_error_message(exc: Exception) -> str:
    message = str(exc).strip()
    normalized = message.lower()

    if "incorrect api key" in normalized or "invalid_api_key" in normalized:
        return "LLM 服务鉴权失败，请检查 OPENAI_API_KEY / REWRITE_API_KEY 配置。"
    if "rate limit" in normalized:
        return "LLM 服务当前触发限流，请稍后重试。"
    if not message:
        return "本轮回答失败，请稍后重试。"
    return f"本轮回答失败：{message}"


def _persist_error_result(
    repository,
    *,
    conversation_id: str,
    knowledge_base_id: str,
    query: str,
    error_message: str,
) -> None:
    user_message = repository.create_message(
        conversation_id=conversation_id,
        role="user",
        content=query,
    )
    repository.create_message_context(
        message_id=user_message["id"],
        knowledge_base_id=knowledge_base_id,
    )
    assistant_message = repository.create_message(
        conversation_id=conversation_id,
        role="assistant",
        content=error_message,
    )
    repository.create_message_context(
        message_id=assistant_message["id"],
        knowledge_base_id=knowledge_base_id,
        citations=[],
        debug=None,
    )



def get_rag_pipeline(knowledge_base_id: str, subject: str) -> RAGPipeline:
    return get_default_pipeline_registry().get(knowledge_base_id, subject)



def reset_knowledge_base_index(knowledge_base_id: str, subject: str) -> None:
    get_default_pipeline_registry().reset_index(knowledge_base_id, subject)



def delete_knowledge_base_index(knowledge_base_id: str, subject: str) -> None:
    get_default_pipeline_registry().delete_knowledge_base_index(knowledge_base_id, subject)


def delete_document_index(knowledge_base_id: str, subject: str, document_id: str) -> None:
    get_default_pipeline_registry().delete_document_index(knowledge_base_id, subject, document_id)



def bootstrap_default_knowledge_base(*, pipeline_registry: RAGPipelineRegistry | None = None) -> None:
    settings = _settings()
    if not settings.bootstrap_default_knowledge_base:
        return
    effective_pipeline_registry = pipeline_registry or get_default_pipeline_registry()

    repository = get_metadata_repository()
    source_path = settings.resolved_source_document_path
    if source_path is None:
        return
    if not source_path.exists():
        return

    knowledge_base = repository.get_knowledge_base(settings.default_knowledge_base_id)
    if knowledge_base is None:
        now = datetime.now(UTC).isoformat()
        knowledge_base = {
            "id": settings.default_knowledge_base_id,
            "name": settings.default_knowledge_base_name,
            "description": settings.default_knowledge_base_description,
            "subject": settings.knowledge_base_subject,
            "domain": settings.default_knowledge_base_domain,
            "status": "empty",
            "created_at": now,
            "updated_at": now,
        }
        repository.upsert_knowledge_base(knowledge_base)

    existing_documents = repository.list_documents(settings.default_knowledge_base_id)
    source_document = next((item for item in existing_documents if item["filename"] == source_path.name), None)
    if source_document is None:
        source_document = repository.create_document(
            knowledge_base_id=settings.default_knowledge_base_id,
            filename=source_path.name,
            content_type="text/plain",
            file_size=source_path.stat().st_size,
            content_hash=hashlib.sha256(source_path.read_bytes()).hexdigest(),
            file_path=str(source_path),
        )

    if source_document["status"] != "ready":
        repository.update_document(source_document["id"], status="indexing")
        repository.update_knowledge_base(settings.default_knowledge_base_id, status="indexing")
        pipeline = effective_pipeline_registry.get(settings.default_knowledge_base_id, settings.knowledge_base_subject)
        pipeline.ingest_file(source_path, document_id=source_document["id"])
        repository.update_document(source_document["id"], status="ready")
        repository.update_knowledge_base(settings.default_knowledge_base_id, status="ready")


def bootstrap_default_knowledge_base_with_runtime(
    *,
    repository: MetadataRepository,
    pipeline_registry: RAGPipelineRegistry,
) -> None:
    settings = _settings()
    if not settings.bootstrap_default_knowledge_base:
        return

    source_path = settings.resolved_source_document_path
    if source_path is None or not source_path.exists():
        return

    knowledge_base = repository.get_knowledge_base(settings.default_knowledge_base_id)
    if knowledge_base is None:
        now = datetime.now(UTC).isoformat()
        knowledge_base = {
            "id": settings.default_knowledge_base_id,
            "name": settings.default_knowledge_base_name,
            "description": settings.default_knowledge_base_description,
            "subject": settings.knowledge_base_subject,
            "domain": settings.default_knowledge_base_domain,
            "status": "empty",
            "created_at": now,
            "updated_at": now,
        }
        repository.upsert_knowledge_base(knowledge_base)

    existing_documents = repository.list_documents(settings.default_knowledge_base_id)
    source_document = next((item for item in existing_documents if item["filename"] == source_path.name), None)
    if source_document is None:
        source_document = repository.create_document(
            knowledge_base_id=settings.default_knowledge_base_id,
            filename=source_path.name,
            content_type="text/plain",
            file_size=source_path.stat().st_size,
            content_hash=hashlib.sha256(source_path.read_bytes()).hexdigest(),
            file_path=str(source_path),
        )

    if source_document["status"] != "ready":
        repository.update_document(source_document["id"], status="indexing")
        repository.update_knowledge_base(settings.default_knowledge_base_id, status="indexing")
        pipeline = pipeline_registry.get(settings.default_knowledge_base_id, settings.knowledge_base_subject)
        pipeline.ingest_file(source_path, document_id=source_document["id"])
        repository.update_document(source_document["id"], status="ready")
        repository.update_knowledge_base(settings.default_knowledge_base_id, status="ready")


class RAGService:
    """普通问答和流式问答共用的应用服务。

    这里刻意把请求编排和 ``RAGPipeline`` 的检索生成逻辑分开，使 API、测试和后台
    流程可以复用同一套路由、意图短路、答案保护和持久化逻辑。
    """

    def __init__(
        self,
        *,
        repository: MetadataRepository | None = None,
        intent_router: IntentRouter | None = None,
        pipeline_registry: RAGPipelineRegistry | None = None,
    ) -> None:
        self.repository = repository or get_metadata_repository()
        self.intent_router = intent_router or IntentRouter()
        self.pipeline_registry = pipeline_registry or get_default_pipeline_registry()

    def _get_pipeline(self, knowledge_base_id: str, subject: str) -> RAGPipeline:
        if self.pipeline_registry is get_default_pipeline_registry():
            return get_rag_pipeline(knowledge_base_id, subject)
        return self.pipeline_registry.get(knowledge_base_id, subject)

    @staticmethod
    def _build_conversation_title(query: str) -> str:
        title = re.sub(r"\s+", " ", query).strip()
        return (title[:50] or "新对话")

    def _resolve_conversation(self, knowledge_base_id: str, conversation_id: str | None, query: str) -> dict:
        if conversation_id:
            conversation = self.repository.get_conversation(conversation_id)
            if conversation is None:
                raise ConversationNotFoundError()
            return conversation
        return self.repository.create_conversation(
            title=self._build_conversation_title(query),
            default_knowledge_base_id=knowledge_base_id,
        )

    @dataclass
    class _ResolvedQueryContext:
        payload: ChatQueryRequest
        routed: object | None
        resolved_knowledge_base_id: str
        knowledge_base: dict[str, object]
        conversation: dict[str, object]
        conversation_messages: list[dict[str, object]]

        @property
        def has_history(self) -> bool:
            return bool(self.conversation_messages)

    @dataclass
    class _PreparedPipelineContext:
        started_at: float
        pipeline: RAGPipeline
        answer_style: str
        source_name_by_document_id: dict[str, str]
        context_resolution: object
        history_rewritten_query: str | None
        query_prep: dict[str, object] | None

    def _resolve_query_context(self, payload: ChatQueryRequest) -> _ResolvedQueryContext:
        routed = None
        if _settings().chat_auto_route_enabled and payload.enable_auto_route:
            # 自动路由会结合知识库元数据决定最终 KB，解析结果会写入 debug。
            routed = KnowledgeBaseRouter(self.repository).route(
                query=payload.query,
                requested_knowledge_base_id=payload.knowledge_base_id,
            )

        resolved_knowledge_base_id = routed.knowledge_base_id if routed else payload.knowledge_base_id
        if not resolved_knowledge_base_id:
            raise KnowledgeBaseNotFoundError()

        knowledge_base = self.repository.get_knowledge_base(resolved_knowledge_base_id)
        if knowledge_base is None:
            raise KnowledgeBaseNotFoundError()
        if knowledge_base["status"] not in {"ready", "indexing"}:
            raise KnowledgeBaseNotReadyError()

        conversation = self._resolve_conversation(
            knowledge_base_id=resolved_knowledge_base_id,
            conversation_id=payload.conversation_id,
            query=payload.query,
        )
        conversation_messages = list(self.repository.list_messages(conversation["id"]))
        return self._ResolvedQueryContext(
            payload=payload,
            routed=routed,
            resolved_knowledge_base_id=resolved_knowledge_base_id,
            knowledge_base=knowledge_base,
            conversation=conversation,
            conversation_messages=conversation_messages,
        )

    @staticmethod
    def _build_intent_history_messages(conversation_messages: list[dict[str, object]]) -> list[dict[str, str]]:
        return [
            {
                "role": str(message.get("role") or ""),
                "content": str(message.get("content") or ""),
            }
            for message in conversation_messages[-6:]
        ]

    def _classify_intent(self, context: _ResolvedQueryContext) -> IntentDecision:
        return self.intent_router.classify(
            context.payload.query,
            has_history=context.has_history,
            history_messages=self._build_intent_history_messages(context.conversation_messages),
        )

    def _build_start_event(self, context: _ResolvedQueryContext) -> dict[str, object]:
        return {
            "event": "start",
            "data": {
                "conversation_id": context.conversation["id"],
                "knowledge_base_id": context.resolved_knowledge_base_id,
                "knowledge_base_name": context.knowledge_base.get("name"),
                "auto_routed": bool(context.routed and context.routed.auto_routed),
            },
        }

    def _build_cross_domain_short_circuit_result(
        self,
        context: _ResolvedQueryContext,
        *,
        detected_domains: list[str],
        include_stream_debug_fields: bool,
    ) -> dict:
        debug = {
            "rewritten_query": context.payload.query,
            "retrieved_count": 0,
            "reranked_count": 0,
            "rewrite_ms": 0,
            "retrieve_ms": 0,
            "rerank_ms": 0,
            "generate_ms": 0,
            "latency_ms": 0,
            "llm_finish_reason": "cross_domain_guard_short_circuit",
            "llm_model": None,
            "llm_prompt_tokens": None,
            "llm_completion_tokens": None,
            "llm_total_tokens": None,
        }
        if include_stream_debug_fields:
            debug["llm_first_token_ms"] = None
        result = {
            "answer": _build_cross_domain_guard_response(detected_domains),
            "conversation_id": context.conversation["id"],
            "citations": [],
            "debug": debug,
        }
        _apply_route_debug_fields(
            result,
            payload=context.payload,
            routed=context.routed,
            resolved_knowledge_base_id=context.resolved_knowledge_base_id,
            knowledge_base_name=context.knowledge_base.get("name"),
        )
        _set_guard_debug_defaults(result)
        result["debug"]["cross_domain_guard_applied"] = True
        result["debug"]["detected_domains"] = detected_domains
        result["debug"]["cross_domain_guard_reason"] = "检测到跨业务域复合问题，转为拆问提示"
        return result

    def _build_intent_short_circuit_result(
        self,
        context: _ResolvedQueryContext,
        decision: IntentDecision,
    ) -> dict | None:
        result: dict | None = None
        if decision.intent == IntentType.LIGHT_INTENT and decision.light_intent_rule is not None:
            result = _build_light_intent_response(
                context.payload.query,
                decision.light_intent_rule,
                context.knowledge_base["subject"],
            )
        elif decision.intent == IntentType.FAQ_SHORTCUT and decision.faq_shortcut_rule is not None:
            result = _build_faq_shortcut_response(
                context.payload.query,
                decision.faq_shortcut_rule,
                context.knowledge_base["subject"],
            )
        elif decision.intent in {IntentType.OFF_TOPIC, IntentType.MEANINGLESS_INPUT}:
            result = _build_off_topic_response(
                context.payload.query,
                context.knowledge_base["subject"],
                finish_reason=decision.finish_reason or "off_topic_short_circuit",
            )
        if result is None:
            return None

        result["conversation_id"] = context.conversation["id"]
        _apply_route_debug_fields(
            result,
            payload=context.payload,
            routed=context.routed,
            resolved_knowledge_base_id=context.resolved_knowledge_base_id,
            knowledge_base_name=context.knowledge_base.get("name"),
        )
        _set_guard_debug_defaults(result)
        if decision.intent == IntentType.FAQ_SHORTCUT and decision.faq_shortcut_rule is not None:
            result["debug"]["faq_short_circuit_applied"] = True
            result["debug"]["faq_rule_name"] = decision.faq_shortcut_rule.name
            result["debug"]["faq_topic"] = decision.faq_shortcut_rule.topic
        _apply_intent_debug_fields(result, decision)
        return result

    def _resolve_short_circuit_result(
        self,
        context: _ResolvedQueryContext,
        *,
        include_stream_debug_fields: bool,
    ) -> tuple[IntentDecision | None, dict | None]:
        # 低成本保护逻辑先于检索执行，避免问候、FAQ 快捷问答、跨业务域复合问题消耗模型/向量资源。
        detected_domains = _detect_cross_domain_compound_query(context.payload.query)
        if detected_domains:
            return None, self._build_cross_domain_short_circuit_result(
                context,
                detected_domains=detected_domains,
                include_stream_debug_fields=include_stream_debug_fields,
            )

        intent_decision = self._classify_intent(context)
        return intent_decision, self._build_intent_short_circuit_result(context, intent_decision)

    def _persist_short_circuit_result(
        self,
        context: _ResolvedQueryContext,
        *,
        result: dict,
    ) -> None:
        _persist_short_circuit_result(
            self.repository,
            conversation_id=context.conversation["id"],
            knowledge_base_id=context.resolved_knowledge_base_id,
            query=context.payload.query,
            result=result,
            include_debug=context.payload.debug,
        )

    def _persist_failure_result(self, context: _ResolvedQueryContext, exc: Exception) -> None:
        try:
            _persist_error_result(
                self.repository,
                conversation_id=context.conversation["id"],
                knowledge_base_id=context.resolved_knowledge_base_id,
                query=context.payload.query,
                error_message=_build_persistable_error_message(exc),
            )
        except Exception:
            logger.exception(
                "rag_query_failure_persist_failed conversation_id=%s knowledge_base_id=%s",
                context.conversation.get("id"),
                context.resolved_knowledge_base_id,
            )

    def _prepare_pipeline_context(self, context: _ResolvedQueryContext) -> _PreparedPipelineContext:
        started_at = perf_counter()
        pipeline = self._get_pipeline(context.resolved_knowledge_base_id, context.knowledge_base["subject"])
        answer_style = _resolve_answer_style(context.payload)
        source_name_by_document_id = _build_source_name_map(
            self.repository.list_documents(context.resolved_knowledge_base_id)
        )
        history_rewrite_started = perf_counter()
        history_rewrite_ms = int((perf_counter() - history_rewrite_started) * 1000)
        context_resolver = ConversationContextResolver(self.repository, pipeline.rewriter)
        try:
            # 追问会先改写成独立问题，但会话里仍保存和展示用户原始输入。
            context_resolution = context_resolver.resolve(
                conversation_id=context.conversation["id"],
                knowledge_base_id=context.resolved_knowledge_base_id,
                query=context.payload.query,
                history_rewrite_ms=history_rewrite_ms,
            )
        except Exception:
            from .conversation_context_resolver import ConversationContextResolution

            context_resolution = ConversationContextResolution(
                standalone_query=context.payload.query,
                used_history=False,
                history_message_count=0,
                history_rewrite_ms=history_rewrite_ms,
                history_message_ids=[],
            )
        history_rewritten_query = context_resolution.standalone_query if context_resolution.used_history else None
        query_prep = _prepare_pipeline_query_inputs(
            pipeline,
            query=context.payload.query,
            enable_rewrite=context.payload.enable_rewrite,
            history_rewritten_query=history_rewritten_query,
        )
        return self._PreparedPipelineContext(
            started_at=started_at,
            pipeline=pipeline,
            answer_style=answer_style,
            source_name_by_document_id=source_name_by_document_id,
            context_resolution=context_resolution,
            history_rewritten_query=history_rewritten_query,
            query_prep=query_prep,
        )

    def _build_pipeline_kwargs(
        self,
        context: _ResolvedQueryContext,
        pipeline_context: _PreparedPipelineContext,
    ) -> dict[str, object]:
        return {
            "query": context.payload.query,
            "top_k_retrieve": context.payload.top_k_retrieve,
            "top_k_rerank": context.payload.top_k_rerank,
            "knowledge_base_id": context.resolved_knowledge_base_id,
            "source_name_by_document_id": pipeline_context.source_name_by_document_id,
            "enable_rewrite": context.payload.enable_rewrite,
            "enable_rerank": context.payload.enable_rerank,
            "answer_style": pipeline_context.answer_style,
            "query_prep": pipeline_context.query_prep,
            "history_rewritten_query": pipeline_context.history_rewritten_query,
            "history_message_count": pipeline_context.context_resolution.history_message_count,
            "history_rewrite_ms": pipeline_context.context_resolution.history_rewrite_ms,
        }

    def _finalize_answer_result(
        self,
        context: _ResolvedQueryContext,
        pipeline_context: _PreparedPipelineContext,
        *,
        result: dict,
        intent_decision: IntentDecision,
    ) -> dict:
        result["debug"]["latency_ms"] = int((perf_counter() - pipeline_context.started_at) * 1000)
        _update_stage_timing_debug_fields(result)
        result["conversation_id"] = context.conversation["id"]
        result["knowledge_base_id"] = context.resolved_knowledge_base_id
        result["knowledge_base_name"] = context.knowledge_base.get("name")
        result["auto_routed"] = bool(context.routed and context.routed.auto_routed)
        result["debug"]["requested_knowledge_base_id"] = context.payload.knowledge_base_id
        result["debug"]["resolved_knowledge_base_id"] = context.resolved_knowledge_base_id
        result["debug"]["resolved_knowledge_base_name"] = context.knowledge_base.get("name")
        result["debug"]["auto_routed"] = bool(context.routed and context.routed.auto_routed)
        result["debug"]["routing_reason"] = context.routed.reason if context.routed else "未启用自动路由"
        _apply_intent_debug_fields(result, intent_decision)
        result["debug"]["cross_domain_guard_applied"] = False
        result["debug"]["detected_domains"] = []
        result["debug"]["cross_domain_guard_reason"] = None
        result["debug"]["faq_short_circuit_applied"] = False
        result["debug"]["faq_rule_name"] = None
        result["debug"]["faq_topic"] = None

        unsupported_qualifiers = _find_unsupported_qualifiers(
            context.payload.query,
            list(result["debug"].get("final_context_chunks") or []),
        )
        if unsupported_qualifiers:
            # 用户问题里的限定词如果没有被检索证据覆盖，优先给保守回答，避免硬编引用。
            result["answer"] = _build_answer_guard_response(unsupported_qualifiers)
            result["citations"] = []
            result["debug"]["answer_guard_applied"] = True
            result["debug"]["unsupported_qualifiers"] = unsupported_qualifiers
            result["debug"]["answer_guard_reason"] = "限定词未被检索证据覆盖，转为保守回答"
        else:
            result["debug"]["answer_guard_applied"] = False
            result["debug"]["unsupported_qualifiers"] = []
            result["debug"]["answer_guard_reason"] = None
            if _looks_like_no_answer_response(str(result.get("answer") or "")):
                _clear_final_context_for_no_answer(result)

        _persist_answer_result(
            self.repository,
            conversation_id=context.conversation["id"],
            knowledge_base_id=context.resolved_knowledge_base_id,
            query=context.payload.query,
            result=result,
            include_debug=context.payload.debug,
        )
        return result

    async def query(self, payload: ChatQueryRequest) -> dict:
        """处理一次非流式问答，并在结束后持久化结果。"""
        context = self._resolve_query_context(payload)
        try:
            intent_decision, short_circuit_result = self._resolve_short_circuit_result(
                context,
                include_stream_debug_fields=False,
            )
            if short_circuit_result is not None:
                self._persist_short_circuit_result(context, result=short_circuit_result)
                return short_circuit_result

            if intent_decision is None:
                raise RuntimeError("intent decision is required for pipeline execution")

            pipeline_context = self._prepare_pipeline_context(context)
            pipeline_kwargs = self._build_pipeline_kwargs(context, pipeline_context)
            result = pipeline_context.pipeline.ask(**pipeline_kwargs)
            result["answer_source"] = "legacy_pipeline"
            if payload.debug and payload.debug_force_graph_response:
                # 仅调试用：强制返回 graph 实现，但保持 legacy 路径的响应结构。
                shadow_result, shadow_latency_ms = _run_shadow_graph_result(
                    pipeline_context.pipeline,
                    **pipeline_kwargs,
                )
                shadow_result["conversation_id"] = context.conversation["id"]
                shadow_result["knowledge_base_id"] = context.resolved_knowledge_base_id
                shadow_result["knowledge_base_name"] = context.knowledge_base.get("name")
                shadow_result["auto_routed"] = bool(context.routed and context.routed.auto_routed)
                shadow_result["answer_source"] = "shadow_graph"
                shadow_result["debug"]["latency_ms"] = int((perf_counter() - pipeline_context.started_at) * 1000)
                _update_stage_timing_debug_fields(shadow_result)
                shadow_result["debug"]["requested_knowledge_base_id"] = payload.knowledge_base_id
                shadow_result["debug"]["resolved_knowledge_base_id"] = context.resolved_knowledge_base_id
                shadow_result["debug"]["resolved_knowledge_base_name"] = context.knowledge_base.get("name")
                shadow_result["debug"]["auto_routed"] = bool(context.routed and context.routed.auto_routed)
                shadow_result["debug"]["routing_reason"] = context.routed.reason if context.routed else "未启用自动路由"
                shadow_result["debug"]["cross_domain_guard_applied"] = False
                shadow_result["debug"]["detected_domains"] = []
                shadow_result["debug"]["cross_domain_guard_reason"] = None
                shadow_result["debug"]["shadow_compare"] = {
                    "status": "forced_graph_response",
                    "compared_stage": "pipeline_core",
                    "answer_match": True,
                    "citation_count_match": True,
                    "citation_ids_match": True,
                    "final_context_ids_match": True,
                    "rewritten_query_match": True,
                    "mismatch_fields": [],
                    "legacy_answer_hash": None,
                    "shadow_answer_hash": _stable_hash(str(shadow_result.get("answer") or "")),
                    "legacy_citation_count": 0,
                    "shadow_citation_count": len(list(shadow_result.get("citations") or [])),
                    "legacy_final_context_count": 0,
                    "shadow_final_context_count": len(list((shadow_result.get("debug") or {}).get("final_context_chunks") or [])),
                    "shadow_latency_ms": shadow_latency_ms,
                    "error": None,
                }
                result = shadow_result
            if result.get("answer_source") != "shadow_graph" and _should_run_shadow_compare(payload):
                # shadow 对比只做旁路观测，除非显式强制 graph 响应，否则不能改变用户答案。
                compare_summary = _run_shadow_graph_compare(
                    pipeline_context.pipeline,
                    **pipeline_kwargs,
                    legacy_result=result,
                )
                result["debug"]["shadow_compare"] = compare_summary
                _log_shadow_compare_result(
                    compare_summary,
                    knowledge_base_id=context.resolved_knowledge_base_id,
                    query=context.payload.query,
                )
            return self._finalize_answer_result(
                context,
                pipeline_context,
                result=result,
                intent_decision=intent_decision,
            )
        except Exception as exc:
            self._persist_failure_result(context, exc)
            raise

    def stream_query(self, payload: ChatQueryRequest) -> Generator[dict[str, object], None, None]:
        """生成 SSE 风格事件，并在流式输出结束后持久化结果。"""
        context = self._resolve_query_context(payload)
        yield self._build_start_event(context)
        persisted = False

        try:
            intent_decision, short_circuit_result = self._resolve_short_circuit_result(
                context,
                include_stream_debug_fields=True,
            )
            if short_circuit_result is not None:
                self._persist_short_circuit_result(context, result=short_circuit_result)
                persisted = True
                yield {"event": "token", "data": {"delta": short_circuit_result["answer"]}}
                yield {"event": "end", "data": short_circuit_result}
                return

            if intent_decision is None:
                raise RuntimeError("intent decision is required for pipeline execution")

            pipeline_context = self._prepare_pipeline_context(context)
            stream = pipeline_context.pipeline.ask_stream(
                **self._build_pipeline_kwargs(context, pipeline_context),
            )
            while True:
                try:
                    delta = next(stream)
                except StopIteration as stop:
                    result = stop.value
                    break
                if delta:
                    yield {"event": "token", "data": {"delta": delta}}

            result["answer_source"] = "legacy_pipeline"
            finalized_result = self._finalize_answer_result(
                context,
                pipeline_context,
                result=result,
                intent_decision=intent_decision,
            )
            persisted = True
            yield {
                "event": "end",
                "data": finalized_result,
            }
        except Exception as exc:
            if not persisted:
                self._persist_failure_result(context, exc)
            raise



