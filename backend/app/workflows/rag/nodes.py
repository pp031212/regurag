"""RAG shadow graph 节点。

节点拆分粒度对齐 legacy pipeline 的主要阶段：查询准备、检索、父块选择、答案生成。
这样可以把 graph 结果和 legacy pipeline 的中间 debug 数据做对照。
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from ...rag.retrieval_utils import mmr_rerank
from .pipeline_steps import (
    apply_source_names,
    build_citations,
    build_debug,
    build_debug_chunks,
    build_final_context_parents,
    build_parent_docs_for_rerank,
    dedupe_retrieved_docs,
    extract_query_keywords,
    format_context_parent,
    is_overview_query,
    preserve_overview_candidate_docs,
    preserve_policy_keyword_docs,
    preserve_query_keyword_docs,
    should_treat_as_policy_question,
    sort_reranked_parents,
    sort_retrieved_docs,
)
from .state import RagWorkflowState


@dataclass(slots=True)
class RagWorkflowDependencies:
    """shadow graph 运行时依赖，全部从 RAGPipeline 注入。"""

    rewriter: Any
    query_alias_expander: Any
    vector_store: Any
    reranker: Any
    llm: Any
    retrieval_rules: Any
    top_k_mmr: int
    retriever: Any | None = None


def prepare_query_node(state: RagWorkflowState, deps: RagWorkflowDependencies) -> dict[str, object]:
    """准备检索 query，优先复用 service/pipeline 已预计算的 query_prep。"""
    if state.get("precomputed_search_query"):
        return {
            "effective_query": state.get("precomputed_effective_query") or state.get("history_rewritten_query") or state["query"],
            "expanded_keywords": state.get("precomputed_expanded_keywords") or "",
            "alias_keywords": state.get("precomputed_alias_keywords") or "",
            "query_keywords": list(state.get("precomputed_query_keywords") or []),
            "search_query": str(state["precomputed_search_query"]),
            "rewrite_ms": int(state.get("precomputed_rewrite_ms", 0)),
        }

    effective_query = state.get("history_rewritten_query") or state["query"]
    rewrite_started = perf_counter()
    expanded_keywords = ""
    if state.get("enable_rewrite", True):
        try:
            # shadow graph 不能因为 rewrite 失败而中断，要和 legacy pipeline 一样退化。
            expanded_keywords = deps.rewriter.rewrite(effective_query)
        except Exception:
            expanded_keywords = ""
    rewrite_ms = int((perf_counter() - rewrite_started) * 1000)

    alias_keywords = deps.query_alias_expander.expand(effective_query)
    query_keywords = extract_query_keywords(alias_keywords)
    if not query_keywords:
        query_keywords = extract_query_keywords(expanded_keywords)

    return {
        "effective_query": effective_query,
        "expanded_keywords": expanded_keywords,
        "alias_keywords": alias_keywords,
        "query_keywords": query_keywords,
        "search_query": f"{effective_query} {expanded_keywords} {alias_keywords}".strip(),
        "rewrite_ms": rewrite_ms,
    }


def retrieve_documents_node(state: RagWorkflowState, deps: RagWorkflowDependencies) -> dict[str, object]:
    """执行统一 HybridRetriever，产出后续 MMR/rerank 所需候选。"""
    retrieve_started = perf_counter()
    retriever = deps.retriever
    if retriever is None:
        from ...rag.retrievers import build_default_hybrid_retriever

        retriever = build_default_hybrid_retriever(deps.vector_store)
    retrieval = retriever.retrieve(
        search_query=state["search_query"],
        query_keywords=state["query_keywords"],
        top_k_retrieve=state["top_k_retrieve"],
        source_name_by_document_id=state.get("source_name_by_document_id"),
    )
    retrieve_ms = int((perf_counter() - retrieve_started) * 1000)
    update = retrieval.as_state_update()
    update["retrieve_ms"] = retrieve_ms
    return update


def select_parents_node(state: RagWorkflowState, deps: RagWorkflowDependencies) -> dict[str, object]:
    """从检索候选里选出用于 parent 聚合的 chunk。"""
    valid_docs = state.get("valid_docs", [])
    if not valid_docs:
        return {
            "mmr_selected_docs": [],
            "mmr_debug_chunks": [],
            "parent_docs_for_rerank": [],
            "is_policy_question": False,
        }

    if state.get("enable_rerank", True):
        # MMR 先在 chunk 级别控重复，再聚合到 parent 文本。
        selected_indices = mmr_rerank(
            query_embedding=state["query_vector"],
            doc_embeddings=state["doc_embeddings"],
            top_k=min(deps.top_k_mmr, len(valid_docs)),
            lambda_mult=0.6,
        )
        mmr_selected_docs = [valid_docs[index] for index in selected_indices]
    else:
        mmr_selected_docs = valid_docs[: min(state["top_k_rerank"], len(valid_docs))]

    is_policy_question = should_treat_as_policy_question(
        state["query"],
        policy_trigger_words=deps.retrieval_rules.policy_trigger_words,
        policy_trigger_phrases=deps.retrieval_rules.policy_trigger_phrases,
        behavior_keywords=deps.retrieval_rules.behavior_keywords,
    )
    # 规则类问题容易被短 chunk 稀释，需要把强关键词候选补回选集。
    mmr_selected_docs = preserve_policy_keyword_docs(
        selected_docs=mmr_selected_docs,
        candidate_docs=state["deduped_docs"],
        is_policy_question=is_policy_question,
        must_keep_child_keywords=deps.retrieval_rules.must_keep_child_keywords,
    )
    mmr_selected_docs = preserve_query_keyword_docs(
        selected_docs=mmr_selected_docs,
        candidate_docs=state["deduped_docs"],
        query_keywords=state["query_keywords"],
    )
    if is_overview_query(state["query"]):
        mmr_selected_docs = preserve_overview_candidate_docs(
            selected_docs=mmr_selected_docs,
            candidate_docs=state["deduped_docs"],
            max_extra_docs=max(4, int(state["top_k_rerank"])),
        )
    parent_docs_for_rerank = build_parent_docs_for_rerank(mmr_selected_docs)

    return {
        "is_policy_question": is_policy_question,
        "mmr_selected_docs": mmr_selected_docs,
        "mmr_debug_chunks": build_debug_chunks(mmr_selected_docs),
        "parent_docs_for_rerank": parent_docs_for_rerank,
    }


def generate_answer_node(state: RagWorkflowState, deps: RagWorkflowDependencies) -> dict[str, object]:
    """执行 parent rerank、最终上下文构建和 LLM 生成。"""
    rerank_started = perf_counter()
    parent_docs_for_rerank = state.get("parent_docs_for_rerank", [])

    if not state.get("valid_docs"):
        # 完全没有有效检索候选时仍调用 LLM，让提示词生成“资料不足”类回答。
        llm_result, generate_ms = _generate_without_context(state, deps)
        return {
            "answer": str(llm_result["answer"]),
            "citations": [],
            "llm_result": llm_result,
            "generate_ms": generate_ms,
            "rerank_ms": 0,
            "debug": _build_result_debug(state, llm_result=llm_result, rerank_ms=0, generate_ms=generate_ms),
        }

    if not parent_docs_for_rerank and is_overview_query(state["query"]):
        # overview 问题需要更宽的上下文，没有 parent 时从 deduped 候选补一批。
        parent_docs_for_rerank = build_parent_docs_for_rerank(
            list(state.get("deduped_docs") or [])[: max(state["top_k_rerank"] + 2, 4)]
        )

    if not parent_docs_for_rerank:
        llm_result, generate_ms = _generate_without_context(state, deps)
        rerank_ms = int((perf_counter() - rerank_started) * 1000)
        return {
            "answer": str(llm_result["answer"]),
            "citations": [],
            "llm_result": llm_result,
            "generate_ms": generate_ms,
            "rerank_ms": rerank_ms,
            "debug": _build_result_debug(state, llm_result=llm_result, rerank_ms=rerank_ms, generate_ms=generate_ms),
        }

    if state.get("enable_rerank", True):
        reranked_parents = deps.reranker.rerank(
            state["search_query"],
            parent_docs_for_rerank,
            top_k=min(state["top_k_rerank"], len(parent_docs_for_rerank)),
        )
    else:
        reranked_parents = parent_docs_for_rerank[: min(state["top_k_rerank"], len(parent_docs_for_rerank))]
    reranked_parents = sort_reranked_parents(reranked_parents)
    reranked_debug_chunks = build_debug_chunks(reranked_parents)

    final_context_parents = build_final_context_parents(
        parent_docs_for_rerank=parent_docs_for_rerank,
        reranked_parents=reranked_parents,
        query=state["query"],
        query_keywords=state["query_keywords"],
        is_policy_question=bool(state.get("is_policy_question")),
        must_keep_parent_keywords=deps.retrieval_rules.must_keep_child_keywords,
        low_value_parent_keywords=deps.retrieval_rules.low_value_parent_keywords,
        rule_signal_keywords=deps.retrieval_rules.rule_signal_keywords,
        behavior_keywords=deps.retrieval_rules.behavior_keywords,
        top_k_rerank=state["top_k_rerank"],
    )
    if not final_context_parents and is_overview_query(state["query"]):
        # overview 问题宁可多给一点候选，也不要空上下文进入生成。
        final_context_parents = parent_docs_for_rerank[: max(state["top_k_rerank"] + 1, 4)]

    rerank_ms = int((perf_counter() - rerank_started) * 1000)
    final_context_debug_chunks = build_debug_chunks(final_context_parents)
    context = "\n\n".join(format_context_parent(doc) for doc in final_context_parents)

    generate_started = perf_counter()
    llm_result = deps.llm.generate(
        state["query"],
        context,
        standalone_query=state["effective_query"],
        answer_style=state.get("answer_style", "concise"),
    )
    generate_ms = int((perf_counter() - generate_started) * 1000)
    citations = build_citations(final_context_parents, state["knowledge_base_id"])

    return {
        "reranked_parents": reranked_parents,
        "reranked_debug_chunks": reranked_debug_chunks,
        "final_context_parents": final_context_parents,
        "final_context_debug_chunks": final_context_debug_chunks,
        "llm_result": llm_result,
        "answer": str(llm_result["answer"]),
        "citations": citations,
        "rerank_ms": rerank_ms,
        "generate_ms": generate_ms,
        "debug": _build_result_debug(
            state,
            llm_result=llm_result,
            rerank_ms=rerank_ms,
            generate_ms=generate_ms,
            reranked_count=len(final_context_parents),
            reranked_chunks=reranked_debug_chunks,
            final_context_chunks=final_context_debug_chunks,
        ),
    }


def _generate_without_context(
    state: RagWorkflowState,
    deps: RagWorkflowDependencies,
) -> tuple[dict[str, object], int]:
    """无上下文生成路径，保持 debug 计时和正常路径一致。"""
    generate_started = perf_counter()
    llm_result = deps.llm.generate(
        state["query"],
        "",
        standalone_query=state["effective_query"],
        answer_style=state.get("answer_style", "concise"),
    )
    generate_ms = int((perf_counter() - generate_started) * 1000)
    return llm_result, generate_ms


def _build_result_debug(
    state: RagWorkflowState,
    *,
    llm_result: dict[str, object],
    rerank_ms: int,
    generate_ms: int,
    reranked_count: int = 0,
    reranked_chunks: list[dict[str, object]] | None = None,
    final_context_chunks: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """把 graph state 转成和 legacy pipeline 对齐的 debug payload。"""
    return build_debug(
        rewritten_query=state["search_query"],
        retrieved_count=len(state.get("retrieved_docs", [])),
        reranked_count=reranked_count,
        enable_rewrite=state.get("enable_rewrite", True),
        enable_rerank=state.get("enable_rerank", True),
        rewrite_ms=int(state.get("rewrite_ms", 0)),
        retrieve_ms=int(state.get("retrieve_ms", 0)),
        rerank_ms=rerank_ms,
        generate_ms=generate_ms,
        llm_result=llm_result,
        retrieved_chunks=state.get("retrieved_debug_chunks"),
        mmr_selected_chunks=state.get("mmr_debug_chunks"),
        reranked_chunks=reranked_chunks or state.get("reranked_debug_chunks"),
        final_context_chunks=final_context_chunks or state.get("final_context_debug_chunks"),
        used_conversation_history=state.get("history_rewritten_query") is not None,
        history_message_count=int(state.get("history_message_count", 0)),
        history_rewritten_query=state.get("history_rewritten_query"),
        history_rewrite_ms=int(state.get("history_rewrite_ms", 0)),
    )

