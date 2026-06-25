"""shadow graph 共享状态定义。"""

from __future__ import annotations

from typing import TypedDict

from .pipeline_steps import RagDoc


class RagWorkflowState(TypedDict, total=False):
    """节点之间传递的可选状态集合。"""

    query: str
    effective_query: str
    search_query: str
    expanded_keywords: str
    alias_keywords: str
    query_keywords: list[str]
    precomputed_effective_query: str
    precomputed_expanded_keywords: str
    precomputed_alias_keywords: str
    precomputed_query_keywords: list[str]
    precomputed_search_query: str
    precomputed_rewrite_ms: int
    top_k_retrieve: int
    top_k_rerank: int
    knowledge_base_id: str
    source_name_by_document_id: dict[str, str] | None
    enable_rewrite: bool
    enable_rerank: bool
    answer_style: str
    history_rewritten_query: str | None
    history_message_count: int
    history_rewrite_ms: int
    query_vector: list[float] | None
    retrieved_docs: list[RagDoc]
    supplemental_docs: list[RagDoc]
    deduped_docs: list[RagDoc]
    valid_docs: list[RagDoc]
    doc_embeddings: list[list[float]]
    mmr_selected_docs: list[RagDoc]
    parent_docs_for_rerank: list[RagDoc]
    reranked_parents: list[RagDoc]
    final_context_parents: list[RagDoc]
    answer: str
    citations: list[RagDoc]
    debug: RagDoc
    retrieved_debug_chunks: list[RagDoc]
    mmr_debug_chunks: list[RagDoc]
    reranked_debug_chunks: list[RagDoc]
    final_context_debug_chunks: list[RagDoc]
    llm_result: RagDoc
    is_policy_question: bool
    rewrite_ms: int
    retrieve_ms: int
    rerank_ms: int
    generate_ms: int
