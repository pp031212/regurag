from typing import Literal

from pydantic import BaseModel, Field


class ChatQueryRequest(BaseModel):
    knowledge_base_id: str | None = Field(default=None, min_length=1)
    query: str = Field(..., min_length=1)
    conversation_id: str | None = None
    top_k_retrieve: int = Field(default=15, ge=1, le=50)
    top_k_rerank: int = Field(default=3, ge=1, le=20)
    enable_rewrite: bool = Field(default=True)
    enable_rerank: bool = Field(default=True)
    enable_auto_route: bool = Field(default=True)
    debug: bool = Field(default=False)
    debug_chunks: bool = Field(default=False)
    debug_shadow_compare: bool = Field(default=False)
    debug_force_graph_response: bool = Field(default=False)
    debug_answer_style: Literal["concise", "structured"] | None = Field(default=None)


class ChatCitation(BaseModel):
    document_id: str
    chunk_id: str
    content: str
    score: float
    source_name: str | None = None
    source_type: str | None = None
    page_number: int | None = None
    block_index: int | None = None


class ChatDebugChunk(BaseModel):
    chunk_id: str
    parent_id: str | None = None
    child_text: str | None = None
    parent_text: str
    distance: float | None = None
    rerank_score: float | None = None
    source_name: str | None = None
    source_type: str | None = None
    page_number: int | None = None
    block_index: int | None = None


class ChatShadowCompare(BaseModel):
    status: str
    compared_stage: str = "pipeline_core"
    answer_match: bool = False
    citation_count_match: bool = False
    citation_ids_match: bool = False
    final_context_ids_match: bool = False
    rewritten_query_match: bool = False
    mismatch_fields: list[str] = Field(default_factory=list)
    legacy_answer_hash: str | None = None
    shadow_answer_hash: str | None = None
    legacy_citation_count: int = 0
    shadow_citation_count: int = 0
    legacy_final_context_count: int = 0
    shadow_final_context_count: int = 0
    shadow_latency_ms: int | None = None
    error: str | None = None


class ChatStageTimings(BaseModel):
    history_rewrite_ms: int | None = None
    rewrite_ms: int | None = None
    retrieve_ms: int | None = None
    rerank_ms: int | None = None
    context_build_ms: int | None = None
    generate_ms: int | None = None
    llm_first_token_ms: int | None = None
    llm_after_first_token_ms: int | None = None
    service_overhead_ms: int | None = None


class ChatDebug(BaseModel):
    intent_name: str | None = None
    intent_source: str | None = None
    intent_classifier_source: str | None = None
    intent_classifier_mode: str | None = None
    intent_classifier_score: float | None = None
    intent_classifier_margin: float | None = None
    rewritten_query: str
    retrieved_count: int
    reranked_count: int
    enable_rewrite: bool = True
    enable_rerank: bool = True
    rewrite_ms: int
    retrieve_ms: int
    rerank_ms: int
    generate_ms: int
    latency_ms: int
    llm_finish_reason: str | None = None
    llm_model: str | None = None
    llm_prompt_tokens: int | None = None
    llm_completion_tokens: int | None = None
    llm_total_tokens: int | None = None
    llm_first_token_ms: int | None = None
    context_build_ms: int = 0
    stage_timings_ms: ChatStageTimings | None = None
    used_conversation_history: bool = False
    history_message_count: int = 0
    history_rewritten_query: str | None = None
    history_rewrite_ms: int = 0
    requested_knowledge_base_id: str | None = None
    resolved_knowledge_base_id: str | None = None
    resolved_knowledge_base_name: str | None = None
    auto_routed: bool = False
    routing_reason: str | None = None
    cross_domain_guard_applied: bool = False
    detected_domains: list[str] = Field(default_factory=list)
    cross_domain_guard_reason: str | None = None
    faq_short_circuit_applied: bool = False
    faq_rule_name: str | None = None
    faq_topic: str | None = None
    answer_guard_applied: bool = False
    unsupported_qualifiers: list[str] = Field(default_factory=list)
    answer_guard_reason: str | None = None
    shadow_compare: ChatShadowCompare | None = None
    retrieved_chunks: list[ChatDebugChunk] = Field(default_factory=list)
    mmr_selected_chunks: list[ChatDebugChunk] = Field(default_factory=list)
    reranked_chunks: list[ChatDebugChunk] = Field(default_factory=list)
    final_context_chunks: list[ChatDebugChunk] = Field(default_factory=list)


class ChatQueryResponse(BaseModel):
    answer: str
    answer_source: str = "legacy_pipeline"
    conversation_id: str
    knowledge_base_id: str
    knowledge_base_name: str | None = None
    auto_routed: bool = False
    citations: list[ChatCitation]
    debug: ChatDebug | None = None
