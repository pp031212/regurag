"""混合检索器。

dense 检索负责语义相似，sparse 检索负责关键词兜底。这里把两路召回合并、去重、
排序，并返回 pipeline 后续 MMR/rerank 所需的统一中间结果。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...workflows.rag.pipeline_steps import (
    RagDoc,
    apply_source_names,
    build_debug_chunks,
    dedupe_retrieved_docs,
    sort_retrieved_docs,
)
from .base import DenseRetriever, SparseRetriever
from .policy import RetrievalPolicy
from .types import HybridRetrievalResult


@dataclass(slots=True)
class HybridRetriever:
    """把 dense/sparse 两路候选整理成统一 retrieval result。"""

    dense_retriever: DenseRetriever
    sparse_retriever: SparseRetriever | None = None
    policy: RetrievalPolicy = field(default_factory=RetrievalPolicy)

    def retrieve(
        self,
        *,
        search_query: str,
        query_keywords: list[str],
        top_k_retrieve: int,
        source_name_by_document_id: dict[str, str] | None = None,
        policy: RetrievalPolicy | None = None,
    ) -> HybridRetrievalResult:
        active_policy = policy or self.policy
        dense_top_k = active_policy.resolve_dense_top_k(top_k_retrieve)
        query_vector, retrieved_docs = self.dense_retriever.search(search_query, top_k=dense_top_k)
        supplemental_docs: list[RagDoc] = []
        if active_policy.enable_sparse and self.sparse_retriever is not None:
            # sparse 只作为补充召回，不替代 dense；后续统一 dedupe/sort。
            supplemental_docs = self.sparse_retriever.search(
                query_keywords,
                min_hits=active_policy.sparse_min_hits,
                top_k=active_policy.resolve_sparse_top_k(top_k_retrieve),
            )

        # embedding 只来自 dense 结果，后续 MMR 只能对带 embedding 的候选执行。
        deduped_docs = sort_retrieved_docs(dedupe_retrieved_docs(retrieved_docs, supplemental_docs))
        deduped_docs = apply_source_names(deduped_docs, source_name_by_document_id)
        valid_docs = [doc for doc in deduped_docs if doc.get("embedding") is not None]
        doc_embeddings = [list(doc["embedding"]) for doc in valid_docs]

        return HybridRetrievalResult(
            query_vector=query_vector,
            retrieved_docs=retrieved_docs,
            supplemental_docs=supplemental_docs,
            deduped_docs=deduped_docs,
            valid_docs=valid_docs,
            doc_embeddings=doc_embeddings,
            retrieved_debug_chunks=build_debug_chunks(deduped_docs),
        )
