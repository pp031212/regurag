"""检索结果数据结构。"""

from __future__ import annotations

from dataclasses import dataclass

from ...workflows.rag.pipeline_steps import RagDoc


@dataclass(slots=True)
class HybridRetrievalResult:
    """HybridRetriever 输出给 RAGPipeline 的完整中间结果。"""

    query_vector: list[float]
    retrieved_docs: list[RagDoc]
    supplemental_docs: list[RagDoc]
    deduped_docs: list[RagDoc]
    valid_docs: list[RagDoc]
    doc_embeddings: list[list[float]]
    retrieved_debug_chunks: list[RagDoc]

    def as_state_update(self) -> dict[str, object]:
        """转换成 shadow graph 节点可直接合并进 state 的结构。"""
        return {
            "query_vector": self.query_vector,
            "retrieved_docs": self.retrieved_docs,
            "supplemental_docs": self.supplemental_docs,
            "deduped_docs": self.deduped_docs,
            "valid_docs": self.valid_docs,
            "doc_embeddings": self.doc_embeddings,
            "retrieved_debug_chunks": self.retrieved_debug_chunks,
        }
