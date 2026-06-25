"""检索器协议定义。"""

from __future__ import annotations

from typing import Protocol

from ...workflows.rag.pipeline_steps import RagDoc

SparseIndexItem = tuple[dict[str, object], str]


class DenseRetriever(Protocol):
    """语义向量召回接口，返回 query 向量和候选 chunk。"""

    def search(self, query: str, top_k: int) -> tuple[list[float], list[RagDoc]]: ...


class SparseRetriever(Protocol):
    """关键词补充召回接口。"""

    def search(
        self,
        query_keywords: list[str],
        *,
        min_hits: int = 2,
        top_k: int = 5,
    ) -> list[RagDoc]: ...


class SparseIndex(SparseRetriever, Protocol):
    """可写入的 sparse 索引，供向量库写入/删除时同步维护。"""

    def upsert_documents(self, items: list[SparseIndexItem]) -> None: ...

    def delete_document(self, document_id: str) -> None: ...

    def clear_collection(self) -> None: ...
