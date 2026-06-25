"""空 sparse 索引实现。"""

from __future__ import annotations

from .base import SparseIndexItem
from ...workflows.rag.pipeline_steps import RagDoc


class NoOpSparseIndex:
    """用于 bm25/scan/none 等不需要写入侧索引的 provider。"""

    def upsert_documents(self, items: list[SparseIndexItem]) -> None:
        return None

    def delete_document(self, document_id: str) -> None:
        return None

    def clear_collection(self) -> None:
        return None

    def search(
        self,
        query_keywords: list[str],
        *,
        min_hits: int = 2,
        top_k: int = 5,
    ) -> list[RagDoc]:
        return []
