"""简单关键词扫描补充召回。"""

from __future__ import annotations
from collections.abc import Callable

from ...workflows.rag.pipeline_steps import RagDoc


def scan_keyword_matches(
    chunks: list[RagDoc],
    query_keywords: list[str],
    *,
    min_hits: int = 2,
    top_k: int = 5,
) -> list[RagDoc]:
    """按关键词命中数扫描 chunk，适合小数据集或测试兜底。"""
    if not query_keywords:
        return []

    matches: list[tuple[int, int, RagDoc]] = []
    for chunk in chunks:
        parent_text = str(chunk.get("parent_text") or "")
        child_text = str(chunk.get("child_text") or "")
        text = f"{child_text}\n{parent_text}"
        hit_count = sum(1 for keyword in query_keywords if keyword and keyword in text)
        if hit_count < min_hits:
            continue

        matches.append(
            (
                hit_count,
                len(child_text),
                {
                    "id": chunk.get("id"),
                    "child_text": child_text,
                    "parent_text": parent_text,
                    "parent_id": chunk.get("parent_id"),
                    "document_id": chunk.get("document_id"),
                    "source_type": chunk.get("source_type"),
                    "page_number": chunk.get("page_number"),
                    "block_index": chunk.get("block_index"),
                    "distance": None,
                    "embedding": None,
                    "keyword_hit_count": hit_count,
                },
            )
        )

    matches.sort(key=lambda item: (-item[0], -item[1]))
    return [item[2] for item in matches[:top_k]]


class CollectionScanningSparseRetriever:
    """通过 list_chunks 拉全量数据后做关键词扫描。"""

    def __init__(self, chunk_supplier: Callable[[], list[RagDoc]]) -> None:
        self.chunk_supplier = chunk_supplier

    def search(
        self,
        query_keywords: list[str],
        *,
        min_hits: int = 2,
        top_k: int = 5,
    ) -> list[RagDoc]:
        return scan_keyword_matches(
            self.chunk_supplier(),
            query_keywords,
            min_hits=min_hits,
            top_k=top_k,
        )


class KeywordSearchCallableSparseRetriever:
    """适配已有 keyword_search 方法，保持 HybridRetriever 依赖统一。"""

    def __init__(self, keyword_search: object) -> None:
        self.keyword_search = keyword_search

    def search(
        self,
        query_keywords: list[str],
        *,
        min_hits: int = 2,
        top_k: int = 5,
    ) -> list[RagDoc]:
        return self.keyword_search(query_keywords, min_hits=min_hits, top_k=top_k)
