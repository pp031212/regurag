"""基于内存扫描的 BM25 风格关键词补充召回。"""

from __future__ import annotations

import math
from collections.abc import Callable

from ...workflows.rag.pipeline_steps import RagDoc


def _term_frequency(text: str, keyword: str) -> int:
    if not text or not keyword:
        return 0
    return text.count(keyword)


def bm25_keyword_matches(
    chunks: list[RagDoc],
    query_keywords: list[str],
    *,
    min_hits: int = 2,
    top_k: int = 5,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[RagDoc]:
    """对当前 collection 全量 chunk 做轻量 BM25 评分。"""
    if not query_keywords:
        return []

    normalized_keywords = [keyword.strip() for keyword in query_keywords if keyword and keyword.strip()]
    if not normalized_keywords:
        return []

    if not chunks:
        return []

    corpus_texts = [f"{str(chunk.get('child_text') or '')}\n{str(chunk.get('parent_text') or '')}" for chunk in chunks]
    avg_doc_length = sum(len(text) for text in corpus_texts) / max(1, len(corpus_texts))

    doc_frequencies: dict[str, int] = {}
    for keyword in normalized_keywords:
        doc_frequencies[keyword] = sum(1 for text in corpus_texts if keyword in text)

    matches: list[tuple[int, float, int, RagDoc]] = []
    total_docs = len(corpus_texts)
    for chunk, text in zip(chunks, corpus_texts):
        # 这里不追求完整搜索引擎能力，只作为 sqlite_fts 不可用时的可解释兜底。
        hit_count = sum(1 for keyword in normalized_keywords if keyword in text)
        if hit_count < min_hits:
            continue

        doc_length = max(1, len(text))
        score = 0.0
        for keyword in normalized_keywords:
            tf = _term_frequency(text, keyword)
            if tf <= 0:
                continue
            doc_freq = doc_frequencies.get(keyword, 0)
            idf = math.log(1.0 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            numerator = tf * (k1 + 1.0)
            denominator = tf + k1 * (1.0 - b + b * doc_length / max(1.0, avg_doc_length))
            score += idf * (numerator / denominator)

        child_text = str(chunk.get("child_text") or "")
        matches.append(
            (
                hit_count,
                score,
                len(child_text),
                {
                    "id": chunk.get("id"),
                    "child_text": child_text,
                    "parent_text": str(chunk.get("parent_text") or ""),
                    "parent_id": chunk.get("parent_id"),
                    "document_id": chunk.get("document_id"),
                    "source_type": chunk.get("source_type"),
                    "page_number": chunk.get("page_number"),
                    "block_index": chunk.get("block_index"),
                    "distance": None,
                    "embedding": None,
                    "keyword_hit_count": hit_count,
                    "bm25_score": round(score, 6),
                },
            )
        )

    matches.sort(key=lambda item: (-item[0], -item[1], -item[2]))
    return [item[3] for item in matches[:top_k]]


class CollectionBM25SparseRetriever:
    """从 vector store 拉全量 chunk 后执行 BM25 扫描。"""

    def __init__(self, chunk_supplier: Callable[[], list[RagDoc]]) -> None:
        self.chunk_supplier = chunk_supplier

    def search(
        self,
        query_keywords: list[str],
        *,
        min_hits: int = 2,
        top_k: int = 5,
    ) -> list[RagDoc]:
        return bm25_keyword_matches(
            self.chunk_supplier(),
            query_keywords,
            min_hits=min_hits,
            top_k=top_k,
        )
