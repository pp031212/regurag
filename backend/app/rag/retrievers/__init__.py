"""检索器工厂。

RAGPipeline 只关心统一的 HybridRetriever；具体 sparse provider 在这里解析成不同实现。
这样 Chroma/Milvus 后端、sqlite_fts/BM25/scan sparse 策略可以独立切换。
"""

from __future__ import annotations

from .base import SparseIndex
from .bm25_sparse import CollectionBM25SparseRetriever
from .hybrid_retriever import HybridRetriever
from .keyword_scanning_sparse import (
    CollectionScanningSparseRetriever,
    KeywordSearchCallableSparseRetriever,
)
from .no_op_sparse import NoOpSparseIndex
from .policy import RetrievalPolicy
from .sqlite_fts_sparse import SQLiteFTSSparseIndex
from .vector_store_dense import VectorStoreDenseRetriever

DEFAULT_SPARSE_PROVIDER = "sqlite_fts"


def normalize_sparse_provider(sparse_provider: str | None) -> str:
    """把空配置归一成默认 sparse provider。"""
    return (sparse_provider or DEFAULT_SPARSE_PROVIDER).strip().lower()


def build_sparse_index(
    *,
    db_path: str | None,
    collection_name: str,
    sparse_provider: str | None,
) -> SparseIndex:
    """创建写入侧 sparse 索引；bm25/scan 是读时扫描，所以写入侧用 NoOp。"""
    provider = normalize_sparse_provider(sparse_provider)
    if provider == "sqlite_fts":
        return SQLiteFTSSparseIndex(db_path=db_path, collection_name=collection_name)
    if provider in {"bm25", "scan", "none"}:
        return NoOpSparseIndex()
    raise ValueError(f"Unsupported retrieval sparse provider: {provider}")


def build_sparse_retriever(vector_store: object, sparse_provider: str | None) -> object | None:
    """按 provider 构造查询侧 sparse retriever。"""
    provider = normalize_sparse_provider(
        sparse_provider or getattr(vector_store, "retrieval_sparse_provider", None)
    )
    if provider == "none":
        return None

    if provider == "bm25":
        list_chunks = getattr(vector_store, "list_chunks", None)
        if callable(list_chunks):
            return CollectionBM25SparseRetriever(list_chunks)
        if hasattr(vector_store, "keyword_search"):
            return KeywordSearchCallableSparseRetriever(vector_store.keyword_search)
        return None

    if provider == "scan":
        list_chunks = getattr(vector_store, "list_chunks", None)
        if callable(list_chunks):
            return CollectionScanningSparseRetriever(list_chunks)
        if hasattr(vector_store, "keyword_search"):
            return KeywordSearchCallableSparseRetriever(vector_store.keyword_search)
        return None

    if provider == "sqlite_fts":
        # 优先复用 vector_store 持有的 sqlite_fts 索引，测试替身缺少该方法时再降级。
        get_sparse_index = getattr(vector_store, "_get_sparse_index", None)
        if callable(get_sparse_index):
            return get_sparse_index()
        sparse_index = getattr(vector_store, "sparse_index", None)
        if sparse_index is not None:
            return sparse_index
        if hasattr(vector_store, "keyword_search"):
            return KeywordSearchCallableSparseRetriever(vector_store.keyword_search)
        list_chunks = getattr(vector_store, "list_chunks", None)
        if callable(list_chunks):
            return CollectionScanningSparseRetriever(list_chunks)
        return None

    raise ValueError(f"Unsupported retrieval sparse provider: {provider}")


def build_default_hybrid_retriever(
    vector_store: object,
    *,
    sparse_provider: str | None = None,
    policy: RetrievalPolicy | None = None,
) -> HybridRetriever:
    """构造默认混合检索器：vector store dense + 当前 sparse provider。"""
    return HybridRetriever(
        dense_retriever=VectorStoreDenseRetriever(vector_store),
        sparse_retriever=build_sparse_retriever(vector_store, sparse_provider),
        policy=policy or RetrievalPolicy(),
    )


__all__ = [
    "CollectionBM25SparseRetriever",
    "CollectionScanningSparseRetriever",
    "DEFAULT_SPARSE_PROVIDER",
    "HybridRetriever",
    "KeywordSearchCallableSparseRetriever",
    "NoOpSparseIndex",
    "RetrievalPolicy",
    "SQLiteFTSSparseIndex",
    "VectorStoreDenseRetriever",
    "build_default_hybrid_retriever",
    "build_sparse_retriever",
    "build_sparse_index",
    "normalize_sparse_provider",
]
