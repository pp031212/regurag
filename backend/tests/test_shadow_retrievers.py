from __future__ import annotations

import numpy as np
from langchain_core.documents import Document

from app.rag.vector_store import ChromaVectorStore
from app.workflows.rag.shadow_retrievers import (
    LangChainChromaDenseRetriever,
    LangChainMilvusDenseRetriever,
    _should_force_rebootstrap,
    build_shadow_retriever,
)


class FakeCollection:
    def __init__(self, embeddings_by_id: dict[str, list[float]]) -> None:
        self.embeddings_by_id = embeddings_by_id

    def get(self, ids: list[str], include: list[str]):
        return {
            "ids": ids,
            "embeddings": np.array([self.embeddings_by_id.get(chunk_id) for chunk_id in ids], dtype=object),
        }


class FakeChroma:
    def __init__(self, *args, **kwargs) -> None:
        self.docs_with_scores = [
            (
                Document(
                    page_content="未依法为劳动者缴纳社会保险费的。",
                    metadata={
                        "chunk_id": "chunk-1",
                        "parent_text": "第三十八条 用人单位有下列情形之一的，劳动者可以解除劳动合同。",
                        "parent_id": "p1",
                        "document_id": "doc_001",
                        "source_type": "text",
                        "page_number": 3,
                        "block_index": 1,
                    },
                ),
                0.12,
            )
        ]

    def similarity_search_with_score(self, query: str, k: int):
        return self.docs_with_scores[:k]


class FakeMilvusStore:
    def __init__(self, docs_with_scores, *, collection_exists: bool = True, search_params=None):
        self.docs_with_scores = docs_with_scores
        self.collection_name = "shadow-test_shadow_milvus"
        self.collection_exists = collection_exists
        self.client = object()
        self.search_params = search_params or {"metric_type": "L2", "params": {}}

    def has_collection(self) -> bool:
        return self.collection_exists

    def ensure_loaded(self) -> None:
        return None

    def search(self, query_vector: list[float], *, top_k: int):
        assert query_vector == [0.91, 0.09]
        return [
            {
                "entity": {
                    "text": doc.page_content,
                    **doc.metadata,
                },
                "distance": score,
            }
            for doc, score in self.docs_with_scores[:top_k]
        ]


def _build_chroma_vector_store() -> ChromaVectorStore:
    vector_store = ChromaVectorStore.__new__(ChromaVectorStore)
    vector_store.client = object()
    vector_store.collection_name = "shadow-test"
    vector_store.collection = FakeCollection({"chunk-1": [0.11, 0.22]})
    vector_store.retrieval_sparse_provider = "none"
    vector_store._get_embedding = lambda text, is_query=False: [0.91, 0.09] if is_query else [0.19, 0.81]
    vector_store.list_chunks = lambda: [
        {
            "id": "chunk-1",
            "child_text": "未依法为劳动者缴纳社会保险费的。",
            "parent_text": "第三十八条 用人单位有下列情形之一的，劳动者可以解除劳动合同。",
            "parent_id": "p1",
            "document_id": "doc_001",
            "source_type": "text",
            "page_number": 3,
            "block_index": 1,
        }
    ]
    return vector_store


def test_langchain_chroma_dense_retriever_maps_documents(monkeypatch) -> None:
    monkeypatch.setattr("app.workflows.rag.shadow_retrievers.Chroma", FakeChroma)
    retriever = LangChainChromaDenseRetriever(_build_chroma_vector_store())

    query_vector, docs = retriever.search("不给劳动者缴纳社保怎么办", top_k=3)

    assert query_vector == [0.91, 0.09]
    assert docs == [
        {
            "id": "chunk-1",
            "child_text": "未依法为劳动者缴纳社会保险费的。",
            "parent_text": "第三十八条 用人单位有下列情形之一的，劳动者可以解除劳动合同。",
            "parent_id": "p1",
            "document_id": "doc_001",
            "source_type": "text",
            "page_number": 3,
            "block_index": 1,
            "distance": 0.12,
            "embedding": [0.11, 0.22],
        }
    ]


def test_build_shadow_retriever_rejects_non_chroma_store_for_langchain_backend() -> None:
    class FakeVectorStore:
        retrieval_sparse_provider = "none"

    try:
        build_shadow_retriever(
            FakeVectorStore(),
            backend="langchain_chroma",
            sparse_provider="none",
        )
    except ValueError as exc:
        assert "requires ChromaVectorStore" in str(exc)
    else:
        raise AssertionError("expected langchain_chroma shadow backend to reject non-Chroma vector stores")


def test_langchain_milvus_dense_retriever_bootstraps_collection_from_chroma_chunks(monkeypatch) -> None:
    monkeypatch.setattr("app.workflows.rag.shadow_retrievers.sys.platform", "linux")
    bootstrap_calls: list[dict[str, object]] = []
    build_calls = {"count": 0}

    def fake_build_milvus_search_store(**kwargs):
        build_calls["count"] += 1
        return FakeMilvusStore(
            [
                (
                    Document(
                        page_content="未依法为劳动者缴纳社会保险费的。",
                        metadata={
                            "chunk_id": "chunk-1",
                            "parent_text": "第三十八条 用人单位有下列情形之一的，劳动者可以解除劳动合同。",
                            "parent_id": "p1",
                            "document_id": "doc_001",
                            "source_type": "text",
                            "page_number": 3,
                            "block_index": 1,
                        },
                    ),
                    0.08,
                )
            ],
            collection_exists=build_calls["count"] > 1,
        )

    monkeypatch.setattr(
        "app.workflows.rag.shadow_retrievers._build_milvus_search_store",
        fake_build_milvus_search_store,
    )
    monkeypatch.setattr(
        "app.workflows.rag.shadow_retrievers._bootstrap_milvus_collection",
        lambda **kwargs: bootstrap_calls.append(kwargs),
    )

    retriever = LangChainMilvusDenseRetriever(
        _build_chroma_vector_store(),
        uri="./data/milvus_shadow.db",
        token="token-123",
        drop_old=True,
    )

    query_vector, docs = retriever.search("不给劳动者缴纳社保怎么办", top_k=3)

    assert query_vector == [0.91, 0.09]
    assert docs == [
        {
            "id": "chunk-1",
            "child_text": "未依法为劳动者缴纳社会保险费的。",
            "parent_text": "第三十八条 用人单位有下列情形之一的，劳动者可以解除劳动合同。",
            "parent_id": "p1",
            "document_id": "doc_001",
            "source_type": "text",
            "page_number": 3,
            "block_index": 1,
            "distance": 0.08,
            "embedding": [0.19, 0.81],
        }
    ]
    assert len(bootstrap_calls) == 1
    assert bootstrap_calls[0]["collection_name"] == "shadow-test_shadow_milvus"
    assert bootstrap_calls[0]["chunks"] == [
        {
            "id": "chunk-1",
            "child_text": "未依法为劳动者缴纳社会保险费的。",
            "parent_text": "第三十八条 用人单位有下列情形之一的，劳动者可以解除劳动合同。",
            "parent_id": "p1",
            "document_id": "doc_001",
            "source_type": "text",
            "page_number": 3,
            "block_index": 1,
        }
    ]
    assert bootstrap_calls[0]["embeddings"] is retriever.embeddings
    assert bootstrap_calls[0]["drop_old"] is True


def test_build_shadow_retriever_supports_langchain_milvus(monkeypatch) -> None:
    monkeypatch.setattr("app.workflows.rag.shadow_retrievers.sys.platform", "linux")
    monkeypatch.setattr(
        "app.workflows.rag.shadow_retrievers._build_milvus_search_store",
        lambda **kwargs: FakeMilvusStore([], collection_exists=False),
    )
    monkeypatch.setattr(
        "app.workflows.rag.shadow_retrievers._bootstrap_milvus_collection",
        lambda **kwargs: None,
    )

    retriever = build_shadow_retriever(
        _build_chroma_vector_store(),
        backend="langchain_milvus",
        sparse_provider="none",
        milvus_uri="./data/milvus_shadow.db",
    )

    assert isinstance(retriever.dense_retriever, LangChainMilvusDenseRetriever)


def test_langchain_milvus_dense_retriever_rejects_local_lite_mode_on_windows(monkeypatch) -> None:
    monkeypatch.setattr("app.workflows.rag.shadow_retrievers.sys.platform", "win32")

    try:
        LangChainMilvusDenseRetriever(
            _build_chroma_vector_store(),
            uri="./data/milvus_shadow.db",
        )
    except RuntimeError as exc:
        assert "not supported on Windows" in str(exc)
    else:
        raise AssertionError("expected Milvus Lite local file mode to be rejected on Windows")


def test_langchain_milvus_dense_retriever_rebuilds_missing_collection(monkeypatch) -> None:
    bootstrap_calls: list[dict[str, object]] = []
    monkeypatch.setattr("app.workflows.rag.shadow_retrievers.sys.platform", "linux")
    monkeypatch.setattr(
        "app.workflows.rag.shadow_retrievers._build_milvus_search_store",
        lambda **kwargs: FakeMilvusStore(
            [
                (
                    Document(
                        page_content="未依法为劳动者缴纳社会保险费的。",
                        metadata={
                            "chunk_id": "chunk-1",
                            "parent_text": "第三十八条 用人单位有下列情形之一的，劳动者可以解除劳动合同。",
                            "parent_id": "p1",
                            "document_id": "doc_001",
                            "source_type": "text",
                            "page_number": 3,
                            "block_index": 1,
                        },
                    ),
                    0.08,
                )
            ],
            collection_exists=False,
        ),
    )
    monkeypatch.setattr(
        "app.workflows.rag.shadow_retrievers._bootstrap_milvus_collection",
        lambda **kwargs: bootstrap_calls.append(kwargs),
    )

    retriever = LangChainMilvusDenseRetriever(
        _build_chroma_vector_store(),
        uri="./data/milvus_shadow.db",
    )
    assert isinstance(retriever.store, FakeMilvusStore)
    retriever.store.collection_exists = False

    query_vector, docs = retriever.search("不给劳动者缴纳社保怎么办", top_k=1)

    assert query_vector == [0.91, 0.09]
    assert len(docs) == 1
    assert len(bootstrap_calls) == 2
    assert bootstrap_calls[-1]["drop_old"] is False


def test_langchain_milvus_dense_retriever_reuses_existing_collection(monkeypatch) -> None:
    bootstrap_calls: list[dict[str, object]] = []
    monkeypatch.setattr("app.workflows.rag.shadow_retrievers.sys.platform", "linux")
    monkeypatch.setattr(
        "app.workflows.rag.shadow_retrievers._build_milvus_search_store",
        lambda **kwargs: FakeMilvusStore([], collection_exists=True),
    )
    monkeypatch.setattr(
        "app.workflows.rag.shadow_retrievers._bootstrap_milvus_collection",
        lambda **kwargs: bootstrap_calls.append(kwargs),
    )

    retriever = LangChainMilvusDenseRetriever(
        _build_chroma_vector_store(),
        uri="./data/milvus_shadow.db",
        drop_old=False,
    )

    assert isinstance(retriever.store, FakeMilvusStore)
    assert bootstrap_calls == []


def test_langchain_milvus_dense_retriever_returns_empty_when_source_chunks_missing(monkeypatch) -> None:
    monkeypatch.setattr("app.workflows.rag.shadow_retrievers.sys.platform", "linux")
    monkeypatch.setattr(
        "app.workflows.rag.shadow_retrievers._build_milvus_search_store",
        lambda **kwargs: FakeMilvusStore([], collection_exists=False),
    )

    vector_store = _build_chroma_vector_store()
    vector_store.list_chunks = lambda: []
    retriever = LangChainMilvusDenseRetriever(
        vector_store,
        uri="./data/milvus_shadow.db",
    )

    query_vector, docs = retriever.search("不给劳动者缴纳社保怎么办", top_k=1)

    assert query_vector == [0.91, 0.09]
    assert docs == []


def test_langchain_milvus_dense_retriever_recovers_when_has_collection_check_fails(monkeypatch) -> None:
    bootstrap_calls: list[dict[str, object]] = []
    monkeypatch.setattr("app.workflows.rag.shadow_retrievers.sys.platform", "linux")
    stores: list[FakeMilvusStore] = []

    def fake_build_milvus_search_store(**kwargs):
        store = FakeMilvusStore(
            [
                (
                    Document(
                        page_content="未依法为劳动者缴纳社会保险费的。",
                        metadata={
                            "chunk_id": "chunk-1",
                            "parent_text": "第三十八条 用人单位有下列情形之一的，劳动者可以解除劳动合同。",
                            "parent_id": "p1",
                            "document_id": "doc_001",
                            "source_type": "text",
                            "page_number": 3,
                            "block_index": 1,
                        },
                    ),
                    0.08,
                )
            ],
            collection_exists=False,
        )
        stores.append(store)
        return store

    monkeypatch.setattr(
        "app.workflows.rag.shadow_retrievers._build_milvus_search_store",
        fake_build_milvus_search_store,
    )
    monkeypatch.setattr(
        "app.workflows.rag.shadow_retrievers._bootstrap_milvus_collection",
        lambda **kwargs: bootstrap_calls.append(kwargs),
    )

    retriever = LangChainMilvusDenseRetriever(
        _build_chroma_vector_store(),
        uri="./data/milvus_shadow.db",
    )

    def broken_has_collection() -> bool:
        raise RuntimeError("should create connection first")

    retriever.store.has_collection = broken_has_collection
    query_vector, docs = retriever.search("不给劳动者缴纳社保怎么办", top_k=1)

    assert query_vector == [0.91, 0.09]
    assert len(docs) == 1
    assert len(bootstrap_calls) == 2
    assert bootstrap_calls[-1]["drop_old"] is False


def test_should_force_rebootstrap_for_incomplete_milvus_collection() -> None:
    assert _should_force_rebootstrap(RuntimeError("index not found")) is True
    assert _should_force_rebootstrap(RuntimeError("collection not loaded")) is True
    assert _should_force_rebootstrap(RuntimeError("should create connection first")) is False
