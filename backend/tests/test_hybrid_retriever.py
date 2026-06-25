from app.rag.retrievers import CollectionBM25SparseRetriever, NoOpSparseIndex, RetrievalPolicy, build_default_hybrid_retriever, build_sparse_index
from app.rag.retrievers.hybrid_retriever import HybridRetriever
from app.rag.retrievers.keyword_scanning_sparse import CollectionScanningSparseRetriever



class FakeDenseRetriever:
    def __init__(self) -> None:
        self.last_top_k: int | None = None

    def search(self, query: str, top_k: int):
        self.last_top_k = top_k
        return [0.1, 0.2], [
            {
                "id": "dense-1",
                "child_text": "alpha 规则",
                "parent_text": "alpha 规则说明",
                "parent_id": "p1",
                "document_id": "doc_1",
                "source_type": "text",
                "page_number": 1,
                "block_index": 0,
                "distance": 0.2,
                "embedding": [0.1, 0.2],
            }
        ]


class FakeSparseRetriever:
    def __init__(self) -> None:
        self.last_min_hits: int | None = None
        self.last_top_k: int | None = None

    def search(self, query_keywords: list[str], *, min_hits: int = 2, top_k: int = 5):
        self.last_min_hits = min_hits
        self.last_top_k = top_k
        return [
            {
                "id": "sparse-1",
                "child_text": "beta 表格",
                "parent_text": "beta 表格要求",
                "parent_id": "p2",
                "document_id": "doc_2",
                "source_type": "table",
                "page_number": 2,
                "block_index": 1,
                "distance": None,
                "embedding": None,
                "keyword_hit_count": 2,
            }
        ]


def test_hybrid_retriever_returns_merged_sorted_and_debug_ready_result() -> None:
    dense = FakeDenseRetriever()
    sparse = FakeSparseRetriever()
    retriever = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
    )

    result = retriever.retrieve(
        search_query="alpha beta",
        query_keywords=["alpha", "beta"],
        top_k_retrieve=5,
        source_name_by_document_id={"doc_1": "《A》", "doc_2": "《B》"},
    )

    assert result.query_vector == [0.1, 0.2]
    assert dense.last_top_k == 5
    assert sparse.last_min_hits == 2
    assert sparse.last_top_k == 5
    assert [doc["id"] for doc in result.deduped_docs] == ["sparse-1", "dense-1"]
    assert result.deduped_docs[0]["source_name"] == "《B》"
    assert result.deduped_docs[1]["source_name"] == "《A》"
    assert len(result.valid_docs) == 1
    assert result.doc_embeddings == [[0.1, 0.2]]
    assert result.retrieved_debug_chunks[0]["chunk_id"] == "sparse-1"


class FakeCollection:
    def get(self, include: list[str]):
        return {"ids": [], "documents": [], "metadatas": []}


class FakeSparseIndex:
    def search(self, query_keywords: list[str], *, min_hits: int = 2, top_k: int = 5):
        return []


class FakeVectorStoreWithSparseIndex:
    def __init__(self) -> None:
        self.collection = FakeCollection()
        self.sparse_index = FakeSparseIndex()
        self.list_chunks_called = False

    def search(self, query: str, top_k: int):
        return [], []

    def _get_sparse_index(self):
        return self.sparse_index

    def list_chunks(self):
        self.list_chunks_called = True
        return []


def test_build_default_hybrid_retriever_prefers_sqlite_sparse_index() -> None:
    retriever = build_default_hybrid_retriever(FakeVectorStoreWithSparseIndex(), sparse_provider="sqlite_fts")

    assert isinstance(retriever.sparse_retriever, FakeSparseIndex)


def test_build_default_hybrid_retriever_can_force_collection_scan() -> None:
    store = FakeVectorStoreWithSparseIndex()
    retriever = build_default_hybrid_retriever(store, sparse_provider="scan")

    assert isinstance(retriever.sparse_retriever, CollectionScanningSparseRetriever)
    retriever.sparse_retriever.search(["alpha"], min_hits=1, top_k=1)
    assert store.list_chunks_called is True


def test_build_default_hybrid_retriever_can_disable_sparse_retrieval() -> None:
    retriever = build_default_hybrid_retriever(FakeVectorStoreWithSparseIndex(), sparse_provider="none")

    assert retriever.sparse_retriever is None


def test_hybrid_retriever_applies_policy_overrides() -> None:
    dense = FakeDenseRetriever()
    sparse = FakeSparseRetriever()
    retriever = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        policy=RetrievalPolicy(dense_top_k=3, sparse_top_k=7, sparse_min_hits=1, enable_sparse=True),
    )

    retriever.retrieve(
        search_query="alpha beta",
        query_keywords=["alpha", "beta"],
        top_k_retrieve=5,
    )

    assert dense.last_top_k == 3
    assert sparse.last_min_hits == 1
    assert sparse.last_top_k == 7


def test_hybrid_retriever_can_disable_sparse_via_policy() -> None:
    dense = FakeDenseRetriever()
    sparse = FakeSparseRetriever()
    retriever = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        policy=RetrievalPolicy(enable_sparse=False),
    )

    result = retriever.retrieve(
        search_query="alpha beta",
        query_keywords=["alpha", "beta"],
        top_k_retrieve=5,
    )

    assert dense.last_top_k == 5
    assert sparse.last_min_hits is None
    assert sparse.last_top_k is None
    assert result.supplemental_docs == []


def test_build_sparse_index_returns_noop_for_scan_and_none() -> None:
    scan_index = build_sparse_index(db_path=None, collection_name="demo", sparse_provider="scan")
    none_index = build_sparse_index(db_path=None, collection_name="demo", sparse_provider="none")

    assert isinstance(scan_index, NoOpSparseIndex)
    assert isinstance(none_index, NoOpSparseIndex)


def test_build_default_hybrid_retriever_can_build_bm25_sparse_retriever() -> None:
    store = FakeVectorStoreWithSparseIndex()
    retriever = build_default_hybrid_retriever(store, sparse_provider="bm25")

    assert isinstance(retriever.sparse_retriever, CollectionBM25SparseRetriever)
    retriever.sparse_retriever.search(["alpha"], min_hits=1, top_k=1)
    assert store.list_chunks_called is True
