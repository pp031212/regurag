from app.rag.pipeline import RAGPipeline


class FakeRewriter:
    def rewrite(self, query: str) -> str:
        return f"{query} 扩写"


class FakeAliasExpander:
    def expand(self, query: str) -> str:
        return f"{query} 别名"


class FakeVectorStore:
    def search(self, query: str, top_k: int):
        return [0.2, 0.8], [
            {
                "id": "doc-1",
                "child_text": "未依法为劳动者缴纳社会保险费的。",
                "parent_text": "第三十八条 用人单位有下列情形之一的，劳动者可以解除劳动合同：（三）未依法为劳动者缴纳社会保险费的；",
                "parent_id": "p1",
                "document_id": "doc_001",
                "source_type": "text",
                "page_number": 1,
                "block_index": 0,
                "distance": 0.1,
                "embedding": [0.1, 0.2],
            }
        ]

    def keyword_search(self, query_keywords: list[str], min_hits: int, top_k: int):
        return [
            {
                "id": "doc-2",
                "child_text": "劳动者可以立即解除劳动合同，不需事先告知用人单位。",
                "parent_text": "劳动者可以立即解除劳动合同，不需事先告知用人单位。",
                "parent_id": "p2",
                "document_id": "doc_002",
                "source_type": "text",
                "page_number": 1,
                "block_index": 1,
                "distance": 0.2,
                "embedding": [0.2, 0.3],
                "keyword_hit_count": 2,
            }
        ]


class FakeReranker:
    def rerank(self, query: str, docs: list[dict[str, object]], top_k: int):
        reranked = []
        for index, doc in enumerate(docs[:top_k], start=1):
            reranked.append({**doc, "rerank_score": float(top_k - index + 1)})
        return reranked


class FakeLLM:
    def generate(
        self,
        query: str,
        context: str,
        standalone_query: str | None = None,
        *,
        answer_style: str = "concise",
    ):
        return {
            "answer": f"{answer_style}:基于上下文回答：{query}",
            "finish_reason": "stop",
            "model": "fake-model",
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        }


class FakeRetrievalRules:
    policy_trigger_words = ("规定", "制度")
    policy_trigger_phrases = ("什么条件",)
    behavior_keywords = ("迟到", "旷课")
    must_keep_child_keywords = ("社会保险费",)
    low_value_parent_keywords = ()
    rule_signal_keywords = ()


class FakeSettings:
    top_k_mmr = 4
    shadow_retrieval_backend = "legacy"
    retrieval_sparse_provider = "sqlite_fts"
    resolved_shadow_milvus_uri = "./data/milvus_shadow.db"
    shadow_milvus_token = None
    shadow_milvus_drop_old = False


def _build_pipeline() -> RAGPipeline:
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.rewriter = FakeRewriter()
    pipeline.query_alias_expander = FakeAliasExpander()
    pipeline.vector_store = FakeVectorStore()
    pipeline.reranker = FakeReranker()
    pipeline.llm = FakeLLM()
    pipeline.retrieval_rules = FakeRetrievalRules()
    pipeline.settings = FakeSettings()
    return pipeline


def test_pipeline_run_shadow_workflow_returns_expected_answer() -> None:
    pipeline = _build_pipeline()

    result = pipeline.run_shadow_workflow(
        query="不给劳动者购置五险可以直接离职吗",
        top_k_retrieve=5,
        top_k_rerank=2,
        knowledge_base_id="kb_001",
        source_name_by_document_id={
            "doc_001": "《中华人民共和国劳动合同法》",
            "doc_002": "《中华人民共和国劳动合同法》",
        },
    )

    assert result["answer"] == "concise:基于上下文回答：不给劳动者购置五险可以直接离职吗"
    assert len(result["citations"]) == 2


def test_pipeline_run_shadow_graph_returns_expected_answer() -> None:
    pipeline = _build_pipeline()

    result = pipeline.run_shadow_graph(
        query="不给劳动者购置五险可以直接离职吗",
        top_k_retrieve=5,
        top_k_rerank=2,
        knowledge_base_id="kb_001",
        source_name_by_document_id={
            "doc_001": "《中华人民共和国劳动合同法》",
            "doc_002": "《中华人民共和国劳动合同法》",
        },
    )

    assert result["answer"] == "concise:基于上下文回答：不给劳动者购置五险可以直接离职吗"
    assert len(result["citations"]) == 2


def test_pipeline_run_shadow_graph_reuses_precomputed_query_prep() -> None:
    pipeline = _build_pipeline()

    class ExplodingRewriter:
        def rewrite(self, query: str) -> str:
            raise AssertionError("rewriter should not be called when query_prep is provided")

    pipeline.rewriter = ExplodingRewriter()

    result = pipeline.run_shadow_graph(
        query="不给劳动者购置五险可以直接离职吗",
        top_k_retrieve=5,
        top_k_rerank=2,
        knowledge_base_id="kb_001",
        source_name_by_document_id={
            "doc_001": "《中华人民共和国劳动合同法》",
            "doc_002": "《中华人民共和国劳动合同法》",
        },
        query_prep={
            "effective_query": "不给劳动者购置五险可以直接离职吗",
            "expanded_keywords": "预计算扩写",
            "alias_keywords": "预计算别名",
            "query_keywords": ["五险", "离职"],
            "search_query": "不给劳动者购置五险可以直接离职吗 预计算扩写 预计算别名",
            "rewrite_ms": 7,
        },
    )

    assert result["debug"]["rewritten_query"] == "不给劳动者购置五险可以直接离职吗 预计算扩写 预计算别名"


def test_pipeline_shadow_workflow_uses_configured_shadow_retrieval_backend(monkeypatch) -> None:
    pipeline = _build_pipeline()
    pipeline.settings.shadow_retrieval_backend = "langchain_chroma"
    sentinel_retriever = object()
    captured: dict[str, object] = {}

    def fake_build_shadow_retriever(
        vector_store,
        *,
        backend: str,
        sparse_provider: str | None,
        policy,
        milvus_uri: str | None,
        milvus_token: str | None,
        milvus_drop_old: bool,
    ):
        captured["vector_store"] = vector_store
        captured["backend"] = backend
        captured["sparse_provider"] = sparse_provider
        captured["policy"] = policy
        captured["milvus_uri"] = milvus_uri
        captured["milvus_token"] = milvus_token
        captured["milvus_drop_old"] = milvus_drop_old
        return sentinel_retriever

    monkeypatch.setattr("app.rag.pipeline.build_shadow_retriever", fake_build_shadow_retriever)

    deps = pipeline._build_shadow_workflow_dependencies()

    assert deps.retriever is sentinel_retriever
    assert captured["vector_store"] is pipeline.vector_store
    assert captured["backend"] == "langchain_chroma"
    assert captured["milvus_uri"] == "./data/milvus_shadow.db"
    assert captured["milvus_token"] is None
    assert captured["milvus_drop_old"] is False
