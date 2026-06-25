from dataclasses import dataclass

import pytest

from app.workflows.rag.graph import build_shadow_graph
from app.workflows.rag.nodes import RagWorkflowDependencies
from app.workflows.rag.runner import run_shadow_workflow


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


@dataclass(slots=True)
class FakeRetrievalRules:
    policy_trigger_words: tuple[str, ...] = ("规定", "制度")
    policy_trigger_phrases: tuple[str, ...] = ("什么条件",)
    behavior_keywords: tuple[str, ...] = ("迟到", "旷课")
    must_keep_child_keywords: tuple[str, ...] = ("社会保险费",)
    low_value_parent_keywords: tuple[str, ...] = ()
    rule_signal_keywords: tuple[str, ...] = ()


def _build_dependencies() -> RagWorkflowDependencies:
    return RagWorkflowDependencies(
        rewriter=FakeRewriter(),
        query_alias_expander=FakeAliasExpander(),
        vector_store=FakeVectorStore(),
        reranker=FakeReranker(),
        llm=FakeLLM(),
        retrieval_rules=FakeRetrievalRules(),
        top_k_mmr=4,
    )


def test_run_shadow_workflow_returns_answer_citations_and_debug() -> None:
    state = run_shadow_workflow(
        {
            "query": "不给劳动者购置五险可以直接离职吗",
            "top_k_retrieve": 5,
            "top_k_rerank": 2,
            "knowledge_base_id": "kb_001",
            "source_name_by_document_id": {
                "doc_001": "《中华人民共和国劳动合同法》",
                "doc_002": "《中华人民共和国劳动合同法》",
            },
            "enable_rewrite": True,
            "enable_rerank": True,
            "history_rewritten_query": None,
            "history_message_count": 0,
            "history_rewrite_ms": 0,
        },
        _build_dependencies(),
    )

    assert state["answer"] == "concise:基于上下文回答：不给劳动者购置五险可以直接离职吗"
    assert len(state["citations"]) == 2
    assert state["debug"]["rewritten_query"]
    assert state["debug"]["retrieved_count"] == 1
    assert state["debug"]["reranked_count"] == 2
    assert state["debug"]["llm_model"] == "fake-model"
    assert state["debug"]["final_context_chunks"]


def test_build_shadow_graph_raises_clear_error_when_langgraph_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    original_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "langgraph.graph":
            raise ImportError("mock missing langgraph")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)

    with pytest.raises(RuntimeError, match="LangGraph is not installed"):
        build_shadow_graph(_build_dependencies())


def test_build_shadow_graph_can_compile_and_invoke_with_langgraph_installed() -> None:
    graph = build_shadow_graph(_build_dependencies())
    result = graph.invoke(
        {
            "query": "不给劳动者购置五险可以直接离职吗",
            "top_k_retrieve": 5,
            "top_k_rerank": 2,
            "knowledge_base_id": "kb_001",
            "source_name_by_document_id": {
                "doc_001": "《中华人民共和国劳动合同法》",
                "doc_002": "《中华人民共和国劳动合同法》",
            },
            "enable_rewrite": True,
            "enable_rerank": True,
            "history_rewritten_query": None,
            "history_message_count": 0,
            "history_rewrite_ms": 0,
        }
    )

    assert result["answer"] == "concise:基于上下文回答：不给劳动者购置五险可以直接离职吗"
    assert len(result["citations"]) == 2
    assert result["debug"]["llm_model"] == "fake-model"


def test_run_shadow_workflow_reuses_precomputed_query_prep() -> None:
    deps = _build_dependencies()

    class ExplodingRewriter:
        def rewrite(self, query: str) -> str:
            raise AssertionError("rewriter should not be called when precomputed query prep is provided")

    deps.rewriter = ExplodingRewriter()
    state = run_shadow_workflow(
        {
            "query": "不给劳动者购置五险可以直接离职吗",
            "top_k_retrieve": 5,
            "top_k_rerank": 2,
            "knowledge_base_id": "kb_001",
            "source_name_by_document_id": {
                "doc_001": "《中华人民共和国劳动合同法》",
                "doc_002": "《中华人民共和国劳动合同法》",
            },
            "enable_rewrite": True,
            "enable_rerank": True,
            "precomputed_effective_query": "不给劳动者购置五险可以直接离职吗",
            "precomputed_expanded_keywords": "预计算扩写",
            "precomputed_alias_keywords": "预计算别名",
            "precomputed_query_keywords": ["五险", "离职"],
            "precomputed_search_query": "不给劳动者购置五险可以直接离职吗 预计算扩写 预计算别名",
            "precomputed_rewrite_ms": 9,
            "history_rewritten_query": None,
            "history_message_count": 0,
            "history_rewrite_ms": 0,
        },
        deps,
    )

    assert state["debug"]["rewritten_query"] == "不给劳动者购置五险可以直接离职吗 预计算扩写 预计算别名"
