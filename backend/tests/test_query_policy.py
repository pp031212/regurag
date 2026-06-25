from app.rag.pipeline import RAGPipeline
from app.rag.query_policy import QueryPolicy
from app.rag.retrievers import RetrievalPolicy


class FakeSettings:
    top_k_mmr = 4


def test_build_query_policy_wraps_retrieval_and_rerank_controls() -> None:
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.settings = FakeSettings()
    pipeline.retrieval_policy = RetrievalPolicy(dense_top_k=6, sparse_top_k=9, sparse_min_hits=1, enable_sparse=False)

    policy = pipeline._build_query_policy(
        top_k_retrieve=5,
        top_k_rerank=3,
        enable_rerank=False,
        answer_style="detailed",
    )

    assert isinstance(policy, QueryPolicy)
    assert policy.retrieval is pipeline.retrieval_policy
    assert policy.top_k_retrieve == 5
    assert policy.top_k_rerank == 3
    assert policy.enable_rerank is False
    assert policy.answer_style == "detailed"


def test_build_shadow_workflow_state_reads_values_from_query_policy() -> None:
    state = RAGPipeline._build_shadow_workflow_state(
        query="alpha",
        query_policy=QueryPolicy(
            retrieval=RetrievalPolicy(),
            top_k_retrieve=8,
            top_k_rerank=2,
            enable_rerank=False,
            answer_style="detailed",
        ),
        knowledge_base_id="kb_001",
        enable_rewrite=False,
    )

    assert state["top_k_retrieve"] == 8
    assert state["top_k_rerank"] == 2
    assert state["enable_rerank"] is False
    assert state["answer_style"] == "detailed"
    assert state["enable_rewrite"] is False
