from __future__ import annotations

from app.evals.shadow_compare import (
    ChainSnapshot,
    EvalCase,
    ShadowCompareCaseResult,
    build_output_payload,
    build_retrieval_drift_cases,
    build_summary,
    extract_retrieval_drift_fields,
    normalize_answer_for_compare,
    select_eval_cases,
)


def test_select_eval_cases_filters_and_limits() -> None:
    cases = [
        EvalCase("q001", "a", "grounded", [], [], "cat", expected_answer_keyword_groups=[]),
        EvalCase("q002", "b", "grounded", [], [], "cat", expected_answer_keyword_groups=[]),
        EvalCase("q003", "c", "grounded", [], [], "cat", expected_answer_keyword_groups=[]),
    ]

    selected = select_eval_cases(cases, case_ids=["q003", "q001"], limit=1)

    assert [case.id for case in selected] == ["q001"]


def test_build_summary_aggregates_match_and_error_metrics() -> None:
    snapshot_ok = ChainSnapshot(
        answer="ok",
        normalized_answer="ok",
        answer_source="legacy_pipeline",
        answer_hit=True,
        answer_hit_ratio=1.0,
        matched_answer_keywords=["ok"],
        missing_answer_keywords=[],
        citation_count=1,
        citation_chunk_ids=["chunk-1"],
        final_context_count=1,
        final_context_chunk_ids=["chunk-1"],
        rewritten_query="ok",
        latency_ms=10,
    )
    snapshot_err = ChainSnapshot(
        answer="",
        normalized_answer="",
        answer_source="shadow_graph",
        answer_hit=False,
        answer_hit_ratio=0.0,
        matched_answer_keywords=[],
        missing_answer_keywords=["x"],
        citation_count=0,
        citation_chunk_ids=[],
        final_context_count=0,
        final_context_chunk_ids=[],
        rewritten_query="",
        latency_ms=20,
        error="boom",
    )
    results = [
        ShadowCompareCaseResult(
            id="q001",
            question="a",
            category="cat",
            answer_mode="grounded",
            notes="",
            final_answer_match=True,
            final_answer_normalized_match=True,
            final_answer_keyword_set_match=True,
            final_answer_hit_parity=True,
            final_citation_ids_match=True,
            final_context_ids_match=True,
            retrieval_drift=False,
            retrieval_drift_fields=[],
            core_compare_status="match",
            core_mismatch_fields=[],
            core_compare={"status": "match"},
            legacy=snapshot_ok,
            graph=snapshot_ok,
        ),
        ShadowCompareCaseResult(
            id="q002",
            question="b",
            category="cat",
            answer_mode="grounded",
            notes="",
            final_answer_match=False,
            final_answer_normalized_match=False,
            final_answer_keyword_set_match=False,
            final_answer_hit_parity=False,
            final_citation_ids_match=False,
            final_context_ids_match=False,
            retrieval_drift=False,
            retrieval_drift_fields=[],
            core_compare_status="error",
            core_mismatch_fields=["shadow_graph_error"],
            core_compare={"status": "error"},
            legacy=snapshot_ok,
            graph=snapshot_err,
        ),
    ]

    summary = build_summary(results)

    assert summary["case_count"] == 2
    assert summary["legacy_answer_hit_rate"] == 1.0
    assert summary["graph_answer_hit_rate"] == 0.5
    assert summary["final_answer_match_rate"] == 0.5
    assert summary["final_answer_normalized_match_rate"] == 0.5
    assert summary["final_answer_keyword_set_match_rate"] == 0.5
    assert summary["final_answer_hit_parity_rate"] == 0.5
    assert summary["retrieval_drift_case_count"] == 0
    assert summary["retrieval_drift_rate"] == 0.0
    assert summary["core_match_rate"] == 0.5
    assert summary["core_error_rate"] == 0.5
    assert summary["graph_error_count"] == 1
    assert summary["mismatch_breakdown"] == {"shadow_graph_error": 1}
    assert summary["retrieval_drift_breakdown"] == {}
    assert summary["retrieval_drift_case_ids"] == []


def test_normalize_answer_for_compare_removes_formatting_noise() -> None:
    legacy = """结论：
- 迟到 20 分钟以内：扣 3 分/次。

依据：
- 《和鸣教育管理制度精简chunk版》：迟到或早退在 20 分钟以内，处罚为扣 3 分/次。
"""
    graph = """【结论】
- 迟到 20 分钟以内：扣 3 分/次。

【直接依据】
- 《和鸣教育管理制度精简chunk版》规定：迟到或早退 20 分钟以内，扣 3 分/次。
"""

    legacy_normalized = normalize_answer_for_compare(legacy)
    graph_normalized = normalize_answer_for_compare(graph)

    assert "结论" not in legacy_normalized
    assert "依据" not in legacy_normalized
    assert "和鸣教育管理制度精简chunk版" not in legacy_normalized
    assert "结论" not in graph_normalized
    assert "直接依据" not in graph_normalized
    assert "和鸣教育管理制度精简chunk版" not in graph_normalized
    assert "迟到20分钟以内扣3分/次" in legacy_normalized
    assert "迟到20分钟以内扣3分/次" in graph_normalized


def test_extract_retrieval_drift_fields_filters_to_non_text_mismatches() -> None:
    mismatch_fields = ["answer", "citation_ids", "final_context_ids", "rewritten_query"]

    assert extract_retrieval_drift_fields(mismatch_fields) == [
        "citation_ids",
        "final_context_ids",
        "rewritten_query",
    ]


def test_build_retrieval_drift_cases_returns_only_true_drift_items() -> None:
    snapshot = ChainSnapshot(
        answer="ok",
        normalized_answer="ok",
        answer_source="legacy_pipeline",
        answer_hit=True,
        answer_hit_ratio=1.0,
        matched_answer_keywords=["ok"],
        missing_answer_keywords=[],
        citation_count=1,
        citation_chunk_ids=["chunk-1"],
        final_context_count=1,
        final_context_chunk_ids=["parent-1"],
        rewritten_query="query a",
        latency_ms=10,
    )
    drift_snapshot = ChainSnapshot(
        answer="ok",
        normalized_answer="ok",
        answer_source="shadow_graph",
        answer_hit=True,
        answer_hit_ratio=1.0,
        matched_answer_keywords=["ok"],
        missing_answer_keywords=[],
        citation_count=1,
        citation_chunk_ids=["chunk-2"],
        final_context_count=1,
        final_context_chunk_ids=["parent-2"],
        rewritten_query="query b",
        latency_ms=12,
    )
    results = [
        ShadowCompareCaseResult(
            id="q001",
            question="a",
            category="cat",
            answer_mode="grounded",
            notes="",
            final_answer_match=False,
            final_answer_normalized_match=False,
            final_answer_keyword_set_match=True,
            final_answer_hit_parity=True,
            final_citation_ids_match=False,
            final_context_ids_match=False,
            retrieval_drift=True,
            retrieval_drift_fields=["citation_ids", "final_context_ids", "rewritten_query"],
            core_compare_status="mismatch",
            core_mismatch_fields=["answer", "citation_ids", "final_context_ids", "rewritten_query"],
            core_compare={"status": "mismatch"},
            legacy=snapshot,
            graph=drift_snapshot,
        ),
        ShadowCompareCaseResult(
            id="q002",
            question="b",
            category="cat",
            answer_mode="grounded",
            notes="",
            final_answer_match=False,
            final_answer_normalized_match=False,
            final_answer_keyword_set_match=True,
            final_answer_hit_parity=True,
            final_citation_ids_match=True,
            final_context_ids_match=True,
            retrieval_drift=False,
            retrieval_drift_fields=[],
            core_compare_status="mismatch",
            core_mismatch_fields=["answer"],
            core_compare={"status": "mismatch"},
            legacy=snapshot,
            graph=snapshot,
        ),
    ]

    drift_cases = build_retrieval_drift_cases(results)

    assert drift_cases == [
        {
            "id": "q001",
            "question": "a",
            "category": "cat",
            "retrieval_drift_fields": ["citation_ids", "final_context_ids", "rewritten_query"],
            "core_mismatch_fields": ["answer", "citation_ids", "final_context_ids", "rewritten_query"],
            "legacy_citation_chunk_ids": ["chunk-1"],
            "graph_citation_chunk_ids": ["chunk-2"],
            "legacy_final_context_chunk_ids": ["parent-1"],
            "graph_final_context_chunk_ids": ["parent-2"],
            "legacy_rewritten_query": "query a",
            "graph_rewritten_query": "query b",
        }
    ]


def test_build_output_payload_records_shadow_retrieval_backend(tmp_path) -> None:
    snapshot = ChainSnapshot(
        answer="ok",
        normalized_answer="ok",
        answer_source="legacy_pipeline",
        answer_hit=True,
        answer_hit_ratio=1.0,
        matched_answer_keywords=["ok"],
        missing_answer_keywords=[],
        citation_count=1,
        citation_chunk_ids=["chunk-1"],
        final_context_count=1,
        final_context_chunk_ids=["parent-1"],
        rewritten_query="query a",
        latency_ms=10,
    )
    results = [
        ShadowCompareCaseResult(
            id="q001",
            question="a",
            category="cat",
            answer_mode="grounded",
            notes="",
            final_answer_match=True,
            final_answer_normalized_match=True,
            final_answer_keyword_set_match=True,
            final_answer_hit_parity=True,
            final_citation_ids_match=True,
            final_context_ids_match=True,
            retrieval_drift=False,
            retrieval_drift_fields=[],
            core_compare_status="match",
            core_mismatch_fields=[],
            core_compare={"status": "match"},
            legacy=snapshot,
            graph=snapshot,
        )
    ]

    payload = build_output_payload(
        dataset=tmp_path / "dataset.jsonl",
        knowledge_base_id="kb_001",
        top_k_retrieve=15,
        top_k_rerank=3,
        enable_rewrite=True,
        enable_rerank=True,
        answer_style="concise",
        shadow_retrieval_backend="langchain_chroma",
        selected_case_ids=["q001"],
        results=results,
    )

    assert payload["config"]["shadow_retrieval_backend"] == "langchain_chroma"
