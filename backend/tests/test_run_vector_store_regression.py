from __future__ import annotations

import json
from pathlib import Path

from scripts.run_vector_store_regression import (
    _build_eval_report_payload,
    _chunk_eval_cases,
    _load_existing_stage_reports,
    _load_partial_eval_results,
    _normalize_batch_size,
    _normalize_stages,
)
from scripts.eval_rag import EvalCase


def test_normalize_stages_defaults_to_all() -> None:
    assert _normalize_stages([]) == ("retrieval", "live_eval", "shadow")


def test_normalize_stages_preserves_order_and_deduplicates() -> None:
    assert _normalize_stages(["live_eval", "retrieval", "live_eval"]) == ("live_eval", "retrieval")


def test_normalize_batch_size_accepts_zero_as_unbatched() -> None:
    assert _normalize_batch_size(0) is None
    assert _normalize_batch_size(3) == 3


def test_load_existing_stage_reports_requires_all_backend_reports(tmp_path: Path) -> None:
    complete = _load_existing_stage_reports(
        output_dir=tmp_path,
        label_prefix="vector-store-regression",
        stage="retrieval",
        backends=("chroma", "milvus"),
    )
    assert complete is None

    chroma_path = tmp_path / "vector-store-regression-retrieval-chroma.json"
    chroma_path.write_text(json.dumps({"summary": {"retrieval_hit_rate": 1.0}}), encoding="utf-8")
    missing_milvus = _load_existing_stage_reports(
        output_dir=tmp_path,
        label_prefix="vector-store-regression",
        stage="retrieval",
        backends=("chroma", "milvus"),
    )
    assert missing_milvus is None

    milvus_path = tmp_path / "vector-store-regression-retrieval-milvus.json"
    milvus_path.write_text(json.dumps({"summary": {"retrieval_hit_rate": 1.0}}), encoding="utf-8")

    loaded_reports, loaded_paths = _load_existing_stage_reports(
        output_dir=tmp_path,
        label_prefix="vector-store-regression",
        stage="retrieval",
        backends=("chroma", "milvus"),
    )
    assert loaded_reports["chroma"]["summary"]["retrieval_hit_rate"] == 1.0
    assert loaded_reports["milvus"]["summary"]["retrieval_hit_rate"] == 1.0
    assert loaded_paths == {
        "chroma": chroma_path,
        "milvus": milvus_path,
    }


def test_load_partial_eval_results_filters_unselected_cases(tmp_path: Path) -> None:
    report_path = tmp_path / "vector-store-regression-eval-milvus.json"
    report_path.write_text(
        json.dumps(
            {
                "results": [
                    {"id": "rk001", "answer_hit": True},
                    {"id": "rk002", "answer_hit": False},
                    {"id": "rk999", "answer_hit": True},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    loaded = _load_partial_eval_results(
        report_path=report_path,
        selected_case_ids=["rk001", "rk002"],
    )

    assert loaded == {
        "rk001": {"id": "rk001", "answer_hit": True},
        "rk002": {"id": "rk002", "answer_hit": False},
    }


def test_chunk_eval_cases_respects_batch_size() -> None:
    cases = [
        EvalCase(
            id=f"rk00{index}",
            question="q",
            answer_mode="grounded",
            expected_answer_keywords=["a"],
            expected_context_keywords=["c"],
            category="test",
        )
        for index in range(1, 6)
    ]

    batches = _chunk_eval_cases(cases, 2)
    assert [[case.id for case in batch] for batch in batches] == [
        ["rk001", "rk002"],
        ["rk003", "rk004"],
        ["rk005"],
    ]


def test_build_eval_report_payload_recomputes_summary_in_selected_order(tmp_path: Path) -> None:
    payload = _build_eval_report_payload(
        dataset_path=tmp_path / "dataset.jsonl",
        fixture_paths=[tmp_path / "fixture.pdf"],
        config_profile="rules_cn",
        vector_store_backend="milvus",
        milvus_uri="http://127.0.0.1:19530",
        top_k_retrieve=15,
        top_k_rerank=3,
        enable_rewrite=True,
        enable_rerank=True,
        knowledge_base_id="kb-test",
        selected_case_ids=["rk002", "rk001"],
        results_by_case_id={
            "rk001": {
                "id": "rk001",
                "question": "q1",
                "category": "test",
                "answer_mode": "grounded",
                "answer": "a1",
                "retrieval_hit": True,
                "final_context_hit": True,
                "citation_hit": True,
                "answer_hit": True,
                "answer_hit_ratio": 1.0,
                "matched_answer_keywords": ["a1"],
                "missing_answer_keywords": [],
                "matched_context_keywords": ["c1"],
                "error_type": "ok",
                "debug": {"latency_ms": 100},
                "citations": [],
            },
            "rk002": {
                "id": "rk002",
                "question": "q2",
                "category": "test",
                "answer_mode": "grounded",
                "answer": "a2",
                "retrieval_hit": True,
                "final_context_hit": True,
                "citation_hit": False,
                "answer_hit": False,
                "answer_hit_ratio": 0.5,
                "matched_answer_keywords": ["a2"],
                "missing_answer_keywords": ["b2"],
                "matched_context_keywords": ["c2"],
                "error_type": "generation_error",
                "debug": {"latency_ms": 200},
                "citations": [],
            },
        },
    )

    assert [item["id"] for item in payload["results"]] == ["rk002", "rk001"]
    assert payload["summary"]["case_count"] == 2
    assert payload["summary"]["answer_hit_rate"] == 0.5
