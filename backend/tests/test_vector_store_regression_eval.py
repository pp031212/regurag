"""向量库灰度门禁汇总测试。

这些测试不跑真实检索，只校验汇总器如何根据 retrieval、live eval 和 shadow graph
报告判断候选向量库是否可以切换上线。
"""

from __future__ import annotations

from pathlib import Path

from app.evals.vector_store_regression import build_vector_store_regression_summary


def test_build_vector_store_regression_summary_marks_answer_regression_as_not_ready(tmp_path: Path) -> None:
    """答案命中率回退时，即使检索指标持平，也不能判定 Milvus 可上线。"""
    retrieval_reports = {
        "chroma": {
            "summary": {
                "retrieval_hit_rate": 1.0,
                "final_context_hit_rate": 1.0,
                "citation_hit_rate": 1.0,
            }
        },
        "milvus": {
            "summary": {
                "retrieval_hit_rate": 1.0,
                "final_context_hit_rate": 1.0,
                "citation_hit_rate": 1.0,
            }
        },
    }
    retrieval_comparisons = {
        "chroma_vs_milvus": {
            "case_drift_count": 0,
            "metric_diffs": {
                "retrieval_hit_rate": 0.0,
                "final_context_hit_rate": 0.0,
                "citation_hit_rate": 0.0,
            },
        }
    }
    eval_reports = {
        "chroma": {"summary": {"answer_hit_rate": 1.0}},
        "milvus": {"summary": {"answer_hit_rate": 0.9375}},
    }
    eval_comparisons = {
        "chroma_vs_milvus": {
            "metrics": {
                "retrieval_hit": {"delta": 0.0},
                "final_context_hit": {"delta": 0.0},
                "citation_hit": {"delta": 0.0},
                "answer_hit": {"delta": -0.0625},
            }
        }
    }
    shadow_reports = {
        "chroma": {
            "summary": {
                "graph_error_count": 0,
                "retrieval_drift_case_count": 0,
                "final_answer_hit_parity_rate": 1.0,
                "final_citation_ids_match_rate": 0.75,
                "final_context_ids_match_rate": 0.75,
                "avg_graph_latency_ms": 18000.0,
            }
        },
        "milvus": {
            "summary": {
                "graph_error_count": 0,
                "retrieval_drift_case_count": 0,
                "final_answer_hit_parity_rate": 1.0,
                "final_citation_ids_match_rate": 1.0,
                "final_context_ids_match_rate": 1.0,
                "avg_graph_latency_ms": 15000.0,
            }
        },
    }

    summary = build_vector_store_regression_summary(
        retrieval_reports=retrieval_reports,
        retrieval_comparisons=retrieval_comparisons,
        eval_reports=eval_reports,
        eval_comparisons=eval_comparisons,
        shadow_reports=shadow_reports,
        baseline_backend="chroma",
        report_paths={
            "chroma": {
                "retrieval": tmp_path / "retrieval-chroma.json",
                "eval": tmp_path / "eval-chroma.json",
                "shadow": tmp_path / "shadow-chroma.json",
            },
            "milvus": {
                "retrieval": tmp_path / "retrieval-milvus.json",
                "eval": tmp_path / "eval-milvus.json",
                "shadow": tmp_path / "shadow-milvus.json",
            },
        },
    )

    milvus = summary["backend_summaries"]["milvus"]
    assert milvus["gates"]["retrieval_drift_free"] is True
    assert milvus["gates"]["live_eval_answer_parity_ok"] is False
    assert milvus["gates"]["shadow_context_alignment_ok"] is True
    assert milvus["ready_for_rollout"] is False
    assert summary["diffs_vs_baseline"]["milvus"]["shadow_compare"]["avg_graph_latency_ms"] == {
        "baseline": 18000,
        "contender": 15000,
        "delta": -3000.0,
    }


def test_build_vector_store_regression_summary_requires_two_backends() -> None:
    """灰度对比至少需要 baseline 和 contender 两个后端。"""
    try:
        build_vector_store_regression_summary(
            retrieval_reports={"chroma": {"summary": {"retrieval_hit_rate": 1.0}}},
            retrieval_comparisons={},
            shadow_reports={"chroma": {"summary": {"graph_error_count": 0}}},
            baseline_backend="chroma",
        )
    except ValueError as exc:
        assert "requires at least two backend reports" in str(exc)
    else:
        raise AssertionError("expected summary builder to reject single-backend input")
