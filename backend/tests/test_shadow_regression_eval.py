from __future__ import annotations

from pathlib import Path

from app.evals.shadow_regression import build_shadow_regression_summary


def test_build_shadow_regression_summary_builds_backend_gates_and_diffs(tmp_path: Path) -> None:
    reports = {
        "langchain_chroma": {
            "summary": {
                "graph_error_count": 0,
                "retrieval_drift_case_count": 0,
                "final_answer_hit_parity_rate": 1.0,
                "final_citation_ids_match_rate": 1.0,
                "final_context_ids_match_rate": 1.0,
                "avg_graph_latency_ms": 18000.0,
            }
        },
        "langchain_milvus": {
            "summary": {
                "graph_error_count": 0,
                "retrieval_drift_case_count": 0,
                "final_answer_hit_parity_rate": 1.0,
                "final_citation_ids_match_rate": 0.8889,
                "final_context_ids_match_rate": 0.9444,
                "avg_graph_latency_ms": 16000.0,
            }
        },
    }

    summary = build_shadow_regression_summary(
        reports=reports,
        baseline_backend="langchain_chroma",
        report_paths={
            "langchain_chroma": tmp_path / "shadow-chroma.json",
            "langchain_milvus": tmp_path / "shadow-milvus.json",
        },
    )

    assert summary["baseline_backend"] == "langchain_chroma"
    assert summary["backend_summaries"]["langchain_chroma"]["gates"] == {
        "graph_error_free": True,
        "retrieval_drift_free": True,
        "citation_alignment_ok": True,
        "context_alignment_ok": True,
        "answer_hit_parity_ok": True,
    }
    assert summary["backend_summaries"]["langchain_milvus"]["gates"] == {
        "graph_error_free": True,
        "retrieval_drift_free": True,
        "citation_alignment_ok": False,
        "context_alignment_ok": False,
        "answer_hit_parity_ok": True,
    }
    assert summary["diffs_vs_baseline"]["langchain_milvus"]["avg_graph_latency_ms"] == {
        "baseline": 18000,
        "contender": 16000,
        "delta": -2000.0,
    }


def test_build_shadow_regression_summary_rejects_invalid_baseline() -> None:
    try:
        build_shadow_regression_summary(
            reports={"langchain_milvus": {"summary": {"graph_error_count": 0}}},
            baseline_backend="langchain_chroma",
        )
    except ValueError as exc:
        assert "Unknown baseline backend" in str(exc)
    else:
        raise AssertionError("expected invalid baseline backend to raise")
