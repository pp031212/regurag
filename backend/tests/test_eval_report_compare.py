from pathlib import Path

from app.evals.eval_report_compare import compare_eval_reports, load_eval_report


def test_compare_eval_reports_summarizes_improvements_and_regressions() -> None:
    baseline_report = {
        "dataset": "baseline.jsonl",
        "results": [
            {
                "id": "sp001",
                "question": "q1",
                "category": "plain",
                "answer_mode": "grounded",
                "retrieval_hit": False,
                "final_context_hit": False,
                "citation_hit": False,
                "answer_hit": True,
                "answer_hit_ratio": 0.5,
                "debug": {"latency_ms": 120, "final_context_chunks": []},
                "citations": [],
            },
            {
                "id": "sp002",
                "question": "q2",
                "category": "ocr",
                "answer_mode": "grounded",
                "retrieval_hit": True,
                "final_context_hit": True,
                "citation_hit": True,
                "answer_hit": True,
                "answer_hit_ratio": 1.0,
                "debug": {"latency_ms": 80, "final_context_chunks": [{"chunk_id": "a"}]},
                "citations": [{"chunk_id": "a"}],
            },
        ],
    }
    contender_report = {
        "dataset": "contender.jsonl",
        "results": [
            {
                "id": "sp001",
                "question": "q1",
                "category": "plain",
                "answer_mode": "grounded",
                "retrieval_hit": True,
                "final_context_hit": True,
                "citation_hit": True,
                "answer_hit": True,
                "answer_hit_ratio": 1.0,
                "debug": {"latency_ms": 90, "final_context_chunks": [{"chunk_id": "b"}]},
                "citations": [{"chunk_id": "b"}],
            },
            {
                "id": "sp002",
                "question": "q2",
                "category": "ocr",
                "answer_mode": "grounded",
                "retrieval_hit": True,
                "final_context_hit": False,
                "citation_hit": True,
                "answer_hit": False,
                "answer_hit_ratio": 0.5,
                "debug": {"latency_ms": 110, "final_context_chunks": []},
                "citations": [{"chunk_id": "a"}],
            },
        ],
    }

    comparison = compare_eval_reports(baseline_report, contender_report)

    assert comparison["shared_case_count"] == 2
    assert comparison["metrics"]["retrieval_hit"]["improved_case_ids"] == ["sp001"]
    assert comparison["metrics"]["final_context_hit"]["regressed_case_ids"] == ["sp002"]
    assert comparison["metrics"]["answer_hit"]["regressed_case_ids"] == ["sp002"]
    assert comparison["metrics"]["answer_hit_ratio"]["delta"] == 0.0
    assert comparison["latency"]["improved_case_ids"] == ["sp001"]
    assert comparison["latency"]["regressed_case_ids"] == ["sp002"]


def test_load_eval_report_rejects_invalid_shape(tmp_path: Path) -> None:
    report_path = tmp_path / "bad.json"
    report_path.write_text('{"summary": {}}', encoding="utf-8")

    try:
        load_eval_report(report_path)
    except ValueError as exc:
        assert "results" in str(exc)
    else:
        raise AssertionError("expected ValueError")
