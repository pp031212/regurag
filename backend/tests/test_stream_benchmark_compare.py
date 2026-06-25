from pathlib import Path

from app.evals.stream_benchmark_compare import compare_stream_benchmark_reports, load_stream_benchmark_report


def _metric(avg: float, p50: float, p95: float, max_value: float) -> dict[str, float]:
    return {
        "avg": avg,
        "min": avg,
        "p50": p50,
        "p95": p95,
        "max": max_value,
    }


def test_compare_stream_benchmark_reports_summarizes_metric_deltas() -> None:
    baseline_report = {
        "label": "baseline",
        "summary": {
            "case_count": 1,
            "run_count": 2,
            "ok_count": 2,
            "error_count": 0,
            "first_token_ms": _metric(100.0, 100.0, 140.0, 150.0),
            "total_ms": _metric(500.0, 500.0, 620.0, 650.0),
            "server_first_token_ms": _metric(80.0, 80.0, 100.0, 110.0),
            "server_latency_ms": _metric(450.0, 450.0, 580.0, 600.0),
            "server_stage_timings_ms": {
                "retrieve_ms": _metric(25.0, 25.0, 30.0, 35.0),
                "context_build_ms": _metric(12.0, 12.0, 15.0, 18.0),
            },
        },
        "results": [
            {
                "id": "bk001",
                "question": "q1",
                "category": "overview",
                "run_count": 2,
                "ok_count": 2,
                "error_count": 0,
                "first_token_ms": _metric(100.0, 100.0, 140.0, 150.0),
                "total_ms": _metric(500.0, 500.0, 620.0, 650.0),
                "server_first_token_ms": _metric(80.0, 80.0, 100.0, 110.0),
                "server_latency_ms": _metric(450.0, 450.0, 580.0, 600.0),
                "server_stage_timings_ms": {
                    "retrieve_ms": _metric(25.0, 25.0, 30.0, 35.0),
                    "context_build_ms": _metric(12.0, 12.0, 15.0, 18.0),
                },
            }
        ],
    }
    contender_report = {
        "label": "contender",
        "summary": {
            "case_count": 1,
            "run_count": 2,
            "ok_count": 1,
            "error_count": 1,
            "first_token_ms": _metric(90.0, 90.0, 120.0, 130.0),
            "total_ms": _metric(540.0, 540.0, 700.0, 720.0),
            "server_first_token_ms": _metric(70.0, 70.0, 95.0, 100.0),
            "server_latency_ms": _metric(430.0, 430.0, 610.0, 640.0),
            "server_stage_timings_ms": {
                "retrieve_ms": _metric(20.0, 20.0, 26.0, 28.0),
                "context_build_ms": _metric(18.0, 18.0, 21.0, 22.0),
            },
        },
        "results": [
            {
                "id": "bk001",
                "question": "q1",
                "category": "overview",
                "run_count": 2,
                "ok_count": 1,
                "error_count": 1,
                "first_token_ms": _metric(90.0, 90.0, 120.0, 130.0),
                "total_ms": _metric(540.0, 540.0, 700.0, 720.0),
                "server_first_token_ms": _metric(70.0, 70.0, 95.0, 100.0),
                "server_latency_ms": _metric(430.0, 430.0, 610.0, 640.0),
                "server_stage_timings_ms": {
                    "retrieve_ms": _metric(20.0, 20.0, 26.0, 28.0),
                    "context_build_ms": _metric(18.0, 18.0, 21.0, 22.0),
                },
            }
        ],
    }

    comparison = compare_stream_benchmark_reports(baseline_report, contender_report)

    assert comparison["summary_counts"]["error_count"]["delta"] == 1
    assert comparison["summary_metrics"]["first_token_ms"]["avg"]["delta"] == -10.0
    assert comparison["summary_metrics"]["first_token_ms"]["avg"]["improved"] is True
    assert comparison["summary_metrics"]["total_ms"]["p95"]["delta"] == 80.0
    assert comparison["summary_metrics"]["total_ms"]["p95"]["improved"] is False
    assert comparison["summary_stage_metrics"]["retrieve_ms"]["avg"]["delta"] == -5.0
    assert comparison["summary_stage_metrics"]["retrieve_ms"]["avg"]["improved"] is True
    assert comparison["case_diffs"][0]["stage_metrics"]["context_build_ms"]["avg"]["delta"] == 6.0
    assert comparison["case_diffs"][0]["error_count"]["delta"] == 1


def test_load_stream_benchmark_report_rejects_invalid_shape(tmp_path: Path) -> None:
    report_path = tmp_path / "bad.json"
    report_path.write_text('{"summary": {}}', encoding="utf-8")

    try:
        load_stream_benchmark_report(report_path)
    except ValueError as exc:
        assert "results" in str(exc)
    else:
        raise AssertionError("expected ValueError")
