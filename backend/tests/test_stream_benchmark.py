from pathlib import Path

from app.evals.stream_benchmark import (
    build_case_summary,
    build_report,
    extract_stage_timings,
    load_stream_benchmark_cases,
    parse_sse_event_block,
    percentile,
    summarize_numeric,
    StreamBenchmarkCase,
    StreamBenchmarkRun,
)


def test_load_stream_benchmark_cases_reads_existing_eval_shape(tmp_path: Path) -> None:
    dataset_path = tmp_path / "benchmark.jsonl"
    dataset_path.write_text(
        '\n'.join(
            [
                '{"id":"bk001","question":"请假一般怎么处理？","category":"overview","notes":"n1"}',
                '{"id":"bk002","question":"课堂违纪一般会怎么处理？","category":"overview","knowledge_base_id":"kb_002"}',
            ]
        ),
        encoding="utf-8",
    )

    cases = load_stream_benchmark_cases(dataset_path)

    assert [case.id for case in cases] == ["bk001", "bk002"]
    assert cases[0].knowledge_base_id is None
    assert cases[1].knowledge_base_id == "kb_002"


def test_parse_sse_event_block_extracts_event_and_json() -> None:
    event, data = parse_sse_event_block(
        "event: token\n"
        'data: {"delta":"测试"}\n'
    )

    assert event == "token"
    assert data == {"delta": "测试"}


def test_percentile_supports_interpolation() -> None:
    values = [10.0, 20.0, 30.0, 40.0]

    assert percentile(values, 0.5) == 25.0
    assert percentile(values, 0.95) == 38.5


def test_build_case_summary_aggregates_ok_and_errors() -> None:
    case = StreamBenchmarkCase(id="bk001", question="测试问题", category="overview")
    runs = [
        StreamBenchmarkRun(
            ok=True,
            run_index=1,
            first_token_ms=120.0,
            total_ms=480.0,
            server_first_token_ms=90,
            server_latency_ms=430,
            token_event_count=5,
            answer_chars=42,
            server_stage_timings_ms={"retrieve_ms": 25, "context_build_ms": 12},
        ),
        StreamBenchmarkRun(
            ok=False,
            run_index=2,
            first_token_ms=None,
            total_ms=300.0,
            server_first_token_ms=None,
            server_latency_ms=None,
            token_event_count=0,
            answer_chars=0,
            error="timeout",
        ),
    ]

    summary = build_case_summary(case, runs)

    assert summary["ok_count"] == 1
    assert summary["error_count"] == 1
    assert summary["first_token_ms"]["avg"] == 120.0
    assert summary["server_latency_ms"]["avg"] == 430.0
    assert summary["server_stage_timings_ms"]["retrieve_ms"]["avg"] == 25.0
    assert summary["errors"] == ["timeout"]


def test_build_report_aggregates_overall_summary(tmp_path: Path) -> None:
    summaries = [
        {
            "id": "bk001",
            "run_count": 2,
            "ok_count": 1,
            "error_count": 1,
            "runs": [
                {
                    "ok": True,
                    "run_index": 1,
                    "first_token_ms": 100.0,
                    "total_ms": 500.0,
                    "server_first_token_ms": 80,
                    "server_latency_ms": 450,
                    "token_event_count": 4,
                    "answer_chars": 20,
                    "server_stage_timings_ms": {"retrieve_ms": 22, "context_build_ms": 8},
                    "error": None,
                },
                {
                    "ok": False,
                    "run_index": 2,
                    "first_token_ms": None,
                    "total_ms": 300.0,
                    "server_first_token_ms": None,
                    "server_latency_ms": None,
                    "token_event_count": 0,
                    "answer_chars": 0,
                    "error": "timeout",
                },
            ],
        }
    ]

    report = build_report(
        label="stream-benchmark",
        dataset_path=tmp_path / "dataset.jsonl",
        base_url="http://127.0.0.1:8000/api/v1",
        repeats=2,
        warmup_runs=1,
        top_k_retrieve=15,
        top_k_rerank=3,
        enable_auto_route=False,
        summaries=summaries,
    )

    assert report["summary"]["case_count"] == 1
    assert report["summary"]["run_count"] == 2
    assert report["summary"]["ok_count"] == 1
    assert report["summary"]["error_count"] == 1
    assert report["summary"]["first_token_ms"]["avg"] == 100.0
    assert report["summary"]["server_latency_ms"]["avg"] == 450.0
    assert report["summary"]["server_stage_timings_ms"]["retrieve_ms"]["avg"] == 22.0


def test_summarize_numeric_returns_none_for_empty_values() -> None:
    summary = summarize_numeric([])

    assert summary == {
        "avg": None,
        "min": None,
        "p50": None,
        "p95": None,
        "max": None,
    }


def test_extract_stage_timings_prefers_nested_stage_block() -> None:
    timings = extract_stage_timings(
        {
            "rewrite_ms": 10,
            "stage_timings_ms": {
                "retrieve_ms": 20,
                "context_build_ms": 5,
            },
        }
    )

    assert timings == {
        "retrieve_ms": 20,
        "context_build_ms": 5,
    }
