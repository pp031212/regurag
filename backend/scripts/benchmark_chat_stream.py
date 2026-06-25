"""压测聊天流式接口。

读取 JSONL 样本，重复调用 /chat/stream 或进程内 stream_query，输出首 token、总耗时、
服务端阶段耗时和错误统计报告。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evals.stream_benchmark import (
    build_case_summary,
    build_report,
    load_stream_benchmark_cases,
    run_inprocess_stream_request,
    run_stream_request,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the streaming chat endpoint.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "evals" / "rag_eval_overview_cases.jsonl",
        help="JSONL dataset. Only id/question/category are required; optional knowledge_base_id is supported.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/api/v1",
        help="Backend API base URL.",
    )
    parser.add_argument(
        "--inprocess",
        action="store_true",
        help="Run benchmark in-process via RAGService.stream_query instead of HTTP SSE. Useful when localhost requests are blocked.",
    )
    parser.add_argument(
        "--knowledge-base-id",
        default=None,
        help="Fallback knowledge base id when the dataset line does not provide one.",
    )
    parser.add_argument(
        "--label",
        default="stream-benchmark",
        help="Output label written to backend/evals/results/<label>.json.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Measured repeats per case.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warmup runs per case before measured repeats.",
    )
    parser.add_argument(
        "--top-k-retrieve",
        type=int,
        default=15,
        help="top_k_retrieve sent to /chat/stream.",
    )
    parser.add_argument(
        "--top-k-rerank",
        type=int,
        default=3,
        help="top_k_rerank sent to /chat/stream.",
    )
    parser.add_argument(
        "--enable-auto-route",
        action="store_true",
        help="Enable auto route during benchmarking. Disabled by default for stable baselines.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="Request timeout for each streaming call.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only benchmark the first N dataset rows.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Benchmark only the given case id. Can be repeated.",
    )
    return parser.parse_args()


def select_cases(cases: list, *, case_ids: list[str], limit: int | None) -> list:
    selected = cases
    if case_ids:
        wanted = set(case_ids)
        selected = [case for case in cases if case.id in wanted]
        missing = [case_id for case_id in case_ids if case_id not in {case.id for case in selected}]
        if missing:
            raise ValueError(f"Unknown case ids: {', '.join(missing)}")
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be greater than 0")
        selected = selected[:limit]
    if not selected:
        raise ValueError("No benchmark cases selected")
    return selected


def main() -> int:
    args = parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be greater than 0")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be zero or greater")

    cases = select_cases(
        load_stream_benchmark_cases(args.dataset),
        case_ids=list(args.case_id),
        limit=args.limit,
    )

    summaries: list[dict[str, object]] = []
    for case in cases:
        knowledge_base_id = case.knowledge_base_id or args.knowledge_base_id
        if not knowledge_base_id:
            raise ValueError(f"Case {case.id} is missing knowledge_base_id and no --knowledge-base-id was provided")

        print(f"[benchmark] {case.id} {case.question}")
        for warmup_index in range(args.warmup_runs):
            warmup_payload = {
                "knowledge_base_id": knowledge_base_id,
                "query": case.question,
                "top_k_retrieve": args.top_k_retrieve,
                "top_k_rerank": args.top_k_rerank,
                "enable_auto_route": args.enable_auto_route,
                "debug": True,
                "debug_chunks": False,
                "_run_index": -(warmup_index + 1),
            }
            warmup_result = (
                run_inprocess_stream_request(payload=warmup_payload)
                if args.inprocess
                else run_stream_request(
                    base_url=args.base_url,
                    payload=warmup_payload,
                    timeout_seconds=args.timeout_seconds,
                )
            )
            status = "ok" if warmup_result.ok else f"error={warmup_result.error}"
            print(f"  warmup {warmup_index + 1}/{args.warmup_runs}: {status}")

        measured_runs = []
        for run_index in range(args.repeats):
            payload = {
                "knowledge_base_id": knowledge_base_id,
                "query": case.question,
                "top_k_retrieve": args.top_k_retrieve,
                "top_k_rerank": args.top_k_rerank,
                "enable_auto_route": args.enable_auto_route,
                "debug": True,
                "debug_chunks": False,
                "_run_index": run_index + 1,
            }
            run_result = (
                run_inprocess_stream_request(payload=payload)
                if args.inprocess
                else run_stream_request(
                    base_url=args.base_url,
                    payload=payload,
                    timeout_seconds=args.timeout_seconds,
                )
            )
            measured_runs.append(run_result)
            if run_result.ok:
                print(
                    "  run {idx}/{total}: first_token={first}ms total={latency}ms server_first={server_first}ms".format(
                        idx=run_index + 1,
                        total=args.repeats,
                        first=run_result.first_token_ms,
                        latency=run_result.total_ms,
                        server_first=run_result.server_first_token_ms,
                    )
                )
            else:
                print(f"  run {run_index + 1}/{args.repeats}: error={run_result.error}")

        summaries.append(build_case_summary(case, measured_runs))

    report = build_report(
        label=args.label,
        dataset_path=args.dataset,
        base_url=args.base_url,
        repeats=args.repeats,
        warmup_runs=args.warmup_runs,
        top_k_retrieve=args.top_k_retrieve,
        top_k_rerank=args.top_k_rerank,
        enable_auto_route=args.enable_auto_route,
        summaries=summaries,
    )
    output_path = ROOT / "evals" / "results" / f"{args.label}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = report["summary"]
    print()
    print(f"[done] wrote {output_path}")
    print(
        "cases={case_count} runs={run_count} ok={ok_count} error={error_count}".format(
            **summary,
        )
    )
    print(
        "first_token avg={avg}ms p50={p50}ms p95={p95}ms".format(
            **summary["first_token_ms"],
        )
    )
    print(
        "total avg={avg}ms p50={p50}ms p95={p95}ms".format(
            **summary["total_ms"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
