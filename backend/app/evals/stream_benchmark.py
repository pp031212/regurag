"""流式聊天性能评测工具。

同时支持 HTTP SSE 和进程内调用，统计首 token、总耗时、服务端阶段耗时、
token 事件数量和错误信息。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any
from urllib import request


@dataclass(slots=True)
class StreamBenchmarkCase:
    """单条流式 benchmark 样本。"""

    id: str
    question: str
    category: str
    knowledge_base_id: str | None = None
    notes: str = ""


@dataclass(slots=True)
class StreamBenchmarkRun:
    """单次请求的流式性能记录。"""

    ok: bool
    run_index: int
    first_token_ms: float | None
    total_ms: float | None
    server_first_token_ms: int | None
    server_latency_ms: int | None
    token_event_count: int
    answer_chars: int
    server_stage_timings_ms: dict[str, float | int | None] = field(default_factory=dict)
    error: str | None = None


def load_stream_benchmark_cases(path: Path) -> list[StreamBenchmarkCase]:
    """从 JSONL 读取 benchmark 样本。"""
    cases: list[StreamBenchmarkCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid benchmark dataset line {line_number}")
            case_id = str(payload.get("id") or "").strip()
            question = str(payload.get("question") or "").strip()
            if not case_id or not question:
                raise ValueError(f"Invalid benchmark dataset line {line_number}: missing id/question")
            cases.append(
                StreamBenchmarkCase(
                    id=case_id,
                    question=question,
                    category=str(payload.get("category") or "unknown"),
                    knowledge_base_id=str(payload["knowledge_base_id"]).strip() if payload.get("knowledge_base_id") else None,
                    notes=str(payload.get("notes") or ""),
                )
            )
    if not cases:
        raise ValueError(f"No benchmark cases found in {path}")
    return cases


def parse_sse_event_block(block: str) -> tuple[str, dict[str, Any] | None]:
    """解析一个 SSE event block。"""
    event = "message"
    data_lines: list[str] = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip() or "message"
            continue
        if line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].strip())
    if not data_lines:
        return event, None
    return event, json.loads("\n".join(data_lines))


def percentile(values: list[float], ratio: float) -> float | None:
    """计算百分位数，样本少时使用线性插值。"""
    if not values:
        return None
    if ratio <= 0:
        return min(values)
    if ratio >= 1:
        return max(values)
    ordered = sorted(values)
    position = (len(ordered) - 1) * ratio
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_numeric(values: list[float]) -> dict[str, float | None]:
    """生成 avg/min/p50/p95/max 摘要。"""
    return {
        "avg": round(mean(values), 2) if values else None,
        "min": round(min(values), 2) if values else None,
        "p50": round(percentile(values, 0.5) or 0.0, 2) if values else None,
        "p95": round(percentile(values, 0.95) or 0.0, 2) if values else None,
        "max": round(max(values), 2) if values else None,
    }


def extract_stage_timings(debug: dict[str, Any] | None) -> dict[str, float | int | None]:
    """从 debug 中提取服务端阶段耗时，兼容新旧字段。"""
    if not isinstance(debug, dict):
        return {}

    raw_stage_timings = debug.get("stage_timings_ms")
    if isinstance(raw_stage_timings, dict) and raw_stage_timings:
        return {
            str(key): value
            for key, value in raw_stage_timings.items()
            if isinstance(key, str)
        }

    return {
        "history_rewrite_ms": debug.get("history_rewrite_ms"),
        "rewrite_ms": debug.get("rewrite_ms"),
        "retrieve_ms": debug.get("retrieve_ms"),
        "rerank_ms": debug.get("rerank_ms"),
        "context_build_ms": debug.get("context_build_ms"),
        "generate_ms": debug.get("generate_ms"),
        "llm_first_token_ms": debug.get("llm_first_token_ms"),
    }


def summarize_stage_timings(runs: list[StreamBenchmarkRun]) -> dict[str, dict[str, float | None]]:
    """汇总多次运行中的服务端阶段耗时。"""
    metric_names = sorted(
        {
            metric_name
            for run in runs
            if run.ok
            for metric_name in run.server_stage_timings_ms
        }
    )
    return {
        metric_name: summarize_numeric(
            [
                float(value)
                for run in runs
                if run.ok
                for value in [run.server_stage_timings_ms.get(metric_name)]
                if value is not None
            ]
        )
        for metric_name in metric_names
    }


def run_stream_request(
    *,
    base_url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> StreamBenchmarkRun:
    """默认通过 HTTP SSE 调用流式接口。"""
    return run_http_stream_request(
        base_url=base_url,
        payload=payload,
        timeout_seconds=timeout_seconds,
    )


def run_http_stream_request(
    *,
    base_url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> StreamBenchmarkRun:
    """实际发起 HTTP SSE 请求并统计客户端观测到的耗时。"""
    target = f"{base_url.rstrip('/')}/chat/stream"
    req = request.Request(
        target,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    started_at = perf_counter()
    first_token_ms: float | None = None
    total_ms: float | None = None
    server_first_token_ms: int | None = None
    server_latency_ms: int | None = None
    server_stage_timings_ms: dict[str, float | int | None] = {}
    token_event_count = 0
    answer_chars = 0
    event_lines: list[str] = []
    run_index = int(payload.get("_run_index") or 0)

    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            while True:
                raw_line = response.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8")
                if line.strip():
                    event_lines.append(line.rstrip("\r\n"))
                    continue

                if not event_lines:
                    continue
                event_name, data = parse_sse_event_block("\n".join(event_lines))
                event_lines.clear()

                if event_name == "token" and isinstance(data, dict):
                    token_event_count += 1
                    delta = str(data.get("delta") or "")
                    answer_chars += len(delta)
                    if first_token_ms is None:
                        first_token_ms = round((perf_counter() - started_at) * 1000, 2)
                    continue

                if event_name == "end" and isinstance(data, dict):
                    total_ms = round((perf_counter() - started_at) * 1000, 2)
                    answer_chars = len(str(data.get("answer") or ""))
                    debug = data.get("debug")
                    if isinstance(debug, dict):
                        raw_first_token = debug.get("llm_first_token_ms")
                        raw_latency = debug.get("latency_ms")
                        server_first_token_ms = int(raw_first_token) if raw_first_token is not None else None
                        server_latency_ms = int(raw_latency) if raw_latency is not None else None
                        server_stage_timings_ms = extract_stage_timings(debug)
                    return StreamBenchmarkRun(
                        ok=True,
                        run_index=run_index,
                        first_token_ms=first_token_ms,
                        total_ms=total_ms,
                        server_first_token_ms=server_first_token_ms,
                        server_latency_ms=server_latency_ms,
                        token_event_count=token_event_count,
                        answer_chars=answer_chars,
                        server_stage_timings_ms=server_stage_timings_ms,
                    )

                if event_name == "error" and isinstance(data, dict):
                    total_ms = round((perf_counter() - started_at) * 1000, 2)
                    return StreamBenchmarkRun(
                        ok=False,
                        run_index=run_index,
                        first_token_ms=first_token_ms,
                        total_ms=total_ms,
                        server_first_token_ms=None,
                        server_latency_ms=None,
                        token_event_count=token_event_count,
                        answer_chars=answer_chars,
                        server_stage_timings_ms={},
                        error=str(data.get("message") or data.get("code") or "stream error"),
                    )
    except Exception as exc:
        total_ms = round((perf_counter() - started_at) * 1000, 2)
        return StreamBenchmarkRun(
            ok=False,
            run_index=run_index,
            first_token_ms=first_token_ms,
            total_ms=total_ms,
            server_first_token_ms=None,
            server_latency_ms=None,
            token_event_count=token_event_count,
            answer_chars=answer_chars,
            server_stage_timings_ms={},
            error=str(exc),
        )

    return StreamBenchmarkRun(
        ok=False,
        run_index=run_index,
        first_token_ms=first_token_ms,
        total_ms=total_ms,
        server_first_token_ms=server_first_token_ms,
        server_latency_ms=server_latency_ms,
        token_event_count=token_event_count,
        answer_chars=answer_chars,
        server_stage_timings_ms=server_stage_timings_ms,
        error="stream ended without end event",
    )


def run_inprocess_stream_request(
    *,
    payload: dict[str, Any],
) -> StreamBenchmarkRun:
    """在当前进程内调用 RAGService.stream_query，避开网络和 Web 服务开销。"""
    from app.schemas.chat import ChatQueryRequest
    from app.services.rag_service import RAGService

    started_at = perf_counter()
    first_token_ms: float | None = None
    total_ms: float | None = None
    server_first_token_ms: int | None = None
    server_latency_ms: int | None = None
    server_stage_timings_ms: dict[str, float | int | None] = {}
    token_event_count = 0
    answer_chars = 0
    run_index = int(payload.get("_run_index") or 0)

    request_payload = ChatQueryRequest(
        knowledge_base_id=payload.get("knowledge_base_id"),
        query=str(payload["query"]),
        conversation_id=payload.get("conversation_id"),
        top_k_retrieve=int(payload.get("top_k_retrieve") or 15),
        top_k_rerank=int(payload.get("top_k_rerank") or 3),
        enable_auto_route=bool(payload.get("enable_auto_route", False)),
        debug=bool(payload.get("debug", True)),
        debug_chunks=bool(payload.get("debug_chunks", False)),
    )

    try:
        service = RAGService()
        for event in service.stream_query(request_payload):
            event_name = str(event.get("event") or "")
            data = event.get("data")
            if not isinstance(data, dict):
                continue

            if event_name == "token":
                token_event_count += 1
                delta = str(data.get("delta") or "")
                answer_chars += len(delta)
                if first_token_ms is None:
                    first_token_ms = round((perf_counter() - started_at) * 1000, 2)
                continue

            if event_name == "end":
                total_ms = round((perf_counter() - started_at) * 1000, 2)
                answer_chars = len(str(data.get("answer") or ""))
                debug = data.get("debug")
                if isinstance(debug, dict):
                    raw_first_token = debug.get("llm_first_token_ms")
                    raw_latency = debug.get("latency_ms")
                    server_first_token_ms = int(raw_first_token) if raw_first_token is not None else None
                    server_latency_ms = int(raw_latency) if raw_latency is not None else None
                    server_stage_timings_ms = extract_stage_timings(debug)
                return StreamBenchmarkRun(
                    ok=True,
                    run_index=run_index,
                    first_token_ms=first_token_ms,
                    total_ms=total_ms,
                    server_first_token_ms=server_first_token_ms,
                    server_latency_ms=server_latency_ms,
                    token_event_count=token_event_count,
                    answer_chars=answer_chars,
                    server_stage_timings_ms=server_stage_timings_ms,
                )

            if event_name == "error":
                total_ms = round((perf_counter() - started_at) * 1000, 2)
                return StreamBenchmarkRun(
                    ok=False,
                    run_index=run_index,
                    first_token_ms=first_token_ms,
                    total_ms=total_ms,
                    server_first_token_ms=None,
                    server_latency_ms=None,
                    token_event_count=token_event_count,
                    answer_chars=answer_chars,
                    server_stage_timings_ms={},
                    error=str(data.get("message") or data.get("code") or "stream error"),
                )
    except Exception as exc:
        total_ms = round((perf_counter() - started_at) * 1000, 2)
        return StreamBenchmarkRun(
            ok=False,
            run_index=run_index,
            first_token_ms=first_token_ms,
            total_ms=total_ms,
            server_first_token_ms=None,
            server_latency_ms=None,
            token_event_count=token_event_count,
            answer_chars=answer_chars,
            server_stage_timings_ms={},
            error=str(exc),
        )

    return StreamBenchmarkRun(
        ok=False,
        run_index=run_index,
        first_token_ms=first_token_ms,
        total_ms=total_ms,
        server_first_token_ms=server_first_token_ms,
        server_latency_ms=server_latency_ms,
        token_event_count=token_event_count,
        answer_chars=answer_chars,
        server_stage_timings_ms=server_stage_timings_ms,
        error="stream ended without end event",
    )


def build_case_summary(case: StreamBenchmarkCase, runs: list[StreamBenchmarkRun]) -> dict[str, Any]:
    """汇总单个样本的多次运行结果。"""
    first_token_values = [item.first_token_ms for item in runs if item.ok and item.first_token_ms is not None]
    total_values = [item.total_ms for item in runs if item.ok and item.total_ms is not None]
    server_first_token_values = [float(item.server_first_token_ms) for item in runs if item.ok and item.server_first_token_ms is not None]
    server_latency_values = [float(item.server_latency_ms) for item in runs if item.ok and item.server_latency_ms is not None]
    ok_runs = [item for item in runs if item.ok]
    errors = [item.error for item in runs if item.error]

    return {
        "id": case.id,
        "question": case.question,
        "category": case.category,
        "knowledge_base_id": case.knowledge_base_id,
        "notes": case.notes,
        "run_count": len(runs),
        "ok_count": len(ok_runs),
        "error_count": len(runs) - len(ok_runs),
        "first_token_ms": summarize_numeric(first_token_values),
        "total_ms": summarize_numeric(total_values),
        "server_first_token_ms": summarize_numeric(server_first_token_values),
        "server_latency_ms": summarize_numeric(server_latency_values),
        "server_stage_timings_ms": summarize_stage_timings(runs),
        "token_event_count_avg": round(mean(item.token_event_count for item in ok_runs), 2) if ok_runs else None,
        "answer_chars_avg": round(mean(item.answer_chars for item in ok_runs), 2) if ok_runs else None,
        "errors": errors,
        "runs": [asdict(item) for item in runs],
    }


def build_report(
    *,
    label: str,
    dataset_path: Path,
    base_url: str,
    repeats: int,
    warmup_runs: int,
    top_k_retrieve: int,
    top_k_rerank: int,
    enable_auto_route: bool,
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """生成整批 benchmark 报告。"""
    all_ok_runs = [
        run
        for summary in summaries
        for run in summary["runs"]
        if run["ok"]
    ]
    first_token_values = [float(run["first_token_ms"]) for run in all_ok_runs if run.get("first_token_ms") is not None]
    total_values = [float(run["total_ms"]) for run in all_ok_runs if run.get("total_ms") is not None]
    server_first_token_values = [
        float(run["server_first_token_ms"])
        for run in all_ok_runs
        if run.get("server_first_token_ms") is not None
    ]
    server_latency_values = [
        float(run["server_latency_ms"])
        for run in all_ok_runs
        if run.get("server_latency_ms") is not None
    ]
    stage_metric_names = sorted(
        {
            metric_name
            for run in all_ok_runs
            for metric_name in dict(run.get("server_stage_timings_ms") or {})
        }
    )

    return {
        "label": label,
        "dataset": str(dataset_path),
        "base_url": base_url,
        "generated_at": datetime.now(UTC).isoformat(),
        "repeats": repeats,
        "warmup_runs": warmup_runs,
        "top_k_retrieve": top_k_retrieve,
        "top_k_rerank": top_k_rerank,
        "enable_auto_route": enable_auto_route,
        "summary": {
            "case_count": len(summaries),
            "run_count": sum(int(summary["run_count"]) for summary in summaries),
            "ok_count": len(all_ok_runs),
            "error_count": sum(int(summary["error_count"]) for summary in summaries),
            "first_token_ms": summarize_numeric(first_token_values),
            "total_ms": summarize_numeric(total_values),
            "server_first_token_ms": summarize_numeric(server_first_token_values),
            "server_latency_ms": summarize_numeric(server_latency_values),
            "server_stage_timings_ms": {
                metric_name: summarize_numeric(
                    [
                        float(value)
                        for run in all_ok_runs
                        for value in [dict(run.get("server_stage_timings_ms") or {}).get(metric_name)]
                        if value is not None
                    ]
                )
                for metric_name in stage_metric_names
            },
        },
        "results": summaries,
    }
