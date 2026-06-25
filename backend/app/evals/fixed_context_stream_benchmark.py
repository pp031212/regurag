"""固定上下文流式生成 benchmark。

先固定检索和上下文构造结果，再只重复测 LLM 流式生成，用来区分检索耗时和生成耗时。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from ..rag.pipeline import RAGPipeline
from ..workflows.rag.pipeline_steps import extract_query_keywords
from .stream_benchmark import StreamBenchmarkCase, summarize_numeric


RewriteMode = Literal["baseline", "skip_overview"]


@dataclass(slots=True)
class FixedContextStreamRun:
    """固定上下文下单次流式生成记录。"""

    ok: bool
    run_index: int
    first_token_ms: float | None
    total_ms: float | None
    token_event_count: int
    answer_chars: int
    finish_reason: str | None = None
    error: str | None = None


def _stable_hash(value: object) -> str:
    """生成稳定哈希，判断两次 benchmark 是否使用同一份上下文。"""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_query_prep(
    pipeline: RAGPipeline,
    *,
    query: str,
    rewrite_mode: RewriteMode,
    enable_rewrite: bool = True,
    history_rewritten_query: str | None = None,
) -> dict[str, object]:
    """构造查询预处理结果，可选择跳过 overview rewrite 作为对照。"""
    if rewrite_mode == "baseline":
        return pipeline.prepare_query_inputs(
            query=query,
            enable_rewrite=enable_rewrite,
            history_rewritten_query=history_rewritten_query,
        )

    effective_query = history_rewritten_query or query
    rewrite_started = perf_counter()
    expanded_keywords = ""
    if enable_rewrite and not pipeline._is_overview_query(effective_query):
        try:
            expanded_keywords = str(pipeline.rewriter.rewrite(effective_query))
        except Exception:
            expanded_keywords = ""
    rewrite_ms = int((perf_counter() - rewrite_started) * 1000)

    alias_keywords = pipeline.query_alias_expander.expand(effective_query)
    overview_keywords = pipeline._expand_overview_query_keywords(effective_query)
    query_keywords = extract_query_keywords(alias_keywords, overview_keywords)
    if not query_keywords:
        query_keywords = extract_query_keywords(expanded_keywords, overview_keywords)

    return {
        "effective_query": effective_query,
        "expanded_keywords": expanded_keywords,
        "alias_keywords": alias_keywords,
        "overview_keywords": overview_keywords,
        "query_keywords": query_keywords,
        "search_query": f"{effective_query} {expanded_keywords} {alias_keywords} {overview_keywords}".strip(),
        "rewrite_ms": rewrite_ms,
    }


def build_fixed_context_inputs(
    pipeline: RAGPipeline,
    *,
    query: str,
    knowledge_base_id: str,
    source_name_by_document_id: dict[str, str] | None,
    top_k_retrieve: int,
    top_k_rerank: int,
    answer_style: str,
    rewrite_mode: RewriteMode,
) -> dict[str, object]:
    """准备一次固定上下文 benchmark 所需的检索、重排和上下文输入。"""
    query_prep = build_query_prep(
        pipeline,
        query=query,
        rewrite_mode=rewrite_mode,
        enable_rewrite=True,
        history_rewritten_query=None,
    )
    generation_inputs = pipeline._prepare_generation_inputs(
        query=query,
        top_k_retrieve=top_k_retrieve,
        top_k_rerank=top_k_rerank,
        knowledge_base_id=knowledge_base_id,
        source_name_by_document_id=source_name_by_document_id,
        enable_rewrite=True,
        enable_rerank=True,
        answer_style=answer_style,
        query_prep=query_prep,
        history_rewritten_query=None,
        history_message_count=0,
        history_rewrite_ms=0,
    )
    return {
        "query_prep": query_prep,
        "generation_inputs": generation_inputs,
    }


def run_fixed_context_stream_sample(
    pipeline: RAGPipeline,
    *,
    query: str,
    context: str,
    standalone_query: str | None,
    answer_style: str,
    run_index: int,
) -> FixedContextStreamRun:
    """在固定上下文上重复调用 LLM stream_generate。"""
    started_at = perf_counter()
    token_event_count = 0
    answer_chars = 0
    first_token_ms: float | None = None

    try:
        stream = pipeline.llm.stream_generate(
            query=query,
            context=context,
            standalone_query=standalone_query,
            answer_style=answer_style,
        )
        while True:
            try:
                delta = next(stream)
            except StopIteration as stop:
                llm_result = dict(stop.value)
                break
            if not delta:
                continue
            token_event_count += 1
            answer_chars += len(delta)
            if first_token_ms is None:
                first_token_ms = round((perf_counter() - started_at) * 1000, 2)
        return FixedContextStreamRun(
            ok=True,
            run_index=run_index,
            first_token_ms=first_token_ms,
            total_ms=round((perf_counter() - started_at) * 1000, 2),
            token_event_count=token_event_count,
            answer_chars=len(str(llm_result.get("answer") or "")),
            finish_reason=str(llm_result.get("finish_reason") or "") or None,
        )
    except Exception as exc:
        return FixedContextStreamRun(
            ok=False,
            run_index=run_index,
            first_token_ms=first_token_ms,
            total_ms=round((perf_counter() - started_at) * 1000, 2),
            token_event_count=token_event_count,
            answer_chars=answer_chars,
            error=str(exc),
        )


def build_fixed_context_report(
    *,
    label: str,
    dataset_path: Path,
    case: StreamBenchmarkCase,
    rewrite_mode: RewriteMode,
    top_k_retrieve: int,
    top_k_rerank: int,
    warmup_runs: int,
    repeats: int,
    fixed_context_inputs: dict[str, object],
    runs: list[FixedContextStreamRun],
) -> dict[str, object]:
    """汇总固定上下文 benchmark 报告。"""
    generation_inputs = dict(fixed_context_inputs["generation_inputs"])
    query_prep = dict(fixed_context_inputs["query_prep"])
    ok_runs = [run for run in runs if run.ok]
    first_token_values = [run.first_token_ms for run in ok_runs if run.first_token_ms is not None]
    total_values = [run.total_ms for run in ok_runs if run.total_ms is not None]
    errors = [run.error for run in runs if run.error]
    final_context_chunks = list(generation_inputs.get("final_context_debug_chunks") or [])
    context = str(generation_inputs.get("context") or "")

    return {
        "label": label,
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": str(dataset_path),
        "case": {
            "id": case.id,
            "question": case.question,
            "category": case.category,
            "knowledge_base_id": case.knowledge_base_id,
            "notes": case.notes,
        },
        "rewrite_mode": rewrite_mode,
        "warmup_runs": warmup_runs,
        "repeats": repeats,
        "top_k_retrieve": top_k_retrieve,
        "top_k_rerank": top_k_rerank,
        "query_prep": {
            "effective_query": query_prep.get("effective_query"),
            "search_query": query_prep.get("search_query"),
            "rewrite_ms": query_prep.get("rewrite_ms"),
            "query_keywords": list(query_prep.get("query_keywords") or []),
        },
        "fixed_context": {
            "context_hash": _stable_hash(context),
            "context_chars": len(context),
            "final_context_count": len(final_context_chunks),
            "final_context_chunk_ids": [str(item.get("chunk_id") or "") for item in final_context_chunks],
            "final_context_previews": [str(item.get("parent_text") or "")[:180] for item in final_context_chunks],
            "prep_stage_timings_ms": {
                "rewrite_ms": generation_inputs.get("rewrite_ms"),
                "retrieve_ms": generation_inputs.get("retrieve_ms"),
                "rerank_ms": generation_inputs.get("rerank_ms"),
                "context_build_ms": generation_inputs.get("context_build_ms"),
            },
        },
        "summary": {
            "run_count": len(runs),
            "ok_count": len(ok_runs),
            "error_count": len(runs) - len(ok_runs),
            "first_token_ms": summarize_numeric([float(value) for value in first_token_values]),
            "total_ms": summarize_numeric([float(value) for value in total_values]),
            "token_event_count_avg": round(
                sum(run.token_event_count for run in ok_runs) / len(ok_runs),
                2,
            )
            if ok_runs
            else None,
            "answer_chars_avg": round(
                sum(run.answer_chars for run in ok_runs) / len(ok_runs),
                2,
            )
            if ok_runs
            else None,
            "errors": errors,
        },
        "runs": [asdict(run) for run in runs],
    }
