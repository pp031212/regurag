"""Legacy pipeline 与 shadow graph 的逐样本对比评测。

同一批问题分别跑 legacy pipeline 和 shadow graph，比较最终答案、引用、上下文和检索漂移。
这份结果是迁移 LangGraph/shadow graph 时的主要回归依据。
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import perf_counter

from app.schemas.chat import ChatQueryRequest
from app.services.rag_service import RAGService

RETRIEVAL_DRIFT_FIELDS = ("citation_count", "citation_ids", "final_context_ids", "rewritten_query")


@dataclass(slots=True)
class EvalCase:
    """评测样本定义。"""

    id: str
    question: str
    answer_mode: str
    expected_answer_keywords: list[str]
    expected_context_keywords: list[str]
    category: str
    expected_answer_keyword_groups: list[list[str]] = field(default_factory=list)
    notes: str = ""


@dataclass(slots=True)
class ChainSnapshot:
    """单条链路运行后的答案、引用、上下文和延迟快照。"""

    answer: str
    normalized_answer: str
    answer_source: str
    answer_hit: bool
    answer_hit_ratio: float
    matched_answer_keywords: list[str]
    missing_answer_keywords: list[str]
    citation_count: int
    citation_chunk_ids: list[str]
    final_context_count: int
    final_context_chunk_ids: list[str]
    rewritten_query: str
    latency_ms: int
    error: str | None = None
    citations: list[dict[str, object]] = field(default_factory=list)
    debug: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ShadowCompareCaseResult:
    """单个样本的 legacy 与 graph 对比结果。"""

    id: str
    question: str
    category: str
    answer_mode: str
    notes: str
    final_answer_match: bool
    final_answer_normalized_match: bool
    final_answer_keyword_set_match: bool
    final_answer_hit_parity: bool
    final_citation_ids_match: bool
    final_context_ids_match: bool
    retrieval_drift: bool
    retrieval_drift_fields: list[str]
    core_compare_status: str
    core_mismatch_fields: list[str]
    core_compare: dict[str, object] | None
    legacy: ChainSnapshot
    graph: ChainSnapshot


def load_eval_cases(path: Path) -> list[EvalCase]:
    """从 JSONL 读取评测样本，并兼容旧版关键词字段。"""
    records: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            try:
                expected_answer_keywords = [str(item) for item in payload["expected_answer_keywords"]]
                raw_answer_groups = payload.get("expected_answer_keyword_groups")
                if raw_answer_groups is None:
                    expected_answer_keyword_groups = [[keyword] for keyword in expected_answer_keywords]
                else:
                    expected_answer_keyword_groups = [
                        [str(item) for item in group]
                        for group in raw_answer_groups
                        if isinstance(group, list) and group
                    ]
                    if not expected_answer_keyword_groups:
                        expected_answer_keyword_groups = [[keyword] for keyword in expected_answer_keywords]
                records.append(
                    EvalCase(
                        id=str(payload["id"]),
                        question=str(payload["question"]),
                        answer_mode=str(payload.get("answer_mode", "grounded")),
                        expected_answer_keywords=expected_answer_keywords,
                        expected_context_keywords=[str(item) for item in payload["expected_context_keywords"]],
                        category=str(payload.get("category", "unknown")),
                        expected_answer_keyword_groups=expected_answer_keyword_groups,
                        notes=str(payload.get("notes", "")),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"Invalid dataset line {line_number}: missing key {exc.args[0]}") from exc
    if not records:
        raise ValueError(f"No evaluation cases found in {path}")
    return records


def select_eval_cases(cases: list[EvalCase], *, case_ids: list[str], limit: int | None) -> list[EvalCase]:
    """按 case id 和 limit 选择本次要跑的样本。"""
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
        raise ValueError("No evaluation cases selected")
    return selected


def normalize_text(text: str) -> str:
    """移除空白和常见标点，用于宽松关键词匹配。"""
    lowered = text.lower()
    collapsed = re.sub(r"\s+", "", lowered)
    return re.sub(r"[!！?？,，.。~～、:：;；'\"“”‘’（）()\-\[\]{}]+", "", collapsed)


def normalize_answer_for_compare(text: str) -> str:
    """去掉回答模板标记，比较更接近真实答案内容。"""
    cleaned = text
    cleaned = re.sub(r"【直接依据】|【推理/判断】|【结论】", "\n", cleaned)
    cleaned = re.sub(r"(?im)^(结论|依据|补充说明)\s*[:：]\s*", "", cleaned)
    cleaned = re.sub(r"《[^》]+》", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*(相关条款|最终结论|直接适用或参照理解|必要时简要说明判断过程)\s*$", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*[-*•]\s*", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*\d+\.\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return normalize_text(cleaned)


def extract_retrieval_drift_fields(mismatch_fields: list[str]) -> list[str]:
    """只保留能代表检索漂移的字段。"""
    return [field for field in mismatch_fields if field in RETRIEVAL_DRIFT_FIELDS]


def keyword_matches(text: str, keywords: list[str]) -> tuple[list[str], list[str]]:
    """返回命中和未命中的关键词。"""
    normalized_text = normalize_text(text)
    matched: list[str] = []
    missing: list[str] = []
    for keyword in keywords:
        if normalize_text(keyword) in normalized_text:
            matched.append(keyword)
        else:
            missing.append(keyword)
    return matched, missing


def keyword_group_matches(text: str, keyword_groups: list[list[str]]) -> tuple[list[str], list[str]]:
    """同一组关键词命中任意一个就算通过，适配等价表述。"""
    normalized_text = normalize_text(text)
    matched: list[str] = []
    missing: list[str] = []
    for group in keyword_groups:
        matched_keyword = next((keyword for keyword in group if normalize_text(keyword) in normalized_text), None)
        if matched_keyword is not None:
            matched.append(matched_keyword)
        elif group:
            missing.append(group[0])
    return matched, missing


def evaluate_answer_hit(case: EvalCase, answer_text: str) -> tuple[bool, float, list[str], list[str]]:
    """判断回答是否覆盖样本要求的答案关键词组。"""
    matched_answer_keywords, missing_answer_keywords = keyword_group_matches(
        answer_text,
        case.expected_answer_keyword_groups,
    )
    if case.answer_mode == "no_answer":
        answer_hit = len(missing_answer_keywords) == 0
        answer_hit_ratio = 1.0 if answer_hit else 0.0
        return answer_hit, answer_hit_ratio, matched_answer_keywords, missing_answer_keywords

    answer_hit_ratio = (
        len(matched_answer_keywords) / len(case.expected_answer_keyword_groups)
        if case.expected_answer_keyword_groups
        else 0.0
    )
    answer_hit = len(missing_answer_keywords) == 0
    return answer_hit, answer_hit_ratio, matched_answer_keywords, missing_answer_keywords


def _extract_chunk_ids(items: list[dict[str, object]]) -> list[str]:
    """从引用或上下文块中提取 chunk_id 序列。"""
    return [str(item.get("chunk_id") or "") for item in items if isinstance(item, dict)]


def build_chain_snapshot(case: EvalCase, result: dict[str, object], *, fallback_source: str) -> ChainSnapshot:
    """把一次 RAGService.query 结果压缩成可比较快照。"""
    debug_payload = dict(result.get("debug") or {})
    citations = list(result.get("citations") or [])
    final_context_chunks = list(debug_payload.get("final_context_chunks") or [])
    answer_text = str(result.get("answer") or "")
    answer_hit, answer_hit_ratio, matched_answer_keywords, missing_answer_keywords = evaluate_answer_hit(case, answer_text)
    return ChainSnapshot(
        answer=answer_text,
        normalized_answer=normalize_answer_for_compare(answer_text),
        answer_source=str(result.get("answer_source") or fallback_source),
        answer_hit=answer_hit,
        answer_hit_ratio=round(answer_hit_ratio, 4),
        matched_answer_keywords=matched_answer_keywords,
        missing_answer_keywords=missing_answer_keywords,
        citation_count=len(citations),
        citation_chunk_ids=_extract_chunk_ids(citations),
        final_context_count=len(final_context_chunks),
        final_context_chunk_ids=_extract_chunk_ids(final_context_chunks),
        rewritten_query=str(debug_payload.get("rewritten_query") or ""),
        latency_ms=int(debug_payload.get("latency_ms") or 0),
        citations=citations,
        debug=debug_payload,
    )


def build_error_snapshot(message: str, *, fallback_source: str) -> ChainSnapshot:
    """链路异常时也生成快照，避免整批评测中断。"""
    return ChainSnapshot(
        answer="",
        normalized_answer="",
        answer_source=fallback_source,
        answer_hit=False,
        answer_hit_ratio=0.0,
        matched_answer_keywords=[],
        missing_answer_keywords=[],
        citation_count=0,
        citation_chunk_ids=[],
        final_context_count=0,
        final_context_chunk_ids=[],
        rewritten_query="",
        latency_ms=0,
        error=message,
    )


async def _run_chain(
    service: RAGService,
    case: EvalCase,
    *,
    knowledge_base_id: str,
    top_k_retrieve: int,
    top_k_rerank: int,
    enable_rewrite: bool,
    enable_rerank: bool,
    answer_style: str,
    force_graph_response: bool,
) -> tuple[ChainSnapshot, dict[str, object] | None]:
    """运行一次 legacy 或 graph 链路，并返回快照和 core shadow 对比信息。"""
    started_at = perf_counter()
    payload = ChatQueryRequest(
        knowledge_base_id=knowledge_base_id,
        query=case.question,
        top_k_retrieve=top_k_retrieve,
        top_k_rerank=top_k_rerank,
        enable_rewrite=enable_rewrite,
        enable_rerank=enable_rerank,
        debug=True,
        debug_chunks=True,
        debug_shadow_compare=not force_graph_response,
        debug_force_graph_response=force_graph_response,
        debug_answer_style=answer_style,
    )
    fallback_source = "shadow_graph" if force_graph_response else "legacy_pipeline"
    try:
        result = await service.query(payload)
    except Exception as exc:
        snapshot = build_error_snapshot(str(exc), fallback_source=fallback_source)
        snapshot.latency_ms = int((perf_counter() - started_at) * 1000)
        return snapshot, None

    snapshot = build_chain_snapshot(case, result, fallback_source=fallback_source)
    if snapshot.latency_ms <= 0:
        snapshot.latency_ms = int((perf_counter() - started_at) * 1000)
    debug_payload = dict(result.get("debug") or {})
    return snapshot, dict(debug_payload.get("shadow_compare") or {}) or None


async def run_case(
    service: RAGService,
    case: EvalCase,
    *,
    knowledge_base_id: str,
    top_k_retrieve: int,
    top_k_rerank: int,
    enable_rewrite: bool,
    enable_rerank: bool,
    answer_style: str,
) -> ShadowCompareCaseResult:
    """同一样本先跑 legacy，再强制跑 graph，最后比较两条链路输出。"""
    legacy_snapshot, core_compare = await _run_chain(
        service,
        case,
        knowledge_base_id=knowledge_base_id,
        top_k_retrieve=top_k_retrieve,
        top_k_rerank=top_k_rerank,
        enable_rewrite=enable_rewrite,
        enable_rerank=enable_rerank,
        answer_style=answer_style,
        force_graph_response=False,
    )
    graph_snapshot, _ = await _run_chain(
        service,
        case,
        knowledge_base_id=knowledge_base_id,
        top_k_retrieve=top_k_retrieve,
        top_k_rerank=top_k_rerank,
        enable_rewrite=enable_rewrite,
        enable_rerank=enable_rerank,
        answer_style=answer_style,
        force_graph_response=True,
    )
    core_mismatch_fields = [str(item) for item in list((core_compare or {}).get("mismatch_fields") or [])]
    retrieval_drift_fields = extract_retrieval_drift_fields(core_mismatch_fields)

    return ShadowCompareCaseResult(
        id=case.id,
        question=case.question,
        category=case.category,
        answer_mode=case.answer_mode,
        notes=case.notes,
        final_answer_match=legacy_snapshot.answer == graph_snapshot.answer,
        final_answer_normalized_match=legacy_snapshot.normalized_answer == graph_snapshot.normalized_answer,
        final_answer_keyword_set_match=set(legacy_snapshot.matched_answer_keywords) == set(graph_snapshot.matched_answer_keywords),
        final_answer_hit_parity=legacy_snapshot.answer_hit == graph_snapshot.answer_hit,
        final_citation_ids_match=legacy_snapshot.citation_chunk_ids == graph_snapshot.citation_chunk_ids,
        final_context_ids_match=legacy_snapshot.final_context_chunk_ids == graph_snapshot.final_context_chunk_ids,
        retrieval_drift=bool(retrieval_drift_fields),
        retrieval_drift_fields=retrieval_drift_fields,
        core_compare_status=str((core_compare or {}).get("status") or "missing"),
        core_mismatch_fields=core_mismatch_fields,
        core_compare=core_compare,
        legacy=legacy_snapshot,
        graph=graph_snapshot,
    )


async def run_shadow_compare_eval(
    service: RAGService,
    cases: list[EvalCase],
    *,
    knowledge_base_id: str,
    top_k_retrieve: int,
    top_k_rerank: int,
    enable_rewrite: bool,
    enable_rerank: bool,
    answer_style: str,
) -> list[ShadowCompareCaseResult]:
    """顺序运行样本，避免并发请求影响延迟和外部模型稳定性。"""
    results: list[ShadowCompareCaseResult] = []
    for case in cases:
        results.append(
            await run_case(
                service,
                case,
                knowledge_base_id=knowledge_base_id,
                top_k_retrieve=top_k_retrieve,
                top_k_rerank=top_k_rerank,
                enable_rewrite=enable_rewrite,
                enable_rerank=enable_rerank,
                answer_style=answer_style,
            )
        )
    return results


def build_summary(results: list[ShadowCompareCaseResult]) -> dict[str, object]:
    """汇总整批样本的命中率、一致性和漂移情况。"""
    if not results:
        raise ValueError("No results to summarize")

    mismatch_breakdown: dict[str, int] = {}
    core_status_breakdown: dict[str, int] = {}
    retrieval_drift_breakdown: dict[str, int] = {}
    retrieval_drift_case_ids: list[str] = []
    for item in results:
        core_status_breakdown[item.core_compare_status] = core_status_breakdown.get(item.core_compare_status, 0) + 1
        for field in item.core_mismatch_fields:
            mismatch_breakdown[field] = mismatch_breakdown.get(field, 0) + 1
        if item.retrieval_drift:
            retrieval_drift_case_ids.append(item.id)
            for field in item.retrieval_drift_fields:
                retrieval_drift_breakdown[field] = retrieval_drift_breakdown.get(field, 0) + 1

    return {
        "case_count": len(results),
        "legacy_answer_hit_rate": round(mean(item.legacy.answer_hit for item in results), 4),
        "graph_answer_hit_rate": round(mean(item.graph.answer_hit for item in results), 4),
        "final_answer_match_rate": round(mean(item.final_answer_match for item in results), 4),
        "final_answer_normalized_match_rate": round(mean(item.final_answer_normalized_match for item in results), 4),
        "final_answer_keyword_set_match_rate": round(mean(item.final_answer_keyword_set_match for item in results), 4),
        "final_answer_hit_parity_rate": round(mean(item.final_answer_hit_parity for item in results), 4),
        "final_citation_ids_match_rate": round(mean(item.final_citation_ids_match for item in results), 4),
        "final_context_ids_match_rate": round(mean(item.final_context_ids_match for item in results), 4),
        "retrieval_drift_case_count": sum(item.retrieval_drift for item in results),
        "retrieval_drift_rate": round(mean(item.retrieval_drift for item in results), 4),
        "core_match_rate": round(mean(item.core_compare_status == "match" for item in results), 4),
        "core_error_rate": round(mean(item.core_compare_status == "error" for item in results), 4),
        "legacy_error_count": sum(item.legacy.error is not None for item in results),
        "graph_error_count": sum(item.graph.error is not None for item in results),
        "avg_legacy_latency_ms": round(mean(item.legacy.latency_ms for item in results), 2),
        "avg_graph_latency_ms": round(mean(item.graph.latency_ms for item in results), 2),
        "core_status_breakdown": core_status_breakdown,
        "mismatch_breakdown": mismatch_breakdown,
        "retrieval_drift_breakdown": retrieval_drift_breakdown,
        "retrieval_drift_case_ids": retrieval_drift_case_ids,
    }


def build_retrieval_drift_cases(results: list[ShadowCompareCaseResult]) -> list[dict[str, object]]:
    """提取发生检索漂移的样本，便于单独排查。"""
    drift_cases: list[dict[str, object]] = []
    for item in results:
        if not item.retrieval_drift:
            continue
        drift_cases.append(
            {
                "id": item.id,
                "question": item.question,
                "category": item.category,
                "retrieval_drift_fields": item.retrieval_drift_fields,
                "core_mismatch_fields": item.core_mismatch_fields,
                "legacy_citation_chunk_ids": item.legacy.citation_chunk_ids,
                "graph_citation_chunk_ids": item.graph.citation_chunk_ids,
                "legacy_final_context_chunk_ids": item.legacy.final_context_chunk_ids,
                "graph_final_context_chunk_ids": item.graph.final_context_chunk_ids,
                "legacy_rewritten_query": item.legacy.rewritten_query,
                "graph_rewritten_query": item.graph.rewritten_query,
            }
        )
    return drift_cases


def build_output_payload(
    *,
    dataset: Path,
    knowledge_base_id: str,
    top_k_retrieve: int,
    top_k_rerank: int,
    enable_rewrite: bool,
    enable_rerank: bool,
    answer_style: str,
    shadow_retrieval_backend: str,
    selected_case_ids: list[str],
    results: list[ShadowCompareCaseResult],
) -> dict[str, object]:
    """组装最终 JSON 报告 payload。"""
    return {
        "run_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "dataset": str(dataset),
        "knowledge_base_id": knowledge_base_id,
        "config": {
            "top_k_retrieve": top_k_retrieve,
            "top_k_rerank": top_k_rerank,
            "enable_rewrite": enable_rewrite,
            "enable_rerank": enable_rerank,
            "answer_style": answer_style,
            "shadow_retrieval_backend": shadow_retrieval_backend,
            "selected_case_ids": selected_case_ids,
        },
        "summary": build_summary(results),
        "retrieval_drift_cases": build_retrieval_drift_cases(results),
        "results": [asdict(item) for item in results],
    }
