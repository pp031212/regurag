"""运行 RAG 问答评测集。

对指定知识库执行 JSONL 样本，统计检索命中、最终上下文命中、引用命中和答案关键词命中。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.schemas.chat import ChatQueryRequest
from app.services.rag_service import RAGService


@dataclass(slots=True)
class EvalCase:
    id: str
    question: str
    answer_mode: str
    expected_answer_keywords: list[str]
    expected_context_keywords: list[str]
    category: str
    expected_answer_keyword_groups: list[list[str]] = field(default_factory=list)
    expected_context_keyword_groups: list[list[str]] = field(default_factory=list)
    notes: str = ""


@dataclass(slots=True)
class EvalCaseResult:
    id: str
    question: str
    category: str
    answer_mode: str
    answer: str
    retrieval_hit: bool
    final_context_hit: bool
    citation_hit: bool
    answer_hit: bool
    answer_hit_ratio: float
    matched_answer_keywords: list[str]
    missing_answer_keywords: list[str]
    matched_context_keywords: list[str]
    error_type: str
    debug: dict[str, object]
    citations: list[dict[str, object]]


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run a minimal RAG evaluation set against ReguRAG.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "evals" / "rag_eval_set.jsonl",
        help="Path to the JSONL evaluation dataset.",
    )
    parser.add_argument(
        "--knowledge-base-id",
        default=settings.default_knowledge_base_id,
        help="Knowledge base id used for evaluation.",
    )
    parser.add_argument(
        "--top-k-retrieve",
        type=int,
        default=15,
        help="Top-k retrieve value passed to the RAG query.",
    )
    parser.add_argument(
        "--top-k-rerank",
        type=int,
        default=3,
        help="Top-k rerank value passed to the RAG query.",
    )
    parser.add_argument(
        "--label",
        default="baseline",
        help="Result file label. Output will be written to backend/evals/results/<label>.json.",
    )
    parser.add_argument(
        "--disable-rewrite",
        action="store_true",
        help="Disable query rewrite for this run.",
    )
    parser.add_argument(
        "--disable-rerank",
        action="store_true",
        help="Disable MMR and reranker for this run.",
    )
    parser.add_argument(
        "--enable-auto-route",
        action="store_true",
        help="Enable knowledge-base auto routing during evaluation. Disabled by default so evals stay on the requested knowledge base.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N cases after case-id filtering.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Run only the specified case id. Can be repeated.",
    )
    return parser.parse_args()


def load_eval_cases(path: Path) -> list[EvalCase]:
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
                expected_context_keywords = [str(item) for item in payload["expected_context_keywords"]]
                raw_context_groups = payload.get("expected_context_keyword_groups")
                if raw_context_groups is None:
                    expected_context_keyword_groups = [[keyword] for keyword in expected_context_keywords]
                else:
                    expected_context_keyword_groups = [
                        [str(item) for item in group]
                        for group in raw_context_groups
                        if isinstance(group, list) and group
                    ]
                    if not expected_context_keyword_groups:
                        expected_context_keyword_groups = [[keyword] for keyword in expected_context_keywords]
                records.append(
                    EvalCase(
                        id=str(payload["id"]),
                        question=str(payload["question"]),
                        answer_mode=str(payload.get("answer_mode", "grounded")),
                        expected_answer_keywords=expected_answer_keywords,
                        expected_context_keywords=expected_context_keywords,
                        category=str(payload.get("category", "unknown")),
                        expected_answer_keyword_groups=expected_answer_keyword_groups,
                        expected_context_keyword_groups=expected_context_keyword_groups,
                        notes=str(payload.get("notes", "")),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"Invalid dataset line {line_number}: missing key {exc.args[0]}") from exc
    if not records:
        raise ValueError(f"No evaluation cases found in {path}")
    return records


def select_eval_cases(cases: list[EvalCase], *, case_ids: list[str], limit: int | None) -> list[EvalCase]:
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
    lowered = text.lower()
    collapsed = re.sub(r"\s+", "", lowered)
    return re.sub(r"[!！?？,，.。~～、:：;；'\"“”‘’（）()\-\[\]{}]+", "", collapsed)


def keyword_matches(text: str, keywords: list[str]) -> tuple[list[str], list[str]]:
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


def join_debug_chunk_texts(debug_payload: dict[str, object], key: str) -> str:
    chunks = debug_payload.get(key, [])
    if not isinstance(chunks, list):
        return ""
    texts: list[str] = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            parent_text = chunk.get("parent_text")
            child_text = chunk.get("child_text")
            if isinstance(parent_text, str):
                texts.append(parent_text)
            if isinstance(child_text, str):
                texts.append(child_text)
    return "\n".join(texts)


def classify_error(
    answer_mode: str,
    retrieval_hit: bool,
    final_context_hit: bool,
    answer_hit: bool,
    citation_hit: bool,
) -> str:
    if answer_mode == "no_answer":
        return "ok" if answer_hit else "fallback_error"
    if retrieval_hit and final_context_hit and answer_hit and citation_hit:
        return "ok"
    if not retrieval_hit:
        return "retrieval_error"
    if not final_context_hit:
        return "rerank_or_filter_error"
    if not answer_hit:
        return "generation_error"
    if not citation_hit:
        return "citation_error"
    return "needs_review"


def _is_retryable_eval_error(exc: Exception) -> bool:
    class_name = exc.__class__.__name__
    return class_name in {"APITimeoutError", "APIConnectionError", "RateLimitError"}


def evaluate_answer_hit(case: EvalCase, answer_text: str) -> tuple[bool, float, list[str], list[str]]:
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


async def run_case(
    service: RAGService,
    case: EvalCase,
    knowledge_base_id: str,
    top_k_retrieve: int,
    top_k_rerank: int,
    enable_rewrite: bool,
    enable_rerank: bool,
    enable_auto_route: bool = False,
    query_retries: int = 2,
) -> EvalCaseResult:
    payload = ChatQueryRequest(
        knowledge_base_id=knowledge_base_id,
        query=case.question,
        top_k_retrieve=top_k_retrieve,
        top_k_rerank=top_k_rerank,
        enable_rewrite=enable_rewrite,
        enable_rerank=enable_rerank,
        enable_auto_route=enable_auto_route,
        debug=True,
        debug_chunks=True,
    )
    last_error: Exception | None = None
    for attempt in range(max(1, query_retries)):
        try:
            result = await service.query(payload)
            break
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= max(1, query_retries) or not _is_retryable_eval_error(exc):
                raise
            await asyncio.sleep(min(2, attempt + 1))
    else:
        raise RuntimeError("eval run_case exhausted retries without returning a result") from last_error
    debug_payload = dict(result["debug"])

    answer_text = str(result["answer"])
    citations = list(result["citations"])
    citation_text = "\n".join(str(item.get("content", "")) for item in citations if isinstance(item, dict))
    retrieved_text = join_debug_chunk_texts(debug_payload, "retrieved_chunks")
    final_context_text = join_debug_chunk_texts(debug_payload, "final_context_chunks")

    answer_hit, answer_hit_ratio, matched_answer_keywords, missing_answer_keywords = evaluate_answer_hit(
        case, answer_text
    )
    matched_retrieval_keywords, missing_retrieval_keywords = keyword_group_matches(
        retrieved_text,
        case.expected_context_keyword_groups,
    )
    matched_final_context_keywords, missing_final_context_keywords = keyword_group_matches(
        final_context_text,
        case.expected_context_keyword_groups,
    )
    matched_citation_keywords, missing_citation_keywords = keyword_group_matches(
        citation_text,
        case.expected_context_keyword_groups,
    )

    retrieval_hit = len(missing_retrieval_keywords) == 0
    final_context_hit = len(missing_final_context_keywords) == 0
    citation_hit = len(missing_citation_keywords) == 0
    if case.answer_mode == "no_answer":
        final_context_chunks = debug_payload.get("final_context_chunks", [])
        retrieval_hit = len(final_context_chunks) == 0
        final_context_hit = len(final_context_chunks) == 0
        citation_hit = len(citations) == 0
    error_type = classify_error(case.answer_mode, retrieval_hit, final_context_hit, answer_hit, citation_hit)

    return EvalCaseResult(
        id=case.id,
        question=case.question,
        category=case.category,
        answer_mode=case.answer_mode,
        answer=answer_text,
        retrieval_hit=retrieval_hit,
        final_context_hit=final_context_hit,
        citation_hit=citation_hit,
        answer_hit=answer_hit,
        answer_hit_ratio=round(answer_hit_ratio, 4),
        matched_answer_keywords=matched_answer_keywords,
        missing_answer_keywords=missing_answer_keywords,
        matched_context_keywords=matched_final_context_keywords,
        error_type=error_type,
        debug=debug_payload,
        citations=citations,
    )


def build_summary(results: list[EvalCaseResult]) -> dict[str, object]:
    total = len(results)
    error_buckets: dict[str, int] = {}
    for item in results:
        error_buckets[item.error_type] = error_buckets.get(item.error_type, 0) + 1

    debug_items = [item.debug for item in results]
    return {
        "case_count": total,
        "grounded_case_count": sum(item.answer_mode == "grounded" for item in results),
        "no_answer_case_count": sum(item.answer_mode == "no_answer" for item in results),
        "retrieval_hit_rate": round(sum(item.retrieval_hit for item in results) / total, 4),
        "final_context_hit_rate": round(sum(item.final_context_hit for item in results) / total, 4),
        "citation_hit_rate": round(sum(item.citation_hit for item in results) / total, 4),
        "answer_hit_rate": round(sum(item.answer_hit for item in results) / total, 4),
        "avg_answer_hit_ratio": round(mean(item.answer_hit_ratio for item in results), 4),
        "avg_rewrite_ms": round(mean(int(item.get("rewrite_ms", 0)) for item in debug_items), 2),
        "avg_retrieve_ms": round(mean(int(item.get("retrieve_ms", 0)) for item in debug_items), 2),
        "avg_rerank_ms": round(mean(int(item.get("rerank_ms", 0)) for item in debug_items), 2),
        "avg_generate_ms": round(mean(int(item.get("generate_ms", 0)) for item in debug_items), 2),
        "avg_latency_ms": round(mean(int(item.get("latency_ms", 0)) for item in debug_items), 2),
        "error_breakdown": error_buckets,
    }


async def main() -> int:
    args = parse_args()
    cases = load_eval_cases(args.dataset)
    selected_cases = select_eval_cases(cases, case_ids=list(args.case_id), limit=args.limit)
    service = RAGService()

    results: list[EvalCaseResult] = []
    for case in selected_cases:
        case_result = await run_case(
            service=service,
            case=case,
            knowledge_base_id=args.knowledge_base_id,
            top_k_retrieve=args.top_k_retrieve,
            top_k_rerank=args.top_k_rerank,
            enable_rewrite=not args.disable_rewrite,
            enable_rerank=not args.disable_rerank,
            enable_auto_route=args.enable_auto_route,
        )
        results.append(case_result)

    summary = build_summary(results)
    output_payload = {
        "run_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "knowledge_base_id": args.knowledge_base_id,
        "dataset": str(args.dataset),
        "config": {
            "top_k_retrieve": args.top_k_retrieve,
            "top_k_rerank": args.top_k_rerank,
            "enable_rewrite": not args.disable_rewrite,
            "enable_rerank": not args.disable_rerank,
            "enable_auto_route": args.enable_auto_route,
            "query_retries": 2,
            "selected_case_ids": [case.id for case in selected_cases],
        },
        "summary": summary,
        "results": [asdict(item) for item in results],
    }

    output_dir = ROOT / "evals" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.label}.json"
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved evaluation report to {output_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))



