"""运行 shadow retrieval backend 回归对比。

脚本会对每个指定 backend 跑同一批样本，分别写出 backend 报告和一份 compact summary。
主要用于 Chroma/Milvus shadow 检索灰度切换前的回归检查。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.evals.shadow_compare import (
    build_output_payload,
    load_eval_cases,
    run_shadow_compare_eval,
    select_eval_cases,
)
from app.evals.shadow_regression import build_shadow_regression_summary
from app.services import rag_service as rag_service_module
from app.services.rag_service import RAGService


def parse_args() -> argparse.Namespace:
    """解析命令行参数，默认同时比较 Chroma 和 Milvus。"""
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Run fixed shadow retrieval regression across multiple backends and save a compact summary."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "evals" / "rag_eval_set.jsonl",
        help="Path to the JSONL evaluation dataset.",
    )
    parser.add_argument(
        "--knowledge-base-id",
        default=settings.default_knowledge_base_id,
        help="Knowledge base id used for comparison.",
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
        "--answer-style",
        choices=("concise", "structured"),
        default="concise",
        help="Answer style used for every backend run.",
    )
    parser.add_argument(
        "--disable-rewrite",
        action="store_true",
        help="Disable query rewrite for all backend runs.",
    )
    parser.add_argument(
        "--disable-rerank",
        action="store_true",
        help="Disable MMR and reranker for all backend runs.",
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
    parser.add_argument(
        "--backend",
        action="append",
        choices=("langchain_chroma", "langchain_milvus"),
        default=[],
        help="Shadow retrieval backend to include. Defaults to langchain_chroma + langchain_milvus.",
    )
    parser.add_argument(
        "--baseline-backend",
        choices=("langchain_chroma", "langchain_milvus"),
        default="langchain_chroma",
        help="Backend used as the regression baseline in the compact summary.",
    )
    parser.add_argument(
        "--label-prefix",
        default="shadow-retrieval-regression",
        help="Output file label prefix. Backend reports and summary will be written under backend/evals/results/.",
    )
    return parser.parse_args()


async def _run_backend_report(
    *,
    backend: str,
    dataset: Path,
    selected_cases,
    knowledge_base_id: str,
    top_k_retrieve: int,
    top_k_rerank: int,
    enable_rewrite: bool,
    enable_rerank: bool,
    answer_style: str,
) -> dict[str, object]:
    """在指定 shadow backend 下运行一次完整评测。"""
    os.environ["SHADOW_RETRIEVAL_BACKEND"] = backend
    # 切换 backend 后必须清理缓存，否则 settings 和 pipeline registry 仍会沿用上一次配置。
    get_settings.cache_clear()
    rag_service_module._settings.cache_clear()
    rag_service_module.get_default_pipeline_registry().clear()

    settings = get_settings()
    service = RAGService()
    results = await run_shadow_compare_eval(
        service,
        selected_cases,
        knowledge_base_id=knowledge_base_id,
        top_k_retrieve=top_k_retrieve,
        top_k_rerank=top_k_rerank,
        enable_rewrite=enable_rewrite,
        enable_rerank=enable_rerank,
        answer_style=answer_style,
    )
    return build_output_payload(
        dataset=dataset,
        knowledge_base_id=knowledge_base_id,
        top_k_retrieve=top_k_retrieve,
        top_k_rerank=top_k_rerank,
        enable_rewrite=enable_rewrite,
        enable_rerank=enable_rerank,
        answer_style=answer_style,
        shadow_retrieval_backend=settings.shadow_retrieval_backend,
        selected_case_ids=[case.id for case in selected_cases],
        results=results,
    )


async def main() -> int:
    """生成每个 backend 的报告，并写出最终回归摘要。"""
    args = parse_args()
    backends = list(dict.fromkeys(args.backend or ["langchain_chroma", "langchain_milvus"]))
    if args.baseline_backend not in backends:
        raise ValueError("--baseline-backend must be included in --backend")

    cases = load_eval_cases(args.dataset)
    selected_cases = select_eval_cases(cases, case_ids=list(args.case_id), limit=args.limit)

    output_dir = ROOT / "evals" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    reports: dict[str, dict[str, object]] = {}
    report_paths: dict[str, Path] = {}
    for backend in backends:
        # 每个 backend 独立跑完整样本，避免同一服务实例残留状态影响对比。
        report = await _run_backend_report(
            backend=backend,
            dataset=args.dataset,
            selected_cases=selected_cases,
            knowledge_base_id=args.knowledge_base_id,
            top_k_retrieve=args.top_k_retrieve,
            top_k_rerank=args.top_k_rerank,
            enable_rewrite=not args.disable_rewrite,
            enable_rerank=not args.disable_rerank,
            answer_style=args.answer_style,
        )
        report_path = output_dir / f"{args.label_prefix}-{backend}.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        reports[backend] = report
        report_paths[backend] = report_path

    summary = build_shadow_regression_summary(
        reports=reports,
        baseline_backend=args.baseline_backend,
        report_paths=report_paths,
    )
    summary_payload = {
        "dataset": str(args.dataset),
        "knowledge_base_id": args.knowledge_base_id,
        "selected_case_ids": [case.id for case in selected_cases],
        "config": {
            "backends": backends,
            "baseline_backend": args.baseline_backend,
            "top_k_retrieve": args.top_k_retrieve,
            "top_k_rerank": args.top_k_rerank,
            "enable_rewrite": not args.disable_rewrite,
            "enable_rerank": not args.disable_rerank,
            "answer_style": args.answer_style,
        },
        "summary": summary,
    }
    summary_path = output_dir / f"{args.label_prefix}-summary.json"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved shadow regression summary to {summary_path}")
    print(json.dumps(summary["backend_summaries"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["diffs_vs_baseline"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
