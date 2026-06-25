"""比较 legacy pipeline 与 shadow graph 的输出。

脚本跑同一批样本并生成逐样本差异，辅助迁移 LangGraph 链路时定位漂移。
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
from app.services.rag_service import RAGService
from app.services import rag_service as rag_service_module


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Compare legacy pipeline and shadow graph without starting the API.")
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
        "--label",
        default="shadow-compare",
        help="Output file label. Report will be written to backend/evals/results/<label>.json.",
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
        "--answer-style",
        choices=("concise", "structured"),
        default="concise",
        help="Answer style used for both legacy and shadow graph runs.",
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
        "--shadow-retrieval-backend",
        choices=("legacy", "langchain_chroma", "langchain_milvus"),
        default=None,
        help="Shadow graph retrieval backend. Defaults to current Settings value.",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    if args.shadow_retrieval_backend:
        os.environ["SHADOW_RETRIEVAL_BACKEND"] = args.shadow_retrieval_backend
        get_settings.cache_clear()
        rag_service_module._settings.cache_clear()
        rag_service_module.get_default_pipeline_registry().clear()

    settings = get_settings()
    cases = load_eval_cases(args.dataset)
    selected_cases = select_eval_cases(cases, case_ids=list(args.case_id), limit=args.limit)
    service = RAGService()

    results = await run_shadow_compare_eval(
        service,
        selected_cases,
        knowledge_base_id=args.knowledge_base_id,
        top_k_retrieve=args.top_k_retrieve,
        top_k_rerank=args.top_k_rerank,
        enable_rewrite=not args.disable_rewrite,
        enable_rerank=not args.disable_rerank,
        answer_style=args.answer_style,
    )
    output_payload = build_output_payload(
        dataset=args.dataset,
        knowledge_base_id=args.knowledge_base_id,
        top_k_retrieve=args.top_k_retrieve,
        top_k_rerank=args.top_k_rerank,
        enable_rewrite=not args.disable_rewrite,
        enable_rerank=not args.disable_rerank,
        answer_style=args.answer_style,
        shadow_retrieval_backend=settings.shadow_retrieval_backend,
        selected_case_ids=[case.id for case in selected_cases],
        results=results,
    )

    output_dir = ROOT / "evals" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.label}.json"
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved shadow compare report to {output_path}")
    print(json.dumps(output_payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
