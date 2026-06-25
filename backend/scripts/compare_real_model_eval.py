"""真实知识库下比较不同模型组合的问答评测表现。

用于评估 embedding/reranker/LLM 配置切换后，答案命中、引用命中和延迟是否回退。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from fastapi import UploadFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.evals.eval_report_compare import compare_eval_reports
from app.rag.document_split_config import load_document_split_config
from app.rag.query_alias_config import load_query_alias_config
from app.rag.query_rewriter_config import load_query_rewriter_prompt_config
from app.rag.retrieval_rules_config import load_retrieval_rules_config
from app.repositories.metadata_repository import MetadataRepository
from app.services.answer_guard_config import load_answer_guard_config
from app.services.cross_domain_guard_config import load_cross_domain_guard_config
from app.services.faq_shortcut_config import load_faq_shortcut_config
from app.services.ingest_service import IngestService
from app.services.knowledge_base_routing_config import load_knowledge_base_routing_config
from app.services.knowledge_base_service import DocumentService, KnowledgeBaseService
from app.services.light_intent_config import load_light_intent_config
from app.services.rag_service import RAGService, _PIPELINES, _settings as rag_service_settings
from app.services.source_name_config import load_source_name_config
from eval_rag import build_summary, load_eval_cases, run_case


DEFAULT_BASELINE_LABEL = "current"
DEFAULT_CONTENDER_LABEL = "bge_m3_pair"
DEFAULT_BASELINE_EMBEDDING = "BAAI/bge-small-zh-v1.5"
DEFAULT_BASELINE_RERANKER = "BAAI/bge-reranker-base"
DEFAULT_CONTENDER_EMBEDDING = "BAAI/bge-m3"
DEFAULT_CONTENDER_RERANKER = "BAAI/bge-reranker-v2-m3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build real PDF knowledge bases for two embedding/reranker pairs, run eval_rag on both, then compare."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "evals" / "rag_eval_real_kb_samples.jsonl",
        help="Path to the JSONL evaluation dataset.",
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        nargs="*",
        default=[
            ROOT / "evals" / "fixtures" / "heming_rules.pdf",
            ROOT / "evals" / "fixtures" / "heming_rules_table.pdf",
            ROOT / "evals" / "fixtures" / "heming_rules_mixed.pdf",
        ],
        help="Fixture documents to import into each comparison knowledge base.",
    )
    parser.add_argument(
        "--config-profile",
        default="rules_cn",
        help="Config profile used for both model-pair runs.",
    )
    parser.add_argument(
        "--baseline-label",
        default=DEFAULT_BASELINE_LABEL,
        help="Baseline model-pair label used in output filenames and KB names.",
    )
    parser.add_argument(
        "--baseline-embedding",
        default=DEFAULT_BASELINE_EMBEDDING,
        help="Baseline embedding model name.",
    )
    parser.add_argument(
        "--baseline-reranker",
        default=DEFAULT_BASELINE_RERANKER,
        help="Baseline reranker model name.",
    )
    parser.add_argument(
        "--contender-label",
        default=DEFAULT_CONTENDER_LABEL,
        help="Contender model-pair label used in output filenames and KB names.",
    )
    parser.add_argument(
        "--contender-embedding",
        default=DEFAULT_CONTENDER_EMBEDDING,
        help="Contender embedding model name.",
    )
    parser.add_argument(
        "--contender-reranker",
        default=DEFAULT_CONTENDER_RERANKER,
        help="Contender reranker model name.",
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
        default="real-model-pair-ab",
        help="Output file label prefix. Three files will be written under backend/evals/results/.",
    )
    parser.add_argument(
        "--knowledge-base-subject",
        default="和鸣教育管理制度",
        help="Subject used when creating the comparison knowledge bases.",
    )
    parser.add_argument(
        "--knowledge-base-domain",
        default="training_management",
        help="Domain used when creating the comparison knowledge bases.",
    )
    parser.add_argument(
        "--name-prefix",
        default="Real Model Pair Eval",
        help="Knowledge base name prefix.",
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
        "--keep-kbs",
        action="store_true",
        help="Keep generated comparison knowledge bases instead of deleting them after the run.",
    )
    return parser.parse_args()



def _clear_runtime_caches() -> None:
    get_settings.cache_clear()
    rag_service_settings.cache_clear()
    _PIPELINES.clear()
    for loader in (
        load_answer_guard_config,
        load_cross_domain_guard_config,
        load_document_split_config,
        load_faq_shortcut_config,
        load_knowledge_base_routing_config,
        load_light_intent_config,
        load_query_alias_config,
        load_query_rewriter_prompt_config,
        load_retrieval_rules_config,
        load_source_name_config,
    ):
        loader.cache_clear()



def _set_runtime_env(*, profile: str, embedding_model: str, reranker_model: str) -> None:
    os.environ["CONFIG_PROFILE"] = profile
    os.environ["EMBEDDING_MODEL_NAME"] = embedding_model
    os.environ["RERANKER_MODEL_NAME"] = reranker_model
    _clear_runtime_caches()


async def _upload_documents(knowledge_base_id: str, fixture_paths: list[Path]) -> list[dict]:
    service = DocumentService()
    records: list[dict] = []
    for path in fixture_paths:
        with path.open("rb") as handle:
            upload = UploadFile(filename=path.name, file=handle)
            record = await service.upload_document(knowledge_base_id, upload)
            records.append(record)
    return records


async def _build_eval_report(
    *,
    dataset_path: Path,
    config_profile: str,
    pair_label: str,
    embedding_model: str,
    reranker_model: str,
    fixture_paths: list[Path],
    top_k_retrieve: int,
    top_k_rerank: int,
    enable_rewrite: bool,
    enable_rerank: bool,
    knowledge_base_subject: str,
    knowledge_base_domain: str,
    name_prefix: str,
) -> tuple[dict[str, object], dict[str, object]]:
    _set_runtime_env(
        profile=config_profile,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
    )
    cases = load_eval_cases(dataset_path)

    knowledge_base_service = KnowledgeBaseService()
    ingest_service = IngestService()
    repository = MetadataRepository()

    knowledge_base = knowledge_base_service.create_knowledge_base(
        name=f"{name_prefix} {pair_label}",
        description=f"real model-pair eval for {pair_label}",
        subject=knowledge_base_subject,
        domain=knowledge_base_domain,
    )
    uploaded_documents = await _upload_documents(knowledge_base["id"], fixture_paths)
    task = ingest_service.create_ingest_task(
        knowledge_base["id"],
        [str(item["id"]) for item in uploaded_documents],
    )
    task_result = ingest_service.run_ingest_task(task["id"])

    rag_service = RAGService()
    results = []
    for case in cases:
        results.append(
            await run_case(
                service=rag_service,
                case=case,
                knowledge_base_id=knowledge_base["id"],
                top_k_retrieve=top_k_retrieve,
                top_k_rerank=top_k_rerank,
                enable_rewrite=enable_rewrite,
                enable_rerank=enable_rerank,
            )
        )

    report = {
        "run_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "knowledge_base_id": knowledge_base["id"],
        "dataset": str(dataset_path),
        "config": {
            "config_profile": config_profile,
            "pair_label": pair_label,
            "embedding_model_name": embedding_model,
            "reranker_model_name": reranker_model,
            "top_k_retrieve": top_k_retrieve,
            "top_k_rerank": top_k_rerank,
            "enable_rewrite": enable_rewrite,
            "enable_rerank": enable_rerank,
            "fixture_paths": [str(path) for path in fixture_paths],
            "selected_case_ids": [case.id for case in cases],
        },
        "summary": build_summary(results),
        "results": [asdict(item) for item in results],
    }
    metadata = {
        "knowledge_base": knowledge_base,
        "uploaded_documents": uploaded_documents,
        "task": task_result,
        "knowledge_base_status": repository.get_knowledge_base(knowledge_base["id"]),
    }
    return report, metadata


async def _delete_knowledge_base(knowledge_base_id: str) -> None:
    _clear_runtime_caches()
    KnowledgeBaseService().delete_knowledge_base(knowledge_base_id)


async def main() -> int:
    args = parse_args()
    fixture_paths = [path.resolve() for path in args.fixtures]
    dataset_path = args.dataset.resolve()
    output_dir = ROOT / "evals" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_report, baseline_metadata = await _build_eval_report(
        dataset_path=dataset_path,
        config_profile=args.config_profile,
        pair_label=args.baseline_label,
        embedding_model=args.baseline_embedding,
        reranker_model=args.baseline_reranker,
        fixture_paths=fixture_paths,
        top_k_retrieve=args.top_k_retrieve,
        top_k_rerank=args.top_k_rerank,
        enable_rewrite=not args.disable_rewrite,
        enable_rerank=not args.disable_rerank,
        knowledge_base_subject=args.knowledge_base_subject,
        knowledge_base_domain=args.knowledge_base_domain,
        name_prefix=args.name_prefix,
    )
    contender_report, contender_metadata = await _build_eval_report(
        dataset_path=dataset_path,
        config_profile=args.config_profile,
        pair_label=args.contender_label,
        embedding_model=args.contender_embedding,
        reranker_model=args.contender_reranker,
        fixture_paths=fixture_paths,
        top_k_retrieve=args.top_k_retrieve,
        top_k_rerank=args.top_k_rerank,
        enable_rewrite=not args.disable_rewrite,
        enable_rerank=not args.disable_rerank,
        knowledge_base_subject=args.knowledge_base_subject,
        knowledge_base_domain=args.knowledge_base_domain,
        name_prefix=args.name_prefix,
    )

    compare_report = compare_eval_reports(baseline_report, contender_report)

    baseline_path = output_dir / f"{args.label}-{args.baseline_label}.json"
    baseline_path.write_text(
        json.dumps({"metadata": baseline_metadata, **baseline_report}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved baseline report to {baseline_path}")

    contender_path = output_dir / f"{args.label}-{args.contender_label}.json"
    contender_path.write_text(
        json.dumps({"metadata": contender_metadata, **contender_report}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved contender report to {contender_path}")

    compare_path = output_dir / f"{args.label}-compare.json"
    compare_path.write_text(
        json.dumps(
            {
                "run_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "baseline_label": args.baseline_label,
                "contender_label": args.contender_label,
                "baseline_embedding_model": args.baseline_embedding,
                "baseline_reranker_model": args.baseline_reranker,
                "contender_embedding_model": args.contender_embedding,
                "contender_reranker_model": args.contender_reranker,
                "comparison": compare_report,
                "baseline_report_path": str(baseline_path.resolve()),
                "contender_report_path": str(contender_path.resolve()),
                "baseline_metadata": baseline_metadata,
                "contender_metadata": contender_metadata,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved model-pair comparison report to {compare_path}")
    print(json.dumps({
        args.baseline_label: baseline_report["summary"],
        args.contender_label: contender_report["summary"],
    }, ensure_ascii=False, indent=2))
    print(json.dumps(compare_report, ensure_ascii=False, indent=2))

    if not args.keep_kbs:
        await _delete_knowledge_base(str(baseline_metadata["knowledge_base"]["id"]))
        await _delete_knowledge_base(str(contender_metadata["knowledge_base"]["id"]))
        print("Deleted generated comparison knowledge bases.")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
