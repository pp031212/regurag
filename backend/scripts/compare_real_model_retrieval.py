"""真实知识库下比较不同模型组合的检索表现。

只关注检索和最终上下文，不调用完整问答生成，适合快速排查召回差异。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import perf_counter

from fastapi import UploadFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.rag.pipeline import RAGPipeline
from app.rag.retrieval_utils import mmr_rerank
from app.repositories.metadata_repository import MetadataRepository
from app.services.answer_guard_config import load_answer_guard_config
from app.services.cross_domain_guard_config import load_cross_domain_guard_config
from app.services.faq_shortcut_config import load_faq_shortcut_config
from app.services.ingest_service import IngestService
from app.services.knowledge_base_routing_config import load_knowledge_base_routing_config
from app.services.knowledge_base_service import DocumentService, KnowledgeBaseService
from app.services.light_intent_config import load_light_intent_config
from app.services.rag_service import _PIPELINES, _settings as rag_service_settings
from app.services.source_name_config import load_source_name_config
from app.workflows.rag.pipeline_steps import (
    build_debug_chunks,
    build_final_context_parents,
    build_parent_docs_for_rerank,
    preserve_policy_keyword_docs,
    preserve_query_keyword_docs,
    should_treat_as_policy_question,
    sort_reranked_parents,
)
from eval_rag import EvalCase, keyword_group_matches, load_eval_cases

DEFAULT_BASELINE_LABEL = "current"
DEFAULT_CONTENDER_LABEL = "bge_m3_pair"
DEFAULT_BASELINE_EMBEDDING = "BAAI/bge-small-zh-v1.5"
DEFAULT_BASELINE_RERANKER = "BAAI/bge-reranker-base"
DEFAULT_CONTENDER_EMBEDDING = "BAAI/bge-m3"
DEFAULT_CONTENDER_RERANKER = "BAAI/bge-reranker-v2-m3"


@dataclass(slots=True)
class RetrievalModelCaseResult:
    id: str
    question: str
    category: str
    answer_mode: str
    retrieval_hit: bool
    final_context_hit: bool
    matched_retrieval_keywords: list[str]
    missing_retrieval_keywords: list[str]
    matched_final_context_keywords: list[str]
    missing_final_context_keywords: list[str]
    debug: dict[str, object]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build real PDF knowledge bases for two model pairs, then run retrieval-only benchmark on both."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "evals" / "rag_eval_real_kb_samples.jsonl",
        help="Path to the JSONL retrieval benchmark dataset.",
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
        help="Top-k retrieve value passed to the retriever.",
    )
    parser.add_argument(
        "--top-k-rerank",
        type=int,
        default=3,
        help="Top-k rerank value used for parent rerank and final context.",
    )
    parser.add_argument(
        "--label",
        default="real-model-pair-retrieval-ab",
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
        default="Real Model Retrieval Eval",
        help="Knowledge base name prefix.",
    )
    parser.add_argument(
        "--enable-rewrite",
        action="store_true",
        help="Enable query rewrite before retrieval. Disabled by default for retrieval-only benchmarking.",
    )
    parser.add_argument(
        "--disable-rerank",
        action="store_true",
        help="Disable MMR and parent reranker for this run.",
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
        load_faq_shortcut_config,
        load_knowledge_base_routing_config,
        load_light_intent_config,
        load_source_name_config,
    ):
        loader.cache_clear()



def _set_runtime_env(*, profile: str, embedding_model: str, reranker_model: str) -> None:
    os.environ["CONFIG_PROFILE"] = profile
    os.environ["EMBEDDING_MODEL_NAME"] = embedding_model
    os.environ["RERANKER_MODEL_NAME"] = reranker_model
    _clear_runtime_caches()



def _collection_name(base_name: str, knowledge_base_id: str) -> str:
    suffix = "".join(char if char.isalnum() or char == "_" else "_" for char in knowledge_base_id)
    return f"{base_name}_{suffix}"



def _join_doc_texts(docs: list[dict[str, object]], *, key: str) -> str:
    return "\n".join(str(doc.get(key) or "") for doc in docs)



def _compare_case_result(case: EvalCase, *, retrieved_docs: list[dict[str, object]], final_context_docs: list[dict[str, object]]) -> tuple[bool, bool, list[str], list[str], list[str], list[str]]:
    if case.answer_mode == "no_answer":
        retrieval_hit = len(retrieved_docs) == 0
        final_context_hit = len(final_context_docs) == 0
        return retrieval_hit, final_context_hit, [], [], [], []

    retrieved_text = "\n".join(
        f"{str(doc.get('child_text') or '')}\n{str(doc.get('parent_text') or doc.get('text') or '')}"
        for doc in retrieved_docs
    )
    final_context_text = _join_doc_texts(final_context_docs, key="text")
    matched_retrieval, missing_retrieval = keyword_group_matches(
        retrieved_text,
        case.expected_context_keyword_groups,
    )
    matched_final_context, missing_final_context = keyword_group_matches(
        final_context_text,
        case.expected_context_keyword_groups,
    )
    return (
        len(missing_retrieval) == 0,
        len(missing_final_context) == 0,
        matched_retrieval,
        missing_retrieval,
        matched_final_context,
        missing_final_context,
    )


async def _upload_documents(knowledge_base_id: str, fixture_paths: list[Path]) -> list[dict]:
    service = DocumentService()
    records: list[dict] = []
    for path in fixture_paths:
        with path.open("rb") as handle:
            upload = UploadFile(filename=path.name, file=handle)
            record = await service.upload_document(knowledge_base_id, upload)
            records.append(record)
    return records



def _run_case(
    *,
    pipeline: RAGPipeline,
    case: EvalCase,
    top_k_retrieve: int,
    top_k_rerank: int,
    enable_rewrite: bool,
    enable_rerank: bool,
) -> RetrievalModelCaseResult:
    query_policy = pipeline._build_query_policy(
        top_k_retrieve=top_k_retrieve,
        top_k_rerank=top_k_rerank,
        enable_rerank=enable_rerank,
        answer_style="concise",
    )
    prepared_query = pipeline.prepare_query_inputs(query=case.question, enable_rewrite=enable_rewrite)
    search_query = str(prepared_query["search_query"])
    query_keywords = list(prepared_query["query_keywords"])
    rewrite_ms = int(prepared_query["rewrite_ms"])

    retrieve_started = perf_counter()
    retrieval = pipeline._get_retriever().retrieve(
        search_query=search_query,
        query_keywords=query_keywords,
        top_k_retrieve=query_policy.top_k_retrieve,
        source_name_by_document_id=None,
        policy=query_policy.retrieval,
    )
    retrieve_ms = int((perf_counter() - retrieve_started) * 1000)

    query_vector = retrieval.query_vector
    deduped_docs = retrieval.deduped_docs
    valid_docs = retrieval.valid_docs

    rerank_started = perf_counter()
    if valid_docs:
        if query_policy.enable_rerank:
            selected_indices = mmr_rerank(
                query_embedding=query_vector,
                doc_embeddings=retrieval.doc_embeddings,
                top_k=min(pipeline.settings.top_k_mmr, len(valid_docs)),
                lambda_mult=0.6,
            )
            mmr_selected_docs = [valid_docs[index] for index in selected_indices]
        else:
            mmr_selected_docs = valid_docs[: min(query_policy.top_k_rerank, len(valid_docs))]

        is_policy_question = should_treat_as_policy_question(
            case.question,
            policy_trigger_words=pipeline.retrieval_rules.policy_trigger_words,
            policy_trigger_phrases=pipeline.retrieval_rules.policy_trigger_phrases,
            behavior_keywords=pipeline.retrieval_rules.behavior_keywords,
        )
        mmr_selected_docs = preserve_policy_keyword_docs(
            selected_docs=mmr_selected_docs,
            candidate_docs=deduped_docs,
            is_policy_question=is_policy_question,
            must_keep_child_keywords=pipeline.retrieval_rules.must_keep_child_keywords,
        )
        mmr_selected_docs = preserve_query_keyword_docs(
            selected_docs=mmr_selected_docs,
            candidate_docs=deduped_docs,
            query_keywords=query_keywords,
        )
        parent_docs_for_rerank = build_parent_docs_for_rerank(mmr_selected_docs)
        if parent_docs_for_rerank:
            if query_policy.enable_rerank:
                reranked_parents = pipeline.reranker.rerank(
                    search_query,
                    parent_docs_for_rerank,
                    top_k=min(query_policy.top_k_rerank, len(parent_docs_for_rerank)),
                )
            else:
                reranked_parents = parent_docs_for_rerank[: min(query_policy.top_k_rerank, len(parent_docs_for_rerank))]
            reranked_parents = sort_reranked_parents(reranked_parents)
            final_context_parents = build_final_context_parents(
                parent_docs_for_rerank=parent_docs_for_rerank,
                reranked_parents=reranked_parents,
                query=case.question,
                query_keywords=query_keywords,
                is_policy_question=is_policy_question,
                must_keep_parent_keywords=pipeline.retrieval_rules.must_keep_child_keywords,
                low_value_parent_keywords=pipeline.retrieval_rules.low_value_parent_keywords,
                rule_signal_keywords=pipeline.retrieval_rules.rule_signal_keywords,
                behavior_keywords=pipeline.retrieval_rules.behavior_keywords,
                top_k_rerank=top_k_rerank,
            )
        else:
            reranked_parents = []
            final_context_parents = []
    else:
        mmr_selected_docs = []
        parent_docs_for_rerank = []
        reranked_parents = []
        final_context_parents = []

    rerank_ms = int((perf_counter() - rerank_started) * 1000)
    total_ms = rewrite_ms + retrieve_ms + rerank_ms

    (
        retrieval_hit,
        final_context_hit,
        matched_retrieval_keywords,
        missing_retrieval_keywords,
        matched_final_context_keywords,
        missing_final_context_keywords,
    ) = _compare_case_result(
        case,
        retrieved_docs=deduped_docs,
        final_context_docs=final_context_parents,
    )

    return RetrievalModelCaseResult(
        id=case.id,
        question=case.question,
        category=case.category,
        answer_mode=case.answer_mode,
        retrieval_hit=retrieval_hit,
        final_context_hit=final_context_hit,
        matched_retrieval_keywords=matched_retrieval_keywords,
        missing_retrieval_keywords=missing_retrieval_keywords,
        matched_final_context_keywords=matched_final_context_keywords,
        missing_final_context_keywords=missing_final_context_keywords,
        debug={
            "rewrite_ms": rewrite_ms,
            "retrieve_ms": retrieve_ms,
            "rerank_ms": rerank_ms,
            "latency_ms": total_ms,
            "retrieved_count": len(deduped_docs),
            "valid_count": len(valid_docs),
            "mmr_selected_count": len(mmr_selected_docs),
            "parent_docs_count": len(parent_docs_for_rerank),
            "reranked_parent_count": len(reranked_parents),
            "final_context_count": len(final_context_parents),
            "query_keywords": query_keywords,
            "search_query": search_query,
            "retrieved_chunks": build_debug_chunks(deduped_docs),
            "mmr_selected_chunks": build_debug_chunks(mmr_selected_docs),
            "reranked_chunks": build_debug_chunks(reranked_parents),
            "final_context_chunks": build_debug_chunks(final_context_parents),
        },
    )



def _build_summary(results: list[RetrievalModelCaseResult]) -> dict[str, object]:
    total = len(results)
    debug_items = [item.debug for item in results]
    return {
        "case_count": total,
        "grounded_case_count": sum(item.answer_mode == "grounded" for item in results),
        "no_answer_case_count": sum(item.answer_mode == "no_answer" for item in results),
        "retrieval_hit_rate": round(sum(item.retrieval_hit for item in results) / total, 4),
        "final_context_hit_rate": round(sum(item.final_context_hit for item in results) / total, 4),
        "avg_rewrite_ms": round(mean(int(item.get("rewrite_ms", 0)) for item in debug_items), 2),
        "avg_retrieve_ms": round(mean(int(item.get("retrieve_ms", 0)) for item in debug_items), 2),
        "avg_rerank_ms": round(mean(int(item.get("rerank_ms", 0)) for item in debug_items), 2),
        "avg_latency_ms": round(mean(int(item.get("latency_ms", 0)) for item in debug_items), 2),
        "avg_retrieved_count": round(mean(int(item.get("retrieved_count", 0)) for item in debug_items), 2),
        "avg_mmr_selected_count": round(mean(int(item.get("mmr_selected_count", 0)) for item in debug_items), 2),
        "avg_final_context_count": round(mean(int(item.get("final_context_count", 0)) for item in debug_items), 2),
    }



def _compare_reports(
    baseline_report: dict[str, object],
    contender_report: dict[str, object],
) -> dict[str, object]:
    baseline_results_raw = {item["id"]: item for item in baseline_report["results"]}
    contender_results_raw = {item["id"]: item for item in contender_report["results"]}
    shared_case_ids = sorted(set(baseline_results_raw) & set(contender_results_raw))

    def avg_bool(case_ids: list[str], field: str, report_results: dict[str, dict[str, object]]) -> float:
        return round(sum(bool(report_results[case_id].get(field)) for case_id in case_ids) / len(case_ids), 4)

    def avg_debug(case_ids: list[str], field: str, report_results: dict[str, dict[str, object]]) -> float:
        return round(
            mean(float((report_results[case_id].get("debug") or {}).get(field) or 0.0) for case_id in case_ids),
            2,
        )

    case_diffs: list[dict[str, object]] = []
    for case_id in shared_case_ids:
        baseline_item = baseline_results_raw[case_id]
        contender_item = contender_results_raw[case_id]
        case_diffs.append(
            {
                "id": case_id,
                "question": contender_item.get("question") or baseline_item.get("question"),
                "category": contender_item.get("category") or baseline_item.get("category"),
                "answer_mode": contender_item.get("answer_mode") or baseline_item.get("answer_mode"),
                "retrieval_hit": {
                    "baseline": bool(baseline_item.get("retrieval_hit")),
                    "contender": bool(contender_item.get("retrieval_hit")),
                },
                "final_context_hit": {
                    "baseline": bool(baseline_item.get("final_context_hit")),
                    "contender": bool(contender_item.get("final_context_hit")),
                },
                "latency_ms": {
                    "baseline": float((baseline_item.get("debug") or {}).get("latency_ms") or 0.0),
                    "contender": float((contender_item.get("debug") or {}).get("latency_ms") or 0.0),
                },
                "retrieve_ms": {
                    "baseline": float((baseline_item.get("debug") or {}).get("retrieve_ms") or 0.0),
                    "contender": float((contender_item.get("debug") or {}).get("retrieve_ms") or 0.0),
                },
                "rerank_ms": {
                    "baseline": float((baseline_item.get("debug") or {}).get("rerank_ms") or 0.0),
                    "contender": float((contender_item.get("debug") or {}).get("rerank_ms") or 0.0),
                },
                "final_context_count": {
                    "baseline": int((baseline_item.get("debug") or {}).get("final_context_count") or 0),
                    "contender": int((contender_item.get("debug") or {}).get("final_context_count") or 0),
                },
            }
        )

    return {
        "shared_case_count": len(shared_case_ids),
        "shared_case_ids": shared_case_ids,
        "metrics": {
            "retrieval_hit": {
                "baseline_avg": avg_bool(shared_case_ids, "retrieval_hit", baseline_results_raw),
                "contender_avg": avg_bool(shared_case_ids, "retrieval_hit", contender_results_raw),
            },
            "final_context_hit": {
                "baseline_avg": avg_bool(shared_case_ids, "final_context_hit", baseline_results_raw),
                "contender_avg": avg_bool(shared_case_ids, "final_context_hit", contender_results_raw),
            },
        },
        "latency": {
            "baseline_avg_ms": avg_debug(shared_case_ids, "latency_ms", baseline_results_raw),
            "contender_avg_ms": avg_debug(shared_case_ids, "latency_ms", contender_results_raw),
        },
        "retrieve": {
            "baseline_avg_ms": avg_debug(shared_case_ids, "retrieve_ms", baseline_results_raw),
            "contender_avg_ms": avg_debug(shared_case_ids, "retrieve_ms", contender_results_raw),
        },
        "rerank": {
            "baseline_avg_ms": avg_debug(shared_case_ids, "rerank_ms", baseline_results_raw),
            "contender_avg_ms": avg_debug(shared_case_ids, "rerank_ms", contender_results_raw),
        },
        "case_diffs": case_diffs,
    }


async def _build_report(
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
    settings = get_settings()
    cases = load_eval_cases(dataset_path)

    knowledge_base_service = KnowledgeBaseService()
    ingest_service = IngestService()
    repository = MetadataRepository()

    knowledge_base = knowledge_base_service.create_knowledge_base(
        name=f"{name_prefix} {pair_label}",
        description=f"real model retrieval eval for {pair_label}",
        subject=knowledge_base_subject,
        domain=knowledge_base_domain,
    )
    uploaded_documents = await _upload_documents(knowledge_base["id"], fixture_paths)
    task = ingest_service.create_ingest_task(
        knowledge_base["id"],
        [str(item["id"]) for item in uploaded_documents],
    )
    task_result = ingest_service.run_ingest_task(task["id"])

    pipeline = RAGPipeline(
        settings=settings,
        collection_name=_collection_name(settings.chroma_collection_name, knowledge_base["id"]),
        subject=knowledge_base_subject,
    )
    results = [
        _run_case(
            pipeline=pipeline,
            case=case,
            top_k_retrieve=top_k_retrieve,
            top_k_rerank=top_k_rerank,
            enable_rewrite=enable_rewrite,
            enable_rerank=enable_rerank,
        )
        for case in cases
    ]

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
        "summary": _build_summary(results),
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
    dataset_path = args.dataset.resolve()
    fixture_paths = [path.resolve() for path in args.fixtures]
    output_dir = ROOT / "evals" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_report, baseline_metadata = await _build_report(
        dataset_path=dataset_path,
        config_profile=args.config_profile,
        pair_label=args.baseline_label,
        embedding_model=args.baseline_embedding,
        reranker_model=args.baseline_reranker,
        fixture_paths=fixture_paths,
        top_k_retrieve=args.top_k_retrieve,
        top_k_rerank=args.top_k_rerank,
        enable_rewrite=args.enable_rewrite,
        enable_rerank=not args.disable_rerank,
        knowledge_base_subject=args.knowledge_base_subject,
        knowledge_base_domain=args.knowledge_base_domain,
        name_prefix=args.name_prefix,
    )
    contender_report, contender_metadata = await _build_report(
        dataset_path=dataset_path,
        config_profile=args.config_profile,
        pair_label=args.contender_label,
        embedding_model=args.contender_embedding,
        reranker_model=args.contender_reranker,
        fixture_paths=fixture_paths,
        top_k_retrieve=args.top_k_retrieve,
        top_k_rerank=args.top_k_rerank,
        enable_rewrite=args.enable_rewrite,
        enable_rerank=not args.disable_rerank,
        knowledge_base_subject=args.knowledge_base_subject,
        knowledge_base_domain=args.knowledge_base_domain,
        name_prefix=args.name_prefix,
    )

    compare_payload = {
        "run_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "baseline_label": args.baseline_label,
        "contender_label": args.contender_label,
        "baseline_embedding_model": args.baseline_embedding,
        "baseline_reranker_model": args.baseline_reranker,
        "contender_embedding_model": args.contender_embedding,
        "contender_reranker_model": args.contender_reranker,
        "comparison": _compare_reports(baseline_report, contender_report),
    }

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
    compare_path.write_text(json.dumps(compare_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved model retrieval comparison report to {compare_path}")
    print(
        json.dumps(
            {
                args.baseline_label: baseline_report["summary"],
                args.contender_label: contender_report["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(json.dumps(compare_payload["comparison"], ensure_ascii=False, indent=2))

    if not args.keep_kbs:
        await _delete_knowledge_base(str(baseline_metadata["knowledge_base"]["id"]))
        await _delete_knowledge_base(str(contender_metadata["knowledge_base"]["id"]))
        print("Deleted generated comparison knowledge bases.")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
