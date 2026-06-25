"""真实知识库下比较不同稀疏检索 provider 的召回效果。

只跑检索侧指标，便于快速定位稀疏召回策略带来的候选集变化。
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
from app.evals.eval_report_compare import compare_eval_reports
from app.rag.query_alias_config import QueryAliasExpander, load_query_alias_config
from app.rag.query_rewriter import QueryRewriter
from app.rag.query_rewriter_config import load_query_rewriter_prompt_config
from app.rag.retrievers import RetrievalPolicy, build_default_hybrid_retriever
from app.rag.vector_store import VectorStore
from app.repositories.metadata_repository import MetadataRepository
from app.services.answer_guard_config import load_answer_guard_config
from app.services.cross_domain_guard_config import load_cross_domain_guard_config
from app.services.faq_shortcut_config import load_faq_shortcut_config
from app.services.ingest_service import IngestService
from app.services.knowledge_base_routing_config import load_knowledge_base_routing_config
from app.services.knowledge_base_service import DocumentService, KnowledgeBaseService
from app.services.light_intent_config import load_light_intent_config
from app.services.rag_service import _PIPELINES, _settings as rag_service_settings
from app.services.source_name_config import load_source_name_config, resolve_source_name
from app.workflows.rag.pipeline_steps import (
    apply_source_names,
    build_debug_chunks,
    dedupe_retrieved_docs,
    extract_query_keywords,
    sort_retrieved_docs,
)
from eval_rag import EvalCase, keyword_group_matches, load_eval_cases

DEFAULT_PROVIDERS = ("sqlite_fts", "bm25", "scan", "none")


@dataclass(slots=True)
class RetrievalBenchmarkCaseResult:
    id: str
    question: str
    category: str
    answer_mode: str
    retrieval_hit: bool
    matched_context_keywords: list[str]
    missing_context_keywords: list[str]
    query_keywords: list[str]
    dense_count: int
    sparse_count: int
    deduped_count: int
    debug: dict[str, object]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build real PDF knowledge bases for multiple sparse providers, then run retrieval-only benchmark on each."
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
        help="Config profile used for every provider run.",
    )
    parser.add_argument(
        "--providers",
        nargs="*",
        default=list(DEFAULT_PROVIDERS),
        help="Sparse providers to compare. The first provider is treated as baseline.",
    )
    parser.add_argument(
        "--top-k-retrieve",
        type=int,
        default=15,
        help="Top-k retrieve value passed to the retriever.",
    )
    parser.add_argument(
        "--label",
        default="real-sparse-provider-retrieval-bench",
        help="Output file label prefix. Provider reports and compare summary will be written under backend/evals/results/.",
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
        default="Real Sparse Retrieval Bench",
        help="Knowledge base name prefix.",
    )
    parser.add_argument(
        "--enable-rewrite",
        action="store_true",
        help="Enable query rewrite before retrieval. Disabled by default for retrieval-only benchmarking.",
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
        load_query_alias_config,
        load_query_rewriter_prompt_config,
        load_source_name_config,
    ):
        loader.cache_clear()



def _set_runtime_env(*, profile: str, sparse_provider: str) -> None:
    os.environ["CONFIG_PROFILE"] = profile
    os.environ["RETRIEVAL_SPARSE_PROVIDER"] = sparse_provider
    _clear_runtime_caches()



def _provider_label(provider: str) -> str:
    return provider.strip().lower() or "unknown"



def _collection_name(base_name: str, knowledge_base_id: str) -> str:
    suffix = "".join(char if char.isalnum() or char == "_" else "_" for char in knowledge_base_id)
    return f"{base_name}_{suffix}"



def _build_source_name_map(documents: list[dict]) -> dict[str, str]:
    return {
        str(item["id"]): resolve_source_name(str(item.get("filename") or ""))
        for item in documents
        if item.get("id")
    }


async def _upload_documents(knowledge_base_id: str, fixture_paths: list[Path]) -> list[dict]:
    service = DocumentService()
    records: list[dict] = []
    for path in fixture_paths:
        with path.open("rb") as handle:
            upload = UploadFile(filename=path.name, file=handle)
            record = await service.upload_document(knowledge_base_id, upload)
            records.append(record)
    return records



def _prepare_query_inputs(
    *,
    query: str,
    enable_rewrite: bool,
    query_alias_expander: QueryAliasExpander,
    rewriter: QueryRewriter | None,
) -> dict[str, object]:
    rewrite_started = perf_counter()
    expanded_keywords = ""
    if enable_rewrite and rewriter is not None:
        try:
            expanded_keywords = rewriter.rewrite(query)
        except Exception:
            expanded_keywords = ""
    rewrite_ms = int((perf_counter() - rewrite_started) * 1000)

    alias_keywords = query_alias_expander.expand(query)
    query_keywords = extract_query_keywords(alias_keywords)
    if not query_keywords:
        query_keywords = extract_query_keywords(expanded_keywords)
    if not query_keywords:
        query_keywords = extract_query_keywords(query)

    return {
        "effective_query": query,
        "expanded_keywords": expanded_keywords,
        "alias_keywords": alias_keywords,
        "query_keywords": query_keywords,
        "search_query": f"{query} {expanded_keywords} {alias_keywords}".strip(),
        "rewrite_ms": rewrite_ms,
    }



def _evaluate_retrieval_hit(case: EvalCase, deduped_docs: list[dict[str, object]]) -> tuple[bool, list[str], list[str]]:
    if case.answer_mode == "no_answer":
        return len(deduped_docs) == 0, [], []

    retrieved_text = "\n".join(
        f"{str(doc.get('child_text') or '')}\n{str(doc.get('parent_text') or '')}" for doc in deduped_docs
    )
    matched = [keyword for keyword in case.expected_context_keywords if keyword in retrieved_text]
    missing = [keyword for keyword in case.expected_context_keywords if keyword not in retrieved_text]
    return len(missing) == 0, matched, missing



def _run_retrieval_case(
    *,
    case: EvalCase,
    retriever: object,
    policy: RetrievalPolicy,
    source_name_by_document_id: dict[str, str],
    enable_rewrite: bool,
    query_alias_expander: QueryAliasExpander,
    rewriter: QueryRewriter | None,
    top_k_retrieve: int,
) -> RetrievalBenchmarkCaseResult:
    prepared = _prepare_query_inputs(
        query=case.question,
        enable_rewrite=enable_rewrite,
        query_alias_expander=query_alias_expander,
        rewriter=rewriter,
    )
    search_query = str(prepared["search_query"])
    query_keywords = list(prepared["query_keywords"])

    dense_top_k = policy.resolve_dense_top_k(top_k_retrieve)
    sparse_top_k = policy.resolve_sparse_top_k(top_k_retrieve)

    retrieve_started = perf_counter()
    dense_started = perf_counter()
    query_vector, retrieved_docs = retriever.dense_retriever.search(search_query, top_k=dense_top_k)
    dense_ms = int((perf_counter() - dense_started) * 1000)

    supplemental_docs: list[dict[str, object]] = []
    sparse_ms = 0
    if policy.enable_sparse and retriever.sparse_retriever is not None:
        sparse_started = perf_counter()
        supplemental_docs = retriever.sparse_retriever.search(
            query_keywords,
            min_hits=policy.sparse_min_hits,
            top_k=sparse_top_k,
        )
        sparse_ms = int((perf_counter() - sparse_started) * 1000)

    merge_started = perf_counter()
    deduped_docs = sort_retrieved_docs(dedupe_retrieved_docs(retrieved_docs, supplemental_docs))
    deduped_docs = apply_source_names(deduped_docs, source_name_by_document_id)
    merge_ms = int((perf_counter() - merge_started) * 1000)
    retrieve_ms = int((perf_counter() - retrieve_started) * 1000)

    retrieval_hit, matched_keywords, missing_keywords = _evaluate_retrieval_hit(case, deduped_docs)
    debug_payload = {
        "rewrite_ms": int(prepared["rewrite_ms"]),
        "dense_ms": dense_ms,
        "sparse_ms": sparse_ms,
        "merge_ms": merge_ms,
        "latency_ms": retrieve_ms,
        "dense_count": len(retrieved_docs),
        "sparse_count": len(supplemental_docs),
        "deduped_count": len(deduped_docs),
        "retrieved_chunks": build_debug_chunks(deduped_docs),
        "query_keywords": query_keywords,
        "search_query": search_query,
        "query_vector_dim": len(query_vector),
    }

    return RetrievalBenchmarkCaseResult(
        id=case.id,
        question=case.question,
        category=case.category,
        answer_mode=case.answer_mode,
        retrieval_hit=retrieval_hit,
        matched_context_keywords=matched_keywords,
        missing_context_keywords=missing_keywords,
        query_keywords=query_keywords,
        dense_count=len(retrieved_docs),
        sparse_count=len(supplemental_docs),
        deduped_count=len(deduped_docs),
        debug=debug_payload,
    )



def _build_summary(results: list[RetrievalBenchmarkCaseResult]) -> dict[str, object]:
    total = len(results)
    debug_items = [item.debug for item in results]
    return {
        "case_count": total,
        "grounded_case_count": sum(item.answer_mode == "grounded" for item in results),
        "no_answer_case_count": sum(item.answer_mode == "no_answer" for item in results),
        "retrieval_hit_rate": round(sum(item.retrieval_hit for item in results) / total, 4),
        "avg_rewrite_ms": round(mean(int(item.get("rewrite_ms", 0)) for item in debug_items), 2),
        "avg_dense_ms": round(mean(int(item.get("dense_ms", 0)) for item in debug_items), 2),
        "avg_sparse_ms": round(mean(int(item.get("sparse_ms", 0)) for item in debug_items), 2),
        "avg_merge_ms": round(mean(int(item.get("merge_ms", 0)) for item in debug_items), 2),
        "avg_retrieve_ms": round(mean(int(item.get("latency_ms", 0)) for item in debug_items), 2),
        "avg_dense_count": round(mean(int(item.get("dense_count", 0)) for item in debug_items), 2),
        "avg_sparse_count": round(mean(int(item.get("sparse_count", 0)) for item in debug_items), 2),
        "avg_deduped_count": round(mean(int(item.get("deduped_count", 0)) for item in debug_items), 2),
        "avg_query_keyword_count": round(mean(len(item.query_keywords) for item in results), 2),
    }


async def _build_benchmark_report(
    *,
    dataset_path: Path,
    config_profile: str,
    sparse_provider: str,
    fixture_paths: list[Path],
    top_k_retrieve: int,
    enable_rewrite: bool,
    knowledge_base_subject: str,
    knowledge_base_domain: str,
    name_prefix: str,
) -> tuple[dict[str, object], dict[str, object]]:
    _set_runtime_env(profile=config_profile, sparse_provider=sparse_provider)
    settings = get_settings()
    cases = load_eval_cases(dataset_path)

    knowledge_base_service = KnowledgeBaseService()
    ingest_service = IngestService()
    repository = MetadataRepository()

    provider_label = _provider_label(sparse_provider)
    knowledge_base = knowledge_base_service.create_knowledge_base(
        name=f"{name_prefix} {provider_label}",
        description=f"real sparse retrieval bench for {provider_label}",
        subject=knowledge_base_subject,
        domain=knowledge_base_domain,
    )
    uploaded_documents = await _upload_documents(knowledge_base["id"], fixture_paths)
    task = ingest_service.create_ingest_task(
        knowledge_base["id"],
        [str(item["id"]) for item in uploaded_documents],
    )
    task_result = ingest_service.run_ingest_task(task["id"])

    source_name_by_document_id = _build_source_name_map(uploaded_documents)
    collection_name = _collection_name(settings.chroma_collection_name, knowledge_base["id"])
    vector_store = VectorStore(
        model_name=settings.embedding_model_name,
        db_path=str(settings.resolved_chroma_path),
        collection_name=collection_name,
        sparse_provider=settings.retrieval_sparse_provider,
    )
    policy = RetrievalPolicy(
        dense_top_k=settings.retrieval_dense_top_k,
        sparse_top_k=settings.retrieval_sparse_top_k,
        sparse_min_hits=settings.retrieval_sparse_min_hits,
        enable_sparse=settings.retrieval_enable_sparse,
    )
    retriever = build_default_hybrid_retriever(
        vector_store,
        sparse_provider=settings.retrieval_sparse_provider,
        policy=policy,
    )
    query_alias_expander = QueryAliasExpander()
    rewriter = None
    if enable_rewrite:
        rewriter = QueryRewriter(
            api_key=settings.rewrite_api_key,
            base_url=settings.rewrite_base_url,
            model_name=settings.rewrite_model,
        )

    results = [
        _run_retrieval_case(
            case=case,
            retriever=retriever,
            policy=policy,
            source_name_by_document_id=source_name_by_document_id,
            enable_rewrite=enable_rewrite,
            query_alias_expander=query_alias_expander,
            rewriter=rewriter,
            top_k_retrieve=top_k_retrieve,
        )
        for case in cases
    ]

    report = {
        "run_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "knowledge_base_id": knowledge_base["id"],
        "dataset": str(dataset_path),
        "config": {
            "config_profile": config_profile,
            "retrieval_sparse_provider": provider_label,
            "top_k_retrieve": top_k_retrieve,
            "enable_rewrite": enable_rewrite,
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
    providers = tuple(dict.fromkeys(_provider_label(provider) for provider in args.providers if provider and provider.strip()))
    if not providers:
        raise ValueError("At least one sparse provider is required")

    fixture_paths = [path.resolve() for path in args.fixtures]
    dataset_path = args.dataset.resolve()
    output_dir = ROOT / "evals" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    reports: dict[str, dict[str, object]] = {}
    metadatas: dict[str, dict[str, object]] = {}
    for provider in providers:
        report, metadata = await _build_benchmark_report(
            dataset_path=dataset_path,
            config_profile=args.config_profile,
            sparse_provider=provider,
            fixture_paths=fixture_paths,
            top_k_retrieve=args.top_k_retrieve,
            enable_rewrite=args.enable_rewrite,
            knowledge_base_subject=args.knowledge_base_subject,
            knowledge_base_domain=args.knowledge_base_domain,
            name_prefix=args.name_prefix,
        )
        reports[provider] = report
        metadatas[provider] = metadata

    baseline_provider = providers[0]
    comparisons: dict[str, dict[str, object]] = {}
    for provider in providers[1:]:
        comparisons[f"{baseline_provider}_vs_{provider}"] = compare_eval_reports(
            reports[baseline_provider],
            reports[provider],
        )

    for provider in providers:
        report_path = output_dir / f"{args.label}-{provider}.json"
        payload = {
            "metadata": metadatas[provider],
            **reports[provider],
        }
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {provider} report to {report_path}")

    compare_path = output_dir / f"{args.label}-compare.json"
    compare_payload = {
        "run_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "baseline_provider": baseline_provider,
        "providers": list(providers),
        "comparisons": comparisons,
        "report_paths": {
            provider: str((output_dir / f"{args.label}-{provider}.json").resolve())
            for provider in providers
        },
        "metadata": metadatas,
    }
    compare_path.write_text(json.dumps(compare_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved provider comparison report to {compare_path}")
    print(json.dumps({provider: reports[provider]["summary"] for provider in providers}, ensure_ascii=False, indent=2))
    print(json.dumps(comparisons, ensure_ascii=False, indent=2))

    if not args.keep_kbs:
        for provider in providers:
            await _delete_knowledge_base(str(metadatas[provider]["knowledge_base"]["id"]))
        print("Deleted generated comparison knowledge bases.")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
