"""真实知识库下比较不同向量库后端的检索效果。

上传同一批 fixture 到候选后端，按样本评估召回、上下文和引用候选差异。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from fastapi import UploadFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
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
from app.services.rag_service import (
    _PIPELINES,
    _collection_name,
    _settings as rag_service_settings,
    get_rag_pipeline,
)
from app.services.source_name_config import load_source_name_config, resolve_source_name
try:
    from eval_rag import EvalCase, keyword_group_matches, load_eval_cases, select_eval_cases
except ModuleNotFoundError:
    from scripts.eval_rag import EvalCase, keyword_group_matches, load_eval_cases, select_eval_cases


DEFAULT_BACKENDS = ("chroma", "milvus")


@dataclass(slots=True)
class RetrievalCaseResult:
    id: str
    question: str
    category: str
    answer_mode: str
    retrieval_hit: bool
    final_context_hit: bool
    citation_hit: bool
    matched_retrieval_keywords: list[str]
    missing_retrieval_keywords: list[str]
    matched_final_context_keywords: list[str]
    missing_final_context_keywords: list[str]
    matched_citation_keywords: list[str]
    missing_citation_keywords: list[str]
    retrieved_chunk_ids: list[str]
    final_context_chunk_ids: list[str]
    citation_chunk_ids: list[str]
    retrieved_signatures: list[str]
    final_context_signatures: list[str]
    citation_signatures: list[str]
    debug: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build real PDF knowledge bases for multiple vector store backends, then run retrieval-only parity checks."
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
        help="Config profile used for every backend run.",
    )
    parser.add_argument(
        "--backends",
        nargs="*",
        default=list(DEFAULT_BACKENDS),
        help="Vector store backends to compare. The first backend is treated as baseline.",
    )
    parser.add_argument(
        "--milvus-uri",
        default="http://127.0.0.1:19530",
        help="Milvus URI used when backend=milvus.",
    )
    parser.add_argument(
        "--milvus-token",
        default=None,
        help="Optional Milvus token used when backend=milvus.",
    )
    parser.add_argument(
        "--top-k-retrieve",
        type=int,
        default=15,
        help="Top-k retrieve value passed to the pipeline.",
    )
    parser.add_argument(
        "--top-k-rerank",
        type=int,
        default=3,
        help="Top-k rerank value passed to the pipeline.",
    )
    parser.add_argument(
        "--label",
        default="real-vector-store-retrieval-compare",
        help="Output file label prefix. Backend reports and compare summary will be written under backend/evals/results/.",
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
        default="Real Vector Store Retrieval Compare",
        help="Knowledge base name prefix.",
    )
    parser.add_argument(
        "--enable-rewrite",
        action="store_true",
        help="Enable query rewrite before retrieval. Disabled by default to avoid external model dependency.",
    )
    parser.add_argument(
        "--disable-rerank",
        action="store_true",
        help="Disable MMR and reranker for this run.",
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


def _set_runtime_env(
    *,
    profile: str,
    vector_store_backend: str,
    milvus_uri: str,
    milvus_token: str | None,
) -> None:
    os.environ["CONFIG_PROFILE"] = profile
    os.environ["VECTOR_STORE_BACKEND"] = vector_store_backend
    os.environ["VECTOR_STORE_MILVUS_URI"] = milvus_uri
    if milvus_token:
        os.environ["VECTOR_STORE_MILVUS_TOKEN"] = milvus_token
    else:
        os.environ.pop("VECTOR_STORE_MILVUS_TOKEN", None)
    _clear_runtime_caches()


def _backend_label(backend: str) -> str:
    return backend.strip().lower() or "unknown"


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


def _extract_chunk_ids(items: list[dict[str, object]]) -> list[str]:
    return [str(item.get("chunk_id") or "") for item in items if isinstance(item, dict)]


def _normalize_signature_text(value: object) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"\s+", " ", text)


def _normalize_signature_number(value: object) -> str:
    if value is None:
        return ""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return str(value)
    return "" if number < 0 else str(number)


def _build_chunk_signature(item: dict[str, object]) -> str:
    return "|".join(
        [
            str(item.get("source_type") or ""),
            _normalize_signature_number(item.get("page_number")),
            _normalize_signature_number(item.get("block_index")),
            _normalize_signature_text(item.get("parent_text")),
            _normalize_signature_text(item.get("child_text")),
        ]
    )


def _build_content_signature(*, source_type: object, content: object) -> str:
    return "|".join(
        [
            str(source_type or ""),
            _normalize_signature_text(content),
        ]
    )


def _build_citation_signature(item: dict[str, object]) -> str:
    return _build_content_signature(source_type=item.get("source_type"), content=item.get("content"))


def _join_debug_chunk_texts(items: list[dict[str, object]]) -> str:
    texts: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        parent_text = item.get("parent_text")
        child_text = item.get("child_text")
        if isinstance(parent_text, str):
            texts.append(parent_text)
        if isinstance(child_text, str):
            texts.append(child_text)
    return "\n".join(texts)


def _evaluate_case(
    *,
    case: EvalCase,
    pipeline,
    knowledge_base_id: str,
    source_name_by_document_id: dict[str, str],
    top_k_retrieve: int,
    top_k_rerank: int,
    enable_rewrite: bool,
    enable_rerank: bool,
) -> RetrievalCaseResult:
    generation_inputs = pipeline._prepare_generation_inputs(
        query=case.question,
        top_k_retrieve=top_k_retrieve,
        top_k_rerank=top_k_rerank,
        knowledge_base_id=knowledge_base_id,
        source_name_by_document_id=source_name_by_document_id,
        enable_rewrite=enable_rewrite,
        enable_rerank=enable_rerank,
        answer_style="concise",
    )

    retrieved_chunks = list(generation_inputs["retrieved_debug_chunks"])
    final_context_chunks = list(generation_inputs["final_context_debug_chunks"])
    citations = list(generation_inputs["citations"])

    retrieved_text = _join_debug_chunk_texts(retrieved_chunks)
    final_context_text = _join_debug_chunk_texts(final_context_chunks)
    citation_text = "\n".join(str(item.get("content", "")) for item in citations if isinstance(item, dict))

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
        retrieval_hit = len(final_context_chunks) == 0
        final_context_hit = len(final_context_chunks) == 0
        citation_hit = len(citations) == 0

    return RetrievalCaseResult(
        id=case.id,
        question=case.question,
        category=case.category,
        answer_mode=case.answer_mode,
        retrieval_hit=retrieval_hit,
        final_context_hit=final_context_hit,
        citation_hit=citation_hit,
        matched_retrieval_keywords=matched_retrieval_keywords,
        missing_retrieval_keywords=missing_retrieval_keywords,
        matched_final_context_keywords=matched_final_context_keywords,
        missing_final_context_keywords=missing_final_context_keywords,
        matched_citation_keywords=matched_citation_keywords,
        missing_citation_keywords=missing_citation_keywords,
        retrieved_chunk_ids=_extract_chunk_ids(retrieved_chunks),
        final_context_chunk_ids=_extract_chunk_ids(final_context_chunks),
        citation_chunk_ids=_extract_chunk_ids(citations),
        retrieved_signatures=[_build_chunk_signature(item) for item in retrieved_chunks if isinstance(item, dict)],
        final_context_signatures=[
            _build_content_signature(
                source_type=item.get("source_type"),
                content=item.get("parent_text") or item.get("child_text"),
            )
            for item in final_context_chunks
            if isinstance(item, dict)
        ],
        citation_signatures=[_build_citation_signature(item) for item in citations if isinstance(item, dict)],
        debug={
            "rewritten_query": generation_inputs["search_query"],
            "retrieved_count": generation_inputs["retrieved_count"],
            "reranked_count": generation_inputs["reranked_count"],
            "rewrite_ms": generation_inputs["rewrite_ms"],
            "retrieve_ms": generation_inputs["retrieve_ms"],
            "rerank_ms": generation_inputs["rerank_ms"],
            "context_build_ms": generation_inputs["context_build_ms"],
            "retrieved_chunks": retrieved_chunks,
            "final_context_chunks": final_context_chunks,
        },
    )


def _build_summary(results: list[RetrievalCaseResult]) -> dict[str, object]:
    if not results:
        raise ValueError("No results to summarize")
    debug_items = [item.debug for item in results]
    return {
        "case_count": len(results),
        "grounded_case_count": sum(item.answer_mode == "grounded" for item in results),
        "no_answer_case_count": sum(item.answer_mode == "no_answer" for item in results),
        "retrieval_hit_rate": round(sum(item.retrieval_hit for item in results) / len(results), 4),
        "final_context_hit_rate": round(sum(item.final_context_hit for item in results) / len(results), 4),
        "citation_hit_rate": round(sum(item.citation_hit for item in results) / len(results), 4),
        "avg_rewrite_ms": round(mean(int(item.get("rewrite_ms", 0)) for item in debug_items), 2),
        "avg_retrieve_ms": round(mean(int(item.get("retrieve_ms", 0)) for item in debug_items), 2),
        "avg_rerank_ms": round(mean(int(item.get("rerank_ms", 0)) for item in debug_items), 2),
        "avg_context_build_ms": round(mean(int(item.get("context_build_ms", 0)) for item in debug_items), 2),
        "avg_retrieved_count": round(mean(int(item.get("retrieved_count", 0)) for item in debug_items), 2),
        "avg_final_context_count": round(mean(int(item.get("reranked_count", 0)) for item in debug_items), 2),
    }


def _compare_reports(baseline: dict[str, object], contender: dict[str, object]) -> dict[str, object]:
    baseline_results = {
        str(item["id"]): item
        for item in baseline["results"]
    }
    contender_results = {
        str(item["id"]): item
        for item in contender["results"]
    }

    def comparable_signatures(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return sorted({str(item) for item in value})

    drift_cases: list[dict[str, object]] = []
    for case_id, baseline_item in baseline_results.items():
        contender_item = contender_results.get(case_id)
        if contender_item is None:
            drift_cases.append({"id": case_id, "missing_in_contender": True})
            continue

        mismatch_fields: list[str] = []
        for field_name in ("retrieval_hit", "final_context_hit", "citation_hit"):
            if baseline_item.get(field_name) != contender_item.get(field_name):
                mismatch_fields.append(field_name)
        for field_name in ("final_context_signatures", "citation_signatures"):
            if comparable_signatures(baseline_item.get(field_name)) != comparable_signatures(contender_item.get(field_name)):
                mismatch_fields.append(field_name)
        if mismatch_fields:
            drift_cases.append(
                {
                    "id": case_id,
                    "question": baseline_item.get("question"),
                    "mismatch_fields": mismatch_fields,
                    "baseline": {
                        "retrieved_chunk_ids": baseline_item.get("retrieved_chunk_ids"),
                        "final_context_chunk_ids": baseline_item.get("final_context_chunk_ids"),
                        "citation_chunk_ids": baseline_item.get("citation_chunk_ids"),
                        "retrieved_signatures": baseline_item.get("retrieved_signatures"),
                        "final_context_signatures": baseline_item.get("final_context_signatures"),
                        "citation_signatures": baseline_item.get("citation_signatures"),
                    },
                    "contender": {
                        "retrieved_chunk_ids": contender_item.get("retrieved_chunk_ids"),
                        "final_context_chunk_ids": contender_item.get("final_context_chunk_ids"),
                        "citation_chunk_ids": contender_item.get("citation_chunk_ids"),
                        "retrieved_signatures": contender_item.get("retrieved_signatures"),
                        "final_context_signatures": contender_item.get("final_context_signatures"),
                        "citation_signatures": contender_item.get("citation_signatures"),
                    },
                }
            )

    baseline_summary = dict(baseline["summary"])
    contender_summary = dict(contender["summary"])
    metric_diffs = {
        key: round(float(contender_summary[key]) - float(baseline_summary[key]), 4)
        for key in (
            "retrieval_hit_rate",
            "final_context_hit_rate",
            "citation_hit_rate",
            "avg_retrieve_ms",
            "avg_rerank_ms",
            "avg_context_build_ms",
        )
    }
    return {
        "case_drift_count": len(drift_cases),
        "case_drift_ids": [item["id"] for item in drift_cases],
        "metric_diffs": metric_diffs,
        "drift_cases": drift_cases,
    }


async def _build_backend_report(
    *,
    dataset_path: Path,
    fixture_paths: list[Path],
    config_profile: str,
    vector_store_backend: str,
    milvus_uri: str,
    milvus_token: str | None,
    top_k_retrieve: int,
    top_k_rerank: int,
    enable_rewrite: bool,
    enable_rerank: bool,
    knowledge_base_subject: str,
    knowledge_base_domain: str,
    name_prefix: str,
    case_ids: list[str],
    limit: int | None,
) -> tuple[dict[str, object], dict[str, object]]:
    _set_runtime_env(
        profile=config_profile,
        vector_store_backend=vector_store_backend,
        milvus_uri=milvus_uri,
        milvus_token=milvus_token,
    )
    settings = get_settings()
    cases = select_eval_cases(load_eval_cases(dataset_path), case_ids=case_ids, limit=limit)

    knowledge_base_service = KnowledgeBaseService()
    ingest_service = IngestService()
    repository = MetadataRepository()

    backend_label = _backend_label(vector_store_backend)
    knowledge_base = knowledge_base_service.create_knowledge_base(
        name=f"{name_prefix} {backend_label}",
        description=f"real vector store retrieval compare for {backend_label}",
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
    pipeline = get_rag_pipeline(knowledge_base["id"], knowledge_base["subject"])
    collection_name = _collection_name(settings.chroma_collection_name, knowledge_base["id"])

    results = [
        _evaluate_case(
            case=case,
            pipeline=pipeline,
            knowledge_base_id=knowledge_base["id"],
            source_name_by_document_id=source_name_by_document_id,
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
            "vector_store_backend": backend_label,
            "vector_store_milvus_uri": milvus_uri if backend_label == "milvus" else None,
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
        "vector_collection_name": collection_name,
    }
    return report, metadata


async def _delete_knowledge_base(knowledge_base_id: str) -> None:
    _clear_runtime_caches()
    KnowledgeBaseService().delete_knowledge_base(knowledge_base_id)


async def main() -> int:
    args = parse_args()
    backends = tuple(dict.fromkeys(_backend_label(backend) for backend in args.backends if backend and backend.strip()))
    if not backends:
        raise ValueError("At least one vector store backend is required")

    fixture_paths = [path.resolve() for path in args.fixtures]
    dataset_path = args.dataset.resolve()
    output_dir = ROOT / "evals" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    reports: dict[str, dict[str, object]] = {}
    metadatas: dict[str, dict[str, object]] = {}
    for backend in backends:
        report, metadata = await _build_backend_report(
            dataset_path=dataset_path,
            fixture_paths=fixture_paths,
            config_profile=args.config_profile,
            vector_store_backend=backend,
            milvus_uri=args.milvus_uri,
            milvus_token=args.milvus_token,
            top_k_retrieve=args.top_k_retrieve,
            top_k_rerank=args.top_k_rerank,
            enable_rewrite=args.enable_rewrite,
            enable_rerank=not args.disable_rerank,
            knowledge_base_subject=args.knowledge_base_subject,
            knowledge_base_domain=args.knowledge_base_domain,
            name_prefix=args.name_prefix,
            case_ids=list(args.case_id),
            limit=args.limit,
        )
        reports[backend] = report
        metadatas[backend] = metadata

    baseline_backend = backends[0]
    comparisons: dict[str, dict[str, object]] = {}
    for backend in backends[1:]:
        comparisons[f"{baseline_backend}_vs_{backend}"] = _compare_reports(
            reports[baseline_backend],
            reports[backend],
        )

    for backend in backends:
        report_path = output_dir / f"{args.label}-{backend}.json"
        payload = {
            "metadata": metadatas[backend],
            **reports[backend],
        }
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {backend} report to {report_path}")

    compare_path = output_dir / f"{args.label}-compare.json"
    compare_payload = {
        "run_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "baseline_backend": baseline_backend,
        "backends": list(backends),
        "comparisons": comparisons,
        "report_paths": {
            backend: str((output_dir / f"{args.label}-{backend}.json").resolve())
            for backend in backends
        },
        "metadata": metadatas,
    }
    compare_path.write_text(json.dumps(compare_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved vector store comparison report to {compare_path}")
    print(json.dumps({backend: reports[backend]["summary"] for backend in backends}, ensure_ascii=False, indent=2))
    print(json.dumps(comparisons, ensure_ascii=False, indent=2))

    if not args.keep_kbs:
        for backend in backends:
            await _delete_knowledge_base(str(metadatas[backend]["knowledge_base"]["id"]))
        print("Deleted generated comparison knowledge bases.")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
