"""真实知识库下比较不同向量库后端的问答评测表现。

用于 Chroma/Milvus 切换前确认最终答案、引用和上下文指标没有回退。
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
try:
    from eval_rag import build_summary, load_eval_cases, run_case, select_eval_cases
except ModuleNotFoundError:
    from scripts.eval_rag import build_summary, load_eval_cases, run_case, select_eval_cases


DEFAULT_BACKENDS = ("chroma", "milvus")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build real PDF knowledge bases for multiple vector store backends, run eval_rag on each, then compare."
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
        default="real-vector-store-eval",
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
        default="Real Vector Store Eval",
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
    cases = select_eval_cases(load_eval_cases(dataset_path), case_ids=case_ids, limit=limit)

    knowledge_base_service = KnowledgeBaseService()
    ingest_service = IngestService()
    repository = MetadataRepository()

    backend_label = _backend_label(vector_store_backend)
    knowledge_base = knowledge_base_service.create_knowledge_base(
        name=f"{name_prefix} {backend_label}",
        description=f"real vector store eval for {backend_label}",
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
            "vector_store_backend": backend_label,
            "vector_store_milvus_uri": milvus_uri if backend_label == "milvus" else None,
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
        report, metadata = await _build_eval_report(
            dataset_path=dataset_path,
            fixture_paths=fixture_paths,
            config_profile=args.config_profile,
            vector_store_backend=backend,
            milvus_uri=args.milvus_uri,
            milvus_token=args.milvus_token,
            top_k_retrieve=args.top_k_retrieve,
            top_k_rerank=args.top_k_rerank,
            enable_rewrite=not args.disable_rewrite,
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
        comparisons[f"{baseline_backend}_vs_{backend}"] = compare_eval_reports(
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
    print(f"Saved vector store eval comparison report to {compare_path}")
    print(json.dumps({backend: reports[backend]["summary"] for backend in backends}, ensure_ascii=False, indent=2))
    print(json.dumps(comparisons, ensure_ascii=False, indent=2))

    if not args.keep_kbs:
        for backend in backends:
            await _delete_knowledge_base(str(metadatas[backend]["knowledge_base"]["id"]))
        print("Deleted generated comparison knowledge bases.")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
