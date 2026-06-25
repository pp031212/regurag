"""运行向量库完整灰度回归。

脚本会临时导入 fixture 到多个后端，串联检索对比、真实问答评测和 shadow graph 对比，
最后生成候选后端的 rollout summary。
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.evals.eval_report_compare import compare_eval_reports
from app.evals.shadow_compare import (
    build_output_payload,
    load_eval_cases as load_shadow_eval_cases,
    run_shadow_compare_eval,
    select_eval_cases as select_shadow_eval_cases,
)
from app.evals.vector_store_regression import build_vector_store_regression_summary
from app.repositories.metadata_repository import MetadataRepository
from app.services import rag_service as rag_service_module
from app.services.ingest_service import IngestService
from app.services.knowledge_base_service import KnowledgeBaseService
from app.services.rag_service import RAGService
try:
    from compare_real_vector_store_retrieval import (
        _backend_label,
        _upload_documents,
        _build_backend_report as _build_retrieval_report,
        _compare_reports as _compare_retrieval_reports,
        _delete_knowledge_base,
        _set_runtime_env,
    )
    from eval_rag import EvalCase, EvalCaseResult, build_summary, load_eval_cases, run_case, select_eval_cases
except ModuleNotFoundError:
    from scripts.compare_real_vector_store_retrieval import (
        _backend_label,
        _upload_documents,
        _build_backend_report as _build_retrieval_report,
        _compare_reports as _compare_retrieval_reports,
        _delete_knowledge_base,
        _set_runtime_env,
    )
    from scripts.eval_rag import EvalCase, EvalCaseResult, build_summary, load_eval_cases, run_case, select_eval_cases


DEFAULT_BACKENDS = ("chroma", "milvus")
STAGES = ("retrieval", "live_eval", "shadow")
SUMMARY_REQUIRED_STAGES = ("retrieval", "shadow")
STAGE_LABEL_SUFFIXES = {
    "retrieval": "retrieval",
    "live_eval": "eval",
    "shadow": "shadow",
}


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Run fixed vector store regression across retrieval-only, live eval, and shadow smoke, then write a compact rollout summary."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "evals" / "rag_eval_real_kb_samples.jsonl",
        help="Path to the JSONL evaluation dataset used by retrieval and live eval runs.",
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
        help="Fixture documents imported into temporary comparison knowledge bases.",
    )
    parser.add_argument(
        "--config-profile",
        default="rules_cn",
        help="Config profile used for every backend run.",
    )
    parser.add_argument(
        "--backend",
        action="append",
        default=[],
        help="Vector store backend to include. Defaults to chroma + milvus.",
    )
    parser.add_argument(
        "--baseline-backend",
        default="chroma",
        help="Backend used as the rollout baseline.",
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
        help="Top-k retrieve value passed to all runs.",
    )
    parser.add_argument(
        "--top-k-rerank",
        type=int,
        default=3,
        help="Top-k rerank value passed to all runs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N retrieval/eval cases after case-id filtering.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Run only the specified retrieval/eval case id. Can be repeated.",
    )
    parser.add_argument(
        "--enable-retrieval-rewrite",
        action="store_true",
        help="Enable query rewrite for retrieval-only parity checks.",
    )
    parser.add_argument(
        "--disable-rewrite",
        action="store_true",
        help="Disable query rewrite for live eval and shadow smoke.",
    )
    parser.add_argument(
        "--disable-rerank",
        action="store_true",
        help="Disable MMR and reranker for all runs.",
    )
    parser.add_argument(
        "--skip-live-eval",
        action="store_true",
        help="Skip live eval even if model credentials are available.",
    )
    parser.add_argument(
        "--live-eval-batch-size",
        type=int,
        default=4,
        help="Run live eval in batches of N cases and write partial reports after each batch. Use 0 to disable batching.",
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=STAGES,
        default=[],
        help="Only execute the specified stage. Can be repeated. Defaults to retrieval + live_eval + shadow.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing stage reports under the same label-prefix when available instead of rerunning that stage.",
    )
    parser.add_argument(
        "--shadow-dataset",
        type=Path,
        default=ROOT / "evals" / "rag_eval_set.jsonl",
        help="Path to the shadow compare dataset.",
    )
    parser.add_argument(
        "--shadow-knowledge-base-id",
        default=settings.default_knowledge_base_id,
        help="Knowledge base id used for shadow smoke.",
    )
    parser.add_argument(
        "--shadow-limit",
        type=int,
        default=4,
        help="Only run the first N shadow cases after case-id filtering.",
    )
    parser.add_argument(
        "--shadow-case-id",
        action="append",
        default=[],
        help="Run only the specified shadow case id. Can be repeated.",
    )
    parser.add_argument(
        "--shadow-answer-style",
        choices=("concise", "structured"),
        default="concise",
        help="Answer style used for shadow smoke.",
    )
    parser.add_argument(
        "--shadow-retrieval-backend",
        choices=("legacy", "langchain_chroma", "langchain_milvus"),
        default=None,
        help="Shadow graph retrieval backend. Defaults to current Settings value.",
    )
    parser.add_argument(
        "--knowledge-base-subject",
        default="和鸣教育管理制度",
        help="Subject used when creating temporary comparison knowledge bases.",
    )
    parser.add_argument(
        "--knowledge-base-domain",
        default="training_management",
        help="Domain used when creating temporary comparison knowledge bases.",
    )
    parser.add_argument(
        "--name-prefix",
        default="Vector Store Regression",
        help="Knowledge base name prefix for temporary comparison knowledge bases.",
    )
    parser.add_argument(
        "--label-prefix",
        default="vector-store-regression",
        help="Output file label prefix written under backend/evals/results/.",
    )
    parser.add_argument(
        "--keep-kbs",
        action="store_true",
        help="Keep generated temporary comparison knowledge bases instead of deleting them after the run.",
    )
    parser.add_argument(
        "--cleanup-timeout-seconds",
        type=float,
        default=20.0,
        help="Best-effort timeout budget for deleting each generated knowledge base after reports are written.",
    )
    return parser.parse_args()


def _normalize_backends(raw_backends: list[str], baseline_backend: str) -> tuple[str, ...]:
    backends = tuple(
        dict.fromkeys(
            _backend_label(item)
            for item in (raw_backends or list(DEFAULT_BACKENDS))
            if item and item.strip()
        )
    )
    if not backends:
        raise ValueError("At least one backend is required")
    if baseline_backend not in backends:
        raise ValueError("--baseline-backend must be included in --backend")
    return backends


def _normalize_stages(raw_stages: list[str]) -> tuple[str, ...]:
    stages = tuple(dict.fromkeys(stage for stage in (raw_stages or list(STAGES)) if stage in STAGES))
    if not stages:
        raise ValueError("At least one stage is required")
    return stages


def _normalize_batch_size(batch_size: int) -> int | None:
    if batch_size < 0:
        raise ValueError("--live-eval-batch-size must be >= 0")
    return None if batch_size == 0 else batch_size


def _has_live_eval_credentials() -> bool:
    settings = get_settings()

    def usable(value: str | None) -> bool:
        normalized = str(value or "").strip().lower()
        return bool(normalized) and normalized not in {"replace-me", "placeholder", "your-api-key"}

    return usable(settings.openai_api_key) and usable(settings.rewrite_api_key)


def _stage_label(label_prefix: str, stage: str) -> str:
    return f"{label_prefix}-{STAGE_LABEL_SUFFIXES[stage]}"


def _stage_report_path(output_dir: Path, label_prefix: str, stage: str, backend: str) -> Path:
    return output_dir / f"{_stage_label(label_prefix, stage)}-{backend}.json"


def _stage_compare_path(output_dir: Path, label_prefix: str, stage: str) -> Path:
    return output_dir / f"{_stage_label(label_prefix, stage)}-compare.json"


def _load_json_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON payload: {path}")
    return payload


def _load_existing_stage_reports(
    *,
    output_dir: Path,
    label_prefix: str,
    stage: str,
    backends: tuple[str, ...],
) -> tuple[dict[str, dict[str, Any]], dict[str, Path]] | None:
    report_paths = {
        backend: _stage_report_path(output_dir, label_prefix, stage, backend)
        for backend in backends
    }
    if not all(path.exists() for path in report_paths.values()):
        return None
    return (
        {backend: _load_json_payload(path) for backend, path in report_paths.items()},
        report_paths,
    )


def _coerce_eval_result(payload: dict[str, Any]) -> EvalCaseResult:
    return EvalCaseResult(
        id=str(payload.get("id") or ""),
        question=str(payload.get("question") or ""),
        category=str(payload.get("category") or "unknown"),
        answer_mode=str(payload.get("answer_mode") or "grounded"),
        answer=str(payload.get("answer") or ""),
        retrieval_hit=bool(payload.get("retrieval_hit")),
        final_context_hit=bool(payload.get("final_context_hit")),
        citation_hit=bool(payload.get("citation_hit")),
        answer_hit=bool(payload.get("answer_hit")),
        answer_hit_ratio=round(float(payload.get("answer_hit_ratio") or 0.0), 4),
        matched_answer_keywords=[str(item) for item in list(payload.get("matched_answer_keywords") or [])],
        missing_answer_keywords=[str(item) for item in list(payload.get("missing_answer_keywords") or [])],
        matched_context_keywords=[str(item) for item in list(payload.get("matched_context_keywords") or [])],
        error_type=str(payload.get("error_type") or "needs_review"),
        debug=dict(payload.get("debug") or {}),
        citations=[
            dict(item) if isinstance(item, dict) else {"content": str(item)}
            for item in list(payload.get("citations") or [])
        ],
    )


def _ordered_eval_result_payloads(
    *,
    selected_case_ids: list[str],
    results_by_case_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        dict(results_by_case_id[case_id])
        for case_id in selected_case_ids
        if case_id in results_by_case_id
    ]


def _build_eval_report_payload(
    *,
    dataset_path: Path,
    fixture_paths: list[Path],
    config_profile: str,
    vector_store_backend: str,
    milvus_uri: str,
    top_k_retrieve: int,
    top_k_rerank: int,
    enable_rewrite: bool,
    enable_rerank: bool,
    knowledge_base_id: str,
    selected_case_ids: list[str],
    results_by_case_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ordered_results = _ordered_eval_result_payloads(
        selected_case_ids=selected_case_ids,
        results_by_case_id=results_by_case_id,
    )
    summary = build_summary([_coerce_eval_result(item) for item in ordered_results])
    return {
        "run_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "knowledge_base_id": knowledge_base_id,
        "dataset": str(dataset_path),
        "config": {
            "config_profile": config_profile,
            "vector_store_backend": _backend_label(vector_store_backend),
            "vector_store_milvus_uri": milvus_uri if _backend_label(vector_store_backend) == "milvus" else None,
            "top_k_retrieve": top_k_retrieve,
            "top_k_rerank": top_k_rerank,
            "enable_rewrite": enable_rewrite,
            "enable_rerank": enable_rerank,
            "fixture_paths": [str(path) for path in fixture_paths],
            "selected_case_ids": selected_case_ids,
        },
        "summary": summary,
        "results": ordered_results,
    }


def _load_partial_eval_results(
    *,
    report_path: Path,
    selected_case_ids: list[str],
) -> dict[str, dict[str, Any]]:
    if not report_path.exists():
        return {}
    payload = _load_json_payload(report_path)
    results = payload.get("results")
    if not isinstance(results, list):
        return {}
    allowed_case_ids = set(selected_case_ids)
    loaded: dict[str, dict[str, Any]] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        case_id = str(item.get("id") or "").strip()
        if case_id and case_id in allowed_case_ids:
            loaded[case_id] = dict(item)
    return loaded


def _chunk_eval_cases(cases: list[EvalCase], batch_size: int | None) -> list[list[EvalCase]]:
    if batch_size is None or batch_size >= len(cases):
        return [cases]
    return [cases[index:index + batch_size] for index in range(0, len(cases), batch_size)]


async def _build_batched_eval_report(
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
    batch_size: int | None,
    report_path: Path,
    reuse_existing: bool,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    selected_cases = select_eval_cases(load_eval_cases(dataset_path), case_ids=case_ids, limit=limit)
    selected_case_ids = [case.id for case in selected_cases]
    existing_results_by_case_id = (
        _load_partial_eval_results(report_path=report_path, selected_case_ids=selected_case_ids)
        if reuse_existing
        else {}
    )
    pending_cases = [case for case in selected_cases if case.id not in existing_results_by_case_id]
    if not pending_cases and existing_results_by_case_id:
        existing_payload = _load_json_payload(report_path)
        return existing_payload, existing_payload.get("metadata") or {}, True

    _set_runtime_env(
        profile=config_profile,
        vector_store_backend=vector_store_backend,
        milvus_uri=milvus_uri,
        milvus_token=milvus_token,
    )
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

    merged_results_by_case_id = dict(existing_results_by_case_id)
    batch_case_ids: list[list[str]] = []
    for batch_cases in _chunk_eval_cases(pending_cases, batch_size):
        batch_case_ids.append([case.id for case in batch_cases])
        for case in batch_cases:
            case_result = await run_case(
                service=rag_service,
                case=case,
                knowledge_base_id=knowledge_base["id"],
                top_k_retrieve=top_k_retrieve,
                top_k_rerank=top_k_rerank,
                enable_rewrite=enable_rewrite,
                enable_rerank=enable_rerank,
            )
            merged_results_by_case_id[case.id] = asdict(case_result)

        partial_report = _build_eval_report_payload(
            dataset_path=dataset_path,
            fixture_paths=fixture_paths,
            config_profile=config_profile,
            vector_store_backend=vector_store_backend,
            milvus_uri=milvus_uri,
            top_k_retrieve=top_k_retrieve,
            top_k_rerank=top_k_rerank,
            enable_rewrite=enable_rewrite,
            enable_rerank=enable_rerank,
            knowledge_base_id=str(knowledge_base["id"]),
            selected_case_ids=selected_case_ids,
            results_by_case_id=merged_results_by_case_id,
        )
        partial_metadata = {
            "knowledge_base": knowledge_base,
            "uploaded_documents": uploaded_documents,
            "task": task_result,
            "knowledge_base_status": repository.get_knowledge_base(knowledge_base["id"]),
            "batch_case_ids": batch_case_ids,
            "completed_case_ids": sorted(merged_results_by_case_id),
            "resumed_case_ids": sorted(existing_results_by_case_id),
        }
        report_path.write_text(
            json.dumps({"metadata": partial_metadata, **partial_report}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"[live_eval] {backend_label} wrote partial report with "
            f"{len(merged_results_by_case_id)}/{len(selected_case_ids)} cases"
        )

    final_report = _build_eval_report_payload(
        dataset_path=dataset_path,
        fixture_paths=fixture_paths,
        config_profile=config_profile,
        vector_store_backend=vector_store_backend,
        milvus_uri=milvus_uri,
        top_k_retrieve=top_k_retrieve,
        top_k_rerank=top_k_rerank,
        enable_rewrite=enable_rewrite,
        enable_rerank=enable_rerank,
        knowledge_base_id=str(knowledge_base["id"]),
        selected_case_ids=selected_case_ids,
        results_by_case_id=merged_results_by_case_id,
    )
    final_metadata = {
        "knowledge_base": knowledge_base,
        "uploaded_documents": uploaded_documents,
        "task": task_result,
        "knowledge_base_status": repository.get_knowledge_base(knowledge_base["id"]),
        "batch_case_ids": batch_case_ids,
        "completed_case_ids": sorted(merged_results_by_case_id),
        "resumed_case_ids": sorted(existing_results_by_case_id),
    }
    return final_report, final_metadata, False


def _stage_status(
    *,
    stage: str,
    requested_stages: tuple[str, ...],
    reused: bool,
    executed: bool,
    available: bool,
) -> dict[str, Any]:
    return {
        "requested": stage in requested_stages,
        "reused_existing": reused,
        "executed": executed,
        "available": available,
    }


def _set_shadow_runtime_env(
    *,
    profile: str,
    vector_store_backend: str,
    milvus_uri: str,
    milvus_token: str | None,
    shadow_retrieval_backend: str | None,
) -> None:
    _set_runtime_env(
        profile=profile,
        vector_store_backend=vector_store_backend,
        milvus_uri=milvus_uri,
        milvus_token=milvus_token,
    )
    if shadow_retrieval_backend:
        os.environ["SHADOW_RETRIEVAL_BACKEND"] = shadow_retrieval_backend
    get_settings.cache_clear()
    rag_service_module._settings.cache_clear()
    rag_service_module.get_default_pipeline_registry().clear()


async def _build_shadow_report(
    *,
    backend: str,
    dataset_path: Path,
    knowledge_base_id: str,
    top_k_retrieve: int,
    top_k_rerank: int,
    enable_rewrite: bool,
    enable_rerank: bool,
    answer_style: str,
    case_ids: list[str],
    limit: int | None,
    config_profile: str,
    milvus_uri: str,
    milvus_token: str | None,
    shadow_retrieval_backend: str | None,
) -> dict[str, object]:
    _set_shadow_runtime_env(
        profile=config_profile,
        vector_store_backend=backend,
        milvus_uri=milvus_uri,
        milvus_token=milvus_token,
        shadow_retrieval_backend=shadow_retrieval_backend,
    )
    settings = get_settings()
    cases = load_shadow_eval_cases(dataset_path)
    selected_cases = select_shadow_eval_cases(cases, case_ids=case_ids, limit=limit)
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
        dataset=dataset_path,
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
    args = parse_args()
    backends = _normalize_backends(args.backend, _backend_label(args.baseline_backend))
    requested_stages = _normalize_stages(args.stage)
    live_eval_batch_size = _normalize_batch_size(args.live_eval_batch_size)
    baseline_backend = _backend_label(args.baseline_backend)
    output_dir = ROOT / "evals" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    retrieval_reports: dict[str, dict[str, object]] = {}
    retrieval_comparisons: dict[str, dict[str, object]] = {}
    eval_reports: dict[str, dict[str, object]] = {}
    eval_comparisons: dict[str, dict[str, object]] = {}
    shadow_reports: dict[str, dict[str, object]] = {}
    report_paths: dict[str, dict[str, Path]] = {backend: {} for backend in backends}
    generated_kb_ids: set[str] = set()
    stage_status: dict[str, dict[str, Any]] = {}

    try:
        fixture_paths = [path.resolve() for path in args.fixtures]
        dataset_path = args.dataset.resolve()

        retrieval_reused = False
        retrieval_executed = False
        loaded_retrieval = None
        if args.reuse_existing or "retrieval" not in requested_stages:
            loaded_retrieval = _load_existing_stage_reports(
                output_dir=output_dir,
                label_prefix=args.label_prefix,
                stage="retrieval",
                backends=backends,
            )
        if loaded_retrieval is not None and ("retrieval" not in requested_stages or args.reuse_existing):
            retrieval_reports, loaded_paths = loaded_retrieval
            for backend, path in loaded_paths.items():
                report_paths[backend]["retrieval"] = path
            retrieval_reused = True
        elif "retrieval" in requested_stages:
            retrieval_executed = True
            for backend in backends:
                report, metadata = await _build_retrieval_report(
                    dataset_path=dataset_path,
                    fixture_paths=fixture_paths,
                    config_profile=args.config_profile,
                    vector_store_backend=backend,
                    milvus_uri=args.milvus_uri,
                    milvus_token=args.milvus_token,
                    top_k_retrieve=args.top_k_retrieve,
                    top_k_rerank=args.top_k_rerank,
                    enable_rewrite=args.enable_retrieval_rewrite,
                    enable_rerank=not args.disable_rerank,
                    knowledge_base_subject=args.knowledge_base_subject,
                    knowledge_base_domain=args.knowledge_base_domain,
                    name_prefix=f"{args.name_prefix} Retrieval",
                    case_ids=list(args.case_id),
                    limit=args.limit,
                )
                retrieval_reports[backend] = report
                generated_kb_ids.add(str(metadata["knowledge_base"]["id"]))
                report_path = _stage_report_path(output_dir, args.label_prefix, "retrieval", backend)
                report_path.write_text(
                    json.dumps({"metadata": metadata, **report}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                report_paths[backend]["retrieval"] = report_path
        stage_status["retrieval"] = _stage_status(
            stage="retrieval",
            requested_stages=requested_stages,
            reused=retrieval_reused,
            executed=retrieval_executed,
            available=bool(retrieval_reports),
        )

        if retrieval_reports:
            for backend in backends:
                if backend == baseline_backend:
                    continue
                retrieval_comparisons[f"{baseline_backend}_vs_{backend}"] = _compare_retrieval_reports(
                    retrieval_reports[baseline_backend],
                    retrieval_reports[backend],
                )

            retrieval_compare_payload = {
                "run_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "baseline_backend": baseline_backend,
                "backends": list(backends),
                "comparisons": retrieval_comparisons,
                "report_paths": {
                    backend: str(report_paths[backend]["retrieval"].resolve())
                    for backend in backends
                },
            }
            _stage_compare_path(output_dir, args.label_prefix, "retrieval").write_text(
                json.dumps(retrieval_compare_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        live_eval_enabled = (not args.skip_live_eval) and _has_live_eval_credentials()
        eval_reused = False
        eval_executed = False
        loaded_eval = None
        if args.reuse_existing or "live_eval" not in requested_stages or not live_eval_enabled:
            loaded_eval = _load_existing_stage_reports(
                output_dir=output_dir,
                label_prefix=args.label_prefix,
                stage="live_eval",
                backends=backends,
            )
        if loaded_eval is not None and ("live_eval" not in requested_stages or args.reuse_existing or not live_eval_enabled):
            eval_reports, loaded_paths = loaded_eval
            for backend, path in loaded_paths.items():
                report_paths[backend]["eval"] = path
            eval_reused = True
        elif "live_eval" in requested_stages and live_eval_enabled:
            for backend in backends:
                report_path = _stage_report_path(output_dir, args.label_prefix, "live_eval", backend)
                report, metadata, reused_existing_eval = await _build_batched_eval_report(
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
                    name_prefix=f"{args.name_prefix} Eval",
                    case_ids=list(args.case_id),
                    limit=args.limit,
                    batch_size=live_eval_batch_size,
                    report_path=report_path,
                    reuse_existing=args.reuse_existing,
                )
                eval_reused = eval_reused or reused_existing_eval
                eval_executed = eval_executed or (not reused_existing_eval)
                eval_reports[backend] = report
                if metadata.get("knowledge_base") and not reused_existing_eval:
                    generated_kb_ids.add(str(metadata["knowledge_base"]["id"]))
                report_path.write_text(
                    json.dumps({"metadata": metadata, **report}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                report_paths[backend]["eval"] = report_path
        stage_status["live_eval"] = _stage_status(
            stage="live_eval",
            requested_stages=requested_stages,
            reused=eval_reused,
            executed=eval_executed,
            available=bool(eval_reports),
        )

        if eval_reports:
            for backend in backends:
                if backend == baseline_backend:
                    continue
                eval_comparisons[f"{baseline_backend}_vs_{backend}"] = compare_eval_reports(
                    eval_reports[baseline_backend],
                    eval_reports[backend],
                )

            eval_compare_payload = {
                "run_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "baseline_backend": baseline_backend,
                "backends": list(backends),
                "comparisons": eval_comparisons,
                "report_paths": {
                    backend: str(report_paths[backend]["eval"].resolve())
                    for backend in backends
                },
            }
            _stage_compare_path(output_dir, args.label_prefix, "live_eval").write_text(
                json.dumps(eval_compare_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        shadow_reused = False
        shadow_executed = False
        loaded_shadow = None
        if args.reuse_existing or "shadow" not in requested_stages:
            loaded_shadow = _load_existing_stage_reports(
                output_dir=output_dir,
                label_prefix=args.label_prefix,
                stage="shadow",
                backends=backends,
            )
        if loaded_shadow is not None and ("shadow" not in requested_stages or args.reuse_existing):
            shadow_reports, loaded_paths = loaded_shadow
            for backend, path in loaded_paths.items():
                report_paths[backend]["shadow"] = path
            shadow_reused = True
        elif "shadow" in requested_stages:
            shadow_executed = True
            for backend in backends:
                report = await _build_shadow_report(
                    backend=backend,
                    dataset_path=args.shadow_dataset.resolve(),
                    knowledge_base_id=args.shadow_knowledge_base_id,
                    top_k_retrieve=args.top_k_retrieve,
                    top_k_rerank=args.top_k_rerank,
                    enable_rewrite=not args.disable_rewrite,
                    enable_rerank=not args.disable_rerank,
                    answer_style=args.shadow_answer_style,
                    case_ids=list(args.shadow_case_id),
                    limit=args.shadow_limit,
                    config_profile=args.config_profile,
                    milvus_uri=args.milvus_uri,
                    milvus_token=args.milvus_token,
                    shadow_retrieval_backend=args.shadow_retrieval_backend,
                )
                shadow_reports[backend] = report
                report_path = _stage_report_path(output_dir, args.label_prefix, "shadow", backend)
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                report_paths[backend]["shadow"] = report_path
        stage_status["shadow"] = _stage_status(
            stage="shadow",
            requested_stages=requested_stages,
            reused=shadow_reused,
            executed=shadow_executed,
            available=bool(shadow_reports),
        )

        summary = None
        summary_skip_reason = None
        missing_summary_stages = [
            stage for stage in SUMMARY_REQUIRED_STAGES
            if stage == "retrieval" and not retrieval_reports or stage == "shadow" and not shadow_reports
        ]
        if not missing_summary_stages:
            summary = build_vector_store_regression_summary(
                retrieval_reports=retrieval_reports,
                retrieval_comparisons=retrieval_comparisons,
                shadow_reports=shadow_reports,
                baseline_backend=baseline_backend,
                eval_reports=eval_reports or None,
                eval_comparisons=eval_comparisons or None,
                report_paths=report_paths,
            )
        else:
            summary_skip_reason = (
                "Summary requires retrieval and shadow reports for every backend. "
                f"Missing stages: {', '.join(missing_summary_stages)}"
            )
        summary_payload = {
            "run_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "config": {
                "backends": list(backends),
                "baseline_backend": baseline_backend,
                "requested_stages": list(requested_stages),
                "reuse_existing": args.reuse_existing,
                "config_profile": args.config_profile,
                "top_k_retrieve": args.top_k_retrieve,
                "top_k_rerank": args.top_k_rerank,
                "retrieval_enable_rewrite": args.enable_retrieval_rewrite,
                "live_eval_enabled": live_eval_enabled,
                "live_eval_enable_rewrite": not args.disable_rewrite,
                "enable_rerank": not args.disable_rerank,
                "shadow_knowledge_base_id": args.shadow_knowledge_base_id,
                "shadow_dataset": str(args.shadow_dataset.resolve()),
                "shadow_limit": args.shadow_limit,
                "shadow_retrieval_backend": args.shadow_retrieval_backend or get_settings().shadow_retrieval_backend,
            },
            "stage_status": stage_status,
            "summary": summary,
            "summary_skipped_reason": summary_skip_reason,
        }
        summary_path = output_dir / f"{args.label_prefix}-summary.json"
        summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"Saved vector store regression summary to {summary_path}")
        print(json.dumps(stage_status, ensure_ascii=False, indent=2))
        if summary is not None:
            print(json.dumps(summary["backend_summaries"], ensure_ascii=False, indent=2))
            print(json.dumps(summary["diffs_vs_baseline"], ensure_ascii=False, indent=2))
        else:
            print(summary_skip_reason)
        return 0
    finally:
        if generated_kb_ids and not args.keep_kbs:
            for knowledge_base_id in sorted(generated_kb_ids):
                try:
                    await asyncio.wait_for(
                        _delete_knowledge_base(knowledge_base_id),
                        timeout=args.cleanup_timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    print(
                        f"Cleanup timed out for generated knowledge base {knowledge_base_id}; reports were already written.",
                        file=sys.stderr,
                    )
                except Exception as exc:  # pragma: no cover - best effort cleanup path
                    print(
                        f"Cleanup failed for generated knowledge base {knowledge_base_id}: {exc}",
                        file=sys.stderr,
                    )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
