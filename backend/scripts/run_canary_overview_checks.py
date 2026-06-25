"""汇总运行 Milvus/PostgreSQL canary 检查。

可选择重新执行子脚本或复用已有 summary，最终写出统一 canary overview 报告。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
DEFAULT_OUTPUT_PATH = ROOT / "evals" / "results" / "canary-overview-summary.json"

CHECK_DEFINITIONS: dict[str, dict[str, str]] = {
    "milvus_canary": {
        "script": "scripts/run_milvus_canary_checks.py",
        "summary": "evals/results/milvus-canary-checks-summary.json",
    },
    "postgresql_canary": {
        "script": "scripts/run_postgresql_canary_checks.py",
        "summary": "evals/results/postgresql-canary-checks-summary.json",
    },
    "postgresql_task_queue_integration": {
        "script": "scripts/run_postgresql_task_queue_integration_checks.py",
        "summary": "evals/results/postgresql-task-queue-integration-summary.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Milvus and PostgreSQL canary checks and write a unified summary."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to write the unified canary summary JSON.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing per-check summaries instead of rerunning the child scripts.",
    )
    parser.add_argument(
        "--skip-milvus",
        action="store_true",
        help="Skip the Milvus canary check.",
    )
    parser.add_argument(
        "--skip-postgresql",
        action="store_true",
        help="Skip the PostgreSQL canary check.",
    )
    parser.add_argument(
        "--skip-postgresql-task-queue",
        action="store_true",
        help="Skip the PostgreSQL task queue integration check.",
    )
    parser.add_argument(
        "--child-timeout-seconds",
        type=int,
        default=900,
        help="Per-child timeout when executing individual canary scripts.",
    )
    return parser.parse_args()


def _log_progress(message: str) -> None:
    print(f"[canary-overview] {message}", flush=True)


def _summary_path(key: str) -> Path:
    return ROOT / CHECK_DEFINITIONS[key]["summary"]


def _script_path(key: str) -> Path:
    return ROOT / CHECK_DEFINITIONS[key]["script"]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _detect_milvus_failed_stage(summary: dict[str, Any]) -> str | None:
    if summary.get("canary_ok"):
        return None
    readiness = summary.get("readiness") or {}
    if not readiness.get("can_roll_out", False):
        return "readiness"
    api_env = ((summary.get("container_env") or {}).get("backend_api") or {}).get("VECTOR_STORE_BACKEND")
    worker_env = ((summary.get("container_env") or {}).get("backend_worker") or {}).get("VECTOR_STORE_BACKEND")
    if api_env != "milvus" or worker_env != "milvus":
        return "container_env"
    retrieval = ((summary.get("checks") or {}).get("retrieval_canary") or {}).get("summary") or {}
    if float(retrieval.get("retrieval_hit_rate") or 0.0) < 1.0:
        return "retrieval_canary"
    if float(retrieval.get("final_context_hit_rate") or 0.0) < 1.0:
        return "retrieval_canary"
    if float(retrieval.get("citation_hit_rate") or 0.0) < 1.0:
        return "retrieval_canary"
    if ((summary.get("checks") or {}).get("retry_smoke") or {}).get("final_status") != "completed":
        return "retry_smoke"
    concurrency = (summary.get("checks") or {}).get("concurrency_smoke") or {}
    if concurrency.get("retried_task_final_status") != "completed":
        return "concurrency_smoke"
    if concurrency.get("companion_task_final_status") != "completed":
        return "concurrency_smoke"
    mixed = (summary.get("checks") or {}).get("mixed_recovery_smoke") or {}
    if mixed.get("retry_task_final_status") != "completed":
        return "mixed_recovery_smoke"
    if mixed.get("stale_task_final_status") != "completed":
        return "mixed_recovery_smoke"
    if not mixed.get("stale_reclaimed"):
        return "mixed_recovery_smoke"
    return "unknown"


def _detect_postgresql_failed_stage(summary: dict[str, Any]) -> str | None:
    if summary.get("canary_ok"):
        return None
    metadata = summary.get("metadata_check") or {}
    if not metadata.get("connection_ok", False) or not metadata.get("schema_ready", False):
        return "metadata_check"
    api_env = (summary.get("container_env") or {}).get("backend_api") or {}
    worker_env = (summary.get("container_env") or {}).get("backend_worker") or {}
    if (
        api_env.get("normalized_database_dialect") != "postgresql"
        or api_env.get("normalized_task_queue_backend") != "sql"
        or not api_env.get("has_psycopg")
        or worker_env.get("normalized_database_dialect") != "postgresql"
        or worker_env.get("normalized_task_queue_backend") != "sql"
        or not worker_env.get("has_psycopg")
    ):
        return "container_env"
    if ((summary.get("checks") or {}).get("retry_smoke") or {}).get("final_status") != "completed":
        return "retry_smoke"
    concurrency = (summary.get("checks") or {}).get("concurrency_smoke") or {}
    if concurrency.get("retried_task_final_status") != "completed":
        return "concurrency_smoke"
    if concurrency.get("companion_task_final_status") != "completed":
        return "concurrency_smoke"
    mixed = (summary.get("checks") or {}).get("mixed_recovery_smoke") or {}
    if mixed.get("retry_task_final_status") != "completed":
        return "mixed_recovery_smoke"
    if mixed.get("stale_task_final_status") != "completed":
        return "mixed_recovery_smoke"
    if not mixed.get("stale_reclaimed"):
        return "mixed_recovery_smoke"
    return "unknown"


def _detect_postgresql_task_queue_failed_stage(summary: dict[str, Any]) -> str | None:
    if summary.get("canary_ok"):
        return None
    api_env = (summary.get("container_env") or {}).get("backend_api") or {}
    if (
        api_env.get("normalized_database_dialect") != "postgresql"
        or api_env.get("normalized_task_queue_backend") != "sql"
        or not api_env.get("has_psycopg")
    ):
        return "container_env"
    return "pytest"


def detect_failed_stage(key: str, summary: dict[str, Any]) -> str | None:
    detectors = {
        "milvus_canary": _detect_milvus_failed_stage,
        "postgresql_canary": _detect_postgresql_failed_stage,
        "postgresql_task_queue_integration": _detect_postgresql_task_queue_failed_stage,
    }
    return detectors[key](summary)


def _run_check(key: str, *, reuse_existing: bool, child_timeout_seconds: int) -> dict[str, Any]:
    script_path = _script_path(key)
    summary_path = _summary_path(key)

    if reuse_existing:
        if not summary_path.exists():
            raise FileNotFoundError(f"Summary not found for {key}: {summary_path}")
        summary = _load_json(summary_path)
        return {
            "key": key,
            "script_path": str(script_path.resolve()),
            "summary_path": str(summary_path.resolve()),
            "executed": False,
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "summary": summary,
            "canary_ok": bool(summary.get("canary_ok")),
            "failed_stage": detect_failed_stage(key, summary),
        }

    _log_progress(f"running {key}")
    command = [sys.executable, str(script_path)]
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            text=True,
            capture_output=True,
            timeout=child_timeout_seconds,
        )
        timed_out = False
        exit_code = result.returncode
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = 124
        stdout = (exc.stdout or "").strip()
        stderr = (exc.stderr or "").strip()

    summary = _load_json(summary_path) if summary_path.exists() else {}
    canary_ok = bool(summary.get("canary_ok")) if summary and not timed_out else False
    return {
        "key": key,
        "script_path": str(script_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "executed": True,
        "timed_out": timed_out,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "summary": summary,
        "canary_ok": canary_ok,
        "failed_stage": (
            "timeout" if timed_out else detect_failed_stage(key, summary) if summary else "summary_missing"
        ),
    }


def build_overview_summary(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    overall_ok = True
    summary_paths: dict[str, str] = {}

    for key, result in results.items():
        checks[key] = {
            "script_path": result["script_path"],
            "summary_path": result["summary_path"],
            "executed": result["executed"],
            "timed_out": result.get("timed_out", False),
            "exit_code": result["exit_code"],
            "canary_ok": result["canary_ok"],
            "failed_stage": result["failed_stage"],
            "summary_run_at": (result.get("summary") or {}).get("run_at"),
        }
        summary_paths[key] = result["summary_path"]
        overall_ok = overall_ok and bool(result["canary_ok"])

    return {
        "run_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "overall_ok": overall_ok,
        "milvus_canary_ok": checks.get("milvus_canary", {}).get("canary_ok"),
        "postgresql_canary_ok": checks.get("postgresql_canary", {}).get("canary_ok"),
        "postgresql_task_queue_integration_ok": checks.get("postgresql_task_queue_integration", {}).get("canary_ok"),
        "summary_paths": summary_paths,
        "checks": checks,
    }


def main() -> int:
    args = parse_args()
    requested_checks: list[str] = []
    if not args.skip_milvus:
        requested_checks.append("milvus_canary")
    if not args.skip_postgresql:
        requested_checks.append("postgresql_canary")
    if not args.skip_postgresql_task_queue:
        requested_checks.append("postgresql_task_queue_integration")

    if not requested_checks:
        raise ValueError("No checks selected.")

    results = {
        key: _run_check(key, reuse_existing=args.reuse_existing, child_timeout_seconds=args.child_timeout_seconds)
        for key in requested_checks
    }
    summary = build_overview_summary(results)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved canary overview summary to {output_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["overall_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
