"""运行 Milvus 灰度 canary 检查。

确认容器配置、向量库读写和检索回归样本在 Milvus 后端下满足上线门禁。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_vector_store_rollout_readiness import build_rollout_assessment, load_summary
from scripts.run_docker_task_queue_concurrency_smoke import run_smoke as run_concurrency_smoke
from scripts.run_docker_task_queue_mixed_recovery_smoke import run_smoke as run_mixed_recovery_smoke
from scripts.run_docker_task_queue_retry_smoke import (
    DEFAULT_SAMPLE_FILE,
    PROJECT_ROOT as SMOKE_PROJECT_ROOT,
    run_smoke as run_retry_smoke,
)

DEFAULT_SUMMARY_PATH = ROOT / "evals" / "results" / "vector-store-regression-live-v1-summary.json"
DEFAULT_OUTPUT_PATH = ROOT / "evals" / "results" / "milvus-canary-checks-summary.json"
DEFAULT_CANARY_LABEL = "vector-store-retrieval-compose-canary-milvus"
DEFAULT_COMPOSE_FILES = "docker-compose.yml;docker-compose.milvus-canary.yml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fixed Milvus canary checklist and write a compact JSON summary."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Vector store regression summary used for readiness assessment.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to write the canary summary JSON.",
    )
    parser.add_argument(
        "--compose-files",
        default=DEFAULT_COMPOSE_FILES,
        help="Compose files used for the canary environment. Use ';' as the separator on Windows.",
    )
    parser.add_argument(
        "--api-service",
        default="backend-api",
        help="Compose service name for the backend API container.",
    )
    parser.add_argument(
        "--worker-service",
        default="backend-worker",
        help="Compose service name for the backend worker container.",
    )
    parser.add_argument(
        "--candidate-backend",
        default="milvus",
        help="Candidate backend to validate.",
    )
    parser.add_argument(
        "--retrieval-label",
        default=DEFAULT_CANARY_LABEL,
        help="Label used for the retrieval-only canary report.",
    )
    parser.add_argument(
        "--retrieval-limit",
        type=int,
        default=4,
        help="Retrieval-only canary case limit.",
    )
    parser.add_argument(
        "--mixed-timeout-seconds",
        type=int,
        default=240,
        help="Total timeout passed to the mixed recovery smoke.",
    )
    parser.add_argument(
        "--sample-file",
        type=Path,
        default=DEFAULT_SAMPLE_FILE,
        help="Sample file used by retry/concurrency/mixed recovery smokes.",
    )
    parser.add_argument(
        "--max-live-eval-latency-delta-ms",
        type=float,
        default=None,
        help="Optional latency budget passed to the readiness assessment.",
    )
    return parser.parse_args()


def _candidate_backend(value: str) -> str:
    return value.strip().lower() or "milvus"


def _docker_compose_command(*args: str) -> list[str]:
    return ["docker", "compose", *args]


def _run_command(*, command: list[str], cwd: Path, capture_output: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def _container_env_script() -> str:
    return (
        "import json, os; "
        "print(json.dumps({"
        "'VECTOR_STORE_BACKEND': os.getenv('VECTOR_STORE_BACKEND'), "
        "'VECTOR_STORE_MILVUS_URI': os.getenv('VECTOR_STORE_MILVUS_URI')"
        "}, ensure_ascii=False))"
    )


def _load_container_backend_env(*, service: str) -> dict[str, Any]:
    result = _run_command(
        command=_docker_compose_command("exec", "-T", service, "python", "-c", _container_env_script()),
        cwd=PROJECT_ROOT,
    )
    return json.loads(result.stdout.strip())


def _retrieval_report_path(label: str, backend: str) -> Path:
    return ROOT / "evals" / "results" / f"{label}-{backend}.json"


def _run_retrieval_canary(*, service: str, label: str, retrieval_limit: int) -> dict[str, Any]:
    report_path = _retrieval_report_path(label, "milvus")
    if report_path.exists():
        report_path.unlink()
    _run_command(
        command=_docker_compose_command(
            "exec",
            "-T",
            service,
            "python",
            "scripts/compare_real_vector_store_retrieval.py",
            "--backends",
            "milvus",
            "--milvus-uri",
            "http://milvus-standalone:19530",
            "--label",
            label,
            "--limit",
            str(retrieval_limit),
        ),
        cwd=PROJECT_ROOT,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"Retrieval canary report missing summary: {report_path}")
    return {
        "label": label,
        "report_path": str(report_path.resolve()),
        "summary": summary,
    }


def _smoke_result_to_dict(result: Any) -> dict[str, Any]:
    return dict(asdict(result))


def _run_retry_smoke(*, sample_file: Path) -> dict[str, Any]:
    return _smoke_result_to_dict(
        run_retry_smoke(
            api_base_url="http://127.0.0.1:8000/api/v1",
            sample_file=sample_file,
            keep_knowledge_base=False,
        )
    )


def _run_concurrency_smoke(*, sample_file: Path) -> dict[str, Any]:
    return _smoke_result_to_dict(
        run_concurrency_smoke(
            api_base_url="http://127.0.0.1:8000/api/v1",
            sample_file=sample_file,
            keep_knowledge_bases=False,
        )
    )


def _run_mixed_recovery_smoke(*, sample_file: Path, mixed_timeout_seconds: int) -> dict[str, Any]:
    return _smoke_result_to_dict(
        run_mixed_recovery_smoke(
            sample_file=sample_file,
            keep_knowledge_bases=False,
            total_timeout_seconds=mixed_timeout_seconds,
        )
    )


@contextmanager
def _compose_env(compose_files: str) -> Iterator[None]:
    previous_compose = os.environ.get("COMPOSE_FILE")
    previous_pythonpath = os.environ.get("PYTHONPATH")
    os.environ["COMPOSE_FILE"] = compose_files
    os.environ["PYTHONPATH"] = str(ROOT)
    try:
        yield
    finally:
        if previous_compose is None:
            os.environ.pop("COMPOSE_FILE", None)
        else:
            os.environ["COMPOSE_FILE"] = previous_compose
        if previous_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = previous_pythonpath


def build_canary_summary(
    *,
    readiness: dict[str, Any],
    api_env: dict[str, Any],
    worker_env: dict[str, Any],
    retrieval_canary: dict[str, Any],
    retry_smoke: dict[str, Any],
    concurrency_smoke: dict[str, Any],
    mixed_recovery_smoke: dict[str, Any],
    compose_files: str,
) -> dict[str, Any]:
    return {
        "run_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "compose_files": compose_files,
        "readiness": readiness,
        "container_env": {
            "backend_api": api_env,
            "backend_worker": worker_env,
        },
        "checks": {
            "retrieval_canary": retrieval_canary,
            "retry_smoke": retry_smoke,
            "concurrency_smoke": concurrency_smoke,
            "mixed_recovery_smoke": mixed_recovery_smoke,
        },
        "canary_ok": (
            readiness["can_roll_out"]
            and api_env.get("VECTOR_STORE_BACKEND") == "milvus"
            and worker_env.get("VECTOR_STORE_BACKEND") == "milvus"
            and float(retrieval_canary["summary"].get("retrieval_hit_rate") or 0.0) >= 1.0
            and float(retrieval_canary["summary"].get("final_context_hit_rate") or 0.0) >= 1.0
            and float(retrieval_canary["summary"].get("citation_hit_rate") or 0.0) >= 1.0
            and retry_smoke.get("final_status") == "completed"
            and concurrency_smoke.get("retried_task_final_status") == "completed"
            and concurrency_smoke.get("companion_task_final_status") == "completed"
            and mixed_recovery_smoke.get("retry_task_final_status") == "completed"
            and mixed_recovery_smoke.get("stale_task_final_status") == "completed"
            and bool(mixed_recovery_smoke.get("stale_reclaimed"))
        ),
    }


def main() -> int:
    args = parse_args()
    sample_file = args.sample_file.resolve()
    if not sample_file.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_file}")

    readiness = build_rollout_assessment(
        load_summary(args.summary.resolve()),
        backend=_candidate_backend(args.candidate_backend),
        max_live_eval_latency_delta_ms=args.max_live_eval_latency_delta_ms,
    )

    with _compose_env(args.compose_files):
        api_env = _load_container_backend_env(service=args.api_service)
        worker_env = _load_container_backend_env(service=args.worker_service)
        retrieval_canary = _run_retrieval_canary(
            service=args.api_service,
            label=args.retrieval_label,
            retrieval_limit=args.retrieval_limit,
        )
        retry_smoke = _run_retry_smoke(sample_file=sample_file)
        concurrency_smoke = _run_concurrency_smoke(sample_file=sample_file)
        mixed_recovery_smoke = _run_mixed_recovery_smoke(
            sample_file=sample_file,
            mixed_timeout_seconds=args.mixed_timeout_seconds,
        )

    summary = build_canary_summary(
        readiness=readiness,
        api_env=api_env,
        worker_env=worker_env,
        retrieval_canary=retrieval_canary,
        retry_smoke=retry_smoke,
        concurrency_smoke=concurrency_smoke,
        mixed_recovery_smoke=mixed_recovery_smoke,
        compose_files=args.compose_files,
    )
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved Milvus canary summary to {output_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["canary_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
