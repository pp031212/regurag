"""运行 PostgreSQL 元数据后端 canary 检查。

用于验证 PostgreSQL 连接、表结构和基础读写行为是否具备替换 MySQL 的条件。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_docker_task_queue_concurrency_smoke import run_smoke as run_concurrency_smoke
from scripts.run_docker_task_queue_mixed_recovery_smoke import run_smoke as run_mixed_recovery_smoke
from scripts.run_docker_task_queue_retry_smoke import DEFAULT_SAMPLE_FILE, run_smoke as run_retry_smoke

DEFAULT_OUTPUT_PATH = ROOT / "evals" / "results" / "postgresql-canary-checks-summary.json"
DEFAULT_COMPOSE_FILES = "docker-compose.yml;docker-compose.postgresql-canary.yml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fixed PostgreSQL canary checklist and write a compact JSON summary."
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
        "--sample-file",
        type=Path,
        default=DEFAULT_SAMPLE_FILE,
        help="Sample file used by retry/concurrency/mixed recovery smokes.",
    )
    parser.add_argument(
        "--mixed-timeout-seconds",
        type=int,
        default=240,
        help="Total timeout passed to the mixed recovery smoke.",
    )
    return parser.parse_args()


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


def _smoke_result_to_dict(result: Any) -> dict[str, Any]:
    return dict(asdict(result))


def _container_env_script() -> str:
    return (
        "import importlib.util, json, os; "
        "from app.core.config import get_settings; "
        "settings = get_settings(); "
        "print(json.dumps({"
        "'DATABASE_URL': os.getenv('DATABASE_URL'), "
        "'TASK_QUEUE_BACKEND': os.getenv('TASK_QUEUE_BACKEND'), "
        "'normalized_database_dialect': settings.normalized_database_dialect, "
        "'normalized_task_queue_backend': settings.normalized_task_queue_backend, "
        "'has_psycopg': importlib.util.find_spec('psycopg') is not None"
        "}, ensure_ascii=False))"
    )


def _load_container_env(*, service: str) -> dict[str, Any]:
    result = _run_command(
        command=_docker_compose_command("exec", "-T", service, "python", "-c", _container_env_script()),
        cwd=PROJECT_ROOT,
    )
    return json.loads(result.stdout.strip())


def _run_metadata_check(*, service: str) -> dict[str, Any]:
    result = _run_command(
        command=_docker_compose_command(
            "exec",
            "-T",
            service,
            "python",
            "scripts/check_metadata_backend.py",
            "--expect-dialect",
            "postgresql",
        ),
        cwd=PROJECT_ROOT,
    )
    return json.loads(result.stdout.strip())


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


def _run_with_retry(
    *,
    label: str,
    func: Callable[[], dict[str, Any]],
    attempts: int = 2,
    delay_seconds: int = 5,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            result = dict(func())
            result["attempts_used"] = attempt
            return result
        except Exception as exc:
            last_error = exc
            print(
                f"[postgresql-canary] {label} attempt {attempt}/{attempts} failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if attempt < attempts:
                time.sleep(delay_seconds)
    assert last_error is not None
    raise last_error


def _log_progress(message: str) -> None:
    print(f"[postgresql-canary] {message}", flush=True)


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
    api_env: dict[str, Any],
    worker_env: dict[str, Any],
    metadata_check: dict[str, Any],
    retry_smoke: dict[str, Any],
    concurrency_smoke: dict[str, Any],
    mixed_recovery_smoke: dict[str, Any],
    compose_files: str,
) -> dict[str, Any]:
    expected_dialect = "postgresql"
    expected_queue_backend = "sql"
    metadata_ok = (
        bool(metadata_check.get("connection_ok"))
        and bool(metadata_check.get("schema_ready"))
        and str(metadata_check.get("dialect") or "").strip().lower() == expected_dialect
    )
    api_env_ok = (
        str(api_env.get("normalized_database_dialect") or "").strip().lower() == expected_dialect
        and str(api_env.get("normalized_task_queue_backend") or "").strip().lower() == expected_queue_backend
        and bool(api_env.get("has_psycopg"))
    )
    worker_env_ok = (
        str(worker_env.get("normalized_database_dialect") or "").strip().lower() == expected_dialect
        and str(worker_env.get("normalized_task_queue_backend") or "").strip().lower() == expected_queue_backend
        and bool(worker_env.get("has_psycopg"))
    )
    return {
        "run_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "compose_files": compose_files,
        "metadata_check": metadata_check,
        "container_env": {
            "backend_api": api_env,
            "backend_worker": worker_env,
        },
        "checks": {
            "retry_smoke": retry_smoke,
            "concurrency_smoke": concurrency_smoke,
            "mixed_recovery_smoke": mixed_recovery_smoke,
        },
        "canary_ok": (
            metadata_ok
            and api_env_ok
            and worker_env_ok
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

    with _compose_env(args.compose_files):
        _log_progress("loading container environment")
        api_env = _load_container_env(service=args.api_service)
        worker_env = _load_container_env(service=args.worker_service)
        _log_progress("running metadata check")
        metadata_check = _run_metadata_check(service=args.api_service)
        _log_progress("running retry smoke")
        retry_smoke = _run_retry_smoke(sample_file=sample_file)
        _log_progress("running concurrency smoke")
        concurrency_smoke = _run_with_retry(
            label="concurrency_smoke",
            func=lambda: _run_concurrency_smoke(sample_file=sample_file),
        )
        _log_progress("running mixed recovery smoke")
        mixed_recovery_smoke = _run_with_retry(
            label="mixed_recovery_smoke",
            func=lambda: _run_mixed_recovery_smoke(
                sample_file=sample_file,
                mixed_timeout_seconds=args.mixed_timeout_seconds,
            ),
        )

    _log_progress("writing summary")
    summary = build_canary_summary(
        api_env=api_env,
        worker_env=worker_env,
        metadata_check=metadata_check,
        retry_smoke=retry_smoke,
        concurrency_smoke=concurrency_smoke,
        mixed_recovery_smoke=mixed_recovery_smoke,
        compose_files=args.compose_files,
    )
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved PostgreSQL canary summary to {output_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["canary_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
