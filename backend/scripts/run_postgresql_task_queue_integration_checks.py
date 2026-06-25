"""运行 PostgreSQL 任务队列集成检查。

重点验证任务入队、认领、重试、租约恢复和事件记录在 PostgreSQL 下是否正常。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent

DEFAULT_OUTPUT_PATH = ROOT / "evals" / "results" / "postgresql-task-queue-integration-summary.json"
DEFAULT_COMPOSE_FILES = "docker-compose.yml;docker-compose.postgresql-canary.yml"
DEFAULT_TEST_TARGETS = [
    "tests/test_task_queue_sql_integration.py",
    "tests/test_task_worker_sql_recovery_integration.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SQL task queue integration suite inside the PostgreSQL canary container."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to write the integration summary JSON.",
    )
    parser.add_argument(
        "--compose-files",
        default=DEFAULT_COMPOSE_FILES,
        help="Compose files used for the canary environment. Use ';' as the separator on Windows.",
    )
    parser.add_argument(
        "--api-service",
        default="backend-api",
        help="Compose service name used to run pytest.",
    )
    parser.add_argument(
        "test_targets",
        nargs="*",
        help="Optional pytest targets. Defaults to the SQL task queue integration suite.",
    )
    return parser.parse_args()


def _docker_compose_command(*args: str) -> list[str]:
    return ["docker", "compose", *args]


def _run_command(*, command: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=check,
        text=True,
        capture_output=True,
    )


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


def parse_pytest_quiet_output(stdout: str) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for label in ("passed", "failed", "skipped", "error", "errors", "xfailed", "xpassed", "deselected", "warnings"):
        match = re.search(rf"(?P<count>\d+)\s+{label}\b", stdout)
        if match:
            normalized_label = "errors" if label == "error" else label
            counts[normalized_label] = int(match.group("count"))

    duration_seconds: float | None = None
    duration_match = re.search(r"\bin\s+(?P<seconds>\d+(?:\.\d+)?)s\b", stdout)
    if duration_match:
        duration_seconds = float(duration_match.group("seconds"))

    return {
        "counts": counts,
        "duration_seconds": duration_seconds,
    }


def _run_pytest_in_container(*, service: str, test_targets: list[str]) -> dict[str, Any]:
    command = _docker_compose_command(
        "exec",
        "-T",
        service,
        "python",
        "-m",
        "pytest",
        *test_targets,
        "-q",
    )
    result = _run_command(command=command, cwd=PROJECT_ROOT, check=False)
    parsed = parse_pytest_quiet_output(result.stdout)
    return {
        "service": service,
        "command": command,
        "test_targets": test_targets,
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "parsed": parsed,
    }


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


def build_integration_summary(
    *,
    api_env: dict[str, Any],
    pytest_result: dict[str, Any],
    compose_files: str,
) -> dict[str, Any]:
    parsed = pytest_result.get("parsed") or {}
    counts = parsed.get("counts") or {}
    env_ok = (
        str(api_env.get("normalized_database_dialect") or "").strip().lower() == "postgresql"
        and str(api_env.get("normalized_task_queue_backend") or "").strip().lower() == "sql"
        and bool(api_env.get("has_psycopg"))
    )
    tests_ok = (
        int(pytest_result.get("exit_code") or 0) == 0
        and int(counts.get("failed") or 0) == 0
        and int(counts.get("errors") or 0) == 0
        and int(counts.get("passed") or 0) > 0
    )
    return {
        "run_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "compose_files": compose_files,
        "container_env": {
            "backend_api": api_env,
        },
        "checks": {
            "pytest": pytest_result,
        },
        "tests_ok": tests_ok,
        "canary_ok": env_ok and tests_ok,
    }


def main() -> int:
    args = parse_args()
    test_targets = list(args.test_targets or DEFAULT_TEST_TARGETS)

    with _compose_env(args.compose_files):
        api_env = _load_container_env(service=args.api_service)
        pytest_result = _run_pytest_in_container(service=args.api_service, test_targets=test_targets)

    summary = build_integration_summary(
        api_env=api_env,
        pytest_result=pytest_result,
        compose_files=args.compose_files,
    )
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved PostgreSQL task queue integration summary to {output_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["canary_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
