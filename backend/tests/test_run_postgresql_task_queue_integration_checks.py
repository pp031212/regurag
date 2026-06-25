from __future__ import annotations

from scripts.run_postgresql_task_queue_integration_checks import (
    build_integration_summary,
    parse_pytest_quiet_output,
)


def _base_env() -> dict:
    return {
        "DATABASE_URL": "postgresql+psycopg://regurag:regurag123@postgresql:5432/regurag",
        "TASK_QUEUE_BACKEND": "sql",
        "normalized_database_dialect": "postgresql",
        "normalized_task_queue_backend": "sql",
        "has_psycopg": True,
    }


def test_parse_pytest_quiet_output_extracts_counts_and_duration() -> None:
    parsed = parse_pytest_quiet_output("11 passed, 2 skipped in 4.82s")

    assert parsed == {
        "counts": {
            "passed": 11,
            "skipped": 2,
        },
        "duration_seconds": 4.82,
    }


def test_build_integration_summary_marks_ok_when_env_and_pytest_are_green() -> None:
    summary = build_integration_summary(
        api_env=_base_env(),
        pytest_result={
            "service": "backend-api",
            "command": ["docker", "compose", "exec", "-T", "backend-api", "python", "-m", "pytest"],
            "test_targets": [
                "tests/test_task_queue_sql_integration.py",
                "tests/test_task_worker_sql_recovery_integration.py",
            ],
            "exit_code": 0,
            "stdout": "11 passed in 4.82s",
            "stderr": "",
            "parsed": {
                "counts": {
                    "passed": 11,
                },
                "duration_seconds": 4.82,
            },
        },
        compose_files="docker-compose.yml;docker-compose.postgresql-canary.yml",
    )

    assert summary["tests_ok"] is True
    assert summary["canary_ok"] is True


def test_build_integration_summary_rejects_failed_pytest_run() -> None:
    summary = build_integration_summary(
        api_env=_base_env(),
        pytest_result={
            "service": "backend-api",
            "command": ["docker", "compose", "exec", "-T", "backend-api", "python", "-m", "pytest"],
            "test_targets": ["tests/test_task_queue_sql_integration.py"],
            "exit_code": 1,
            "stdout": "1 failed, 10 passed in 4.82s",
            "stderr": "",
            "parsed": {
                "counts": {
                    "failed": 1,
                    "passed": 10,
                },
                "duration_seconds": 4.82,
            },
        },
        compose_files="docker-compose.yml;docker-compose.postgresql-canary.yml",
    )

    assert summary["tests_ok"] is False
    assert summary["canary_ok"] is False
