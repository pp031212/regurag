from __future__ import annotations

from scripts.run_postgresql_canary_checks import build_canary_summary


def _base_env() -> dict:
    return {
        "DATABASE_URL": "postgresql+psycopg://regurag:regurag123@postgresql:5432/regurag",
        "TASK_QUEUE_BACKEND": "sql",
        "normalized_database_dialect": "postgresql",
        "normalized_task_queue_backend": "sql",
        "has_psycopg": True,
    }


def _base_metadata_check() -> dict:
    return {
        "database_url": "postgresql+psycopg://regurag:regurag123@postgresql:5432/regurag",
        "dialect": "postgresql",
        "driver": "postgresql+psycopg",
        "database_name": "regurag",
        "required_tables": [
            "knowledge_bases",
            "documents",
            "tasks",
            "task_events",
            "conversations",
            "messages",
            "message_contexts",
        ],
        "existing_tables": [
            "conversations",
            "documents",
            "knowledge_bases",
            "message_contexts",
            "messages",
            "task_events",
            "tasks",
        ],
        "missing_required_tables": [],
        "connection_ok": True,
        "schema_ready": True,
    }


def _base_retry_smoke() -> dict:
    return {
        "final_status": "completed",
        "retry_event_count": 1,
    }


def _base_concurrency_smoke() -> dict:
    return {
        "retried_task_final_status": "completed",
        "companion_task_final_status": "completed",
    }


def _base_mixed_smoke() -> dict:
    return {
        "retry_task_final_status": "completed",
        "stale_task_final_status": "completed",
        "stale_reclaimed": True,
    }


def test_build_canary_summary_marks_canary_ok_when_all_checks_pass() -> None:
    summary = build_canary_summary(
        api_env=_base_env(),
        worker_env=_base_env(),
        metadata_check=_base_metadata_check(),
        retry_smoke=_base_retry_smoke(),
        concurrency_smoke=_base_concurrency_smoke(),
        mixed_recovery_smoke=_base_mixed_smoke(),
        compose_files="docker-compose.yml;docker-compose.postgresql-canary.yml",
    )

    assert summary["metadata_check"]["dialect"] == "postgresql"
    assert summary["checks"]["retry_smoke"]["final_status"] == "completed"
    assert summary["canary_ok"] is True


def test_build_canary_summary_rejects_wrong_queue_backend() -> None:
    worker_env = _base_env()
    worker_env["normalized_task_queue_backend"] = "redis"

    summary = build_canary_summary(
        api_env=_base_env(),
        worker_env=worker_env,
        metadata_check=_base_metadata_check(),
        retry_smoke=_base_retry_smoke(),
        concurrency_smoke=_base_concurrency_smoke(),
        mixed_recovery_smoke=_base_mixed_smoke(),
        compose_files="docker-compose.yml;docker-compose.postgresql-canary.yml",
    )

    assert summary["canary_ok"] is False


def test_build_canary_summary_rejects_failed_metadata_check() -> None:
    metadata_check = _base_metadata_check()
    metadata_check["schema_ready"] = False

    summary = build_canary_summary(
        api_env=_base_env(),
        worker_env=_base_env(),
        metadata_check=metadata_check,
        retry_smoke=_base_retry_smoke(),
        concurrency_smoke=_base_concurrency_smoke(),
        mixed_recovery_smoke=_base_mixed_smoke(),
        compose_files="docker-compose.yml;docker-compose.postgresql-canary.yml",
    )

    assert summary["canary_ok"] is False
