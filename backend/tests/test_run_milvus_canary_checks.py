from __future__ import annotations

from scripts.run_milvus_canary_checks import build_canary_summary


def _base_readiness() -> dict:
    return {
        "can_roll_out": True,
        "ready_for_rollout": True,
        "failed_gates": [],
        "missing_or_unavailable_stages": [],
    }


def _base_env() -> dict:
    return {
        "VECTOR_STORE_BACKEND": "milvus",
        "VECTOR_STORE_MILVUS_URI": "http://milvus-standalone:19530",
    }


def _base_retrieval_canary() -> dict:
    return {
        "label": "vector-store-retrieval-compose-canary-milvus",
        "report_path": "D:/fake/report.json",
        "summary": {
            "retrieval_hit_rate": 1.0,
            "final_context_hit_rate": 1.0,
            "citation_hit_rate": 1.0,
        },
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
        readiness=_base_readiness(),
        api_env=_base_env(),
        worker_env=_base_env(),
        retrieval_canary=_base_retrieval_canary(),
        retry_smoke=_base_retry_smoke(),
        concurrency_smoke=_base_concurrency_smoke(),
        mixed_recovery_smoke=_base_mixed_smoke(),
        compose_files="docker-compose.yml;docker-compose.milvus-canary.yml",
    )

    assert summary["container_env"]["backend_api"]["VECTOR_STORE_BACKEND"] == "milvus"
    assert summary["checks"]["retry_smoke"]["final_status"] == "completed"
    assert summary["canary_ok"] is True


def test_build_canary_summary_rejects_wrong_container_backend() -> None:
    worker_env = _base_env()
    worker_env["VECTOR_STORE_BACKEND"] = "chroma"

    summary = build_canary_summary(
        readiness=_base_readiness(),
        api_env=_base_env(),
        worker_env=worker_env,
        retrieval_canary=_base_retrieval_canary(),
        retry_smoke=_base_retry_smoke(),
        concurrency_smoke=_base_concurrency_smoke(),
        mixed_recovery_smoke=_base_mixed_smoke(),
        compose_files="docker-compose.yml;docker-compose.milvus-canary.yml",
    )

    assert summary["canary_ok"] is False


def test_build_canary_summary_rejects_failed_mixed_recovery() -> None:
    mixed = _base_mixed_smoke()
    mixed["stale_reclaimed"] = False

    summary = build_canary_summary(
        readiness=_base_readiness(),
        api_env=_base_env(),
        worker_env=_base_env(),
        retrieval_canary=_base_retrieval_canary(),
        retry_smoke=_base_retry_smoke(),
        concurrency_smoke=_base_concurrency_smoke(),
        mixed_recovery_smoke=mixed,
        compose_files="docker-compose.yml;docker-compose.milvus-canary.yml",
    )

    assert summary["canary_ok"] is False
