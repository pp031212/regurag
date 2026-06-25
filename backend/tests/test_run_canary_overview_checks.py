from __future__ import annotations

from scripts.run_canary_overview_checks import build_overview_summary


def _base_result(*, canary_ok: bool = True, failed_stage: str | None = None) -> dict:
    return {
        "key": "placeholder",
        "script_path": "D:/fake/script.py",
        "summary_path": "D:/fake/summary.json",
        "executed": True,
        "exit_code": 0 if canary_ok else 1,
        "stdout": "",
        "stderr": "",
        "summary": {
            "run_at": "2026-04-19T11:00:00Z",
            "canary_ok": canary_ok,
        },
        "canary_ok": canary_ok,
        "failed_stage": failed_stage,
    }


def test_build_overview_summary_marks_overall_ok_when_all_checks_pass() -> None:
    summary = build_overview_summary(
        {
            "milvus_canary": _base_result(),
            "postgresql_canary": _base_result(),
            "postgresql_task_queue_integration": _base_result(),
        }
    )

    assert summary["overall_ok"] is True
    assert summary["milvus_canary_ok"] is True
    assert summary["postgresql_canary_ok"] is True
    assert summary["postgresql_task_queue_integration_ok"] is True


def test_build_overview_summary_surfaces_failed_stage_and_nonzero_overall_state() -> None:
    summary = build_overview_summary(
        {
            "milvus_canary": _base_result(),
            "postgresql_canary": _base_result(canary_ok=False, failed_stage="metadata_check"),
        }
    )

    assert summary["overall_ok"] is False
    assert summary["postgresql_canary_ok"] is False
    assert summary["checks"]["postgresql_canary"]["failed_stage"] == "metadata_check"
