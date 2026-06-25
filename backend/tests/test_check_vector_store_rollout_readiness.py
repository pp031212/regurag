from __future__ import annotations

from scripts.check_vector_store_rollout_readiness import build_rollout_assessment


def _base_payload() -> dict:
    return {
        "run_at": "2026-04-18T11:35:38.753009Z",
        "stage_status": {
            "retrieval": {"available": True},
            "live_eval": {"available": True},
            "shadow": {"available": True},
        },
        "summary": {
            "backend_summaries": {
                "milvus": {
                    "gates": {
                        "retrieval_drift_free": True,
                        "live_eval_answer_parity_ok": True,
                    },
                    "ready_for_rollout": True,
                }
            },
            "diffs_vs_baseline": {
                "milvus": {
                    "live_eval_compare": {
                        "latency": {
                            "delta_ms": 9365.31,
                        }
                    }
                }
            },
        },
    }


def test_build_rollout_assessment_passes_when_all_requirements_hold() -> None:
    assessment = build_rollout_assessment(_base_payload(), backend="milvus")

    assert assessment["can_roll_out"] is True
    assert assessment["missing_or_unavailable_stages"] == []
    assert assessment["failed_gates"] == []
    assert assessment["live_eval_latency_delta_ms"] == 9365.31


def test_build_rollout_assessment_fails_when_stage_missing() -> None:
    payload = _base_payload()
    payload["stage_status"]["shadow"]["available"] = False

    assessment = build_rollout_assessment(payload, backend="milvus")

    assert assessment["can_roll_out"] is False
    assert assessment["missing_or_unavailable_stages"] == ["shadow"]


def test_build_rollout_assessment_fails_when_latency_budget_exceeded() -> None:
    assessment = build_rollout_assessment(
        _base_payload(),
        backend="milvus",
        max_live_eval_latency_delta_ms=5000,
    )

    assert assessment["can_roll_out"] is False
    assert assessment["latency_gate_ok"] is False
    assert "exceeds" in str(assessment["latency_gate_reason"])
