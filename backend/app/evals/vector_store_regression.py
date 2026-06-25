"""向量库灰度回归汇总。

把检索对比、真实问答评测和 shadow graph 对比结果合并成一份门禁报告，
用于判断候选向量库是否可以替换 baseline。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


RETRIEVAL_PARITY_METRICS = (
    "retrieval_hit_rate",
    "final_context_hit_rate",
    "citation_hit_rate",
)

EVAL_PARITY_METRICS = (
    "retrieval_hit",
    "final_context_hit",
    "citation_hit",
    "answer_hit",
)

SHADOW_PRIMARY_SUMMARY_METRICS = (
    "graph_error_count",
    "retrieval_drift_case_count",
    "final_answer_hit_parity_rate",
    "final_citation_ids_match_rate",
    "final_context_ids_match_rate",
    "avg_graph_latency_ms",
)

ROLLOUT_GATES = (
    "retrieval_drift_free",
    "retrieval_hit_parity_ok",
    "final_context_parity_ok",
    "citation_parity_ok",
    "live_eval_available",
    "live_eval_retrieval_parity_ok",
    "live_eval_final_context_parity_ok",
    "live_eval_citation_parity_ok",
    "live_eval_answer_parity_ok",
    "shadow_graph_error_free",
    "shadow_retrieval_drift_free",
    "shadow_citation_alignment_ok",
    "shadow_context_alignment_ok",
    "shadow_answer_hit_parity_ok",
)


def build_vector_store_regression_summary(
    *,
    retrieval_reports: dict[str, dict[str, Any]],
    retrieval_comparisons: dict[str, dict[str, Any]],
    shadow_reports: dict[str, dict[str, Any]],
    baseline_backend: str,
    eval_reports: dict[str, dict[str, Any]] | None = None,
    eval_comparisons: dict[str, dict[str, Any]] | None = None,
    report_paths: dict[str, dict[str, Path]] | None = None,
) -> dict[str, Any]:
    """生成跨后端汇总报告和 rollout gate 结果。"""
    if baseline_backend not in retrieval_reports:
        raise ValueError(f"Unknown baseline backend: {baseline_backend}")
    if len(retrieval_reports) < 2:
        raise ValueError("Vector store regression summary requires at least two backend reports")

    eval_reports = eval_reports or {}
    eval_comparisons = eval_comparisons or {}
    normalized_paths = {
        backend: {report_type: str(path) for report_type, path in paths.items()}
        for backend, paths in (report_paths or {}).items()
    }

    backends = tuple(retrieval_reports.keys())
    backend_summaries: dict[str, dict[str, Any]] = {}
    diffs_vs_baseline: dict[str, dict[str, Any]] = {}
    for backend in backends:
        # 每个后端先独立提取报告摘要，再对非 baseline 后端计算门禁。
        retrieval_summary = _extract_summary(retrieval_reports[backend], backend=backend, report_kind="retrieval")
        eval_summary = (
            _extract_summary(eval_reports[backend], backend=backend, report_kind="eval")
            if backend in eval_reports
            else None
        )
        shadow_summary = (
            _extract_summary(shadow_reports[backend], backend=backend, report_kind="shadow")
            if backend in shadow_reports
            else None
        )

        if backend == baseline_backend:
            gates = None
            ready_for_rollout = None
        else:
            retrieval_comparison = _lookup_comparison(
                comparisons=retrieval_comparisons,
                baseline_backend=baseline_backend,
                contender_backend=backend,
            )
            eval_comparison = _lookup_comparison(
                comparisons=eval_comparisons,
                baseline_backend=baseline_backend,
                contender_backend=backend,
            )
            gates = _build_rollout_gates(
                retrieval_comparison=retrieval_comparison,
                eval_comparison=eval_comparison,
                shadow_summary=shadow_summary,
            )
            ready_for_rollout = all(gates.values())
            diffs_vs_baseline[backend] = {
                "retrieval_compare": retrieval_comparison,
                "live_eval_compare": eval_comparison,
                "shadow_compare": _build_shadow_diff(
                    baseline_summary=_extract_summary(
                        shadow_reports[baseline_backend],
                        backend=baseline_backend,
                        report_kind="shadow",
                    )
                    if baseline_backend in shadow_reports
                    else None,
                    contender_summary=shadow_summary,
                ),
            }

        backend_summaries[backend] = {
            "report_paths": normalized_paths.get(backend, {}),
            "retrieval_summary": retrieval_summary,
            "eval_summary": eval_summary,
            "shadow_summary": shadow_summary,
            "gates": gates,
            "ready_for_rollout": ready_for_rollout,
        }

    return {
        "baseline_backend": baseline_backend,
        "backends": list(backends),
        "rollout_gates": list(ROLLOUT_GATES),
        "rollout_policy": {
            "description": "Candidate backend is ready only when all rollout gates are true.",
            "note": "retrieval parity and live eval parity are measured against the baseline backend; shadow health is measured on the candidate backend itself.",
        },
        "backend_summaries": backend_summaries,
        "diffs_vs_baseline": diffs_vs_baseline,
    }


def _extract_summary(report: dict[str, Any], *, backend: str, report_kind: str) -> dict[str, Any]:
    """从单份报告中取 summary，缺失时直接失败，避免生成误导性门禁。"""
    summary = dict(report.get("summary") or {})
    if not summary:
        raise ValueError(f"Missing {report_kind} summary for backend: {backend}")
    return summary


def _lookup_comparison(
    *,
    comparisons: dict[str, dict[str, Any]],
    baseline_backend: str,
    contender_backend: str,
) -> dict[str, Any] | None:
    """按 baseline_vs_contender 的命名约定查找对比报告。"""
    if not comparisons:
        return None
    return comparisons.get(f"{baseline_backend}_vs_{contender_backend}")


def _build_rollout_gates(
    *,
    retrieval_comparison: dict[str, Any] | None,
    eval_comparison: dict[str, Any] | None,
    shadow_summary: dict[str, Any] | None,
) -> dict[str, bool]:
    """把多个报告指标压成布尔门禁，全部为真才允许切换。"""
    gates = {
        "retrieval_drift_free": int((retrieval_comparison or {}).get("case_drift_count") or 0) == 0,
        "retrieval_hit_parity_ok": _metric_diff_at_least(
            (retrieval_comparison or {}).get("metric_diffs"),
            "retrieval_hit_rate",
            minimum=0.0,
        ),
        "final_context_parity_ok": _metric_diff_at_least(
            (retrieval_comparison or {}).get("metric_diffs"),
            "final_context_hit_rate",
            minimum=0.0,
        ),
        "citation_parity_ok": _metric_diff_at_least(
            (retrieval_comparison or {}).get("metric_diffs"),
            "citation_hit_rate",
            minimum=0.0,
        ),
        "live_eval_available": eval_comparison is not None,
        "live_eval_retrieval_parity_ok": _eval_metric_delta_at_least(
            eval_comparison,
            "retrieval_hit",
            minimum=0.0,
        ),
        "live_eval_final_context_parity_ok": _eval_metric_delta_at_least(
            eval_comparison,
            "final_context_hit",
            minimum=0.0,
        ),
        "live_eval_citation_parity_ok": _eval_metric_delta_at_least(
            eval_comparison,
            "citation_hit",
            minimum=0.0,
        ),
        "live_eval_answer_parity_ok": _eval_metric_delta_at_least(
            eval_comparison,
            "answer_hit",
            minimum=0.0,
        ),
        "shadow_graph_error_free": int((shadow_summary or {}).get("graph_error_count") or 0) == 0,
        "shadow_retrieval_drift_free": int((shadow_summary or {}).get("retrieval_drift_case_count") or 0) == 0,
        "shadow_citation_alignment_ok": float((shadow_summary or {}).get("final_citation_ids_match_rate") or 0.0) >= 0.95,
        "shadow_context_alignment_ok": float((shadow_summary or {}).get("final_context_ids_match_rate") or 0.0) >= 0.95,
        "shadow_answer_hit_parity_ok": float((shadow_summary or {}).get("final_answer_hit_parity_rate") or 0.0) >= 1.0,
    }
    return gates


def _metric_diff_at_least(metric_diffs: object, metric_name: str, *, minimum: float) -> bool:
    """检索指标只接受不低于 baseline 的候选后端。"""
    if not isinstance(metric_diffs, dict):
        return False
    try:
        return float(metric_diffs.get(metric_name)) >= minimum
    except (TypeError, ValueError):
        return False


def _eval_metric_delta_at_least(comparison: dict[str, Any] | None, metric_name: str, *, minimum: float) -> bool:
    """真实问答评测指标也不能比 baseline 回退。"""
    if not comparison:
        return False
    metrics = comparison.get("metrics")
    if not isinstance(metrics, dict):
        return False
    metric_payload = metrics.get(metric_name)
    if not isinstance(metric_payload, dict):
        return False
    try:
        return float(metric_payload.get("delta")) >= minimum
    except (TypeError, ValueError):
        return False


def _build_shadow_diff(
    *,
    baseline_summary: dict[str, Any] | None,
    contender_summary: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """构造 shadow graph 关键指标相对 baseline 的差异。"""
    if baseline_summary is None or contender_summary is None:
        return None

    metrics: dict[str, dict[str, int | float | None]] = {}
    for metric_name in SHADOW_PRIMARY_SUMMARY_METRICS:
        baseline_value = _to_number(baseline_summary.get(metric_name))
        contender_value = _to_number(contender_summary.get(metric_name))
        metrics[metric_name] = {
            "baseline": baseline_value,
            "contender": contender_value,
            "delta": (
                round(float(contender_value) - float(baseline_value), 4)
                if baseline_value is not None and contender_value is not None
                else None
            ),
        }
    return metrics


def _to_number(value: Any) -> int | float | None:
    """把报告中的数字字段规整为 int/float，无法转换时返回 None。"""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number.is_integer():
        return int(number)
    return round(number, 4)
