"""Shadow graph 回归汇总。

用于比较不同 shadow retrieval backend 的 graph 输出稳定性，关注检索漂移、引用对齐、
最终上下文对齐和答案命中率一致性。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


PRIMARY_SUMMARY_METRICS = (
    "graph_error_count",
    "retrieval_drift_case_count",
    "final_answer_hit_parity_rate",
    "final_citation_ids_match_rate",
    "final_context_ids_match_rate",
    "avg_graph_latency_ms",
)


def build_shadow_regression_summary(
    *,
    reports: dict[str, dict[str, Any]],
    baseline_backend: str,
    report_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """生成 shadow backend 对比摘要。"""
    if baseline_backend not in reports:
        raise ValueError(f"Unknown baseline backend: {baseline_backend}")
    if len(reports) < 2:
        raise ValueError("Shadow regression summary requires at least two backend reports")

    baseline_report = reports[baseline_backend]
    baseline_summary = _extract_summary(baseline_report, backend=baseline_backend)
    normalized_paths = {backend: str(path) for backend, path in (report_paths or {}).items()}

    backend_summaries = {
        backend: {
            "report_path": normalized_paths.get(backend),
            "summary": _extract_summary(report, backend=backend),
            "gates": _build_backend_gates(_extract_summary(report, backend=backend)),
        }
        for backend, report in reports.items()
    }

    diffs: dict[str, dict[str, Any]] = {}
    for backend, report in reports.items():
        if backend == baseline_backend:
            continue
        contender_summary = _extract_summary(report, backend=backend)
        diffs[backend] = _build_summary_diff(
            baseline_summary=baseline_summary,
            contender_summary=contender_summary,
        )

    return {
        "baseline_backend": baseline_backend,
        "backend_summaries": backend_summaries,
        "primary_metrics": list(PRIMARY_SUMMARY_METRICS),
        "diffs_vs_baseline": diffs,
    }


def _extract_summary(report: dict[str, Any], *, backend: str) -> dict[str, Any]:
    """读取单个 backend 报告的 summary。"""
    summary = dict(report.get("summary") or {})
    if not summary:
        raise ValueError(f"Missing summary for backend report: {backend}")
    return summary


def _build_backend_gates(summary: dict[str, Any]) -> dict[str, Any]:
    """把单个 backend 的 shadow 指标转换成上线检查项。"""
    return {
        "graph_error_free": int(summary.get("graph_error_count") or 0) == 0,
        "retrieval_drift_free": int(summary.get("retrieval_drift_case_count") or 0) == 0,
        "citation_alignment_ok": float(summary.get("final_citation_ids_match_rate") or 0.0) >= 0.95,
        "context_alignment_ok": float(summary.get("final_context_ids_match_rate") or 0.0) >= 0.95,
        "answer_hit_parity_ok": float(summary.get("final_answer_hit_parity_rate") or 0.0) >= 1.0,
    }


def _build_summary_diff(*, baseline_summary: dict[str, Any], contender_summary: dict[str, Any]) -> dict[str, Any]:
    """计算候选 backend 相对 baseline 的关键指标差异。"""
    metrics: dict[str, dict[str, Any]] = {}
    for metric_name in PRIMARY_SUMMARY_METRICS:
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
    """把报告值转成可比较数字。"""
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
