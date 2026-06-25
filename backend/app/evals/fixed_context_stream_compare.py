"""固定上下文流式生成报告对比。

对比两份固定上下文 benchmark 报告，先确认 case 和 context hash 一致，再比较生成耗时。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STAT_NAMES = ("avg", "p50", "p95", "max")
METRIC_NAMES = ("first_token_ms", "total_ms")


def load_fixed_context_report(path: Path) -> dict[str, Any]:
    """加载并校验固定上下文报告结构。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid fixed-context report: {path}")
    if not isinstance(payload.get("summary"), dict):
        raise ValueError(f"Invalid fixed-context report summary: {path}")
    if not isinstance(payload.get("fixed_context"), dict):
        raise ValueError(f"Invalid fixed-context report fixed_context: {path}")
    return payload


def _to_float(value: object) -> float | None:
    """把指标值转成 float。"""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compare_metric_block(
    baseline_block: dict[str, Any],
    contender_block: dict[str, Any],
) -> dict[str, Any]:
    """比较 avg/p50/p95/max 指标块。"""
    comparison: dict[str, Any] = {}
    for stat_name in STAT_NAMES:
        baseline_value = _to_float(baseline_block.get(stat_name))
        contender_value = _to_float(contender_block.get(stat_name))
        delta = None
        improved = None
        if baseline_value is not None and contender_value is not None:
            delta = round(contender_value - baseline_value, 2)
            improved = contender_value < baseline_value
        comparison[stat_name] = {
            "baseline": baseline_value,
            "contender": contender_value,
            "delta": delta,
            "improved": improved,
        }
    return comparison


def compare_fixed_context_reports(
    baseline_report: dict[str, Any],
    contender_report: dict[str, Any],
) -> dict[str, Any]:
    """比较两份固定上下文报告。"""
    baseline_summary = dict(baseline_report.get("summary") or {})
    contender_summary = dict(contender_report.get("summary") or {})
    baseline_context = dict(baseline_report.get("fixed_context") or {})
    contender_context = dict(contender_report.get("fixed_context") or {})

    return {
        "baseline_label": baseline_report.get("label") or "baseline",
        "contender_label": contender_report.get("label") or "contender",
        "same_case": dict(baseline_report.get("case") or {}).get("id") == dict(contender_report.get("case") or {}).get("id"),
        "same_context_hash": baseline_context.get("context_hash") == contender_context.get("context_hash"),
        "context_hash": {
            "baseline": baseline_context.get("context_hash"),
            "contender": contender_context.get("context_hash"),
        },
        "summary_counts": {
            "run_count": {
                "baseline": int(baseline_summary.get("run_count") or 0),
                "contender": int(contender_summary.get("run_count") or 0),
            },
            "ok_count": {
                "baseline": int(baseline_summary.get("ok_count") or 0),
                "contender": int(contender_summary.get("ok_count") or 0),
                "delta": int(contender_summary.get("ok_count") or 0) - int(baseline_summary.get("ok_count") or 0),
            },
            "error_count": {
                "baseline": int(baseline_summary.get("error_count") or 0),
                "contender": int(contender_summary.get("error_count") or 0),
                "delta": int(contender_summary.get("error_count") or 0) - int(baseline_summary.get("error_count") or 0),
            },
        },
        "summary_metrics": {
            metric_name: _compare_metric_block(
                dict(baseline_summary.get(metric_name) or {}),
                dict(contender_summary.get(metric_name) or {}),
            )
            for metric_name in METRIC_NAMES
        },
        "answer_chars_avg": {
            "baseline": _to_float(baseline_summary.get("answer_chars_avg")),
            "contender": _to_float(contender_summary.get("answer_chars_avg")),
            "delta": (
                round(float(contender_summary.get("answer_chars_avg")) - float(baseline_summary.get("answer_chars_avg")), 2)
                if baseline_summary.get("answer_chars_avg") is not None and contender_summary.get("answer_chars_avg") is not None
                else None
            ),
        },
        "token_event_count_avg": {
            "baseline": _to_float(baseline_summary.get("token_event_count_avg")),
            "contender": _to_float(contender_summary.get("token_event_count_avg")),
            "delta": (
                round(float(contender_summary.get("token_event_count_avg")) - float(baseline_summary.get("token_event_count_avg")), 2)
                if baseline_summary.get("token_event_count_avg") is not None and contender_summary.get("token_event_count_avg") is not None
                else None
            ),
        },
        "errors": {
            "baseline": list(baseline_summary.get("errors") or []),
            "contender": list(contender_summary.get("errors") or []),
        },
    }
