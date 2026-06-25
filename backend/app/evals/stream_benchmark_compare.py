"""流式聊天 benchmark 报告对比。

按样本 ID 对齐两份报告，比较首 token、总耗时、服务端阶段耗时和错误数量。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

METRIC_NAMES = (
    "first_token_ms",
    "total_ms",
    "server_first_token_ms",
    "server_latency_ms",
)
STAT_NAMES = ("avg", "p50", "p95", "max")


def load_stream_benchmark_report(path: Path) -> dict[str, Any]:
    """加载并校验 benchmark 报告结构。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid stream benchmark report: {path}")
    if not isinstance(payload.get("summary"), dict):
        raise ValueError(f"Invalid stream benchmark report summary: {path}")
    if not isinstance(payload.get("results"), list):
        raise ValueError(f"Invalid stream benchmark report results: {path}")
    return payload


def _to_float(value: object) -> float | None:
    """把报告字段转换为 float，无法转换时返回 None。"""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _index_results(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """按 case id 建索引，后续只比较共有样本。"""
    indexed: dict[str, dict[str, Any]] = {}
    for item in results:
        case_id = str(item.get("id") or "").strip()
        if not case_id:
            raise ValueError("Stream benchmark report contains a result without id")
        indexed[case_id] = item
    return indexed


def _compare_metric_block(
    baseline_block: dict[str, Any],
    contender_block: dict[str, Any],
) -> dict[str, Any]:
    """比较一个指标块中的 avg/p50/p95/max。"""
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


def _compare_stage_metric_blocks(
    baseline_stage_metrics: dict[str, Any],
    contender_stage_metrics: dict[str, Any],
) -> dict[str, Any]:
    """比较服务端各阶段耗时。"""
    stage_metric_names = sorted(set(baseline_stage_metrics) | set(contender_stage_metrics))
    return {
        metric_name: _compare_metric_block(
            dict(baseline_stage_metrics.get(metric_name) or {}),
            dict(contender_stage_metrics.get(metric_name) or {}),
        )
        for metric_name in stage_metric_names
    }


def compare_stream_benchmark_reports(
    baseline_report: dict[str, Any],
    contender_report: dict[str, Any],
) -> dict[str, Any]:
    """比较 baseline 和 contender 两份流式 benchmark 报告。"""
    baseline_results_raw = baseline_report.get("results")
    contender_results_raw = contender_report.get("results")
    if not isinstance(baseline_results_raw, list) or not isinstance(contender_results_raw, list):
        raise ValueError("Stream benchmark report missing results")

    baseline_results = _index_results([item for item in baseline_results_raw if isinstance(item, dict)])
    contender_results = _index_results([item for item in contender_results_raw if isinstance(item, dict)])
    shared_case_ids = sorted(set(baseline_results) & set(contender_results))

    summary_comparison = {
        metric_name: _compare_metric_block(
            dict((baseline_report.get("summary") or {}).get(metric_name) or {}),
            dict((contender_report.get("summary") or {}).get(metric_name) or {}),
        )
        for metric_name in METRIC_NAMES
    }
    summary_stage_metrics = _compare_stage_metric_blocks(
        dict((baseline_report.get("summary") or {}).get("server_stage_timings_ms") or {}),
        dict((contender_report.get("summary") or {}).get("server_stage_timings_ms") or {}),
    )
    summary_counts = {
        "case_count": {
            "baseline": int((baseline_report.get("summary") or {}).get("case_count") or 0),
            "contender": int((contender_report.get("summary") or {}).get("case_count") or 0),
        },
        "run_count": {
            "baseline": int((baseline_report.get("summary") or {}).get("run_count") or 0),
            "contender": int((contender_report.get("summary") or {}).get("run_count") or 0),
        },
        "ok_count": {
            "baseline": int((baseline_report.get("summary") or {}).get("ok_count") or 0),
            "contender": int((contender_report.get("summary") or {}).get("ok_count") or 0),
        },
        "error_count": {
            "baseline": int((baseline_report.get("summary") or {}).get("error_count") or 0),
            "contender": int((contender_report.get("summary") or {}).get("error_count") or 0),
            "delta": int((contender_report.get("summary") or {}).get("error_count") or 0)
            - int((baseline_report.get("summary") or {}).get("error_count") or 0),
        },
    }

    case_diffs: list[dict[str, Any]] = []
    for case_id in shared_case_ids:
        baseline_item = baseline_results[case_id]
        contender_item = contender_results[case_id]
        case_diffs.append(
            {
                "id": case_id,
                "question": contender_item.get("question") or baseline_item.get("question"),
                "category": contender_item.get("category") or baseline_item.get("category"),
                "run_count": {
                    "baseline": int(baseline_item.get("run_count") or 0),
                    "contender": int(contender_item.get("run_count") or 0),
                },
                "ok_count": {
                    "baseline": int(baseline_item.get("ok_count") or 0),
                    "contender": int(contender_item.get("ok_count") or 0),
                },
                "error_count": {
                    "baseline": int(baseline_item.get("error_count") or 0),
                    "contender": int(contender_item.get("error_count") or 0),
                    "delta": int(contender_item.get("error_count") or 0) - int(baseline_item.get("error_count") or 0),
                },
                "metrics": {
                    metric_name: _compare_metric_block(
                        dict(baseline_item.get(metric_name) or {}),
                        dict(contender_item.get(metric_name) or {}),
                    )
                    for metric_name in METRIC_NAMES
                },
                "stage_metrics": _compare_stage_metric_blocks(
                    dict(baseline_item.get("server_stage_timings_ms") or {}),
                    dict(contender_item.get("server_stage_timings_ms") or {}),
                ),
            }
        )

    return {
        "baseline_label": baseline_report.get("label") or "baseline",
        "contender_label": contender_report.get("label") or "contender",
        "shared_case_count": len(shared_case_ids),
        "shared_case_ids": shared_case_ids,
        "missing_in_baseline": sorted(set(contender_results) - set(baseline_results)),
        "missing_in_contender": sorted(set(baseline_results) - set(contender_results)),
        "summary_counts": summary_counts,
        "summary_metrics": summary_comparison,
        "summary_stage_metrics": summary_stage_metrics,
        "case_diffs": case_diffs,
    }
