"""问答评测报告对比工具。

把两份 eval_rag 输出按 case id 对齐，统计命中率、答案覆盖率和延迟差异，
用于判断某次模型/检索配置调整是否回退。
"""

from __future__ import annotations

import json
from pathlib import Path

BOOL_METRICS = ("retrieval_hit", "final_context_hit", "citation_hit", "answer_hit")
FLOAT_METRICS = ("answer_hit_ratio",)
LATENCY_FIELD = "latency_ms"


def load_eval_report(path: Path) -> dict[str, object]:
    """加载并校验评测报告基本结构。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid eval report: {path}")
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"Invalid eval report results: {path}")
    return payload


def _index_results(results: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    """按 case id 建索引，后续只比较两份报告共有的样本。"""
    indexed: dict[str, dict[str, object]] = {}
    for item in results:
        case_id = str(item.get("id") or "").strip()
        if not case_id:
            raise ValueError("Eval report contains a result without id")
        indexed[case_id] = item
    return indexed


def _metric_value(item: dict[str, object], metric: str) -> float:
    """把 bool/number 指标统一成浮点数，便于求平均和 delta。"""
    value = item.get(metric)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _latency_value(item: dict[str, object]) -> float:
    """延迟记录在 debug 中，缺失时按 0 处理。"""
    debug_payload = item.get("debug")
    if not isinstance(debug_payload, dict):
        return 0.0
    try:
        return float(debug_payload.get(LATENCY_FIELD) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def compare_eval_reports(baseline_report: dict[str, object], contender_report: dict[str, object]) -> dict[str, object]:
    """比较 baseline 和 contender 两份评测报告。"""
    baseline_results_raw = baseline_report.get("results")
    contender_results_raw = contender_report.get("results")
    if not isinstance(baseline_results_raw, list) or not isinstance(contender_results_raw, list):
        raise ValueError("Eval report missing results")

    baseline_results = _index_results([item for item in baseline_results_raw if isinstance(item, dict)])
    contender_results = _index_results([item for item in contender_results_raw if isinstance(item, dict)])

    shared_case_ids = sorted(set(baseline_results) & set(contender_results))
    if not shared_case_ids:
        raise ValueError("No overlapping eval case ids between reports")

    metric_summary: dict[str, dict[str, object]] = {}
    for metric in (*BOOL_METRICS, *FLOAT_METRICS):
        # 对每个指标分别统计提升、回退和持平的样本，便于定位具体 case。
        improved: list[str] = []
        regressed: list[str] = []
        unchanged: list[str] = []
        baseline_total = 0.0
        contender_total = 0.0

        for case_id in shared_case_ids:
            baseline_value = _metric_value(baseline_results[case_id], metric)
            contender_value = _metric_value(contender_results[case_id], metric)
            baseline_total += baseline_value
            contender_total += contender_value
            if contender_value > baseline_value:
                improved.append(case_id)
            elif contender_value < baseline_value:
                regressed.append(case_id)
            else:
                unchanged.append(case_id)

        metric_summary[metric] = {
            "baseline_avg": round(baseline_total / len(shared_case_ids), 4),
            "contender_avg": round(contender_total / len(shared_case_ids), 4),
            "delta": round((contender_total - baseline_total) / len(shared_case_ids), 4),
            "improved_case_ids": improved,
            "regressed_case_ids": regressed,
            "unchanged_case_ids": unchanged,
        }

    latency_improved: list[str] = []
    latency_regressed: list[str] = []
    baseline_latency_total = 0.0
    contender_latency_total = 0.0
    for case_id in shared_case_ids:
        baseline_latency = _latency_value(baseline_results[case_id])
        contender_latency = _latency_value(contender_results[case_id])
        baseline_latency_total += baseline_latency
        contender_latency_total += contender_latency
        if contender_latency < baseline_latency:
            latency_improved.append(case_id)
        elif contender_latency > baseline_latency:
            latency_regressed.append(case_id)

    case_diffs: list[dict[str, object]] = []
    for case_id in shared_case_ids:
        # case_diffs 保留逐样本差异，排查时不用再手动打开两份完整报告。
        baseline_item = baseline_results[case_id]
        contender_item = contender_results[case_id]
        case_diffs.append(
            {
                "id": case_id,
                "question": contender_item.get("question") or baseline_item.get("question"),
                "category": contender_item.get("category") or baseline_item.get("category"),
                "answer_mode": contender_item.get("answer_mode") or baseline_item.get("answer_mode"),
                "retrieval_hit": {
                    "baseline": bool(baseline_item.get("retrieval_hit")),
                    "contender": bool(contender_item.get("retrieval_hit")),
                },
                "final_context_hit": {
                    "baseline": bool(baseline_item.get("final_context_hit")),
                    "contender": bool(contender_item.get("final_context_hit")),
                },
                "citation_hit": {
                    "baseline": bool(baseline_item.get("citation_hit")),
                    "contender": bool(contender_item.get("citation_hit")),
                },
                "answer_hit": {
                    "baseline": bool(baseline_item.get("answer_hit")),
                    "contender": bool(contender_item.get("answer_hit")),
                },
                "answer_hit_ratio": {
                    "baseline": round(_metric_value(baseline_item, "answer_hit_ratio"), 4),
                    "contender": round(_metric_value(contender_item, "answer_hit_ratio"), 4),
                },
                "latency_ms": {
                    "baseline": round(_latency_value(baseline_item), 2),
                    "contender": round(_latency_value(contender_item), 2),
                },
                "citation_count": {
                    "baseline": len(list(baseline_item.get("citations") or [])),
                    "contender": len(list(contender_item.get("citations") or [])),
                },
                "final_context_count": {
                    "baseline": len(list((baseline_item.get("debug") or {}).get("final_context_chunks") or []))
                    if isinstance(baseline_item.get("debug"), dict)
                    else 0,
                    "contender": len(list((contender_item.get("debug") or {}).get("final_context_chunks") or []))
                    if isinstance(contender_item.get("debug"), dict)
                    else 0,
                },
            }
        )

    return {
        "baseline_label": baseline_report.get("dataset") or baseline_report.get("label") or "baseline",
        "contender_label": contender_report.get("dataset") or contender_report.get("label") or "contender",
        "shared_case_count": len(shared_case_ids),
        "shared_case_ids": shared_case_ids,
        "missing_in_baseline": sorted(set(contender_results) - set(baseline_results)),
        "missing_in_contender": sorted(set(baseline_results) - set(contender_results)),
        "metrics": metric_summary,
        "latency": {
            "baseline_avg_ms": round(baseline_latency_total / len(shared_case_ids), 2),
            "contender_avg_ms": round(contender_latency_total / len(shared_case_ids), 2),
            "delta_ms": round((contender_latency_total - baseline_latency_total) / len(shared_case_ids), 2),
            "improved_case_ids": latency_improved,
            "regressed_case_ids": latency_regressed,
        },
        "case_diffs": case_diffs,
    }
