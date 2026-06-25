"""读取向量库回归汇总并判断候选后端是否可上线。

脚本不重新跑评测，只消费已有 summary JSON，适合部署前做快速门禁检查。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_PATH = ROOT / "evals" / "results" / "vector-store-regression-live-v1-summary.json"
REQUIRED_STAGES = ("retrieval", "live_eval", "shadow")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a vector-store regression summary and print a compact rollout-readiness assessment."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Path to the vector store regression summary JSON.",
    )
    parser.add_argument(
        "--backend",
        default="milvus",
        help="Candidate backend to assess. Defaults to milvus.",
    )
    parser.add_argument(
        "--max-live-eval-latency-delta-ms",
        type=float,
        default=None,
        help="Optional upper bound for contender minus baseline live-eval latency delta. If omitted, latency is reported but not gated.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid summary payload: {path}")
    return payload


def build_rollout_assessment(
    payload: dict[str, Any],
    *,
    backend: str,
    max_live_eval_latency_delta_ms: float | None = None,
) -> dict[str, Any]:
    stage_status = payload.get("stage_status")
    if not isinstance(stage_status, dict):
        raise ValueError("Summary payload missing stage_status")

    missing_or_unavailable_stages = [
        stage_name
        for stage_name in REQUIRED_STAGES
        if not isinstance(stage_status.get(stage_name), dict) or not bool(stage_status[stage_name].get("available"))
    ]

    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("Summary payload missing summary")
    backend_summaries = summary.get("backend_summaries")
    if not isinstance(backend_summaries, dict):
        raise ValueError("Summary payload missing backend_summaries")
    backend_summary = backend_summaries.get(backend)
    if not isinstance(backend_summary, dict):
        raise ValueError(f"Unknown backend in summary: {backend}")

    ready_for_rollout = bool(backend_summary.get("ready_for_rollout"))
    gates = backend_summary.get("gates")
    failed_gates = [
        gate_name
        for gate_name, gate_value in (gates.items() if isinstance(gates, dict) else [])
        if not bool(gate_value)
    ]

    diffs_vs_baseline = summary.get("diffs_vs_baseline")
    if not isinstance(diffs_vs_baseline, dict):
        raise ValueError("Summary payload missing diffs_vs_baseline")
    backend_diffs = diffs_vs_baseline.get(backend)
    if not isinstance(backend_diffs, dict):
        raise ValueError(f"Summary payload missing diffs for backend: {backend}")

    latency_delta_ms = None
    live_eval_compare = backend_diffs.get("live_eval_compare")
    if isinstance(live_eval_compare, dict):
        latency = live_eval_compare.get("latency")
        if isinstance(latency, dict):
            raw_delta = latency.get("delta_ms")
            if isinstance(raw_delta, int | float):
                latency_delta_ms = float(raw_delta)

    latency_gate_ok = (
        True
        if max_live_eval_latency_delta_ms is None or latency_delta_ms is None
        else latency_delta_ms <= max_live_eval_latency_delta_ms
    )
    latency_gate_reason = None
    if max_live_eval_latency_delta_ms is not None:
        if latency_delta_ms is None:
            latency_gate_reason = "missing_live_eval_latency_delta"
        elif latency_delta_ms > max_live_eval_latency_delta_ms:
            latency_gate_reason = (
                f"live_eval_latency_delta_ms={latency_delta_ms:.2f} exceeds "
                f"max_live_eval_latency_delta_ms={max_live_eval_latency_delta_ms:.2f}"
            )

    can_roll_out = not missing_or_unavailable_stages and ready_for_rollout and latency_gate_ok
    return {
        "backend": backend,
        "summary_run_at": payload.get("run_at"),
        "required_stages": list(REQUIRED_STAGES),
        "missing_or_unavailable_stages": missing_or_unavailable_stages,
        "ready_for_rollout": ready_for_rollout,
        "failed_gates": failed_gates,
        "live_eval_latency_delta_ms": latency_delta_ms,
        "latency_gate_ok": latency_gate_ok,
        "latency_gate_reason": latency_gate_reason,
        "can_roll_out": can_roll_out,
    }


def main() -> int:
    args = parse_args()
    payload = load_summary(args.summary.resolve())
    assessment = build_rollout_assessment(
        payload,
        backend=args.backend.strip().lower(),
        max_live_eval_latency_delta_ms=args.max_live_eval_latency_delta_ms,
    )
    print(json.dumps(assessment, ensure_ascii=False, indent=2))
    return 0 if assessment["can_roll_out"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
