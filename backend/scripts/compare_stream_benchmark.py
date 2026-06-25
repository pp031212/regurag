"""比较两份流式聊天 benchmark 报告。

用于观察首 token、总耗时、服务端阶段耗时和错误数是否相对 baseline 改善或回退。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evals.stream_benchmark_compare import compare_stream_benchmark_reports, load_stream_benchmark_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two stream benchmark result reports.")
    parser.add_argument("--baseline", type=Path, required=True, help="Path to the baseline stream benchmark JSON.")
    parser.add_argument("--contender", type=Path, required=True, help="Path to the contender stream benchmark JSON.")
    parser.add_argument(
        "--label",
        default="stream-benchmark-compare",
        help="Output label. Report will be written to backend/evals/results/<label>.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline_report = load_stream_benchmark_report(args.baseline.resolve())
    contender_report = load_stream_benchmark_report(args.contender.resolve())
    comparison = compare_stream_benchmark_reports(baseline_report, contender_report)

    output_payload = {
        "run_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "baseline": str(args.baseline.resolve()),
        "contender": str(args.contender.resolve()),
        "comparison": comparison,
    }

    output_dir = ROOT / "evals" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.label}.json"
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved stream benchmark comparison report to {output_path}")
    print(json.dumps(comparison["summary_counts"], ensure_ascii=False, indent=2))
    print(json.dumps(comparison["summary_metrics"], ensure_ascii=False, indent=2))
    if comparison.get("summary_stage_metrics"):
        print(json.dumps(comparison["summary_stage_metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
