"""比较两份 RAG 问答评测报告。

按样本 ID 对齐 baseline 和 contender，输出命中率、答案覆盖率和延迟差异。
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

from app.evals.eval_report_compare import compare_eval_reports, load_eval_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two eval_rag result reports.")
    parser.add_argument("--baseline", type=Path, required=True, help="Path to the baseline eval report JSON.")
    parser.add_argument("--contender", type=Path, required=True, help="Path to the contender eval report JSON.")
    parser.add_argument(
        "--label",
        default="eval-report-compare",
        help="Output file label. Report will be written to backend/evals/results/<label>.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline_report = load_eval_report(args.baseline.resolve())
    contender_report = load_eval_report(args.contender.resolve())
    comparison = compare_eval_reports(baseline_report, contender_report)
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

    print(f"Saved eval comparison report to {output_path}")
    print(json.dumps(comparison["metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(comparison["latency"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
