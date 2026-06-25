"""比较不同分块 profile 或稀疏检索 provider 的离线评测结果。

主要用于调整文档分块策略、中文规则 profile 和稀疏召回策略时做快速 A/B。
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

from app.evals.split_profile_ab import run_split_profile_ab_compare


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a strict split-profile A/B comparison on isolated fixture documents.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "evals" / "rag_eval_split_profile_strict.jsonl",
        help="Path to the JSONL evaluation dataset.",
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=ROOT / "evals" / "fixtures" / "split_profile_strict",
        help="Fixture directory containing plain_rules.txt and ocr_rules.json.",
    )
    parser.add_argument(
        "--baseline-profile",
        default="",
        help="Baseline config profile. Empty string means use root config only.",
    )
    parser.add_argument(
        "--contender-profile",
        default="rules_cn",
        help="Contender config profile.",
    )
    parser.add_argument(
        "--label",
        default="split-profile-strict-ab",
        help="Output file label. Report will be written to backend/evals/results/<label>.json.",
    )
    parser.add_argument(
        "--child-chunk-size",
        type=int,
        default=18,
        help="Child chunk size used in the isolated split-profile A/B harness.",
    )
    parser.add_argument(
        "--parent-chunk-size",
        type=int,
        default=32,
        help="Parent chunk size used in the isolated split-profile A/B harness.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_payload = {
        "run_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        **run_split_profile_ab_compare(
            dataset_path=args.dataset.resolve(),
            fixture_dir=args.fixture_dir.resolve(),
            baseline_profile=args.baseline_profile,
            contender_profile=args.contender_profile,
            child_chunk_size=args.child_chunk_size,
            parent_chunk_size=args.parent_chunk_size,
        ),
    }
    output_dir = ROOT / "evals" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.label}.json"
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved strict split-profile A/B report to {output_path}")
    print(json.dumps(output_payload["comparison"]["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
