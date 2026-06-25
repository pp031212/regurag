"""校验意图分类数据集。

检查 JSONL 字段、标签、split、历史消息和重复 ID，输出数据集摘要。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evals.intent_dataset import load_intent_dataset, summarize_intent_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and summarize the cold-start intent dataset.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "evals" / "intent_dataset_seed.jsonl",
        help="Path to the JSONL intent dataset.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    examples = load_intent_dataset(args.dataset)
    print(json.dumps(summarize_intent_dataset(examples), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
