"""评测意图分类器。

读取意图分类数据集，运行当前分类逻辑并输出分类准确率与错误样本。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as functional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.evals.intent_dataset import IntentDatasetExample, load_intent_dataset
from app.services.intent_local_classifier import IntentLocalClassifier, build_intent_classifier_text


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Evaluate a trained local intent classifier artifact.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "evals" / "intent_dataset_seed.jsonl",
        help="Path to the JSONL intent dataset.",
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=settings.resolved_intent_local_classifier_artifact_path,
        help="Path to the trained classifier artifact.",
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=("train", "validation", "test", "all"),
        help="Which split to evaluate.",
    )
    return parser.parse_args()


def select_examples(examples: list[IntentDatasetExample], split: str) -> list[IntentDatasetExample]:
    if split == "all":
        return examples
    return [example for example in examples if example.split == split]


def main() -> int:
    args = parse_args()
    if not args.artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found: {args.artifact_path}")

    artifact = torch.load(args.artifact_path, map_location="cpu")
    labels = [str(item) for item in list(artifact["labels"])]
    label_to_index = {label: index for index, label in enumerate(labels)}
    model_name = str(artifact["model_name"])
    weight = torch.as_tensor(artifact["classifier_weight"], dtype=torch.float32)
    bias = torch.as_tensor(artifact["classifier_bias"], dtype=torch.float32)

    examples = select_examples(load_intent_dataset(args.dataset), args.split)
    if not examples:
        raise ValueError(f"No examples found for split={args.split}")

    classifier = IntentLocalClassifier(
        model_name=model_name,
        artifact_path=args.artifact_path,
        min_score=0.0,
        min_margin=0.0,
    )

    correct = 0
    per_label: dict[str, dict[str, int]] = {label: {"total": 0, "correct": 0} for label in labels}
    rows: list[dict[str, object]] = []

    for example in examples:
        text = build_intent_classifier_text(
            example.query,
            history_messages=[
                {"role": message.role, "content": message.content}
                for message in example.history_messages
            ],
        )
        vector = classifier.embed_text(text)
        logits = torch.matmul(weight, vector) + bias
        probabilities = torch.softmax(logits, dim=0)
        predicted_index = int(torch.argmax(probabilities).item())
        predicted_label = labels[predicted_index]
        expected_label = example.label
        is_correct = predicted_label == expected_label
        if is_correct:
            correct += 1
        per_label[expected_label]["total"] += 1
        if is_correct:
            per_label[expected_label]["correct"] += 1

        rows.append(
            {
                "id": example.id,
                "query": example.query,
                "expected_label": expected_label,
                "predicted_label": predicted_label,
                "top_score": round(float(probabilities[predicted_index].item()), 4),
                "correct": is_correct,
            }
        )

    accuracy = round(correct / len(examples), 4)
    macro_recall = round(
        sum(
            item["correct"] / item["total"] if item["total"] else 0.0
            for item in per_label.values()
        )
        / len(per_label),
        4,
    )
    loss = round(
        float(
            functional.cross_entropy(
                torch.stack(
                    [torch.matmul(weight, classifier.embed_text(build_intent_classifier_text(
                        example.query,
                        history_messages=[
                            {"role": message.role, "content": message.content}
                            for message in example.history_messages
                        ],
                    ))) + bias for example in examples]
                ),
                torch.tensor([label_to_index[example.label] for example in examples], dtype=torch.long),
            ).item()
        ),
        4,
    )

    print(
        json.dumps(
            {
                "artifact_path": str(args.artifact_path),
                "dataset": str(args.dataset),
                "split": args.split,
                "count": len(examples),
                "accuracy": accuracy,
                "macro_recall": macro_recall,
                "loss": loss,
                "per_label": per_label,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
