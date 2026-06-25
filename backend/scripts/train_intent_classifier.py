"""训练意图分类器。

读取意图分类数据集，训练本地分类模型并写出模型产物和训练指标。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as functional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.evals.intent_dataset import IntentDatasetExample, load_intent_dataset
from app.services.intent_local_classifier import IntentLocalClassifier, build_intent_classifier_text


@dataclass(slots=True)
class SplitMetrics:
    split: str
    count: int
    accuracy: float
    macro_f1: float
    loss: float


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Train a local intent classifier head on top of the frozen encoder.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "evals" / "intent_dataset_seed.jsonl",
        help="Path to the JSONL intent dataset.",
    )
    parser.add_argument(
        "--model-name",
        default=settings.intent_local_classifier_model or settings.embedding_model_name,
        help="Encoder model name used to embed samples.",
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=settings.resolved_intent_local_classifier_artifact_path,
        help="Where to save the trained classifier artifact.",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs for the linear head.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate for Adam.")
    parser.add_argument("--weight-decay", type=float, default=0.001, help="Weight decay for Adam.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def select_split(examples: list[IntentDatasetExample], split: str) -> list[IntentDatasetExample]:
    return [example for example in examples if example.split == split]


def build_label_index(examples: list[IntentDatasetExample]) -> tuple[list[str], dict[str, int]]:
    labels = sorted({example.label for example in examples})
    return labels, {label: index for index, label in enumerate(labels)}


def build_feature_matrix(
    classifier: IntentLocalClassifier,
    examples: list[IntentDatasetExample],
    *,
    label_to_index: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    vectors: list[torch.Tensor] = []
    labels: list[int] = []
    for example in examples:
        text = build_intent_classifier_text(
            example.query,
            history_messages=[
                {"role": message.role, "content": message.content}
                for message in example.history_messages
            ],
        )
        vectors.append(classifier.embed_text(text))
        labels.append(label_to_index[example.label])
    return torch.stack(vectors), torch.tensor(labels, dtype=torch.long)


def compute_macro_f1(predictions: torch.Tensor, labels: torch.Tensor, *, label_count: int) -> float:
    f1_scores: list[float] = []
    for label_index in range(label_count):
        true_positive = int(((predictions == label_index) & (labels == label_index)).sum().item())
        false_positive = int(((predictions == label_index) & (labels != label_index)).sum().item())
        false_negative = int(((predictions != label_index) & (labels == label_index)).sum().item())
        if true_positive == 0 and false_positive == 0 and false_negative == 0:
            f1_scores.append(0.0)
            continue
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        if precision == 0.0 and recall == 0.0:
            f1_scores.append(0.0)
            continue
        f1_scores.append(2 * precision * recall / (precision + recall))
    return sum(f1_scores) / len(f1_scores)


def evaluate_split(features: torch.Tensor, labels: torch.Tensor, *, weight: torch.Tensor, bias: torch.Tensor, split: str) -> SplitMetrics:
    logits = functional.linear(features, weight, bias)
    loss = float(functional.cross_entropy(logits, labels).item())
    predictions = torch.argmax(logits, dim=1)
    accuracy = float((predictions == labels).float().mean().item())
    macro_f1 = compute_macro_f1(predictions, labels, label_count=weight.shape[0])
    return SplitMetrics(
        split=split,
        count=int(labels.shape[0]),
        accuracy=round(accuracy, 4),
        macro_f1=round(macro_f1, 4),
        loss=round(loss, 4),
    )


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    examples = load_intent_dataset(args.dataset)
    train_examples = select_split(examples, "train")
    validation_examples = select_split(examples, "validation")
    if not train_examples:
        raise ValueError("Intent dataset does not contain any train samples")

    labels, label_to_index = build_label_index(examples)
    classifier = IntentLocalClassifier(
        model_name=args.model_name,
        artifact_path=ROOT / "data" / "models" / "__intent_training_placeholder__.pt",
    )

    train_features, train_targets = build_feature_matrix(classifier, train_examples, label_to_index=label_to_index)
    validation_features = validation_targets = None
    if validation_examples:
        validation_features, validation_targets = build_feature_matrix(
            classifier,
            validation_examples,
            label_to_index=label_to_index,
        )

    input_dim = int(train_features.shape[1])
    output_dim = len(labels)
    weight = torch.nn.Parameter(torch.zeros(output_dim, input_dim))
    bias = torch.nn.Parameter(torch.zeros(output_dim))
    optimizer = torch.optim.Adam([weight, bias], lr=args.learning_rate, weight_decay=args.weight_decay)

    for _ in range(args.epochs):
        optimizer.zero_grad()
        logits = functional.linear(train_features, weight, bias)
        loss = functional.cross_entropy(logits, train_targets)
        loss.backward()
        optimizer.step()

    metrics = [evaluate_split(train_features, train_targets, weight=weight, bias=bias, split="train")]
    if validation_features is not None and validation_targets is not None:
        metrics.append(
            evaluate_split(validation_features, validation_targets, weight=weight, bias=bias, split="validation")
        )

    artifact_payload = {
        "version": 1,
        "model_name": args.model_name,
        "labels": labels,
        "classifier_weight": weight.detach().cpu(),
        "classifier_bias": bias.detach().cpu(),
        "metrics": [asdict(item) for item in metrics],
        "dataset": str(args.dataset),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
    }
    args.artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact_payload, args.artifact_path)

    print(
        json.dumps(
            {
                "artifact_path": str(args.artifact_path),
                "model_name": args.model_name,
                "metrics": [asdict(item) for item in metrics],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
