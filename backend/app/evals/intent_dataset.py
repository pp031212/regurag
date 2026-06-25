"""意图分类数据集加载和校验。

数据集用于冷启动本地意图分类器，字段校验放在这里，训练、评测和 validate 脚本共用。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


ALLOWED_INTENT_LABELS = frozenset(
    {
        "business_query",
        "follow_up_query",
        "off_topic",
        "meaningless_input",
    }
)
ALLOWED_SPLITS = frozenset({"train", "validation", "test"})
ALLOWED_MESSAGE_ROLES = frozenset({"user", "assistant"})


@dataclass(slots=True, frozen=True)
class IntentDatasetMessage:
    """历史消息样本。"""

    role: str
    content: str


@dataclass(slots=True, frozen=True)
class IntentDatasetExample:
    """单条意图分类监督样本。"""

    id: str
    label: str
    query: str
    history_messages: tuple[IntentDatasetMessage, ...]
    split: str
    source: str
    notes: str = ""


def _load_history_messages(payload: object, *, line_number: int) -> tuple[IntentDatasetMessage, ...]:
    """解析并校验 history_messages 字段。"""
    if payload is None:
        return ()
    if not isinstance(payload, list):
        raise ValueError(f"Invalid dataset line {line_number}: history_messages must be a list")

    messages: list[IntentDatasetMessage] = []
    for message in payload:
        if not isinstance(message, dict):
            raise ValueError(f"Invalid dataset line {line_number}: history message must be an object")
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if role not in ALLOWED_MESSAGE_ROLES:
            raise ValueError(f"Invalid dataset line {line_number}: unsupported history role {role!r}")
        if not content:
            raise ValueError(f"Invalid dataset line {line_number}: history message content cannot be empty")
        messages.append(IntentDatasetMessage(role=role, content=content))
    return tuple(messages)


def load_intent_dataset(path: Path) -> list[IntentDatasetExample]:
    """读取 JSONL 数据集并执行字段级校验。"""
    examples: list[IntentDatasetExample] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid dataset line {line_number}: expected a JSON object")

            example_id = str(payload.get("id") or "").strip()
            label = str(payload.get("label") or "").strip()
            query = str(payload.get("query") or "").strip()
            split = str(payload.get("split") or "train").strip()
            source = str(payload.get("source") or "").strip()
            notes = str(payload.get("notes") or "").strip()

            if not example_id:
                raise ValueError(f"Invalid dataset line {line_number}: id cannot be empty")
            if example_id in seen_ids:
                raise ValueError(f"Invalid dataset line {line_number}: duplicate id {example_id}")
            if label not in ALLOWED_INTENT_LABELS:
                raise ValueError(f"Invalid dataset line {line_number}: unsupported label {label!r}")
            if not query:
                raise ValueError(f"Invalid dataset line {line_number}: query cannot be empty")
            if split not in ALLOWED_SPLITS:
                raise ValueError(f"Invalid dataset line {line_number}: unsupported split {split!r}")
            if not source:
                raise ValueError(f"Invalid dataset line {line_number}: source cannot be empty")

            history_messages = _load_history_messages(payload.get("history_messages"), line_number=line_number)
            if label == "follow_up_query" and not history_messages:
                raise ValueError(
                    f"Invalid dataset line {line_number}: follow_up_query requires history_messages for supervision"
                )

            examples.append(
                IntentDatasetExample(
                    id=example_id,
                    label=label,
                    query=query,
                    history_messages=history_messages,
                    split=split,
                    source=source,
                    notes=notes,
                )
            )
            seen_ids.add(example_id)

    if not examples:
        raise ValueError(f"No intent dataset examples found in {path}")
    return examples


def summarize_intent_dataset(examples: list[IntentDatasetExample]) -> dict[str, object]:
    """统计标签、split 和追问样本数量。"""
    by_label: dict[str, int] = {}
    by_split: dict[str, int] = {}
    follow_up_with_history = 0

    for example in examples:
        by_label[example.label] = by_label.get(example.label, 0) + 1
        by_split[example.split] = by_split.get(example.split, 0) + 1
        if example.label == "follow_up_query" and example.history_messages:
            follow_up_with_history += 1

    return {
        "count": len(examples),
        "labels": by_label,
        "splits": by_split,
        "follow_up_with_history": follow_up_with_history,
    }
