from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer

from ..core.config import get_settings


_LABEL_PROTOTYPES: dict[str, tuple[str, ...]] = {
    "business_query": (
        "试用期不签劳动合同可以入职吗",
        "劳动合同必须包含哪些内容",
        "加班工资怎么计算",
        "辞退员工需要满足什么条件",
    ),
    "follow_up_query": (
        "那如果是第一次呢",
        "这个情况怎么算",
        "那这种情况呢",
        "这样的话怎么处理",
    ),
    "off_topic": (
        "今天天气怎么样",
        "讲个笑话",
        "现在几点了",
        "帮我写一段 Python 代码",
    ),
    "meaningless_input": (
        "哈哈哈",
        "嗯嗯",
        "？",
        "行吧",
    ),
}


class _EncoderRuntime(NamedTuple):
    tokenizer: Any
    model: Any
    label_vectors: dict[str, torch.Tensor]


class _TrainedClassifierBundle(NamedTuple):
    model_name: str
    labels: tuple[str, ...]
    weight: torch.Tensor
    bias: torch.Tensor


@dataclass(frozen=True)
class IntentLocalClassificationResult:
    label: str
    score: float
    margin: float
    mode: str


def build_intent_classifier_text(query: str, history_messages: list[dict[str, str]] | None = None) -> str:
    history_lines = []
    for message in list(history_messages or [])[-4:]:
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if role and content:
            history_lines.append(f"{role}: {content}")
    history_text = "\n".join(history_lines)
    if not history_text:
        return query
    return f"{history_text}\nuser: {query}"


def embed_text_with_runtime(text: str, *, tokenizer: Any, model: Any) -> torch.Tensor:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )
    with torch.no_grad():
        outputs = model(**inputs)

    hidden_state = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"].unsqueeze(-1)
    pooled = (hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
    return functional.normalize(pooled, p=2, dim=1).squeeze(0)


def _build_label_vectors(*, tokenizer: Any, model: Any) -> dict[str, torch.Tensor]:
    label_vectors: dict[str, torch.Tensor] = {}
    for label, prototypes in _LABEL_PROTOTYPES.items():
        vectors = [embed_text_with_runtime(prototype, tokenizer=tokenizer, model=model) for prototype in prototypes]
        stacked = torch.stack(vectors)
        centroid = functional.normalize(stacked.mean(dim=0, keepdim=True), p=2, dim=1).squeeze(0)
        label_vectors[label] = centroid
    return label_vectors


@lru_cache(maxsize=4)
def _load_encoder_runtime(model_name: str) -> _EncoderRuntime:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    label_vectors = _build_label_vectors(tokenizer=tokenizer, model=model)
    return _EncoderRuntime(
        tokenizer=tokenizer,
        model=model,
        label_vectors=label_vectors,
    )


@lru_cache(maxsize=8)
def _load_trained_classifier_bundle(path_str: str, modified_at_ns: int) -> _TrainedClassifierBundle:
    payload = torch.load(path_str, map_location="cpu")
    labels = tuple(str(item) for item in list(payload["labels"]))
    return _TrainedClassifierBundle(
        model_name=str(payload["model_name"]),
        labels=labels,
        weight=torch.as_tensor(payload["classifier_weight"], dtype=torch.float32),
        bias=torch.as_tensor(payload["classifier_bias"], dtype=torch.float32),
    )


class IntentLocalClassifier:
    def __init__(
        self,
        *,
        model_name: str | None = None,
        artifact_path: str | Path | None = None,
        min_score: float | None = None,
        min_margin: float | None = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.intent_local_classifier_model or settings.embedding_model_name
        self.artifact_path = Path(artifact_path) if artifact_path is not None else settings.resolved_intent_local_classifier_artifact_path
        self.min_score = settings.intent_local_classifier_min_score if min_score is None else min_score
        self.min_margin = settings.intent_local_classifier_min_margin if min_margin is None else min_margin

    def _encoder_runtime(self) -> _EncoderRuntime:
        return _load_encoder_runtime(self.model_name)

    def _trained_bundle(self) -> _TrainedClassifierBundle | None:
        if not self.artifact_path.exists():
            return None
        bundle = _load_trained_classifier_bundle(
            str(self.artifact_path),
            self.artifact_path.stat().st_mtime_ns,
        )
        if bundle.model_name != self.model_name:
            return None
        return bundle

    def embed_text(self, text: str) -> torch.Tensor:
        runtime = self._encoder_runtime()
        return embed_text_with_runtime(text, tokenizer=runtime.tokenizer, model=runtime.model)

    def _classify_with_trained_bundle(
        self,
        query_vector: torch.Tensor,
        bundle: _TrainedClassifierBundle,
    ) -> IntentLocalClassificationResult | None:
        logits = torch.matmul(bundle.weight, query_vector) + bundle.bias
        probabilities = torch.softmax(logits, dim=0)
        top_index = int(torch.argmax(probabilities).item())
        top_score = float(probabilities[top_index].item())
        runner_up_score = (
            float(torch.topk(probabilities, k=min(2, probabilities.shape[0])).values[-1].item())
            if probabilities.shape[0] > 1
            else 0.0
        )
        margin = top_score - runner_up_score
        if top_score < self.min_score or margin < self.min_margin:
            return None
        return IntentLocalClassificationResult(
            label=bundle.labels[top_index],
            score=round(top_score, 4),
            margin=round(margin, 4),
            mode="trained",
        )

    def _classify_with_prototypes(
        self,
        query_vector: torch.Tensor,
        runtime: _EncoderRuntime,
    ) -> IntentLocalClassificationResult | None:
        scored = sorted(
            (
                (label, float(torch.dot(query_vector, label_vector).item()))
                for label, label_vector in runtime.label_vectors.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        top_label, top_score = scored[0]
        runner_up_score = scored[1][1] if len(scored) > 1 else 0.0
        margin = top_score - runner_up_score
        if top_score < self.min_score or margin < self.min_margin:
            return None
        return IntentLocalClassificationResult(
            label=top_label,
            score=round(top_score, 4),
            margin=round(margin, 4),
            mode="prototype",
        )

    def classify_with_debug(
        self,
        query: str,
        *,
        history_messages: list[dict[str, str]] | None = None,
    ) -> IntentLocalClassificationResult | None:
        runtime = self._encoder_runtime()
        query_vector = self.embed_text(build_intent_classifier_text(query, history_messages))
        bundle = self._trained_bundle()
        if bundle is not None:
            result = self._classify_with_trained_bundle(query_vector, bundle)
            if result is not None:
                return result
        return self._classify_with_prototypes(query_vector, runtime)

    def classify(
        self,
        query: str,
        *,
        history_messages: list[dict[str, str]] | None = None,
    ) -> str | None:
        result = self.classify_with_debug(query, history_messages=history_messages)
        return result.label if result is not None else None
