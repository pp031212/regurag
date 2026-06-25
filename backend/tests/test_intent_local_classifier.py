from pathlib import Path

import torch

from app.services import intent_local_classifier as module
from app.services.intent_local_classifier import IntentLocalClassifier, _EncoderRuntime


def test_intent_local_classifier_prefers_trained_artifact_when_available(tmp_path, monkeypatch) -> None:
    artifact_path = tmp_path / "intent_classifier.pt"
    torch.save(
        {
            "version": 1,
            "model_name": "test-model",
            "labels": ["business_query", "off_topic"],
            "classifier_weight": torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32),
            "classifier_bias": torch.tensor([0.0, 0.0], dtype=torch.float32),
        },
        artifact_path,
    )
    monkeypatch.setattr(
        module,
        "_load_encoder_runtime",
        lambda model_name: _EncoderRuntime(tokenizer=None, model=None, label_vectors={}),
    )

    classifier = IntentLocalClassifier(
        model_name="test-model",
        artifact_path=artifact_path,
        min_score=0.0,
        min_margin=0.0,
    )
    monkeypatch.setattr(
        classifier,
        "embed_text",
        lambda text: torch.tensor([1.0, 0.0], dtype=torch.float32),
    )

    result = classifier.classify_with_debug("试用期不签劳动合同可以入职吗")

    assert result is not None
    assert result.label == "business_query"
    assert result.mode == "trained"
    assert result.score is not None
    assert result.margin is not None


def test_intent_local_classifier_falls_back_to_prototypes_without_artifact(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        module,
        "_load_encoder_runtime",
        lambda model_name: _EncoderRuntime(
            tokenizer=None,
            model=None,
            label_vectors={
                "business_query": torch.tensor([1.0, 0.0], dtype=torch.float32),
                "off_topic": torch.tensor([0.0, 1.0], dtype=torch.float32),
            },
        ),
    )

    classifier = IntentLocalClassifier(
        model_name="test-model",
        artifact_path=tmp_path / "missing.pt",
        min_score=0.1,
        min_margin=0.1,
    )
    monkeypatch.setattr(
        classifier,
        "embed_text",
        lambda text: torch.tensor([0.9, 0.1], dtype=torch.float32),
    )

    result = classifier.classify_with_debug("试用期不签劳动合同可以入职吗")

    assert result is not None
    assert result.label == "business_query"
    assert result.mode == "prototype"
