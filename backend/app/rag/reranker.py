import hashlib
import logging
import time
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..core.config import get_settings
from ..core.file_lock import advisory_file_lock

logger = logging.getLogger(__name__)


def _model_lock_path(model_name: str) -> Path:
    settings = get_settings()
    digest = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:16]
    return settings.resolved_chroma_path / ".model-bootstrap-locks" / f"reranker_{digest}.lock"


@lru_cache(maxsize=None)
def _load_reranker_components(model_name: str) -> tuple[object, object]:
    settings = get_settings()
    lock_path = _model_lock_path(model_name)
    wait_started = time.perf_counter()
    logger.info("reranker_model_bootstrap_waiting model=%s lock_path=%s", model_name, lock_path)
    with advisory_file_lock(lock_path, timeout_seconds=float(settings.pipeline_bootstrap_lock_timeout_seconds)):
        logger.info(
            "reranker_model_bootstrap_started model=%s wait_ms=%s",
            model_name,
            int((time.perf_counter() - wait_started) * 1000),
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        logger.info(
            "reranker_model_bootstrap_completed model=%s total_ms=%s",
            model_name,
            int((time.perf_counter() - wait_started) * 1000),
        )
        return tokenizer, model


class Reranker:
    def __init__(self, model_name: str) -> None:
        self.tokenizer, self.model = _load_reranker_components(model_name)

    def rerank(
        self,
        query: str,
        retrieved_docs: list[dict[str, object]],
        top_k: int = 2,
    ) -> list[dict[str, object]]:
        if not retrieved_docs:
            return []

        pairs = [(query, str(doc["text"])) for doc in retrieved_docs]
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            scores = self.model(**inputs, return_dict=True).logits.view(-1).float().tolist()

        ranked_docs: list[dict[str, object]] = []
        for doc, score in zip(retrieved_docs, scores):
            ranked_docs.append({**doc, "rerank_score": float(score)})

        ranked_docs.sort(key=lambda item: float(item["rerank_score"]), reverse=True)
        return ranked_docs[:top_k]
