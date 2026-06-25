from __future__ import annotations

from dataclasses import dataclass

from .retrievers import RetrievalPolicy


@dataclass(slots=True)
class QueryPolicy:
    retrieval: RetrievalPolicy
    top_k_retrieve: int
    top_k_rerank: int
    enable_rerank: bool = True
    answer_style: str = "concise"
