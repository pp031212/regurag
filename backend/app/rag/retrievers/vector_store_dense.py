"""把 VectorStore 适配成 DenseRetriever。"""

from __future__ import annotations

from ...workflows.rag.pipeline_steps import RagDoc


class VectorStoreDenseRetriever:
    """直接复用 vector_store.search 做 dense 召回。"""

    def __init__(self, vector_store: object) -> None:
        self.vector_store = vector_store

    def search(self, query: str, top_k: int) -> tuple[list[float], list[RagDoc]]:
        return self.vector_store.search(query, top_k=top_k)
