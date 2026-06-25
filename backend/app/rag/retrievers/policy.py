"""检索策略配置。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RetrievalPolicy:
    """控制 dense/sparse 召回数量和 sparse 开关。"""

    dense_top_k: int | None = None
    sparse_top_k: int | None = None
    sparse_min_hits: int = 2
    enable_sparse: bool = True

    def resolve_dense_top_k(self, requested_top_k: int) -> int:
        return self.dense_top_k or requested_top_k

    def resolve_sparse_top_k(self, requested_top_k: int) -> int:
        return self.sparse_top_k or max(2, requested_top_k)
