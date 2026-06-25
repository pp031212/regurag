"""预热模型缓存。

在 Docker build 或部署前主动加载 embedding/reranker 依赖，减少首次请求等待时间。
"""

import argparse
from time import perf_counter

from app.core.config import get_settings
from app.rag.reranker import _load_reranker_components
from app.rag.vector_store import _load_embedding_components


def _run_embedding_warmup(model_name: str) -> None:
    print(f"[warmup] embedding model: {model_name}", flush=True)
    started = perf_counter()
    _load_embedding_components(model_name)
    print(f"[warmup] embedding ready in {int((perf_counter() - started) * 1000)}ms", flush=True)


def _run_reranker_warmup(model_name: str) -> None:
    print(f"[warmup] reranker model: {model_name}", flush=True)
    started = perf_counter()
    _load_reranker_components(model_name)
    print(f"[warmup] reranker ready in {int((perf_counter() - started) * 1000)}ms", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Warm shared Hugging Face model cache.")
    parser.add_argument(
        "--only",
        choices=("all", "embedding", "reranker"),
        default="all",
        help="Warm both models or only one component.",
    )
    return parser


def main() -> None:
    settings = get_settings()
    args = _build_parser().parse_args()

    if args.only in {"all", "embedding"}:
        _run_embedding_warmup(settings.embedding_model_name)
    if args.only in {"all", "reranker"}:
        _run_reranker_warmup(settings.reranker_model_name)

    print("[warmup] completed", flush=True)


if __name__ == "__main__":
    main()
