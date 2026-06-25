import threading
import time
from pathlib import Path
from types import SimpleNamespace

from app.rag import reranker as reranker_module
from app.rag import vector_store as vector_store_module
from app.services import rag_service


def test_pipeline_registry_builds_same_knowledge_base_once_under_concurrency(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry = rag_service.RAGPipelineRegistry()
    settings = SimpleNamespace(
        chroma_collection_name="regurag_docs",
        resolved_chroma_path=tmp_path,
        pipeline_bootstrap_lock_timeout_seconds=5,
    )
    monkeypatch.setattr(rag_service, "_settings", lambda: settings)

    created: list[tuple[str, str]] = []

    class StubPipeline:
        def __init__(self, *, settings, collection_name: str, subject: str) -> None:
            created.append((collection_name, subject))
            time.sleep(0.15)
            self.collection_name = collection_name
            self.subject = subject

    monkeypatch.setattr(rag_service, "RAGPipeline", StubPipeline)

    barrier = threading.Barrier(2)
    results: list[StubPipeline] = []

    def worker() -> None:
        barrier.wait()
        results.append(registry.get("kb_demo", "演示主题"))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(created) == 1
    assert len(results) == 2
    assert results[0] is results[1]


def test_embedding_components_are_cached_per_model(tmp_path: Path, monkeypatch) -> None:
    vector_store_module._load_embedding_components.cache_clear()
    settings = SimpleNamespace(
        resolved_chroma_path=tmp_path,
        pipeline_bootstrap_lock_timeout_seconds=5,
    )
    monkeypatch.setattr(vector_store_module, "get_settings", lambda: settings)

    calls = {"tokenizer": 0, "model": 0}

    class StubTokenizer:
        pass

    class StubModel:
        def eval(self) -> None:
            return None

    def load_tokenizer(model_name: str) -> StubTokenizer:
        calls["tokenizer"] += 1
        return StubTokenizer()

    def load_model(model_name: str) -> StubModel:
        calls["model"] += 1
        return StubModel()

    monkeypatch.setattr(vector_store_module.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(vector_store_module.AutoModel, "from_pretrained", load_model)

    first = vector_store_module._load_embedding_components("BAAI/bge-small-zh-v1.5")
    second = vector_store_module._load_embedding_components("BAAI/bge-small-zh-v1.5")

    assert first is second
    assert calls == {"tokenizer": 1, "model": 1}


def test_reranker_components_are_cached_per_model(tmp_path: Path, monkeypatch) -> None:
    reranker_module._load_reranker_components.cache_clear()
    settings = SimpleNamespace(
        resolved_chroma_path=tmp_path,
        pipeline_bootstrap_lock_timeout_seconds=5,
    )
    monkeypatch.setattr(reranker_module, "get_settings", lambda: settings)

    calls = {"tokenizer": 0, "model": 0}

    class StubTokenizer:
        pass

    class StubModel:
        def eval(self) -> None:
            return None

    def load_tokenizer(model_name: str) -> StubTokenizer:
        calls["tokenizer"] += 1
        return StubTokenizer()

    def load_model(model_name: str) -> StubModel:
        calls["model"] += 1
        return StubModel()

    monkeypatch.setattr(reranker_module.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(reranker_module.AutoModelForSequenceClassification, "from_pretrained", load_model)

    first = reranker_module._load_reranker_components("BAAI/bge-reranker-base")
    second = reranker_module._load_reranker_components("BAAI/bge-reranker-base")

    assert first is second
    assert calls == {"tokenizer": 1, "model": 1}
