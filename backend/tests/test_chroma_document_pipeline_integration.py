import json
from pathlib import Path
from uuid import uuid4

import chromadb

from app.rag.document_processor import DocumentProcessor
from app.rag.pipeline import RAGPipeline
from app.rag.structured_document_processor import StructuredDocumentProcessor
from app.rag.retrievers import build_sparse_index
from app.rag.vector_store import ChromaVectorStore, VectorStore, create_vector_store


class DeterministicVectorStore(VectorStore):
    def __init__(self, db_path: str, collection_name: str, *, sparse_provider: str = "sqlite_fts") -> None:
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.tokenizer = None
        self.model = None
        self.sparse_provider = sparse_provider
        self.retrieval_sparse_provider = sparse_provider
        self.sparse_index = build_sparse_index(
            db_path=db_path,
            collection_name=collection_name,
            sparse_provider=sparse_provider,
        )

    def _get_embedding(self, text: str, is_query: bool = False) -> list[float]:
        lowered = text.lower()
        score_alpha = 1.0 if "alpha" in lowered or "阿尔法" in lowered else 0.0
        score_beta = 1.0 if "beta" in lowered or "贝塔" in lowered else 0.0
        score_gamma = 1.0 if "gamma" in lowered or "伽马" in lowered else 0.0
        score_table = 1.0 if "表格" in lowered else 0.0
        score_text = 1.0 if "正文" in lowered else 0.0
        score_image = 1.0 if "图片" in lowered or "ocr" in lowered else 0.0
        vector = [score_alpha, score_beta, score_gamma, score_table, score_text, score_image]
        if not any(vector):
            return [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        return vector


def _build_vector_store(tmp_path: Path) -> DeterministicVectorStore:
    return DeterministicVectorStore(
        db_path=str(tmp_path / "chroma"),
        collection_name=f"test_{uuid4().hex[:8]}",
    )


def test_create_vector_store_returns_chroma_implementation(tmp_path: Path, monkeypatch) -> None:
    class StubTokenizer:
        pass

    class StubModel:
        def eval(self) -> None:
            return None

    monkeypatch.setattr("app.rag.vector_store.AutoTokenizer.from_pretrained", lambda _: StubTokenizer())
    monkeypatch.setattr("app.rag.vector_store.AutoModel.from_pretrained", lambda _: StubModel())

    store = create_vector_store(
        model_name="BAAI/bge-small-zh-v1.5",
        db_path=str(tmp_path / "factory_chroma"),
        collection_name=f"factory_{uuid4().hex[:8]}",
        sparse_provider="none",
    )

    assert isinstance(store, ChromaVectorStore)
    assert isinstance(store, VectorStore)


def test_vector_store_persists_search_and_delete_document(tmp_path: Path) -> None:
    store = _build_vector_store(tmp_path)
    store.add_documents(
        [
            {
                "child_text": "alpha 规则说明",
                "parent_text": "alpha 规则说明，包含阿尔法条件。",
                "parent_id": "rule_alpha",
                "document_id": "doc_alpha",
                "source_type": "text",
                "page_number": 1,
                "block_index": 0,
            },
            {
                "child_text": "beta 处理流程",
                "parent_text": "beta 处理流程，包含贝塔条件。",
                "parent_id": "rule_beta",
                "document_id": "doc_beta",
                "source_type": "table",
                "page_number": 2,
                "block_index": 1,
            },
        ]
    )

    _, results = store.search("alpha", top_k=2)
    assert results
    assert results[0]["document_id"] == "doc_alpha"
    assert results[0]["parent_id"] == "rule_alpha"

    keyword_results = store.keyword_search(["beta"], min_hits=1, top_k=2)
    assert keyword_results
    assert keyword_results[0]["document_id"] == "doc_beta"

    reloaded = DeterministicVectorStore(
        db_path=str(tmp_path / "chroma"),
        collection_name=store.collection_name,
    )
    _, reloaded_results = reloaded.search("beta", top_k=2)
    assert reloaded_results
    assert reloaded_results[0]["document_id"] == "doc_beta"
    reloaded_keyword_results = reloaded.keyword_search(["beta"], min_hits=1, top_k=2)
    assert reloaded_keyword_results
    assert reloaded_keyword_results[0]["document_id"] == "doc_beta"

    store.delete_document("doc_beta")
    _, after_delete_results = store.search("beta", top_k=2)
    assert all(item["document_id"] != "doc_beta" for item in after_delete_results)


def test_pipeline_ingest_file_writes_markdown_chunks_into_chroma(tmp_path: Path) -> None:
    markdown_path = tmp_path / "rules.md"
    markdown_path.write_text(
        "# Alpha 手册\n\n"
        "## 第一章 总则\n\n"
        "**第一条** alpha 要求学员完成阿尔法登记。\n\n"
        "**第二条** beta 要求学员完成贝塔审批。\n",
        encoding="utf-8",
    )

    vector_store = _build_vector_store(tmp_path)
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.processor = DocumentProcessor(child_chunk_size=200)
    pipeline.structured_processor = StructuredDocumentProcessor(pipeline.processor)
    pipeline.vector_store = vector_store

    chunk_count = pipeline.ingest_file(markdown_path, document_id="doc_markdown")

    assert chunk_count >= 2
    stored = vector_store.collection.get(where={"document_id": "doc_markdown"}, include=["documents", "metadatas"])
    assert len(stored["ids"]) == chunk_count
    assert any("Alpha 手册第一章 总则第一条" in meta["parent_text"] for meta in stored["metadatas"])

    _, results = vector_store.search("alpha", top_k=3)
    assert results
    assert any(item["document_id"] == "doc_markdown" for item in results)


def test_pipeline_ingest_file_writes_structured_json_chunks_into_chroma(tmp_path: Path) -> None:
    structured_path = tmp_path / "structured.json"
    structured_payload = {
        "pdf_file_id": "demo_pdf",
        "pages": [
            {
                "page_number": 1,
                "text": "alpha 正文说明。",
                "tables": [
                    {
                        "index": 0,
                        "text": "beta 表格要求",
                        "quality": "normal",
                    }
                ],
                "images": [
                    {
                        "index": 0,
                        "ocr_text": "gamma 图片 OCR 内容",
                        "translated_text": "gamma 图片 OCR 内容",
                        "translated_source_type": "image_ocr",
                    }
                ],
            }
        ],
    }
    structured_path.write_text(json.dumps(structured_payload, ensure_ascii=False), encoding="utf-8")

    vector_store = _build_vector_store(tmp_path)
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.processor = DocumentProcessor(child_chunk_size=200)
    pipeline.structured_processor = StructuredDocumentProcessor(pipeline.processor)
    pipeline.vector_store = vector_store

    chunk_count = pipeline.ingest_file(structured_path, document_id="doc_structured")

    assert chunk_count == 3
    stored = vector_store.collection.get(where={"document_id": "doc_structured"}, include=["documents", "metadatas"])
    assert len(stored["ids"]) == 3
    source_types = {meta["source_type"] for meta in stored["metadatas"]}
    assert source_types == {"text", "table", "image_ocr"}

    _, table_results = vector_store.search("beta 表格", top_k=3)
    assert any(item["source_type"] == "table" for item in table_results)

    _, image_results = vector_store.search("gamma 图片", top_k=3)
    assert any(item["source_type"] == "image_ocr" for item in image_results)


def test_vector_store_keyword_search_scan_provider_uses_collection_scan(tmp_path: Path) -> None:
    store = DeterministicVectorStore(
        db_path=str(tmp_path / "chroma_scan"),
        collection_name=f"scan_{uuid4().hex[:8]}",
        sparse_provider="scan",
    )
    store.add_documents(
        [
            {
                "child_text": "alpha 规则说明",
                "parent_text": "alpha 规则说明，包含阿尔法条件。",
                "parent_id": "rule_alpha",
                "document_id": "doc_alpha",
                "source_type": "text",
                "page_number": 1,
                "block_index": 0,
            },
            {
                "child_text": "beta 处理流程",
                "parent_text": "beta 处理流程，包含贝塔条件。",
                "parent_id": "rule_beta",
                "document_id": "doc_beta",
                "source_type": "table",
                "page_number": 2,
                "block_index": 1,
            },
        ]
    )

    keyword_results = store.keyword_search(["beta"], min_hits=1, top_k=2)

    assert keyword_results
    assert keyword_results[0]["document_id"] == "doc_beta"


def test_vector_store_keyword_search_none_provider_returns_empty(tmp_path: Path) -> None:
    store = DeterministicVectorStore(
        db_path=str(tmp_path / "chroma_none"),
        collection_name=f"none_{uuid4().hex[:8]}",
        sparse_provider="none",
    )
    store.add_documents(
        [
            {
                "child_text": "beta 处理流程",
                "parent_text": "beta 处理流程，包含贝塔条件。",
                "parent_id": "rule_beta",
                "document_id": "doc_beta",
                "source_type": "table",
                "page_number": 2,
                "block_index": 1,
            },
        ]
    )

    assert store.keyword_search(["beta"], min_hits=1, top_k=2) == []


def test_vector_store_keyword_search_bm25_provider_uses_bm25_ranking(tmp_path: Path) -> None:
    store = DeterministicVectorStore(
        db_path=str(tmp_path / "chroma_bm25"),
        collection_name=f"bm25_{uuid4().hex[:8]}",
        sparse_provider="bm25",
    )
    store.add_documents(
        [
            {
                "child_text": "alpha beta beta",
                "parent_text": "alpha beta beta 规则说明。",
                "parent_id": "rule_alpha_beta",
                "document_id": "doc_alpha_beta",
                "source_type": "text",
                "page_number": 1,
                "block_index": 0,
            },
            {
                "child_text": "alpha beta",
                "parent_text": "alpha beta 流程。",
                "parent_id": "rule_alpha_beta_short",
                "document_id": "doc_alpha_beta_short",
                "source_type": "text",
                "page_number": 1,
                "block_index": 1,
            },
        ]
    )

    keyword_results = store.keyword_search(["alpha", "beta"], min_hits=2, top_k=2)

    assert keyword_results
    assert keyword_results[0]["document_id"] == "doc_alpha_beta"
    assert float(keyword_results[0]["bm25_score"]) >= float(keyword_results[1]["bm25_score"])
