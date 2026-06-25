import json
from pathlib import Path
from uuid import uuid4

from app.rag.document_processor import DocumentProcessor
from app.rag.pipeline import RAGPipeline
from app.rag.structured_document_processor import StructuredDocumentProcessor
from app.rag.vector_store import MilvusVectorStore, _load_embedding_components, create_vector_store


class _StubTokenizer:
    pass


class _StubModel:
    def eval(self) -> None:
        return None


class FakeMilvusSchema:
    def __init__(self) -> None:
        self.fields: list[tuple[str, object, dict[str, object]]] = []

    def add_field(self, name: str, data_type: object, **kwargs) -> None:
        self.fields.append((name, data_type, kwargs))


class FakeMilvusIndexParams:
    def __init__(self) -> None:
        self.indexes: list[dict[str, object]] = []

    def add_index(self, **kwargs) -> None:
        self.indexes.append(kwargs)


class FakeMilvusQueryIterator:
    def __init__(self, rows: list[dict[str, object]], *, batch_size: int, output_fields: list[str] | None) -> None:
        self.rows = rows
        self.batch_size = batch_size
        self.output_fields = output_fields
        self.offset = 0
        self.closed = False

    def next(self) -> list[dict[str, object]]:
        if self.offset >= len(self.rows):
            return []
        batch = self.rows[self.offset : self.offset + self.batch_size]
        self.offset += self.batch_size
        return [self._select_fields(row) for row in batch]

    def close(self) -> None:
        self.closed = True

    def _select_fields(self, row: dict[str, object]) -> dict[str, object]:
        if not self.output_fields:
            return dict(row)
        return {field_name: row.get(field_name) for field_name in self.output_fields}


class FakeMilvusClient:
    _databases: dict[str, dict[str, dict[str, object]]] = {}

    def __init__(self, *, uri: str, token: str | None = None) -> None:
        self.database_key = f"{uri}|{token or ''}"
        self.collections = self._databases.setdefault(self.database_key, {})

    @classmethod
    def reset(cls) -> None:
        cls._databases.clear()

    def has_collection(self, collection_name: str, timeout: float | None = None, **kwargs) -> bool:
        return collection_name in self.collections

    def drop_collection(self, collection_name: str, timeout: float | None = None, **kwargs) -> None:
        self.collections.pop(collection_name, None)

    def create_schema(self, **kwargs) -> FakeMilvusSchema:
        return FakeMilvusSchema()

    def prepare_index_params(self, field_name: str = "", **kwargs) -> FakeMilvusIndexParams:
        return FakeMilvusIndexParams()

    def create_collection(
        self,
        collection_name: str,
        dimension: int | None = None,
        primary_field_name: str = "id",
        id_type: str = "int",
        vector_field_name: str = "vector",
        metric_type: str = "COSINE",
        auto_id: bool = False,
        timeout: float | None = None,
        schema: FakeMilvusSchema | None = None,
        index_params: FakeMilvusIndexParams | None = None,
        **kwargs,
    ) -> None:
        self.collections[collection_name] = {
            "rows": {},
            "schema": schema,
            "index_params": index_params,
            "loaded": False,
        }

    def load_collection(self, collection_name: str, timeout: float | None = None, **kwargs) -> None:
        self.collections[collection_name]["loaded"] = True

    def flush(self, collection_name: str, timeout: float | None = None, **kwargs) -> None:
        return None

    def upsert(
        self,
        collection_name: str,
        data: dict[str, object] | list[dict[str, object]],
        timeout: float | None = None,
        partition_name: str | None = "",
        **kwargs,
    ) -> dict[str, object]:
        rows = data if isinstance(data, list) else [data]
        collection_rows = self.collections[collection_name]["rows"]
        for row in rows:
            collection_rows[row["chunk_id"]] = dict(row)
        return {"upsert_count": len(rows)}

    def query_iterator(
        self,
        collection_name: str,
        batch_size: int = 1000,
        limit: int = -1,
        filter: str = "",
        output_fields: list[str] | None = None,
        partition_names: list[str] | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> FakeMilvusQueryIterator:
        rows = list(self.collections[collection_name]["rows"].values())
        if limit >= 0:
            rows = rows[:limit]
        return FakeMilvusQueryIterator(rows, batch_size=batch_size, output_fields=output_fields)

    def search(
        self,
        collection_name: str,
        data: list[list[float]] | list[float] | None = None,
        filter: str = "",
        limit: int = 10,
        output_fields: list[str] | None = None,
        search_params: dict[str, object] | None = None,
        timeout: float | None = None,
        partition_names: list[str] | None = None,
        anns_field: str | None = None,
        ranker=None,
        highlighter=None,
        ids=None,
        **kwargs,
    ) -> list[list[dict[str, object]]]:
        query_vector = list(data[0] if data and isinstance(data[0], list) else data or [])
        rows = list(self.collections[collection_name]["rows"].values())
        ranked = sorted(rows, key=lambda row: self._l2_distance(query_vector, list(row["vector"])))
        return [
            [
                {
                    "id": row["chunk_id"],
                    "distance": self._l2_distance(query_vector, list(row["vector"])),
                    "entity": self._select_fields(row, output_fields),
                }
                for row in ranked[:limit]
            ]
        ]

    def delete(
        self,
        collection_name: str,
        ids: list[str] | str | int | None = None,
        timeout: float | None = None,
        filter: str | None = None,
        partition_name: str | None = None,
        **kwargs,
    ) -> dict[str, int]:
        rows = self.collections[collection_name]["rows"]
        deleted_count = 0
        if filter and filter.startswith('document_id == "'):
            document_id = filter[len('document_id == "') : -1]
            matching_ids = [chunk_id for chunk_id, row in rows.items() if row.get("document_id") == document_id]
            for chunk_id in matching_ids:
                rows.pop(chunk_id, None)
            deleted_count = len(matching_ids)
        elif ids is not None:
            normalized_ids = ids if isinstance(ids, list) else [ids]
            for chunk_id in normalized_ids:
                if rows.pop(chunk_id, None) is not None:
                    deleted_count += 1
        return {"delete_count": deleted_count}

    @staticmethod
    def _select_fields(row: dict[str, object], output_fields: list[str] | None) -> dict[str, object]:
        if not output_fields:
            return dict(row)
        return {field_name: row.get(field_name) for field_name in output_fields}

    @staticmethod
    def _l2_distance(left: list[float], right: list[float]) -> float:
        return float(sum((lhs - rhs) ** 2 for lhs, rhs in zip(left, right)))


class DeterministicMilvusVectorStore(MilvusVectorStore):
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


def _patch_milvus_dependencies(monkeypatch) -> None:
    FakeMilvusClient.reset()
    _load_embedding_components.cache_clear()
    monkeypatch.setattr("app.rag.vector_store._load_pymilvus_client_class", lambda: FakeMilvusClient)
    monkeypatch.setattr("app.rag.vector_store.AutoTokenizer.from_pretrained", lambda _: _StubTokenizer())
    monkeypatch.setattr("app.rag.vector_store.AutoModel.from_pretrained", lambda _: _StubModel())


def _build_vector_store(tmp_path: Path, *, collection_name: str | None = None) -> DeterministicMilvusVectorStore:
    return DeterministicMilvusVectorStore(
        model_name="BAAI/bge-small-zh-v1.5",
        db_path=str(tmp_path / "milvus_sidecar"),
        collection_name=collection_name or f"milvus_{uuid4().hex[:8]}",
        milvus_uri="http://milvus.test:19530",
        sparse_provider="sqlite_fts",
    )


def test_create_vector_store_returns_milvus_implementation(tmp_path: Path, monkeypatch) -> None:
    _patch_milvus_dependencies(monkeypatch)

    store = create_vector_store(
        model_name="BAAI/bge-small-zh-v1.5",
        db_path=str(tmp_path / "factory_milvus"),
        collection_name=f"factory_{uuid4().hex[:8]}",
        sparse_provider="none",
        backend="milvus",
        milvus_uri="http://milvus.test:19530",
    )

    assert isinstance(store, MilvusVectorStore)


def test_milvus_vector_store_persists_search_keyword_delete_and_list(tmp_path: Path, monkeypatch) -> None:
    _patch_milvus_dependencies(monkeypatch)
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

    listed_chunks = store.list_chunks()
    assert len(listed_chunks) == 2
    assert {chunk["document_id"] for chunk in listed_chunks} == {"doc_alpha", "doc_beta"}

    _, results = store.search("alpha", top_k=2)
    assert results
    assert results[0]["document_id"] == "doc_alpha"
    assert results[0]["embedding"] is not None

    keyword_results = store.keyword_search(["beta"], min_hits=1, top_k=2)
    assert keyword_results
    assert keyword_results[0]["document_id"] == "doc_beta"

    reloaded = _build_vector_store(tmp_path, collection_name=store.collection_name)
    _, reloaded_results = reloaded.search("beta", top_k=2)
    assert reloaded_results
    assert reloaded_results[0]["document_id"] == "doc_beta"

    store.delete_document("doc_beta")
    _, after_delete_results = store.search("beta", top_k=2)
    assert all(item["document_id"] != "doc_beta" for item in after_delete_results)
    assert store.keyword_search(["beta"], min_hits=1, top_k=2) == []


def test_milvus_vector_store_reset_and_delete_collection(tmp_path: Path, monkeypatch) -> None:
    _patch_milvus_dependencies(monkeypatch)
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
            }
        ]
    )

    store.reset_collection()
    assert store.list_chunks() == []
    assert store.keyword_search(["alpha"], min_hits=1, top_k=1) == []

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
            }
        ]
    )
    assert store.list_chunks()

    store.delete_collection()
    assert store.list_chunks() == []
    _, results = store.search("beta", top_k=1)
    assert results == []


def test_pipeline_ingest_file_writes_markdown_chunks_into_milvus(tmp_path: Path, monkeypatch) -> None:
    _patch_milvus_dependencies(monkeypatch)
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
    listed_chunks = vector_store.list_chunks()
    assert len(listed_chunks) == chunk_count
    assert any("Alpha 手册第一章 总则第一条" in str(chunk["parent_text"]) for chunk in listed_chunks)

    _, results = vector_store.search("alpha", top_k=3)
    assert results
    assert any(item["document_id"] == "doc_markdown" for item in results)


def test_pipeline_ingest_file_writes_structured_json_chunks_into_milvus(tmp_path: Path, monkeypatch) -> None:
    _patch_milvus_dependencies(monkeypatch)
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
    listed_chunks = vector_store.list_chunks()
    assert len(listed_chunks) == 3
    source_types = {chunk["source_type"] for chunk in listed_chunks}
    assert source_types == {"text", "table", "image_ocr"}

    _, table_results = vector_store.search("beta 表格", top_k=3)
    assert any(item["source_type"] == "table" for item in table_results)

    _, image_results = vector_store.search("gamma 图片", top_k=3)
    assert any(item["source_type"] == "image_ocr" for item in image_results)
