"""向量库适配层。

当前支持本地 Chroma 和 Milvus 两种后端，并在两者外面提供相同的读写接口。
稀疏检索索引仍使用本地实现，作为 dense retrieval 的补充信号。
"""

import hashlib
import logging
import os
import sys
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import chromadb
import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer

from ..core.config import get_settings
from ..core.file_lock import advisory_file_lock
from .retrievers import build_sparse_index, normalize_sparse_provider
from .retrievers.base import SparseIndex
from .retrievers.bm25_sparse import bm25_keyword_matches
from .retrievers.keyword_scanning_sparse import scan_keyword_matches

if os.name == "nt":  # pragma: no cover - platform specific
    import msvcrt
else:  # pragma: no cover - platform specific
    import fcntl

logger = logging.getLogger(__name__)


def _model_lock_path(component: str, model_name: str) -> Path:
    settings = get_settings()
    digest = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:16]
    return settings.resolved_chroma_path / ".model-bootstrap-locks" / f"{component}_{digest}.lock"


@lru_cache(maxsize=None)
def _load_embedding_components(model_name: str) -> tuple[object, object]:
    """加载 embedding tokenizer/model，并用文件锁避免多进程同时下载或初始化。"""
    settings = get_settings()
    lock_path = _model_lock_path("embedding", model_name)
    wait_started = time.perf_counter()
    logger.info("embedding_model_bootstrap_waiting model=%s lock_path=%s", model_name, lock_path)
    with advisory_file_lock(lock_path, timeout_seconds=float(settings.pipeline_bootstrap_lock_timeout_seconds)):
        logger.info(
            "embedding_model_bootstrap_started model=%s wait_ms=%s",
            model_name,
            int((time.perf_counter() - wait_started) * 1000),
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        logger.info(
            "embedding_model_bootstrap_completed model=%s total_ms=%s",
            model_name,
            int((time.perf_counter() - wait_started) * 1000),
        )
        return tokenizer, model


def _load_pymilvus_client_class():
    try:
        from pymilvus import MilvusClient
    except ImportError as exc:
        raise RuntimeError("pymilvus is not installed. Install pymilvus before using the Milvus vector store.") from exc
    return MilvusClient


def _is_local_milvus_uri(uri: str) -> bool:
    return "://" not in uri


def _escape_milvus_string_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


@runtime_checkable
class VectorStoreBackend(Protocol):
    """RAGPipeline 依赖的最小向量库接口。"""

    retrieval_sparse_provider: str

    def reset_collection(self) -> None: ...

    def delete_collection(self) -> None: ...

    def delete_document(self, document_id: str) -> None: ...

    def add_documents(self, chunks_data: list[dict[str, object]]) -> None: ...

    def list_chunks(self) -> list[dict[str, object]]: ...

    def search(self, query: str, top_k: int = 5) -> tuple[list[float], list[dict[str, object]]]: ...

    def keyword_search(
        self,
        query_keywords: list[str],
        *,
        min_hits: int = 2,
        top_k: int = 5,
    ) -> list[dict[str, object]]: ...


class ChromaVectorStore:
    """本地 Chroma 实现，适合开发环境和轻量部署。"""

    def __init__(
        self,
        model_name: str,
        db_path: str,
        collection_name: str,
        *,
        sparse_provider: str = "sqlite_fts",
    ) -> None:
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.tokenizer, self.model = _load_embedding_components(model_name)
        self.sparse_provider = normalize_sparse_provider(sparse_provider)
        self.retrieval_sparse_provider = self.sparse_provider
        self.sparse_index = build_sparse_index(
            db_path=db_path,
            collection_name=collection_name,
            sparse_provider=self.sparse_provider,
        )
        self.write_lock_path = Path(db_path) / ".chroma-write.lock"

    def _get_sparse_index(self) -> SparseIndex:
        sparse_index = getattr(self, "sparse_index", None)
        if sparse_index is None:
            sparse_index = build_sparse_index(
                db_path=getattr(self, "db_path", None),
                collection_name=self.collection_name,
                sparse_provider=getattr(self, "sparse_provider", None),
            )
            self.sparse_index = sparse_index
        return sparse_index

    def _get_embedding(self, text: str, is_query: bool = False) -> list[float]:
        if is_query:
            text = "为这个句子生成表示以用于检索相关文章：" + text

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state[:, 0]
        normalized_embeddings = functional.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings.squeeze().tolist()

    @staticmethod
    def _generate_id(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def reset_collection(self) -> None:
        with self._write_lock():
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            self._get_sparse_index().clear_collection()

    def delete_collection(self) -> None:
        with self._write_lock():
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
            self._get_sparse_index().clear_collection()

    def delete_document(self, document_id: str) -> None:
        with self._write_lock():
            try:
                self.collection.delete(where={"document_id": document_id})
            except Exception:
                pass
            self._get_sparse_index().delete_document(document_id)

    def add_documents(self, chunks_data: list[dict[str, object]]) -> None:
        """写入新 chunk，并同步更新稀疏索引。"""
        if not chunks_data:
            return

        started = time.perf_counter()
        logger.info(
            "chroma_add_documents_started collection=%s chunks=%s",
            self.collection_name,
            len(chunks_data),
        )
        with self._write_lock():
            logger.info(
                "chroma_add_documents_lock_acquired collection=%s chunks=%s",
                self.collection_name,
                len(chunks_data),
            )
            ids = [self._generate_id(item["child_text"] + item["parent_id"]) for item in chunks_data]
            existing = self.collection.get(ids=ids, include=[])
            existing_ids = set(existing.get("ids", []))

            # chunk id 由 child_text + parent_id 决定，重复入库时跳过已存在内容。
            new_items = []
            for item, doc_id in zip(chunks_data, ids):
                if doc_id not in existing_ids:
                    new_items.append((item, doc_id))

            if not new_items:
                logger.info(
                    "chroma_add_documents_skipped_existing collection=%s chunks=%s total_ms=%s",
                    self.collection_name,
                    len(chunks_data),
                    int((time.perf_counter() - started) * 1000),
                )
                return

            embed_started = time.perf_counter()
            embeddings = [self._get_embedding(item["child_text"]) for item, _ in new_items]
            logger.info(
                "chroma_add_documents_embeddings_ready collection=%s new_items=%s embedding_ms=%s",
                self.collection_name,
                len(new_items),
                int((time.perf_counter() - embed_started) * 1000),
            )
            documents = [item["child_text"] for item, _ in new_items]
            metadatas = [
                {
                    "document_id": str(item.get("document_id") or ""),
                    "parent_text": item["parent_text"],
                    "parent_id": item["parent_id"],
                    "source_type": str(item.get("source_type") or "text"),
                    "page_number": int(item.get("page_number", 0) or 0),
                    "block_index": int(item.get("block_index", -1) or -1),
                    "ocr_quality": str(item.get("ocr_quality") or ""),
                    "ocr_avg_confidence": float(item.get("ocr_avg_confidence") or -1.0),
                    "ocr_min_confidence": float(item.get("ocr_min_confidence") or -1.0),
                    "ocr_low_confidence_line_count": int(item.get("ocr_low_confidence_line_count") or 0),
                }
                for item, _ in new_items
            ]
            new_ids = [doc_id for _, doc_id in new_items]

            self.collection.upsert(
                ids=new_ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            sparse_started = time.perf_counter()
            self._get_sparse_index().upsert_documents(new_items)
            logger.info(
                "chroma_add_documents_completed collection=%s new_items=%s sparse_ms=%s total_ms=%s",
                self.collection_name,
                len(new_items),
                int((time.perf_counter() - sparse_started) * 1000),
                int((time.perf_counter() - started) * 1000),
            )

    def list_chunks(self) -> list[dict[str, object]]:
        results = self.collection.get(include=["documents", "metadatas"])
        docs = results.get("documents") or []
        ids = results.get("ids") or []
        metas = results.get("metadatas") or []

        chunks: list[dict[str, object]] = []
        for doc, doc_id, meta in zip(docs, ids, metas):
            chunks.append(
                {
                    "id": doc_id,
                    "child_text": str(doc or ""),
                    "parent_text": str(meta.get("parent_text") or ""),
                    "parent_id": meta.get("parent_id"),
                    "document_id": meta.get("document_id"),
                    "source_type": meta.get("source_type"),
                    "page_number": meta.get("page_number"),
                    "block_index": meta.get("block_index"),
                    "ocr_quality": meta.get("ocr_quality"),
                    "ocr_avg_confidence": meta.get("ocr_avg_confidence"),
                    "ocr_min_confidence": meta.get("ocr_min_confidence"),
                    "ocr_low_confidence_line_count": meta.get("ocr_low_confidence_line_count"),
                    "distance": None,
                    "embedding": None,
                }
            )
        return chunks

    def search(self, query: str, top_k: int = 5) -> tuple[list[float], list[dict[str, object]]]:
        query_vector = self._get_embedding(query, is_query=True)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "distances", "embeddings", "metadatas"],
        )

        structured_results: list[dict[str, object]] = []
        if results["documents"] and results["documents"][0]:
            docs = results["documents"][0]
            dists = results["distances"][0]
            ids = results["ids"][0]
            embs = results["embeddings"][0]
            metas = results["metadatas"][0]

            for doc, dist, doc_id, emb, meta in zip(docs, dists, ids, embs, metas):
                structured_results.append(
                    {
                        "id": doc_id,
                        "child_text": doc,
                        "parent_text": meta["parent_text"],
                        "parent_id": meta["parent_id"],
                        "document_id": meta.get("document_id"),
                        "source_type": meta.get("source_type"),
                        "page_number": meta.get("page_number"),
                        "block_index": meta.get("block_index"),
                        "ocr_quality": meta.get("ocr_quality"),
                        "ocr_avg_confidence": meta.get("ocr_avg_confidence"),
                        "ocr_min_confidence": meta.get("ocr_min_confidence"),
                        "ocr_low_confidence_line_count": meta.get("ocr_low_confidence_line_count"),
                        "distance": float(dist) if dist is not None else None,
                        "embedding": emb,
                    }
                )

        return query_vector, structured_results

    def keyword_search(
        self,
        query_keywords: list[str],
        *,
        min_hits: int = 2,
        top_k: int = 5,
    ) -> list[dict[str, object]]:
        if self.sparse_provider == "bm25":
            return bm25_keyword_matches(
                self.list_chunks(),
                query_keywords,
                min_hits=min_hits,
                top_k=top_k,
            )
        if self.sparse_provider == "scan":
            return scan_keyword_matches(
                self.list_chunks(),
                query_keywords,
                min_hits=min_hits,
                top_k=top_k,
            )
        if self.sparse_provider == "none":
            return []
        return self._get_sparse_index().search(
            query_keywords,
            min_hits=min_hits,
            top_k=top_k,
        )

    @contextmanager
    def _write_lock(self):
        """保护 Chroma 和本地稀疏索引的复合写操作。"""
        lock_path = getattr(self, "write_lock_path", None)
        if lock_path is None:
            lock_path = Path(str(getattr(self, "db_path", "."))) / ".chroma-write.lock"
            self.write_lock_path = lock_path
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        wait_started = time.perf_counter()
        with lock_path.open("a+b") as handle:
            self._acquire_file_lock(handle)
            logger.info(
                "chroma_write_lock_acquired collection=%s wait_ms=%s path=%s",
                self.collection_name,
                int((time.perf_counter() - wait_started) * 1000),
                lock_path,
            )
            try:
                yield
            finally:
                self._release_file_lock(handle)
                logger.info(
                    "chroma_write_lock_released collection=%s path=%s",
                    self.collection_name,
                    lock_path,
                )

    @staticmethod
    def _acquire_file_lock(handle, *, timeout_seconds: float = 120.0, poll_interval_seconds: float = 0.1) -> None:
        if os.name == "nt":  # pragma: no cover - Windows dev fallback
            return
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # pragma: no cover - platform specific
                return
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out waiting for Chroma write lock")
                time.sleep(poll_interval_seconds)

    @staticmethod
    def _release_file_lock(handle) -> None:
        if os.name == "nt":  # pragma: no cover - Windows dev fallback
            return
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # pragma: no cover - platform specific


class MilvusVectorStore:
    """Milvus 实现，面向未来服务器 GPU/独立向量库部署。"""

    def __init__(
        self,
        model_name: str,
        db_path: str,
        collection_name: str,
        *,
        milvus_uri: str,
        milvus_token: str | None = None,
        sparse_provider: str = "sqlite_fts",
    ) -> None:
        if _is_local_milvus_uri(milvus_uri) and sys.platform.startswith("win"):
            raise RuntimeError(
                "Milvus Lite local file mode is not supported on Windows. Set VECTOR_STORE_MILVUS_URI to a remote "
                "Milvus endpoint such as http://127.0.0.1:19530 or a Docker service address."
            )
        MilvusClient = _load_pymilvus_client_class()
        connection_args: dict[str, object] = {"uri": milvus_uri}
        if milvus_token:
            connection_args["token"] = milvus_token

        self.db_path = db_path
        self.client = MilvusClient(**connection_args)
        self.collection_name = collection_name
        self.tokenizer, self.model = _load_embedding_components(model_name)
        self.sparse_provider = normalize_sparse_provider(sparse_provider)
        self.retrieval_sparse_provider = self.sparse_provider
        self.sparse_index = build_sparse_index(
            db_path=db_path,
            collection_name=collection_name,
            sparse_provider=self.sparse_provider,
        )
        self.search_params = {"metric_type": "L2", "params": {}}
        self.write_lock_path = Path(db_path) / ".milvus-write.lock"

    def _get_sparse_index(self) -> SparseIndex:
        """懒加载稀疏索引，兼容测试里手工构造的轻量实例。"""
        sparse_index = getattr(self, "sparse_index", None)
        if sparse_index is None:
            sparse_index = build_sparse_index(
                db_path=getattr(self, "db_path", None),
                collection_name=self.collection_name,
                sparse_provider=getattr(self, "sparse_provider", None),
            )
            self.sparse_index = sparse_index
        return sparse_index

    def _get_embedding(self, text: str, is_query: bool = False) -> list[float]:
        if is_query:
            # E5 类 embedding 模型建议给 query 加检索前缀，文档向量保持原文。
            text = "为这个句子生成表示以用于检索相关文章：" + text

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state[:, 0]
        normalized_embeddings = functional.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings.squeeze().tolist()

    @staticmethod
    def _generate_id(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def reset_collection(self) -> None:
        with self._write_lock():
            self._drop_collection_if_exists()
            self._ensure_collection(vector_dim=len(self._get_embedding("reset collection bootstrap probe")))
            self._get_sparse_index().clear_collection()

    def delete_collection(self) -> None:
        with self._write_lock():
            self._drop_collection_if_exists()
            self._get_sparse_index().clear_collection()

    def delete_document(self, document_id: str) -> None:
        with self._write_lock():
            if self._has_collection():
                self.client.delete(
                    self.collection_name,
                    filter=f'document_id == "{_escape_milvus_string_literal(document_id)}"',
                )
                self.client.flush(self.collection_name)
            self._get_sparse_index().delete_document(document_id)

    def add_documents(self, chunks_data: list[dict[str, object]]) -> None:
        """写入 Milvus collection，并同步维护本地稀疏索引。"""
        if not chunks_data:
            return

        started = time.perf_counter()
        logger.info(
            "milvus_add_documents_started collection=%s chunks=%s",
            self.collection_name,
            len(chunks_data),
        )
        with self._write_lock():
            ids = [self._generate_id(str(item["child_text"]) + str(item["parent_id"])) for item in chunks_data]
            embed_started = time.perf_counter()
            # Milvus collection 需要知道向量维度，因此先生成首批 embedding 再建表。
            embeddings = [self._get_embedding(str(item["child_text"])) for item in chunks_data]
            logger.info(
                "milvus_add_documents_embeddings_ready collection=%s chunks=%s embedding_ms=%s",
                self.collection_name,
                len(chunks_data),
                int((time.perf_counter() - embed_started) * 1000),
            )
            self._ensure_collection(vector_dim=len(embeddings[0]))
            rows = [
                {
                    "chunk_id": chunk_id,
                    "vector": embedding,
                    "parent_text": str(item.get("parent_text") or ""),
                    "parent_id": str(item.get("parent_id") or ""),
                    "document_id": str(item.get("document_id") or ""),
                    "source_type": str(item.get("source_type") or "text"),
                    "page_number": int(item.get("page_number")) if item.get("page_number") is not None else -1,
                    "block_index": int(item.get("block_index")) if item.get("block_index") is not None else -1,
                    "text": str(item.get("child_text") or ""),
                }
                for item, chunk_id, embedding in zip(chunks_data, ids, embeddings)
            ]
            self.client.upsert(self.collection_name, rows)
            self.client.flush(self.collection_name)
            self.client.load_collection(self.collection_name)
            self._get_sparse_index().upsert_documents(list(zip(chunks_data, ids)))
            logger.info(
                "milvus_add_documents_completed collection=%s chunks=%s total_ms=%s",
                self.collection_name,
                len(chunks_data),
                int((time.perf_counter() - started) * 1000),
            )

    def list_chunks(self) -> list[dict[str, object]]:
        if not self._has_collection():
            return []

        # query_iterator 避免一次性把大 collection 拉进内存。
        iterator = self.client.query_iterator(
            self.collection_name,
            batch_size=1000,
            limit=-1,
            filter='chunk_id != ""',
            output_fields=[
                "chunk_id",
                "parent_text",
                "parent_id",
                "document_id",
                "source_type",
                "page_number",
                "block_index",
                "text",
            ],
        )
        chunks: list[dict[str, object]] = []
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                for entity in batch:
                    chunks.append(self._build_result_doc(entity))
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
        return chunks

    def search(self, query: str, top_k: int = 5) -> tuple[list[float], list[dict[str, object]]]:
        query_vector = self._get_embedding(query, is_query=True)
        if not self._has_collection():
            return query_vector, []

        self.client.load_collection(self.collection_name)
        results = self.client.search(
            self.collection_name,
            data=[query_vector],
            anns_field="vector",
            search_params=self.search_params,
            limit=top_k,
            output_fields=[
                "chunk_id",
                "parent_text",
                "parent_id",
                "document_id",
                "source_type",
                "page_number",
                "block_index",
                "text",
                "vector",
            ],
        )

        structured_results: list[dict[str, object]] = []
        for raw_result in (results[0] if results else []):
            entity = raw_result.get("entity") or raw_result
            structured_results.append(
                self._build_result_doc(
                    entity,
                    distance=float(raw_result.get("distance") or 0.0),
                    embedding=entity.get("vector"),
                )
            )
        return query_vector, structured_results

    def keyword_search(
        self,
        query_keywords: list[str],
        *,
        min_hits: int = 2,
        top_k: int = 5,
    ) -> list[dict[str, object]]:
        """按配置选择稀疏检索实现，作为 dense search 的补充召回。"""
        if self.sparse_provider == "bm25":
            return bm25_keyword_matches(
                self.list_chunks(),
                query_keywords,
                min_hits=min_hits,
                top_k=top_k,
            )
        if self.sparse_provider == "scan":
            return scan_keyword_matches(
                self.list_chunks(),
                query_keywords,
                min_hits=min_hits,
                top_k=top_k,
            )
        if self.sparse_provider == "none":
            return []
        return self._get_sparse_index().search(
            query_keywords,
            min_hits=min_hits,
            top_k=top_k,
        )

    def _build_result_doc(
        self,
        entity: dict[str, Any],
        *,
        distance: float | None = None,
        embedding: list[float] | None = None,
    ) -> dict[str, object]:
        return {
            "id": str(entity.get("chunk_id") or ""),
            "child_text": str(entity.get("text") or ""),
            "parent_text": str(entity.get("parent_text") or ""),
            "parent_id": str(entity.get("parent_id") or ""),
            "document_id": str(entity.get("document_id") or ""),
            "source_type": str(entity.get("source_type") or "text"),
            "page_number": self._optional_positive_int(entity.get("page_number")),
            "block_index": self._optional_positive_int(entity.get("block_index")),
            "distance": distance,
            "embedding": list(embedding) if embedding is not None else None,
        }

    @staticmethod
    def _optional_positive_int(value: object) -> int | None:
        if value is None:
            return None
        number = int(value)
        return None if number < 0 else number

    def _has_collection(self) -> bool:
        try:
            return bool(self.client.has_collection(self.collection_name))
        except Exception:
            return False

    def _drop_collection_if_exists(self) -> None:
        if self._has_collection():
            self.client.drop_collection(self.collection_name)

    def _ensure_collection(self, *, vector_dim: int | None = None) -> None:
        if self._has_collection():
            self.client.load_collection(self.collection_name)
            return
        if vector_dim is None:
            raise RuntimeError(f"Cannot create Milvus collection {self.collection_name}: missing vector dimension")

        from pymilvus import DataType

        # schema 字段保持和 Chroma metadata 对齐，便于 pipeline 无差别读取。
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=256)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=vector_dim)
        schema.add_field("parent_text", DataType.VARCHAR, max_length=65535)
        schema.add_field("parent_id", DataType.VARCHAR, max_length=512)
        schema.add_field("document_id", DataType.VARCHAR, max_length=512)
        schema.add_field("source_type", DataType.VARCHAR, max_length=128)
        schema.add_field("page_number", DataType.INT64)
        schema.add_field("block_index", DataType.INT64)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)

        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="FLAT", metric_type="L2")
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        self.client.load_collection(self.collection_name)

    @contextmanager
    def _write_lock(self):
        """Milvus 写入和本地稀疏索引更新必须一起串行化。"""
        settings = get_settings()
        wait_started = time.perf_counter()
        logger.info("milvus_write_lock_waiting collection=%s path=%s", self.collection_name, self.write_lock_path)
        with advisory_file_lock(
            self.write_lock_path,
            timeout_seconds=float(settings.pipeline_bootstrap_lock_timeout_seconds),
        ):
            logger.info(
                "milvus_write_lock_acquired collection=%s wait_ms=%s path=%s",
                self.collection_name,
                int((time.perf_counter() - wait_started) * 1000),
                self.write_lock_path,
            )
            yield
            logger.info("milvus_write_lock_released collection=%s path=%s", self.collection_name, self.write_lock_path)


def create_vector_store(
    *,
    model_name: str,
    db_path: str,
    collection_name: str,
    sparse_provider: str = "sqlite_fts",
    backend: str = "chroma",
    milvus_uri: str | None = None,
    milvus_token: str | None = None,
) -> VectorStoreBackend:
    """根据配置创建具体向量库后端。"""
    normalized_backend = (backend or "chroma").strip().lower()
    if normalized_backend == "chroma":
        return ChromaVectorStore(
            model_name=model_name,
            db_path=db_path,
            collection_name=collection_name,
            sparse_provider=sparse_provider,
        )
    if normalized_backend == "milvus":
        if not milvus_uri:
            raise ValueError("Milvus vector store backend requires a Milvus URI")
        return MilvusVectorStore(
            model_name=model_name,
            db_path=db_path,
            collection_name=collection_name,
            milvus_uri=milvus_uri,
            milvus_token=milvus_token,
            sparse_provider=sparse_provider,
        )
    raise ValueError(f"Unsupported vector store backend: {backend}")


# 兼容旧导入：pipeline 迁移到工厂创建后，仍保留 VectorStore 名称。
VectorStore = ChromaVectorStore
