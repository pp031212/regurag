"""shadow graph 使用的可替换检索后端。

这里把 LangChain Chroma、shadow Milvus 和 legacy retriever 包装成同一个 HybridRetriever
形态，用于对照不同检索实现的结果差异。
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any, cast

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ...rag.retrievers import HybridRetriever, RetrievalPolicy, build_default_hybrid_retriever, build_sparse_retriever
from ...rag.retrievers.base import DenseRetriever
from ...rag.vector_store import ChromaVectorStore
from .pipeline_steps import RagDoc

def _load_pymilvus_client_class():
    try:
        from pymilvus import MilvusClient
    except ImportError as exc:
        raise RuntimeError(
            "pymilvus is not installed. Install pymilvus before using the langchain_milvus shadow retrieval backend."
        ) from exc
    return MilvusClient


class VectorStoreEmbeddings(Embeddings):
    """把现有 VectorStore embedding 方法适配给 LangChain。"""

    def __init__(self, vector_store: ChromaVectorStore) -> None:
        self.vector_store = vector_store

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.vector_store._get_embedding(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.vector_store._get_embedding(text, is_query=True)


class MilvusSearchStore:
    """shadow Milvus 的最小 search 封装。"""

    def __init__(
        self,
        *,
        connection_args: dict[str, object],
        collection_name: str,
        output_fields: list[str],
        search_params: dict[str, object] | None = None,
        timeout: float | None = None,
    ) -> None:
        MilvusClient = _load_pymilvus_client_class()
        self.client = MilvusClient(**connection_args)
        self.collection_name = collection_name
        self.output_fields = output_fields
        self.search_params = search_params or {"metric_type": "L2", "params": {}}
        self.timeout = timeout

    def has_collection(self) -> bool:
        return bool(self.client.has_collection(self.collection_name))

    def ensure_loaded(self) -> None:
        self.client.load_collection(self.collection_name)

    def search(self, query_vector: list[float], *, top_k: int) -> list[dict[str, Any]]:
        results = self.client.search(
            self.collection_name,
            data=[query_vector],
            anns_field="vector",
            search_params=self.search_params,
            limit=top_k,
            output_fields=self.output_fields,
            timeout=self.timeout,
        )
        if not results:
            return []
        return cast(list[dict[str, Any]], results[0])


class LangChainChromaDenseRetriever(DenseRetriever):
    """使用 LangChain Chroma wrapper 的 dense retriever。"""

    def __init__(self, vector_store: ChromaVectorStore) -> None:
        self.vector_store = vector_store
        self.embeddings = VectorStoreEmbeddings(vector_store)
        self.store = Chroma(
            client=vector_store.client,
            collection_name=vector_store.collection_name,
            embedding_function=self.embeddings,
        )

    def search(self, query: str, top_k: int) -> tuple[list[float], list[RagDoc]]:
        query_vector = self.embeddings.embed_query(query)
        docs_with_scores = self.store.similarity_search_with_score(query, k=top_k)
        if not docs_with_scores:
            return query_vector, []

        chunk_ids = [self._derive_chunk_id(doc) for doc, _ in docs_with_scores]
        embedding_lookup = self._load_chunk_embeddings(chunk_ids)
        results: list[RagDoc] = []
        for doc, score in docs_with_scores:
            metadata = cast(dict[str, Any], doc.metadata or {})
            chunk_id = self._derive_chunk_id(doc)
            results.append(
                {
                    "id": chunk_id,
                    "child_text": doc.page_content,
                    "parent_text": str(metadata.get("parent_text") or ""),
                    "parent_id": str(metadata.get("parent_id") or ""),
                    "document_id": str(metadata.get("document_id") or ""),
                    "source_type": str(metadata.get("source_type") or "text"),
                    "page_number": _as_optional_int(metadata.get("page_number")),
                    "block_index": _as_optional_int(metadata.get("block_index")),
                    "distance": float(score),
                    "embedding": embedding_lookup.get(chunk_id),
                }
            )
        return query_vector, results

    def _derive_chunk_id(self, doc: Document) -> str:
        metadata = cast(dict[str, Any], doc.metadata or {})
        chunk_id = metadata.get("chunk_id")
        if chunk_id:
            return str(chunk_id)
        parent_id = str(metadata.get("parent_id") or "")
        return self.vector_store._generate_id(doc.page_content + parent_id)

    def _load_chunk_embeddings(self, chunk_ids: list[str]) -> dict[str, list[float]]:
        """LangChain 查询结果不一定带 embedding，需要回查 Chroma collection。"""
        if not chunk_ids:
            return {}
        results = self.vector_store.collection.get(ids=chunk_ids, include=["embeddings"])
        raw_ids = results.get("ids")
        raw_embeddings = results.get("embeddings")
        ids = cast(list[str], raw_ids if raw_ids is not None else [])
        embeddings = cast(list[list[float] | None], raw_embeddings if raw_embeddings is not None else [])
        return {
            str(chunk_id): list(embedding)
            for chunk_id, embedding in zip(ids, embeddings)
            if embedding is not None
        }


class LangChainMilvusDenseRetriever(DenseRetriever):
    """使用 shadow Milvus collection 的 dense retriever。"""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        *,
        uri: str,
        token: str | None = None,
        drop_old: bool = True,
    ) -> None:
        self.vector_store = vector_store
        self.embeddings = VectorStoreEmbeddings(vector_store)
        self.uri = uri
        self.token = token
        self.drop_old = drop_old
        self.store = self._build_store()

    def _build_store(self, *, drop_old: bool | None = None):
        """按需从当前 vector_store chunk 重建 shadow Milvus collection。"""
        if _is_local_milvus_uri(self.uri) and sys.platform.startswith("win"):
            raise RuntimeError(
                "Milvus Lite local file mode is not supported on Windows. Set SHADOW_MILVUS_URI to a remote "
                "Milvus endpoint such as http://127.0.0.1:19530 or a Docker service address."
            )
        chunks = self.vector_store.list_chunks()
        if not chunks:
            return None
        collection_name = _shadow_milvus_collection_name(self.vector_store.collection_name)
        connection_args: dict[str, object] = {"uri": self.uri}
        if self.token:
            connection_args["token"] = self.token
        effective_drop_old = self.drop_old if drop_old is None else drop_old
        search_store = _build_milvus_search_store(
            connection_args=connection_args,
            collection_name=collection_name,
        )
        if search_store.has_collection() and not effective_drop_old:
            return search_store
        try:
            _bootstrap_milvus_collection(
                client=search_store.client,
                collection_name=collection_name,
                chunks=chunks,
                embeddings=self.embeddings,
                drop_old=effective_drop_old,
            )
            return _build_milvus_search_store(
                connection_args=connection_args,
                collection_name=collection_name,
            )
        except Exception as exc:
            if "milvus-lite is required" in str(exc).lower():
                raise RuntimeError(
                    "Milvus Lite support is missing. Install pymilvus[milvus_lite] before using a local "
                    "SHADOW_MILVUS_URI file."
                ) from exc
            raise

    def search(self, query: str, top_k: int) -> tuple[list[float], list[RagDoc]]:
        query_vector = self.embeddings.embed_query(query)
        if self.store is None:
            return query_vector, []
        raw_results = self._search_with_recovery(query_vector, top_k=top_k)
        if not raw_results:
            return query_vector, []

        results: list[RagDoc] = []
        for raw_result in raw_results:
            entity = cast(dict[str, Any], raw_result.get("entity") or {})
            doc = _parse_milvus_document(entity)
            metadata = cast(dict[str, Any], doc.metadata or {})
            results.append(
                _build_rag_doc(
                    chunk_id=str(metadata.get("chunk_id") or ""),
                    page_content=doc.page_content,
                    metadata=metadata,
                    distance=float(raw_result.get("distance") or 0.0),
                    embedding=self.vector_store._get_embedding(doc.page_content),
                )
            )
        return query_vector, results

    def _search_by_vector(self, query_vector: list[float], *, top_k: int) -> list[dict[str, Any]]:
        return self.store.search(query_vector, top_k=top_k)

    def _search_with_recovery(self, query_vector: list[float], *, top_k: int) -> list[dict[str, Any]]:
        try:
            self._ensure_collection_exists()
            return self._search_by_vector(query_vector, top_k=top_k)
        except Exception as exc:
            # shadow Milvus 只用于对照，遇到可恢复错误时重建 collection 后再试一次。
            if not _is_retryable_milvus_error(exc):
                raise
            self.store = self._build_store(drop_old=_should_force_rebootstrap(exc))
            if self.store is None:
                return []
            return self._search_by_vector(query_vector, top_k=top_k)

    def _ensure_collection_exists(self) -> None:
        if self.store.has_collection():
            self.store.ensure_loaded()
            return
        self.store = self._build_store(drop_old=False)
        if self.store is not None:
            self.store.ensure_loaded()


@dataclass(slots=True)
class ShadowRetrieverBuilder:
    """根据配置构造 shadow graph 使用的 retriever。"""

    vector_store: object
    sparse_provider: str | None
    policy: RetrievalPolicy
    milvus_uri: str | None = None
    milvus_token: str | None = None
    milvus_drop_old: bool = True

    def build(self, *, backend: str) -> HybridRetriever:
        normalized_backend = (backend or "legacy").strip().lower()
        if normalized_backend == "legacy":
            return build_default_hybrid_retriever(
                self.vector_store,
                sparse_provider=self.sparse_provider,
                policy=self.policy,
            )
        if normalized_backend == "langchain_chroma":
            if not isinstance(self.vector_store, ChromaVectorStore):
                raise ValueError("langchain_chroma shadow backend currently requires ChromaVectorStore")
            return HybridRetriever(
                dense_retriever=LangChainChromaDenseRetriever(self.vector_store),
                sparse_retriever=build_sparse_retriever(self.vector_store, self.sparse_provider),
                policy=self.policy,
            )
        if normalized_backend == "langchain_milvus":
            if not isinstance(self.vector_store, ChromaVectorStore):
                raise ValueError("langchain_milvus shadow backend currently requires ChromaVectorStore")
            if not self.milvus_uri:
                raise ValueError("langchain_milvus shadow backend requires a Milvus URI")
            return HybridRetriever(
                dense_retriever=LangChainMilvusDenseRetriever(
                    self.vector_store,
                    uri=self.milvus_uri,
                    token=self.milvus_token,
                    drop_old=self.milvus_drop_old,
                ),
                sparse_retriever=build_sparse_retriever(self.vector_store, self.sparse_provider),
                policy=self.policy,
            )
        raise ValueError(f"Unsupported shadow retrieval backend: {backend}")


def build_shadow_retriever(
    vector_store: object,
    *,
    backend: str,
    sparse_provider: str | None,
    policy: RetrievalPolicy | None = None,
    milvus_uri: str | None = None,
    milvus_token: str | None = None,
    milvus_drop_old: bool = True,
) -> HybridRetriever:
    """对外工厂：legacy/langchain_chroma/langchain_milvus 三种 shadow 后端。"""
    return ShadowRetrieverBuilder(
        vector_store=vector_store,
        sparse_provider=sparse_provider,
        policy=policy or RetrievalPolicy(),
        milvus_uri=milvus_uri,
        milvus_token=milvus_token,
        milvus_drop_old=milvus_drop_old,
    ).build(backend=backend)


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _build_rag_doc(
    *,
    chunk_id: str,
    page_content: str,
    metadata: dict[str, Any],
    distance: float,
    embedding: list[float] | None,
) -> RagDoc:
    return {
        "id": chunk_id,
        "child_text": page_content,
        "parent_text": str(metadata.get("parent_text") or ""),
        "parent_id": str(metadata.get("parent_id") or ""),
        "document_id": str(metadata.get("document_id") or ""),
        "source_type": str(metadata.get("source_type") or "text"),
        "page_number": _as_optional_int(metadata.get("page_number")),
        "block_index": _as_optional_int(metadata.get("block_index")),
        "distance": distance,
        "embedding": embedding,
    }


def _parse_milvus_document(entity: dict[str, Any]) -> Document:
    data = dict(entity)
    data.pop("vector", None)
    for field_name in ("page_number", "block_index"):
        if data.get(field_name) == -1:
            data[field_name] = None
    return Document(
        page_content=str(data.pop("text", "")),
        metadata=data,
    )


def _shadow_milvus_collection_name(base_collection_name: str) -> str:
    return f"{base_collection_name}_shadow_milvus"


def _is_local_milvus_uri(uri: str) -> bool:
    return "://" not in uri


def _is_retryable_milvus_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "collection not found" in message
        or "should create connection first" in message
        or "collection not loaded" in message
        or "index not found" in message
    )


def _should_force_rebootstrap(exc: Exception) -> bool:
    message = str(exc).lower()
    return "index not found" in message or "collection not loaded" in message


def _build_milvus_search_store(
    *,
    connection_args: dict[str, object],
    collection_name: str,
) -> MilvusSearchStore:
    return MilvusSearchStore(
        connection_args=connection_args,
        collection_name=collection_name,
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


def _bootstrap_milvus_collection(
    *,
    client: Any,
    collection_name: str,
    chunks: list[dict[str, object]],
    embeddings: Embeddings,
    drop_old: bool,
) -> None:
    """把当前 chunk 快照写入 shadow Milvus collection。"""
    from pymilvus import DataType

    if not chunks:
        return

    if drop_old and client.has_collection(collection_name):
        client.drop_collection(collection_name)

    texts = [str(chunk.get("child_text") or "") for chunk in chunks]
    vectors = embeddings.embed_documents(texts)
    dimension = len(vectors[0]) if vectors else 0
    if dimension <= 0:
        raise RuntimeError(f"Cannot bootstrap Milvus collection {collection_name}: empty embedding dimension")

    if not client.has_collection(collection_name):
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=256)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field("parent_text", DataType.VARCHAR, max_length=65535)
        schema.add_field("parent_id", DataType.VARCHAR, max_length=512)
        schema.add_field("document_id", DataType.VARCHAR, max_length=512)
        schema.add_field("source_type", DataType.VARCHAR, max_length=128)
        schema.add_field("page_number", DataType.INT64)
        schema.add_field("block_index", DataType.INT64)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)

        index_params = client.prepare_index_params()
        # shadow 对比更重视结果对齐，不追求 ANN 吞吐，所以使用精确 FLAT 检索。
        index_params.add_index(field_name="vector", index_type="FLAT", metric_type="L2")
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )

    rows = [
        {
            "chunk_id": str(chunk.get("id") or ""),
            "vector": vector,
            "parent_text": str(chunk.get("parent_text") or ""),
            "parent_id": str(chunk.get("parent_id") or ""),
            "document_id": str(chunk.get("document_id") or ""),
            "source_type": str(chunk.get("source_type") or "text"),
            "page_number": int(chunk.get("page_number")) if chunk.get("page_number") is not None else -1,
            "block_index": int(chunk.get("block_index")) if chunk.get("block_index") is not None else -1,
            "text": str(chunk.get("child_text") or ""),
        }
        for chunk, vector in zip(chunks, vectors)
    ]
    client.insert(collection_name, rows)
    client.flush(collection_name)
    client.load_collection(collection_name)
