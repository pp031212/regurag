"""RAG 检索生成流水线。

本文件负责单个知识库内部的核心问答流程：文档入库、查询改写、混合检索、
MMR 去冗余、rerank、上下文拼装和 LLM 生成。面向 API 的路由、会话和持久化逻辑
放在 ``services/rag_service.py``，这里保持为可单独测试的 pipeline。
"""

import json
import logging
from pathlib import Path
from time import perf_counter

from ..core.config import Settings
from ..workflows.rag.graph import build_shadow_graph
from ..workflows.rag.nodes import RagWorkflowDependencies
from ..workflows.rag.shadow_retrievers import build_shadow_retriever
from ..workflows.rag.pipeline_steps import (
    RagDoc,
    apply_source_names,
    build_citations,
    build_debug,
    build_debug_chunk,
    build_debug_chunks,
    build_final_context_parents,
    build_parent_docs_for_rerank,
    collect_overview_article_parents,
    collect_related_restriction_article_parents,
    count_query_keyword_hits,
    dedupe_retrieved_docs,
    extract_article_number,
    extract_query_keywords,
    format_context_parent,
    is_overview_query,
    parse_chinese_numeral,
    preserve_overview_candidate_docs,
    preserve_policy_keyword_docs,
    preserve_query_keyword_docs,
    prettify_legal_text,
    should_treat_as_policy_question,
    sort_reranked_parents,
    sort_retrieved_docs,
    source_priority,
)
from ..workflows.rag.runner import run_shadow_workflow
from ..workflows.rag.state import RagWorkflowState
from .document_processor import DocumentProcessor
from .llm_client import LLMGenerator
from .query_alias_config import QueryAliasExpander
from .query_policy import QueryPolicy
from .query_rewriter import QueryRewriter
from .retrievers import RetrievalPolicy, build_default_hybrid_retriever
from .retrieval_rules_config import load_retrieval_rules_config
from .reranker import Reranker
from .retrieval_utils import mmr_rerank
from .structured_document_processor import StructuredDocumentProcessor
from .vector_store import create_vector_store

logger = logging.getLogger(__name__)


class RAGPipeline:
    """单个知识库 collection 对应的 RAG 执行单元。"""

    def __init__(self, settings: Settings, collection_name: str | None = None, subject: str | None = None) -> None:
        self.settings = settings
        self.processor = DocumentProcessor(child_chunk_size=settings.child_chunk_size)
        self.structured_processor = StructuredDocumentProcessor(self.processor)
        self.vector_store = create_vector_store(
            model_name=settings.embedding_model_name,
            db_path=str(settings.resolved_chroma_path),
            collection_name=collection_name or settings.chroma_collection_name,
            sparse_provider=settings.retrieval_sparse_provider,
            backend=settings.normalized_vector_store_backend,
            milvus_uri=settings.resolved_vector_store_milvus_uri,
            milvus_token=settings.vector_store_milvus_token,
        )
        self.vector_store.retrieval_sparse_provider = settings.retrieval_sparse_provider
        self.retrieval_policy = RetrievalPolicy(
            dense_top_k=settings.retrieval_dense_top_k,
            sparse_top_k=settings.retrieval_sparse_top_k,
            sparse_min_hits=settings.retrieval_sparse_min_hits,
            enable_sparse=settings.retrieval_enable_sparse,
        )
        self.retriever = build_default_hybrid_retriever(
            self.vector_store,
            sparse_provider=settings.retrieval_sparse_provider,
            policy=self.retrieval_policy,
        )
        self.reranker = Reranker(model_name=settings.reranker_model_name)
        self.llm = LLMGenerator(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model_name=settings.openai_model,
            subject=subject or settings.knowledge_base_subject,
            timeout_seconds=settings.openai_timeout_seconds,
            max_tokens=settings.openai_max_tokens,
            fallback_api_key=settings.rewrite_api_key,
            fallback_base_url=settings.rewrite_base_url,
            fallback_model_name=settings.rewrite_model,
        )
        self.rewriter = QueryRewriter(
            api_key=settings.rewrite_api_key,
            base_url=settings.rewrite_base_url,
            model_name=settings.rewrite_model,
        )
        self.query_alias_expander = QueryAliasExpander()
        self.retrieval_rules = load_retrieval_rules_config()

    def reset_index(self) -> None:
        self.vector_store.reset_collection()

    def delete_index(self) -> None:
        self.vector_store.delete_collection()

    def delete_document(self, document_id: str) -> None:
        self.vector_store.delete_document(document_id)

    @staticmethod
    def _build_debug_chunk(doc: dict[str, object]) -> dict[str, object]:
        return build_debug_chunk(doc)

    def _build_debug_chunks(self, docs: list[dict[str, object]]) -> list[dict[str, object]]:
        return build_debug_chunks(docs)

    @staticmethod
    def _prettify_legal_text(text: str) -> str:
        return prettify_legal_text(text)

    @classmethod
    def _format_context_parent(cls, doc: dict[str, object]) -> str:
        return format_context_parent(doc)

    @staticmethod
    def _extract_query_keywords(*parts: str) -> list[str]:
        return extract_query_keywords(*parts)

    def _expand_overview_query_keywords(self, query: str) -> str:
        if not is_overview_query(query):
            return ""

        expansions: list[str] = []
        seen: set[str] = set()
        for trigger, aliases in getattr(self.retrieval_rules, "overview_query_expansions", {}).items():
            if trigger not in query:
                continue
            for alias in aliases:
                alias_text = str(alias).strip()
                if not alias_text or alias_text in seen:
                    continue
                expansions.append(alias_text)
                seen.add(alias_text)
        return " ".join(expansions)

    @staticmethod
    def _is_overview_query(query: str) -> bool:
        return is_overview_query(query)

    @staticmethod
    def _parse_chinese_numeral(value: str) -> int | None:
        return parse_chinese_numeral(value)

    @classmethod
    def _extract_article_number(cls, text: str) -> int | None:
        return extract_article_number(text)

    @staticmethod
    def _count_query_keyword_hits(doc: dict[str, object], query_keywords: list[str]) -> int:
        return count_query_keyword_hits(doc, query_keywords)

    def _preserve_query_keyword_docs(
        self,
        selected_docs: list[dict[str, object]],
        candidate_docs: list[dict[str, object]],
        query_keywords: list[str],
        *,
        min_hits: int = 2,
        max_extra_docs: int = 2,
    ) -> list[dict[str, object]]:
        return preserve_query_keyword_docs(
            selected_docs=selected_docs,
            candidate_docs=candidate_docs,
            query_keywords=query_keywords,
            min_hits=min_hits,
            max_extra_docs=max_extra_docs,
        )

    def _collect_overview_article_parents(
        self,
        parent_docs: list[dict[str, object]],
        query: str,
        query_keywords: list[str],
    ) -> list[dict[str, object]]:
        return collect_overview_article_parents(
            parent_docs=parent_docs,
            query=query,
            query_keywords=query_keywords,
        )

    def _collect_related_restriction_article_parents(
        self,
        parent_docs: list[dict[str, object]],
        final_context_parents: list[dict[str, object]],
        query: str,
    ) -> list[dict[str, object]]:
        return collect_related_restriction_article_parents(
            parent_docs=parent_docs,
            final_context_parents=final_context_parents,
            query=query,
        )

    @staticmethod
    def _source_priority(source_type: str | None) -> int:
        return source_priority(source_type)

    def _page_has_table_source(self, docs: list[dict[str, object]], page_number: int | None) -> bool:
        if page_number is None:
            return False
        return any(
            int(doc.get("page_number") or 0) == int(page_number)
            and str(doc.get("source_type") or "") == "table"
            for doc in docs
        )

    def _retrieval_penalty(self, doc: dict[str, object], docs: list[dict[str, object]]) -> float:
        source_type = str(doc.get("source_type") or "text")
        penalties = {
            "table": 0.0,
            "text": 0.03,
            "image_ocr_table": 0.12,
            "image_ocr": 0.2,
        }
        penalty = penalties.get(source_type, 0.05)
        page_number = int(doc.get("page_number") or 0)
        if source_type == "image_ocr_table" and self._page_has_table_source(docs, page_number):
            penalty += 0.08
        if source_type.startswith("image_ocr") and str(doc.get("ocr_quality") or "") == "low":
            penalty += 0.12
        return penalty

    def _sort_retrieved_docs(self, docs: list[dict[str, object]]) -> list[dict[str, object]]:
        return sort_retrieved_docs(docs)

    def _rerank_penalty(self, doc: dict[str, object], docs: list[dict[str, object]]) -> float:
        source_type = str(doc.get("source_type") or "text")
        penalties = {
            "table": 0.0,
            "text": 0.03,
            "image_ocr_table": 0.18,
            "image_ocr": 0.28,
        }
        penalty = penalties.get(source_type, 0.05)
        page_number = int(doc.get("page_number") or 0)
        if source_type == "image_ocr_table" and self._page_has_table_source(docs, page_number):
            penalty += 0.12
        if source_type.startswith("image_ocr") and str(doc.get("ocr_quality") or "") == "low":
            penalty += 0.18
        return penalty

    def _sort_reranked_parents(self, docs: list[dict[str, object]]) -> list[dict[str, object]]:
        return sort_reranked_parents(docs)

    def _build_debug(
        self,
        rewritten_query: str,
        retrieved_count: int,
        reranked_count: int,
        enable_rewrite: bool,
        enable_rerank: bool,
        rewrite_ms: int,
        retrieve_ms: int,
        rerank_ms: int,
        context_build_ms: int,
        generate_ms: int,
        llm_result: dict[str, object] | None = None,
        retrieved_chunks: list[dict[str, object]] | None = None,
        mmr_selected_chunks: list[dict[str, object]] | None = None,
        reranked_chunks: list[dict[str, object]] | None = None,
        final_context_chunks: list[dict[str, object]] | None = None,
        used_conversation_history: bool = False,
        history_message_count: int = 0,
        history_rewritten_query: str | None = None,
        history_rewrite_ms: int = 0,
    ) -> dict[str, object]:
        return build_debug(
            rewritten_query=rewritten_query,
            retrieved_count=retrieved_count,
            reranked_count=reranked_count,
            enable_rewrite=enable_rewrite,
            enable_rerank=enable_rerank,
            rewrite_ms=rewrite_ms,
            retrieve_ms=retrieve_ms,
            rerank_ms=rerank_ms,
            context_build_ms=context_build_ms,
            generate_ms=generate_ms,
            llm_result=llm_result,
            retrieved_chunks=retrieved_chunks,
            mmr_selected_chunks=mmr_selected_chunks,
            reranked_chunks=reranked_chunks,
            final_context_chunks=final_context_chunks,
            used_conversation_history=used_conversation_history,
            history_message_count=history_message_count,
            history_rewritten_query=history_rewritten_query,
            history_rewrite_ms=history_rewrite_ms,
        )

    def ingest_file(self, file_path: Path, document_id: str | None = None) -> int:
        """把预处理后的文件切块并写入向量库，返回成功写入的 chunk 数。"""
        started = perf_counter()
        logger.info(
            "rag_pipeline_ingest_started file_path=%s document_id=%s suffix=%s",
            file_path,
            document_id,
            file_path.suffix.lower(),
        )
        if file_path.suffix.lower() == ".json":
            # JSON 是结构化解析结果，通常来自 PDF/OCR/图片预处理，保留页码和块类型。
            read_started = perf_counter()
            structured_data = json.loads(file_path.read_text(encoding="utf-8"))
            if document_id:
                structured_data["document_id"] = document_id
            chunks = self.structured_processor.process(structured_data)
            logger.info(
                "rag_pipeline_ingest_structured_processed file_path=%s document_id=%s chunks=%s read_and_process_ms=%s",
                file_path,
                document_id,
                len(chunks),
                int((perf_counter() - read_started) * 1000),
            )
        else:
            # Markdown 会保留更强的标题/表格线索；普通文本走通用切块。
            read_started = perf_counter()
            text = file_path.read_text(encoding="utf-8")
            source_format = "markdown" if file_path.suffix.lower() == ".md" else "plain"
            chunks = self.processor.process(text, source_format=source_format)
            if document_id:
                for chunk in chunks:
                    chunk["document_id"] = document_id
            logger.info(
                "rag_pipeline_ingest_text_processed file_path=%s document_id=%s source_format=%s chunks=%s read_and_process_ms=%s",
                file_path,
                document_id,
                source_format,
                len(chunks),
                int((perf_counter() - read_started) * 1000),
            )
        vector_started = perf_counter()
        self.vector_store.add_documents(chunks)
        logger.info(
            "rag_pipeline_ingest_completed file_path=%s document_id=%s chunks=%s vector_store_ms=%s total_ms=%s",
            file_path,
            document_id,
            len(chunks),
            int((perf_counter() - vector_started) * 1000),
            int((perf_counter() - started) * 1000),
        )
        return len(chunks)

    def prepare_query_inputs(
        self,
        *,
        query: str,
        enable_rewrite: bool = True,
        history_rewritten_query: str | None = None,
    ) -> dict[str, object]:
        """准备检索输入，集中处理追问改写、查询扩展和关键词抽取。"""
        effective_query = history_rewritten_query or query
        rewrite_started = perf_counter()
        expanded_keywords = ""
        if enable_rewrite:
            try:
                # rewrite 失败不能中断问答，最多退化为原问题 + 本地别名扩展。
                expanded_keywords = self.rewriter.rewrite(effective_query)
            except Exception:
                expanded_keywords = ""
        rewrite_ms = int((perf_counter() - rewrite_started) * 1000)

        alias_keywords = self.query_alias_expander.expand(effective_query)
        overview_keywords = self._expand_overview_query_keywords(effective_query)
        query_keywords = extract_query_keywords(alias_keywords, overview_keywords)
        if not query_keywords:
            query_keywords = extract_query_keywords(expanded_keywords, overview_keywords)

        return {
            "effective_query": effective_query,
            "expanded_keywords": expanded_keywords,
            "alias_keywords": alias_keywords,
            "overview_keywords": overview_keywords,
            "query_keywords": query_keywords,
            "search_query": f"{effective_query} {expanded_keywords} {alias_keywords} {overview_keywords}".strip(),
            "rewrite_ms": rewrite_ms,
        }

    def _get_retriever(self):
        retriever = getattr(self, "retriever", None)
        if retriever is None:
            retriever = build_default_hybrid_retriever(
                self.vector_store,
                sparse_provider=getattr(self.settings, "retrieval_sparse_provider", None),
                policy=getattr(self, "retrieval_policy", RetrievalPolicy()),
            )
            self.retriever = retriever
        return retriever

    def _get_shadow_retriever(self):
        shadow_retriever = getattr(self, "_shadow_retriever", None)
        shadow_backend = getattr(self.settings, "shadow_retrieval_backend", "legacy")
        normalized_backend = str(shadow_backend).strip().lower()
        if normalized_backend == "legacy":
            self._shadow_retrieval_backend = normalized_backend
            self._shadow_retriever = self._get_retriever()
            return self._shadow_retriever
        if shadow_retriever is None or getattr(self, "_shadow_retrieval_backend", None) != shadow_backend:
            shadow_retriever = build_shadow_retriever(
                self.vector_store,
                backend=normalized_backend,
                sparse_provider=getattr(self.settings, "retrieval_sparse_provider", None),
                policy=getattr(self, "retrieval_policy", RetrievalPolicy()),
                milvus_uri=getattr(self.settings, "resolved_shadow_milvus_uri", None),
                milvus_token=getattr(self.settings, "shadow_milvus_token", None),
                milvus_drop_old=bool(getattr(self.settings, "shadow_milvus_drop_old", True)),
            )
            self._shadow_retriever = shadow_retriever
            self._shadow_retrieval_backend = normalized_backend
        return shadow_retriever

    def _build_shadow_workflow_dependencies(self) -> RagWorkflowDependencies:
        return RagWorkflowDependencies(
            rewriter=self.rewriter,
            query_alias_expander=self.query_alias_expander,
            vector_store=self.vector_store,
            reranker=self.reranker,
            llm=self.llm,
            retrieval_rules=self.retrieval_rules,
            top_k_mmr=self.settings.top_k_mmr,
            retriever=self._get_shadow_retriever(),
        )

    def _build_query_policy(
        self,
        *,
        top_k_retrieve: int,
        top_k_rerank: int,
        enable_rerank: bool,
        answer_style: str,
    ) -> QueryPolicy:
        return QueryPolicy(
            retrieval=getattr(self, "retrieval_policy", RetrievalPolicy()),
            top_k_retrieve=top_k_retrieve,
            top_k_rerank=top_k_rerank,
            enable_rerank=enable_rerank,
            answer_style=answer_style,
        )

    @staticmethod
    def _build_shadow_workflow_state(
        *,
        query: str,
        query_policy: QueryPolicy,
        knowledge_base_id: str,
        source_name_by_document_id: dict[str, str] | None = None,
        enable_rewrite: bool = True,
        query_prep: dict[str, object] | None = None,
        history_rewritten_query: str | None = None,
        history_message_count: int = 0,
        history_rewrite_ms: int = 0,
    ) -> RagWorkflowState:
        return {
            "query": query,
            "top_k_retrieve": query_policy.top_k_retrieve,
            "top_k_rerank": query_policy.top_k_rerank,
            "knowledge_base_id": knowledge_base_id,
            "source_name_by_document_id": source_name_by_document_id,
            "enable_rewrite": enable_rewrite,
            "enable_rerank": query_policy.enable_rerank,
            "answer_style": query_policy.answer_style,
            "precomputed_effective_query": (query_prep or {}).get("effective_query"),
            "precomputed_expanded_keywords": (query_prep or {}).get("expanded_keywords"),
            "precomputed_alias_keywords": (query_prep or {}).get("alias_keywords"),
            "precomputed_query_keywords": (query_prep or {}).get("query_keywords"),
            "precomputed_search_query": (query_prep or {}).get("search_query"),
            "precomputed_rewrite_ms": (query_prep or {}).get("rewrite_ms"),
            "history_rewritten_query": history_rewritten_query,
            "history_message_count": history_message_count,
            "history_rewrite_ms": history_rewrite_ms,
        }

    def run_shadow_workflow(
        self,
        *,
        query: str,
        top_k_retrieve: int,
        top_k_rerank: int,
        knowledge_base_id: str,
        source_name_by_document_id: dict[str, str] | None = None,
        enable_rewrite: bool = True,
        enable_rerank: bool = True,
        answer_style: str = "concise",
        query_prep: dict[str, object] | None = None,
        history_rewritten_query: str | None = None,
        history_message_count: int = 0,
        history_rewrite_ms: int = 0,
    ) -> RagWorkflowState:
        """运行函数式 shadow workflow，用于和 legacy pipeline 做结果对比。"""
        query_policy = self._build_query_policy(
            top_k_retrieve=top_k_retrieve,
            top_k_rerank=top_k_rerank,
            enable_rerank=enable_rerank,
            answer_style=answer_style,
        )
        state = self._build_shadow_workflow_state(
            query=query,
            query_policy=query_policy,
            knowledge_base_id=knowledge_base_id,
            source_name_by_document_id=source_name_by_document_id,
            enable_rewrite=enable_rewrite,
            query_prep=query_prep,
            history_rewritten_query=history_rewritten_query,
            history_message_count=history_message_count,
            history_rewrite_ms=history_rewrite_ms,
        )
        return run_shadow_workflow(state, self._build_shadow_workflow_dependencies())

    def run_shadow_graph(
        self,
        *,
        query: str,
        top_k_retrieve: int,
        top_k_rerank: int,
        knowledge_base_id: str,
        source_name_by_document_id: dict[str, str] | None = None,
        enable_rewrite: bool = True,
        enable_rerank: bool = True,
        answer_style: str = "concise",
        query_prep: dict[str, object] | None = None,
        history_rewritten_query: str | None = None,
        history_message_count: int = 0,
        history_rewrite_ms: int = 0,
    ) -> RagWorkflowState:
        """运行 LangGraph 形态的 shadow graph，验证迁移后的节点编排。"""
        graph = build_shadow_graph(self._build_shadow_workflow_dependencies())
        query_policy = self._build_query_policy(
            top_k_retrieve=top_k_retrieve,
            top_k_rerank=top_k_rerank,
            enable_rerank=enable_rerank,
            answer_style=answer_style,
        )
        state = self._build_shadow_workflow_state(
            query=query,
            query_policy=query_policy,
            knowledge_base_id=knowledge_base_id,
            source_name_by_document_id=source_name_by_document_id,
            enable_rewrite=enable_rewrite,
            query_prep=query_prep,
            history_rewritten_query=history_rewritten_query,
            history_message_count=history_message_count,
            history_rewrite_ms=history_rewrite_ms,
        )
        return graph.invoke(state)

    def ask(
        self,
        query: str,
        top_k_retrieve: int,
        top_k_rerank: int,
        knowledge_base_id: str,
        source_name_by_document_id: dict[str, str] | None = None,
        enable_rewrite: bool = True,
        enable_rerank: bool = True,
        answer_style: str = "concise",
        query_prep: dict[str, object] | None = None,
        history_rewritten_query: str | None = None,
        history_message_count: int = 0,
        history_rewrite_ms: int = 0,
    ) -> dict[str, object]:
        """执行一次完整的非流式 RAG 问答。"""
        generation_inputs = self._prepare_generation_inputs(
            query=query,
            top_k_retrieve=top_k_retrieve,
            top_k_rerank=top_k_rerank,
            knowledge_base_id=knowledge_base_id,
            source_name_by_document_id=source_name_by_document_id,
            enable_rewrite=enable_rewrite,
            enable_rerank=enable_rerank,
            answer_style=answer_style,
            query_prep=query_prep,
            history_rewritten_query=history_rewritten_query,
            history_message_count=history_message_count,
            history_rewrite_ms=history_rewrite_ms,
        )
        generate_started = perf_counter()
        llm_result = self.llm.generate(
            query,
            str(generation_inputs["context"]),
            standalone_query=str(generation_inputs["effective_query"]),
            answer_style=answer_style,
        )
        generate_ms = int((perf_counter() - generate_started) * 1000)

        return {
            "answer": llm_result["answer"],
            "citations": list(generation_inputs["citations"]),
            "debug": build_debug(
                rewritten_query=str(generation_inputs["search_query"]),
                retrieved_count=int(generation_inputs["retrieved_count"]),
                reranked_count=int(generation_inputs["reranked_count"]),
                enable_rewrite=enable_rewrite,
                enable_rerank=bool(generation_inputs["enable_rerank"]),
                rewrite_ms=int(generation_inputs["rewrite_ms"]),
                retrieve_ms=int(generation_inputs["retrieve_ms"]),
                rerank_ms=int(generation_inputs["rerank_ms"]),
                context_build_ms=int(generation_inputs["context_build_ms"]),
                generate_ms=generate_ms,
                llm_result=llm_result,
                retrieved_chunks=list(generation_inputs["retrieved_debug_chunks"]),
                mmr_selected_chunks=list(generation_inputs["mmr_debug_chunks"]),
                reranked_chunks=list(generation_inputs["reranked_debug_chunks"]),
                final_context_chunks=list(generation_inputs["final_context_debug_chunks"]),
                used_conversation_history=history_rewritten_query is not None,
                history_message_count=history_message_count,
                history_rewritten_query=history_rewritten_query,
                history_rewrite_ms=history_rewrite_ms,
            ),
        }

    def ask_stream(
        self,
        query: str,
        top_k_retrieve: int,
        top_k_rerank: int,
        knowledge_base_id: str,
        source_name_by_document_id: dict[str, str] | None = None,
        enable_rewrite: bool = True,
        enable_rerank: bool = True,
        answer_style: str = "concise",
        query_prep: dict[str, object] | None = None,
        history_rewritten_query: str | None = None,
        history_message_count: int = 0,
        history_rewrite_ms: int = 0,
    ):
        """执行一次流式 RAG 问答，先 yield token，最后 return 完整结果。"""
        generation_inputs = self._prepare_generation_inputs(
            query=query,
            top_k_retrieve=top_k_retrieve,
            top_k_rerank=top_k_rerank,
            knowledge_base_id=knowledge_base_id,
            source_name_by_document_id=source_name_by_document_id,
            enable_rewrite=enable_rewrite,
            enable_rerank=enable_rerank,
            answer_style=answer_style,
            query_prep=query_prep,
            history_rewritten_query=history_rewritten_query,
            history_message_count=history_message_count,
            history_rewrite_ms=history_rewrite_ms,
        )
        generate_started = perf_counter()
        stream = self.llm.stream_generate(
            query,
            str(generation_inputs["context"]),
            standalone_query=str(generation_inputs["effective_query"]),
            answer_style=answer_style,
        )
        while True:
            try:
                delta = next(stream)
            except StopIteration as stop:
                llm_result = stop.value
                break
            if delta:
                yield delta

        generate_ms = int((perf_counter() - generate_started) * 1000)
        return {
            "answer": llm_result["answer"],
            "citations": list(generation_inputs["citations"]),
            "debug": build_debug(
                rewritten_query=str(generation_inputs["search_query"]),
                retrieved_count=int(generation_inputs["retrieved_count"]),
                reranked_count=int(generation_inputs["reranked_count"]),
                enable_rewrite=enable_rewrite,
                enable_rerank=bool(generation_inputs["enable_rerank"]),
                rewrite_ms=int(generation_inputs["rewrite_ms"]),
                retrieve_ms=int(generation_inputs["retrieve_ms"]),
                rerank_ms=int(generation_inputs["rerank_ms"]),
                context_build_ms=int(generation_inputs["context_build_ms"]),
                generate_ms=generate_ms,
                llm_result=llm_result,
                retrieved_chunks=list(generation_inputs["retrieved_debug_chunks"]),
                mmr_selected_chunks=list(generation_inputs["mmr_debug_chunks"]),
                reranked_chunks=list(generation_inputs["reranked_debug_chunks"]),
                final_context_chunks=list(generation_inputs["final_context_debug_chunks"]),
                used_conversation_history=history_rewritten_query is not None,
                history_message_count=history_message_count,
                history_rewritten_query=history_rewritten_query,
                history_rewrite_ms=history_rewrite_ms,
            ),
        }

    def _prepare_generation_inputs(
        self,
        *,
        query: str,
        top_k_retrieve: int,
        top_k_rerank: int,
        knowledge_base_id: str,
        source_name_by_document_id: dict[str, str] | None = None,
        enable_rewrite: bool = True,
        enable_rerank: bool = True,
        answer_style: str = "concise",
        query_prep: dict[str, object] | None = None,
        history_rewritten_query: str | None = None,
        history_message_count: int = 0,
        history_rewrite_ms: int = 0,
    ) -> dict[str, object]:
        """生成前的重活都集中在这里，便于非流式和流式路径复用。"""
        query_policy = self._build_query_policy(
            top_k_retrieve=top_k_retrieve,
            top_k_rerank=top_k_rerank,
            enable_rerank=enable_rerank,
            answer_style=answer_style,
        )
        prepared_query = query_prep or self.prepare_query_inputs(
            query=query,
            enable_rewrite=enable_rewrite,
            history_rewritten_query=history_rewritten_query,
        )
        effective_query = str(prepared_query["effective_query"])
        expanded_keywords = str(prepared_query["expanded_keywords"])
        alias_keywords = str(prepared_query["alias_keywords"])
        query_keywords = list(prepared_query["query_keywords"])
        search_query = str(prepared_query["search_query"])
        rewrite_ms = int(prepared_query["rewrite_ms"])

        # 第一阶段：混合检索。返回值同时包含向量结果、稀疏结果和 debug 所需的中间数据。
        retrieve_started = perf_counter()
        retrieval = self._get_retriever().retrieve(
            search_query=search_query,
            query_keywords=query_keywords,
            top_k_retrieve=query_policy.top_k_retrieve,
            source_name_by_document_id=source_name_by_document_id,
            policy=query_policy.retrieval,
        )
        retrieve_ms = int((perf_counter() - retrieve_started) * 1000)
        query_vector = retrieval.query_vector
        retrieved_docs = retrieval.retrieved_docs
        deduped_docs = retrieval.deduped_docs
        retrieved_debug_chunks = retrieval.retrieved_debug_chunks
        valid_docs = retrieval.valid_docs
        doc_embeddings = retrieval.doc_embeddings

        if not valid_docs:
            return {
                "effective_query": effective_query,
                "search_query": search_query,
                "context": "",
                "citations": [],
                "retrieved_count": len(retrieved_docs),
                "reranked_count": 0,
                "enable_rerank": query_policy.enable_rerank,
                "rewrite_ms": rewrite_ms,
                "retrieve_ms": retrieve_ms,
                "rerank_ms": 0,
                "context_build_ms": 0,
                "retrieved_debug_chunks": retrieved_debug_chunks,
                "mmr_debug_chunks": [],
                "reranked_debug_chunks": [],
                "final_context_debug_chunks": [],
                "history_message_count": history_message_count,
                "history_rewrite_ms": history_rewrite_ms,
            }

        # 第二阶段：MMR 控制相似 chunk 的重复度；关闭 rerank 时仍用 top_k 做截断。
        rerank_started = perf_counter()
        if query_policy.enable_rerank:
            selected_indices = mmr_rerank(
                query_embedding=query_vector,
                doc_embeddings=doc_embeddings,
                top_k=min(self.settings.top_k_mmr, len(valid_docs)),
                lambda_mult=0.6,
            )
            mmr_selected_docs = [valid_docs[index] for index in selected_indices]
        else:
            mmr_selected_docs = valid_docs[: min(query_policy.top_k_rerank, len(valid_docs))]

        policy_trigger_words = self.retrieval_rules.policy_trigger_words
        policy_trigger_phrases = self.retrieval_rules.policy_trigger_phrases
        behavior_keywords = self.retrieval_rules.behavior_keywords
        must_keep_child_keywords = self.retrieval_rules.must_keep_child_keywords
        is_policy_question = should_treat_as_policy_question(
            query,
            policy_trigger_words=policy_trigger_words,
            policy_trigger_phrases=policy_trigger_phrases,
            behavior_keywords=behavior_keywords,
        )

        # 第三阶段：保留规则。法律/制度类问题常被短 chunk 稀释，这里把强关键词命中的候选补回来。
        mmr_selected_docs = preserve_policy_keyword_docs(
            selected_docs=mmr_selected_docs,
            candidate_docs=deduped_docs,
            is_policy_question=is_policy_question,
            must_keep_child_keywords=must_keep_child_keywords,
        )
        mmr_selected_docs = preserve_query_keyword_docs(
            selected_docs=mmr_selected_docs,
            candidate_docs=deduped_docs,
            query_keywords=query_keywords,
        )
        if is_overview_query(query):
            mmr_selected_docs = preserve_overview_candidate_docs(
                selected_docs=mmr_selected_docs,
                candidate_docs=deduped_docs,
                max_extra_docs=max(4, query_policy.top_k_rerank),
            )

        mmr_debug_chunks = build_debug_chunks(mmr_selected_docs)
        parent_docs_for_rerank = build_parent_docs_for_rerank(mmr_selected_docs)
        context_parent_pool = build_parent_docs_for_rerank(deduped_docs)
        if not parent_docs_for_rerank and is_overview_query(query):
            parent_docs_for_rerank = build_parent_docs_for_rerank(
                deduped_docs[: max(query_policy.top_k_rerank + 2, 4)]
            )
            context_parent_pool = parent_docs_for_rerank
        if not parent_docs_for_rerank:
            rerank_ms = int((perf_counter() - rerank_started) * 1000)
            return {
                "effective_query": effective_query,
                "search_query": search_query,
                "context": "",
                "citations": [],
                "retrieved_count": len(retrieved_docs),
                "reranked_count": 0,
                "enable_rerank": query_policy.enable_rerank,
                "rewrite_ms": rewrite_ms,
                "retrieve_ms": retrieve_ms,
                "rerank_ms": rerank_ms,
                "context_build_ms": 0,
                "retrieved_debug_chunks": retrieved_debug_chunks,
                "mmr_debug_chunks": mmr_debug_chunks,
                "reranked_debug_chunks": [],
                "final_context_debug_chunks": [],
                "history_message_count": history_message_count,
                "history_rewrite_ms": history_rewrite_ms,
            }

        if query_policy.enable_rerank:
            reranked_parents = self.reranker.rerank(
                search_query,
                parent_docs_for_rerank,
                top_k=min(query_policy.top_k_rerank, len(parent_docs_for_rerank)),
            )
        else:
            reranked_parents = parent_docs_for_rerank[: min(query_policy.top_k_rerank, len(parent_docs_for_rerank))]
        reranked_parents = sort_reranked_parents(reranked_parents)
        reranked_debug_chunks = build_debug_chunks(reranked_parents)

        must_keep_parent_keywords = must_keep_child_keywords
        low_value_parent_keywords = self.retrieval_rules.low_value_parent_keywords
        rule_signal_keywords = self.retrieval_rules.rule_signal_keywords

        rerank_ms = int((perf_counter() - rerank_started) * 1000)
        # 第四阶段：从 rerank 后的 parent 文档里挑最终上下文，优先保留可支撑回答的证据。
        final_context_parents = build_final_context_parents(
            parent_docs_for_rerank=parent_docs_for_rerank,
            reranked_parents=reranked_parents,
            context_parent_pool=context_parent_pool,
            query=query,
            query_keywords=query_keywords,
            is_policy_question=is_policy_question,
            must_keep_parent_keywords=must_keep_parent_keywords,
            low_value_parent_keywords=low_value_parent_keywords,
            rule_signal_keywords=rule_signal_keywords,
            behavior_keywords=behavior_keywords,
            top_k_rerank=top_k_rerank,
        )
        context_build_started = perf_counter()
        # citations 和 debug chunks 必须基于同一批最终上下文，避免前端引用和回答证据不一致。
        final_context_debug_chunks = build_debug_chunks(final_context_parents)
        context = "\n\n".join(format_context_parent(doc) for doc in final_context_parents)
        citations = build_citations(final_context_parents, knowledge_base_id)
        context_build_ms = int((perf_counter() - context_build_started) * 1000)
        return {
            "effective_query": effective_query,
            "search_query": search_query,
            "context": context,
            "citations": citations,
            "retrieved_count": len(retrieved_docs),
            "reranked_count": len(final_context_parents),
            "enable_rerank": query_policy.enable_rerank,
            "rewrite_ms": rewrite_ms,
            "retrieve_ms": retrieve_ms,
            "rerank_ms": rerank_ms,
            "context_build_ms": context_build_ms,
            "retrieved_debug_chunks": retrieved_debug_chunks,
            "mmr_debug_chunks": mmr_debug_chunks,
            "reranked_debug_chunks": reranked_debug_chunks,
            "final_context_debug_chunks": final_context_debug_chunks,
            "history_message_count": history_message_count,
            "history_rewrite_ms": history_rewrite_ms,
        }






