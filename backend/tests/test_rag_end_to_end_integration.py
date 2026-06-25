"""RAG 主链路端到端集成测试。

这些测试使用确定性的向量、重写、重排和 LLM 替身，验证业务行为而不是模型效果：
入库、检索、引用、会话历史、自动路由和 guard 结果必须稳定可断言。
"""

import asyncio
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import chromadb
import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.core.exceptions import ConversationNotFoundError, KnowledgeBaseNotFoundError, KnowledgeBaseNotReadyError
from app.db.session import engine
from app.rag.document_processor import DocumentProcessor
from app.rag.pipeline import RAGPipeline
from app.rag.retrieval_rules_config import load_retrieval_rules_config
from app.rag.structured_document_processor import StructuredDocumentProcessor
from app.rag.vector_store import VectorStore
from app.repositories.metadata_repository import MetadataRepository, get_metadata_repository
from app.schemas.chat import ChatQueryRequest
from app.services import rag_service
from app.services.ingest_service import IngestService
from app.services.knowledge_base_router import RoutedKnowledgeBase
from app.workflows.rag.pipeline_steps import extract_query_keywords


class DeterministicVectorStore(VectorStore):
    """用关键词维度生成固定向量，避免测试依赖真实 embedding 模型。"""

    def __init__(self, db_path: str, collection_name: str) -> None:
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.tokenizer = None
        self.model = None

    def _get_embedding(self, text: str, is_query: bool = False) -> list[float]:
        lowered = text.lower()
        score_alpha = 1.0 if "alpha" in lowered or "阿尔法" in lowered else 0.0
        score_beta = 1.0 if "beta" in lowered or "贝塔" in lowered else 0.0
        score_first = 1.0 if "第一次" in lowered or "首次" in lowered or "first" in lowered else 0.0
        score_register = 1.0 if "登记" in lowered or "注册" in lowered else 0.0
        score_trial = 1.0 if "试用期" in lowered else 0.0
        score_salary = 1.0 if "工资" in lowered or "薪资" in lowered else 0.0
        score_compensation = 1.0 if "赔偿金" in lowered or "补偿" in lowered or "离职" in lowered else 0.0
        vector = [
            score_alpha,
            score_beta,
            score_first,
            score_register,
            score_trial,
            score_salary,
            score_compensation,
        ]
        if not any(vector):
            return [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        return vector


class StaticAliasExpander:
    """测试中不扩展同义词，避免额外变量影响检索断言。"""

    def expand(self, query: str) -> str:
        return query


class DeterministicRewriter:
    """只在追问场景返回可预测的独立问题。"""

    def __init__(self) -> None:
        self.rewrite_calls: list[str] = []
        self.history_calls: list[tuple[str, list[dict[str, str]]]] = []

    def rewrite(self, user_query: str) -> str:
        self.rewrite_calls.append(user_query)
        return ""

    def rewrite_with_history(self, user_query: str, history_messages: list[dict[str, str]]) -> str:
        self.history_calls.append((user_query, history_messages))
        if "第一次" not in user_query:
            return user_query

        last_user_message = next(
            (item["content"] for item in reversed(history_messages) if item.get("role") == "user"),
            "",
        )
        lowered = last_user_message.lower()
        if "alpha" in lowered or "阿尔法" in last_user_message:
            return "alpha 第一次需要做什么？"
        if "beta" in lowered or "贝塔" in last_user_message:
            return "beta 第一次需要做什么？"
        return user_query


class DeterministicReranker:
    """按关键词命中数重排，保证最终上下文顺序稳定。"""

    def rerank(self, query: str, docs: list[dict[str, object]], top_k: int) -> list[dict[str, object]]:
        query_keywords = extract_query_keywords(query)
        reranked: list[dict[str, object]] = []
        for index, doc in enumerate(docs):
            parent_text = str(doc.get("parent_text") or "")
            score = sum(1 for keyword in query_keywords if keyword and keyword in parent_text)
            candidate = dict(doc)
            candidate["rerank_score"] = float(score + max(0, len(docs) - index) / 10)
            reranked.append(candidate)
        reranked.sort(key=lambda item: float(item.get("rerank_score") or 0.0), reverse=True)
        return reranked[:top_k]


class DeterministicLLM:
    """返回首段上下文作为答案，便于断言引用和 guard 行为。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        query: str,
        context: str,
        standalone_query: str | None = None,
        *,
        answer_style: str = "concise",
    ) -> dict[str, object]:
        self.calls.append(
            {
                "query": query,
                "context": context,
                "standalone_query": standalone_query,
                "answer_style": answer_style,
            }
        )
        leading_context = context.split("\n\n", 1)[0].strip() if context else "参考资料未明确"
        answer = f"根据资料：{leading_context}"
        if standalone_query and standalone_query != query:
            answer = f"{answer}\n完整问题：{standalone_query}"
        return {
            "answer": answer,
            "finish_reason": "stop",
            "model": "deterministic-test-model",
            "prompt_tokens": 12,
            "completion_tokens": 24,
            "total_tokens": 36,
        }


@pytest.fixture()
def sql_metadata_repository() -> MetadataRepository:
    """连接真实 SQL 元数据后端；不可用时跳过集成测试。"""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"SQL metadata backend unavailable for integration test: {exc}")
    return get_metadata_repository()


def _cleanup(
    repository: MetadataRepository,
    *,
    conversation_ids: list[str] | None = None,
    knowledge_base_ids: list[str] | None = None,
) -> None:
    """清理测试创建的会话和知识库，忽略已被级联删除的记录。"""
    for conversation_id in conversation_ids or []:
        with suppress(ConversationNotFoundError):
            repository.delete_conversation(conversation_id)
    for knowledge_base_id in knowledge_base_ids or []:
        with suppress(KnowledgeBaseNotFoundError):
            repository.delete_knowledge_base(knowledge_base_id)


def _build_pipeline(tmp_path: Path) -> tuple[RAGPipeline, DeterministicLLM, DeterministicRewriter]:
    """构造一条不依赖真实模型的 RAGPipeline。"""
    llm = DeterministicLLM()
    rewriter = DeterministicRewriter()
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.settings = SimpleNamespace(top_k_mmr=4)
    pipeline.processor = DocumentProcessor(child_chunk_size=200)
    pipeline.structured_processor = StructuredDocumentProcessor(pipeline.processor)
    pipeline.vector_store = DeterministicVectorStore(
        db_path=str(tmp_path / "chroma"),
        collection_name=f"rag_e2e_{uuid4().hex[:8]}",
    )
    pipeline.reranker = DeterministicReranker()
    pipeline.llm = llm
    pipeline.rewriter = rewriter
    pipeline.query_alias_expander = StaticAliasExpander()
    pipeline.retrieval_rules = load_retrieval_rules_config()
    return pipeline, llm, rewriter


def _create_ready_knowledge_base(
    repository: MetadataRepository,
    *,
    name_prefix: str = "RAG E2E Ready KB",
    description: str = "rag e2e ready integration",
    subject: str = "Alpha 学员管理制度",
    domain: str = "general",
) -> dict:
    """创建已 ready 但不一定有文档的知识库。"""
    knowledge_base = repository.create_knowledge_base(
        name=f"{name_prefix} {uuid4().hex[:6]}",
        description=description,
        subject=subject,
        domain=domain,
    )
    return repository.update_knowledge_base(knowledge_base["id"], status="ready")


def _create_ingested_knowledge_base(
    repository: MetadataRepository,
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    markdown_filename: str = "alpha_rules.md",
    markdown_content: str | None = None,
    knowledge_base_name_prefix: str = "RAG E2E KB",
    knowledge_base_description: str = "rag e2e integration",
    knowledge_base_subject: str = "Alpha 学员管理制度",
    knowledge_base_domain: str = "general",
) -> tuple[dict, dict, dict, RAGPipeline, DeterministicLLM, DeterministicRewriter]:
    """创建临时 Markdown 文档、执行入库，并返回可断言的 pipeline 替身。"""
    tmp_path.mkdir(parents=True, exist_ok=True)
    markdown_path = tmp_path / markdown_filename
    markdown_path.write_text(
        markdown_content
        or (
            "# Alpha 手册\n\n"
            "## 第一章 登记要求\n\n"
            "**第一条** alpha 学员需要完成阿尔法登记。\n\n"
            "**第二条** 第一次参加 alpha 课程的学员，需要在当天完成首次登记并提交证件。\n\n"
            "**第三条** beta 学员需要完成贝塔确认。\n"
        ),
        encoding="utf-8",
    )

    knowledge_base = repository.create_knowledge_base(
        name=f"{knowledge_base_name_prefix} {uuid4().hex[:6]}",
        description=knowledge_base_description,
        subject=knowledge_base_subject,
        domain=knowledge_base_domain,
    )
    document = repository.create_document(
        knowledge_base_id=knowledge_base["id"],
        filename=markdown_path.name,
        content_type="text/markdown",
        file_size=markdown_path.stat().st_size,
        content_hash=f"hash-{uuid4().hex}",
        file_path=str(markdown_path),
    )

    pipeline, llm, rewriter = _build_pipeline(tmp_path)
    monkeypatch.setattr("app.services.ingest_service.get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)
    monkeypatch.setattr("app.services.rag_service.get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    ingest_service = IngestService()
    ingest_service.repository = repository
    task = ingest_service.create_ingest_task(knowledge_base["id"], [document["id"]])
    task_result = ingest_service.run_ingest_task(task["id"])

    assert task_result["status"] == "completed"
    assert repository.get_document(document["id"]) is not None
    assert repository.get_document(document["id"])["status"] == "ready"
    assert repository.get_knowledge_base(knowledge_base["id"]) is not None
    assert repository.get_knowledge_base(knowledge_base["id"])["status"] == "ready"

    return knowledge_base, document, task_result, pipeline, llm, rewriter


def test_rag_query_end_to_end_persists_answer_and_citations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sql_metadata_repository: MetadataRepository,
) -> None:
    """普通问答应持久化用户/助手消息、引用和 debug 信息。"""
    repository = sql_metadata_repository
    knowledge_base, _document, _task_result, _pipeline, llm, _rewriter = _create_ingested_knowledge_base(
        repository,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    conversation_ids: list[str] = []

    try:
        service = rag_service.RAGService()
        service.repository = repository

        result = asyncio.run(
            service.query(
                ChatQueryRequest(
                    knowledge_base_id=knowledge_base["id"],
                    query="alpha 学员需要做什么？",
                    enable_auto_route=False,
                    debug=True,
                )
            )
        )
        conversation_ids.append(result["conversation_id"])

        assert result["answer_source"] == "legacy_pipeline"
        assert result["conversation_id"].startswith("conv_")
        assert "alpha 学员需要完成阿尔法登记" in result["answer"]
        assert result["knowledge_base_id"] == knowledge_base["id"]
        assert result["citations"]
        assert all(item["document_id"] == knowledge_base["id"] for item in result["citations"])
        assert any(item["source_name"] for item in result["citations"])
        assert result["debug"]["retrieved_count"] >= 1
        assert result["debug"]["reranked_count"] >= 1
        assert result["debug"]["final_context_chunks"]
        assert llm.calls[-1]["answer_style"] == "structured"

        messages = repository.list_messages(result["conversation_id"])
        assert [item["role"] for item in messages] == ["user", "assistant"]
        assert [item["sequence"] for item in messages] == [1, 2]
        assert messages[0]["knowledge_base_id"] == knowledge_base["id"]
        assert messages[1]["knowledge_base_id"] == knowledge_base["id"]
        assert messages[1]["citations"] == result["citations"]
        assert messages[1]["debug"] is not None
        assert messages[1]["debug"]["resolved_knowledge_base_id"] == knowledge_base["id"]
    finally:
        _cleanup(
            repository,
            conversation_ids=conversation_ids,
            knowledge_base_ids=[knowledge_base["id"]],
        )


def test_rag_query_end_to_end_returns_no_answer_when_nothing_is_retrieved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sql_metadata_repository: MetadataRepository,
) -> None:
    """没有检索结果时仍会返回可持久化答案，但引用和上下文为空。"""
    repository = sql_metadata_repository
    knowledge_base = _create_ready_knowledge_base(
        repository,
        name_prefix="RAG E2E Empty KB",
        description="rag e2e no retrieval integration",
    )
    pipeline, llm, _rewriter = _build_pipeline(tmp_path)
    monkeypatch.setattr("app.services.rag_service.get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)
    conversation_ids: list[str] = []

    try:
        service = rag_service.RAGService()
        service.repository = repository

        result = asyncio.run(
            service.query(
                ChatQueryRequest(
                    knowledge_base_id=knowledge_base["id"],
                    query="完全无关的问题",
                    enable_auto_route=False,
                    debug=True,
                )
            )
        )
        conversation_ids.append(result["conversation_id"])

        assert result["answer"] == "根据资料：参考资料未明确"
        assert result["citations"] == []
        assert result["debug"]["retrieved_count"] == 0
        assert result["debug"]["reranked_count"] == 0
        assert result["debug"]["final_context_chunks"] == []
        assert result["debug"]["answer_guard_applied"] is False
        assert llm.calls[-1]["context"] == ""

        messages = repository.list_messages(result["conversation_id"])
        assert [item["role"] for item in messages] == ["user", "assistant"]
        assert messages[1]["citations"] == []
        assert messages[1]["debug"] is not None
        assert messages[1]["debug"]["retrieved_count"] == 0
    finally:
        _cleanup(
            repository,
            conversation_ids=conversation_ids,
            knowledge_base_ids=[knowledge_base["id"]],
        )


def test_rag_follow_up_end_to_end_uses_persisted_history_for_retrieval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sql_metadata_repository: MetadataRepository,
) -> None:
    """追问应使用已持久化的会话历史补全检索问题。"""
    repository = sql_metadata_repository
    knowledge_base, _document, _task_result, _pipeline, llm, rewriter = _create_ingested_knowledge_base(
        repository,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    conversation_ids: list[str] = []

    try:
        service = rag_service.RAGService()
        service.repository = repository

        first_result = asyncio.run(
            service.query(
                ChatQueryRequest(
                    knowledge_base_id=knowledge_base["id"],
                    query="alpha 学员需要做什么？",
                    enable_auto_route=False,
                    debug=True,
                )
            )
        )
        conversation_ids.append(first_result["conversation_id"])

        second_result = asyncio.run(
            service.query(
                ChatQueryRequest(
                    knowledge_base_id=knowledge_base["id"],
                    conversation_id=first_result["conversation_id"],
                    query="那第一次呢？",
                    enable_auto_route=False,
                    debug=True,
                )
            )
        )

        assert second_result["conversation_id"] == first_result["conversation_id"]
        assert "第一次参加 alpha 课程的学员" in second_result["answer"]
        assert "完整问题：alpha 第一次需要做什么？" in second_result["answer"]
        assert second_result["debug"]["used_conversation_history"] is True
        assert second_result["debug"]["history_message_count"] == 2
        assert second_result["debug"]["history_rewritten_query"] == "alpha 第一次需要做什么？"
        assert llm.calls[-1]["standalone_query"] == "alpha 第一次需要做什么？"
        assert rewriter.history_calls

        messages = repository.list_messages(first_result["conversation_id"])
        assert [item["sequence"] for item in messages] == [1, 2, 3, 4]
        assert [item["role"] for item in messages] == ["user", "assistant", "user", "assistant"]
        assert messages[-1]["debug"] is not None
        assert messages[-1]["debug"]["used_conversation_history"] is True
    finally:
        _cleanup(
            repository,
            conversation_ids=conversation_ids,
            knowledge_base_ids=[knowledge_base["id"]],
        )


def test_rag_query_end_to_end_auto_routes_to_another_knowledge_base_and_persists_resolved_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sql_metadata_repository: MetadataRepository,
) -> None:
    """自动路由到其他知识库后，消息上下文要记录实际命中的知识库。"""
    repository = sql_metadata_repository
    requested_knowledge_base = _create_ready_knowledge_base(
        repository,
        name_prefix="和鸣制度库",
        description="training management knowledge base",
        subject="和鸣教育管理制度",
        domain="training_management",
    )
    labor_knowledge_base, _document, _task_result, _pipeline, labor_llm, _rewriter = _create_ingested_knowledge_base(
        repository,
        tmp_path=tmp_path / "labor_route",
        monkeypatch=monkeypatch,
        markdown_filename="labor_rules.md",
        markdown_content=(
            "# 劳动法试用期规则\n\n"
            "## 第一章 试用期工资\n\n"
            "**第一条** 试用期工资不得低于约定工资的百分之八十。\n\n"
            "**第二条** 用人单位不得以试用期为由不发工资。\n"
        ),
        knowledge_base_name_prefix="劳动&劳动合同",
        knowledge_base_description="劳动相关法律",
        knowledge_base_subject="劳动相关法律",
        knowledge_base_domain="labor_law",
    )
    requested_pipeline, _requested_llm, _requested_rewriter = _build_pipeline(tmp_path / "requested_route")
    monkeypatch.setattr(
        rag_service,
        "KnowledgeBaseRouter",
        lambda repository: SimpleNamespace(
            route=lambda **kwargs: RoutedKnowledgeBase(
                knowledge_base_id=labor_knowledge_base["id"],
                knowledge_base_name=labor_knowledge_base["name"],
                subject=labor_knowledge_base["subject"],
                auto_routed=True,
                requested_knowledge_base_id=requested_knowledge_base["id"],
                score=100,
                reason="命中测试路由:labor",
            )
        ),
    )
    monkeypatch.setattr(
        "app.services.rag_service.get_rag_pipeline",
        lambda knowledge_base_id, subject: {
            requested_knowledge_base["id"]: requested_pipeline,
            labor_knowledge_base["id"]: _pipeline,
        }[knowledge_base_id],
    )
    monkeypatch.setattr(
        rag_service,
        "_settings",
        lambda: SimpleNamespace(
            chat_auto_route_enabled=True,
            chat_shadow_compare_enabled=False,
            chat_shadow_compare_sample_rate=0.0,
        ),
    )
    conversation_ids: list[str] = []

    try:
        service = rag_service.RAGService()
        service.repository = repository

        result = asyncio.run(
            service.query(
                ChatQueryRequest(
                    knowledge_base_id=requested_knowledge_base["id"],
                    query="试用期可以不发工资吗？",
                    debug=True,
                )
            )
        )
        conversation_ids.append(result["conversation_id"])

        assert result["knowledge_base_id"] == labor_knowledge_base["id"]
        assert result["knowledge_base_name"] == labor_knowledge_base["name"]
        assert result["auto_routed"] is True
        assert result["debug"]["requested_knowledge_base_id"] == requested_knowledge_base["id"]
        assert result["debug"]["resolved_knowledge_base_id"] == labor_knowledge_base["id"]
        assert "试用期工资" in result["answer"]
        assert labor_llm.calls

        conversation = repository.get_conversation(result["conversation_id"])
        assert conversation is not None
        assert conversation["default_knowledge_base_id"] == labor_knowledge_base["id"]

        messages = repository.list_messages(result["conversation_id"])
        assert [item["role"] for item in messages] == ["user", "assistant"]
        assert all(item["knowledge_base_id"] == labor_knowledge_base["id"] for item in messages)
        assert messages[1]["debug"] is not None
        assert messages[1]["debug"]["auto_routed"] is True
        assert messages[1]["debug"]["resolved_knowledge_base_id"] == labor_knowledge_base["id"]
    finally:
        _cleanup(
            repository,
            conversation_ids=conversation_ids,
            knowledge_base_ids=[requested_knowledge_base["id"], labor_knowledge_base["id"]],
        )


def test_rag_follow_up_end_to_end_keeps_auto_routed_knowledge_base_within_same_conversation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sql_metadata_repository: MetadataRepository,
) -> None:
    """同一会话的追问应沿用自动路由后的知识库。"""
    repository = sql_metadata_repository
    requested_knowledge_base = _create_ready_knowledge_base(
        repository,
        name_prefix="和鸣制度库",
        description="training management knowledge base",
        subject="和鸣教育管理制度",
        domain="training_management",
    )
    labor_knowledge_base, _document, _task_result, _pipeline, labor_llm, _rewriter = _create_ingested_knowledge_base(
        repository,
        tmp_path=tmp_path / "labor_follow_up",
        monkeypatch=monkeypatch,
        markdown_filename="labor_follow_up_rules.md",
        markdown_content=(
            "# 劳动法试用期与离职规则\n\n"
            "## 第一章 试用期工资\n\n"
            "**第一条** 试用期工资不得低于约定工资的百分之八十。\n\n"
            "## 第二章 解除与补偿\n\n"
            "**第二条** 违法解除劳动合同的，应当依法支付赔偿金。\n"
        ),
        knowledge_base_name_prefix="劳动&劳动合同",
        knowledge_base_description="劳动相关法律",
        knowledge_base_subject="劳动相关法律",
        knowledge_base_domain="labor_law",
    )
    requested_pipeline, _requested_llm, _requested_rewriter = _build_pipeline(tmp_path / "requested_follow_up")
    monkeypatch.setattr(
        rag_service,
        "KnowledgeBaseRouter",
        lambda repository: SimpleNamespace(
            route=lambda **kwargs: RoutedKnowledgeBase(
                knowledge_base_id=labor_knowledge_base["id"],
                knowledge_base_name=labor_knowledge_base["name"],
                subject=labor_knowledge_base["subject"],
                auto_routed=True,
                requested_knowledge_base_id=requested_knowledge_base["id"],
                score=100,
                reason="命中测试路由:labor",
            )
        ),
    )
    monkeypatch.setattr(
        "app.services.rag_service.get_rag_pipeline",
        lambda knowledge_base_id, subject: {
            requested_knowledge_base["id"]: requested_pipeline,
            labor_knowledge_base["id"]: _pipeline,
        }[knowledge_base_id],
    )
    monkeypatch.setattr(
        rag_service,
        "_settings",
        lambda: SimpleNamespace(
            chat_auto_route_enabled=True,
            chat_shadow_compare_enabled=False,
            chat_shadow_compare_sample_rate=0.0,
        ),
    )
    conversation_ids: list[str] = []

    try:
        service = rag_service.RAGService()
        service.repository = repository

        first_result = asyncio.run(
            service.query(
                ChatQueryRequest(
                    knowledge_base_id=requested_knowledge_base["id"],
                    query="试用期可以不发工资吗？",
                    debug=True,
                )
            )
        )
        conversation_ids.append(first_result["conversation_id"])

        second_result = asyncio.run(
            service.query(
                ChatQueryRequest(
                    knowledge_base_id=requested_knowledge_base["id"],
                    conversation_id=first_result["conversation_id"],
                    query="那离职赔偿金呢？",
                    debug=True,
                )
            )
        )

        assert first_result["conversation_id"] == second_result["conversation_id"]
        assert first_result["knowledge_base_id"] == labor_knowledge_base["id"]
        assert second_result["knowledge_base_id"] == labor_knowledge_base["id"]
        assert second_result["auto_routed"] is True
        assert second_result["debug"]["resolved_knowledge_base_id"] == labor_knowledge_base["id"]
        assert "依法支付赔偿金" in second_result["answer"]
        assert len(labor_llm.calls) == 2

        messages = repository.list_messages(first_result["conversation_id"])
        assert [item["sequence"] for item in messages] == [1, 2, 3, 4]
        assert [item["role"] for item in messages] == ["user", "assistant", "user", "assistant"]
        assert all(item["knowledge_base_id"] == labor_knowledge_base["id"] for item in messages)
        assert messages[-1]["debug"] is not None
        assert messages[-1]["debug"]["resolved_knowledge_base_id"] == labor_knowledge_base["id"]
        assert messages[-1]["debug"]["auto_routed"] is True
    finally:
        _cleanup(
            repository,
            conversation_ids=conversation_ids,
            knowledge_base_ids=[requested_knowledge_base["id"], labor_knowledge_base["id"]],
        )


def test_rag_query_end_to_end_applies_answer_guard_and_persists_conservative_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sql_metadata_repository: MetadataRepository,
) -> None:
    """限定条件未被资料覆盖时，answer guard 应给出保守回答并清空引用。"""
    repository = sql_metadata_repository
    knowledge_base, _document, _task_result, _pipeline, _llm, _rewriter = _create_ingested_knowledge_base(
        repository,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    conversation_ids: list[str] = []

    try:
        service = rag_service.RAGService()
        service.repository = repository

        result = asyncio.run(
            service.query(
                ChatQueryRequest(
                    knowledge_base_id=knowledge_base["id"],
                    query="实习期间 alpha 学员需要做什么？",
                    enable_auto_route=False,
                    debug=True,
                )
            )
        )
        conversation_ids.append(result["conversation_id"])

        assert "现有资料中没有明确覆盖“实习期间”这一限定场景" in result["answer"]
        assert result["citations"] == []
        assert result["debug"]["answer_guard_applied"] is True
        assert result["debug"]["unsupported_qualifiers"] == ["实习期间"]
        assert result["debug"]["final_context_chunks"]

        messages = repository.list_messages(result["conversation_id"])
        assert [item["role"] for item in messages] == ["user", "assistant"]
        assert messages[1]["citations"] == []
        assert messages[1]["debug"] is not None
        assert messages[1]["debug"]["answer_guard_applied"] is True
        assert messages[1]["debug"]["unsupported_qualifiers"] == ["实习期间"]
    finally:
        _cleanup(
            repository,
            conversation_ids=conversation_ids,
            knowledge_base_ids=[knowledge_base["id"]],
        )


def test_rag_query_end_to_end_applies_cross_domain_guard_and_persists_short_circuit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sql_metadata_repository: MetadataRepository,
) -> None:
    """跨业务域问题应短路，不进入 LLM 生成。"""
    repository = sql_metadata_repository
    knowledge_base = _create_ready_knowledge_base(
        repository,
        name_prefix="RAG E2E Guard KB",
        description="rag e2e cross domain guard integration",
        subject="培训管理制度",
        domain="training_management",
    )
    pipeline, llm, _rewriter = _build_pipeline(tmp_path)
    monkeypatch.setattr("app.services.rag_service.get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)
    conversation_ids: list[str] = []

    try:
        service = rag_service.RAGService()
        service.repository = repository

        result = asyncio.run(
            service.query(
                ChatQueryRequest(
                    knowledge_base_id=knowledge_base["id"],
                    query="离职和退学分别怎么处理？",
                    enable_auto_route=False,
                    debug=True,
                )
            )
        )
        conversation_ids.append(result["conversation_id"])

        assert "这个问题同时涉及“劳动法”、“培训管理”多个业务域" in result["answer"]
        assert result["citations"] == []
        assert result["debug"]["cross_domain_guard_applied"] is True
        assert result["debug"]["detected_domains"] == ["labor_law", "training_management"]
        assert llm.calls == []

        messages = repository.list_messages(result["conversation_id"])
        assert [item["role"] for item in messages] == ["user", "assistant"]
        assert messages[1]["citations"] == []
        assert messages[1]["debug"] is not None
        assert messages[1]["debug"]["cross_domain_guard_applied"] is True
    finally:
        _cleanup(
            repository,
            conversation_ids=conversation_ids,
            knowledge_base_ids=[knowledge_base["id"]],
        )


def test_rag_query_end_to_end_raises_when_knowledge_base_is_not_ready(
    sql_metadata_repository: MetadataRepository,
) -> None:
    """未 ready 的知识库不能进入问答链路，也不应创建会话。"""
    repository = sql_metadata_repository
    knowledge_base = repository.create_knowledge_base(
        name=f"RAG E2E Not Ready KB {uuid4().hex[:6]}",
        description="rag e2e not ready integration",
        subject="Alpha 学员管理制度",
        domain="general",
    )

    try:
        service = rag_service.RAGService()
        service.repository = repository

        with pytest.raises(KnowledgeBaseNotReadyError):
            asyncio.run(
                service.query(
                    ChatQueryRequest(
                        knowledge_base_id=knowledge_base["id"],
                        query="alpha 学员需要做什么？",
                        enable_auto_route=False,
                        debug=True,
                    )
                )
            )

        assert repository.list_conversations(default_knowledge_base_id=knowledge_base["id"]) == []
    finally:
        _cleanup(repository, knowledge_base_ids=[knowledge_base["id"]])


def test_rag_query_end_to_end_raises_when_conversation_does_not_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sql_metadata_repository: MetadataRepository,
) -> None:
    """传入不存在的 conversation_id 时应报错，并且不创建新会话。"""
    repository = sql_metadata_repository
    knowledge_base = _create_ready_knowledge_base(
        repository,
        name_prefix="RAG E2E Missing Conversation KB",
        description="rag e2e missing conversation integration",
    )
    pipeline, _llm, _rewriter = _build_pipeline(tmp_path)
    monkeypatch.setattr("app.services.rag_service.get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    try:
        service = rag_service.RAGService()
        service.repository = repository

        with pytest.raises(ConversationNotFoundError):
            asyncio.run(
                service.query(
                    ChatQueryRequest(
                        knowledge_base_id=knowledge_base["id"],
                        conversation_id="conv_missing_e2e",
                        query="alpha 学员需要做什么？",
                        enable_auto_route=False,
                        debug=True,
                    )
                )
            )

        assert repository.list_conversations(default_knowledge_base_id=knowledge_base["id"]) == []
    finally:
        _cleanup(repository, knowledge_base_ids=[knowledge_base["id"]])


def test_prepare_query_inputs_adds_overview_expansion_keywords() -> None:
    """概览型问题会补充配置中的扩展关键词，提高召回覆盖。"""
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.rewriter = DeterministicRewriter()
    pipeline.query_alias_expander = StaticAliasExpander()
    pipeline.retrieval_rules = SimpleNamespace(overview_query_expansions={"课堂违纪": ("课堂纪律", "上课睡觉", "课堂捣乱")})

    prepared = pipeline.prepare_query_inputs(query="课堂违纪一般会怎么处理？", enable_rewrite=True)

    assert "课堂纪律" in str(prepared["search_query"])
    assert "上课睡觉" in str(prepared["search_query"])
    assert "课堂捣乱" in list(prepared["query_keywords"])
