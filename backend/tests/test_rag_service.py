import asyncio

import pytest

from app.schemas.chat import ChatQueryRequest
from app.services import rag_service
from app.services.knowledge_base_router import KnowledgeBaseRouter
from app.services.knowledge_base_routing_config import (
    KnowledgeBaseRoutingConfig,
    RoutingEmbeddingConfig,
    KnowledgeBaseQueryRuleConfig,
    KnowledgeBaseRuleConfig,
    RoutingScoreConfig,
)
from app.services.faq_shortcut_config import FAQShortcutConfig, FAQShortcutRule


class FakeMetadataRepository:
    def __init__(self):
        self.conversations = {}
        self.messages = {}
        self.message_contexts = {}

    def get_knowledge_base(self, knowledge_base_id: str):
        mapping = {
            "kb_001": {
                "id": "kb_001",
                "name": "hemingRulesText",
                "subject": "和鸣教育管理制度",
                "description": "",
                "domain": "training_management",
                "status": "ready",
            },
            "kb_law": {
                "id": "kb_law",
                "name": "劳动&劳动合同",
                "subject": "劳动相关法律",
                "description": "",
                "domain": "labor_law",
                "status": "ready",
            },
        }
        return mapping.get(knowledge_base_id)

    def list_knowledge_bases(self):
        return [
            self.get_knowledge_base("kb_001"),
            self.get_knowledge_base("kb_law"),
        ]

    def get_conversation(self, conversation_id: str):
        return self.conversations.get(conversation_id)

    def create_conversation(self, title: str, default_knowledge_base_id: str | None = None):
        conversation = {
            "id": "conv_001",
            "default_knowledge_base_id": default_knowledge_base_id,
            "title": title,
        }
        self.conversations[conversation["id"]] = conversation
        self.messages.setdefault(conversation["id"], [])
        return conversation

    def create_message(self, **kwargs):
        message = {"id": "msg_001", **kwargs}
        self.messages.setdefault(kwargs["conversation_id"], []).append(message)
        return message

    def create_message_context(self, message_id: str, knowledge_base_id=None, citations=None, debug=None):
        self.message_contexts[message_id] = {
            "message_id": message_id,
            "knowledge_base_id": knowledge_base_id,
            "citations": citations or [],
            "debug": debug,
        }
        return self.message_contexts[message_id]

    def list_messages(self, conversation_id: str):
        return list(self.messages.get(conversation_id, []))

    def list_documents(self, knowledge_base_id: str):
        if knowledge_base_id == "kb_law":
            return [
                {
                    "id": "doc_labor_law",
                    "knowledge_base_id": knowledge_base_id,
                    "filename": "中华人民共和国劳动法.md",
                },
                {
                    "id": "doc_labor_contract_law",
                    "knowledge_base_id": knowledge_base_id,
                    "filename": "中华人民共和国劳动合同法.md",
                },
            ]
        return [
            {
                "id": "doc_labor_law",
                "knowledge_base_id": knowledge_base_id,
                "filename": "和鸣教育管理制度精简chunk版.md",
            },
            {
                "id": "doc_labor_contract_law",
                "knowledge_base_id": knowledge_base_id,
                "filename": "学生管理制度.docx",
            },
        ]


class StubPipeline:
    def __init__(self):
        self.called = False
        self.last_kwargs = None
        self.rewriter = self
        self.shadow_called = False

    def ask(self, **kwargs):
        self.called = True
        self.last_kwargs = kwargs
        return {
            "answer": "pipeline answer",
            "conversation_id": "",
            "citations": [],
            "debug": {
                "rewritten_query": kwargs["query"],
                "retrieved_count": 0,
                "reranked_count": 0,
                "enable_rewrite": kwargs["enable_rewrite"],
                "enable_rerank": kwargs["enable_rerank"],
                "rewrite_ms": 0,
                "retrieve_ms": 0,
                "rerank_ms": 0,
                "generate_ms": 0,
                "latency_ms": 0,
                "llm_finish_reason": "stop",
                "llm_model": None,
                "llm_prompt_tokens": None,
                "llm_completion_tokens": None,
                "llm_total_tokens": None,
                "used_conversation_history": kwargs.get("history_rewritten_query") is not None,
                "history_message_count": kwargs.get("history_message_count", 0),
                "history_rewritten_query": kwargs.get("history_rewritten_query"),
                "history_rewrite_ms": kwargs.get("history_rewrite_ms", 0),
            },
        }

    def run_shadow_graph(self, **kwargs):
        self.shadow_called = True
        return self.ask(**kwargs)

    def rewrite_with_history(self, query, history_messages):
        if not history_messages:
            return query
        return f"改写后:{query}"


class ErrorStubPipeline(StubPipeline):
    def ask(self, **kwargs):
        self.called = True
        self.last_kwargs = kwargs
        raise RuntimeError("Incorrect API key provided: replace-me")

    def ask_stream(self, **kwargs):
        self.called = True
        self.last_kwargs = kwargs
        raise RuntimeError("Incorrect API key provided: replace-me")


class GuardedStubPipeline(StubPipeline):
    def ask(self, **kwargs):
        result = super().ask(**kwargs)
        result["citations"] = [
            {
                "document_id": "kb_001",
                "chunk_id": "parent_1-1",
                "content": "宿舍违规一般条款",
                "score": 0.9,
                "source_name": "《和鸣教育管理制度》",
                "source_type": "text",
                "page_number": 1,
                "block_index": 1,
            }
        ]
        result["debug"]["final_context_chunks"] = [
            {
                "chunk_id": "parent_1",
                "parent_id": "parent_1",
                "child_text": "在宿舍打牌、打游戏并影响其他室友学习或休息的，扣10分/次。",
                "parent_text": "在宿舍打牌、打游戏并影响其他室友学习或休息的，扣10分/次。",
                "distance": 0.1,
                "rerank_score": 0.9,
                "source_name": "《和鸣教育管理制度》",
                "source_type": "text",
                "page_number": 1,
                "block_index": 1,
            }
        ]
        return result


class OverviewStubPipeline(GuardedStubPipeline):
    def ask(self, **kwargs):
        result = super().ask(**kwargs)
        result["answer"] = (
            "【直接依据】\n"
            "- 课堂捣乱且影响他人学习的，扣20分/次。\n"
            "- 上课睡觉或做与学习无关的事，扣5分/次。\n\n"
            "【推理/判断】\n"
            "- 资料中未提供关于“课堂违纪”的统一总则，需要区分具体情形，按对应规则处理。\n\n"
            "【结论】\n"
            "- 课堂违纪需要按具体行为对应不同扣分规则处理。"
        )
        result["debug"]["reranked_count"] = 1
        result["debug"]["reranked_chunks"] = [
            {
                "chunk_id": "parent_1",
                "parent_id": "parent_1",
                "parent_text": "在宿舍打牌、打游戏并影响其他室友学习或休息的，扣10分/次。",
            }
        ]
        return result


class NoAnswerStubPipeline(GuardedStubPipeline):
    def ask(self, **kwargs):
        result = super().ask(**kwargs)
        result["answer"] = (
            "【直接依据】\n"
            "- 参考资料中未提供关于“携带宠物进入宿舍”的规定。\n\n"
            "【结论】\n"
            "- 根据现有参考资料，无法确定是否可以带宠物进宿舍。"
        )
        result["debug"]["reranked_count"] = 1
        result["debug"]["reranked_chunks"] = [
            {
                "chunk_id": "parent_1",
                "parent_id": "parent_1",
                "parent_text": "在宿舍打牌、打游戏并影响其他室友学习或休息的，扣10分/次。",
            }
        ]
        return result



class PartialUnknownClauseStubPipeline(GuardedStubPipeline):
    def ask(self, **kwargs):
        result = super().ask(**kwargs)
        result["answer"] = (
            "【直接依据】\n"
            "- 累计扣分 1—10 分：由班主任谈话警告。\n"
            "- 累计扣分 11—20 分：由学工部教导处谈话。\n"
            "- 累计扣分 21—30 分：由班主任联系家长并通报。\n"
            "- 累计扣分 40 分及以上：无条件退学。\n\n"
            "【推理/判断】\n"
            "- 累计扣分需要按分数段处理。\n\n"
            "【结论】\n"
            "- 1—10 分、11—20 分、21—30 分和 40 分及以上都有明确处理规则；31—39 分现有资料不足，无法确定。"
        )
        result["debug"]["reranked_count"] = 1
        result["debug"]["reranked_chunks"] = [
            {
                "chunk_id": "parent_1",
                "parent_id": "parent_1",
                "parent_text": "在宿舍打牌、打游戏并影响其他室友学习或休息的，扣10分/次。",
            }
        ]
        return result


class CitedPartialUnknownClauseStubPipeline(GuardedStubPipeline):
    def ask(self, **kwargs):
        result = super().ask(**kwargs)
        result["answer"] = (
            "结论：\n"
            "- 早退在 20 分钟以内的，扣 3 分/次。\n"
            "- 早退超过 20 分钟的处理方式，参考资料未明确。\n\n"
            "依据：\n"
            "- 《和鸣教育管理制度精简chunk版》规定：迟到或早退，且时间在 20 分钟以内，扣 3 分/次。"
        )
        result["debug"]["reranked_count"] = 1
        result["debug"]["reranked_chunks"] = [
            {
                "chunk_id": "parent_1",
                "parent_id": "parent_1",
                "parent_text": "迟到或早退，且时间在 20 分钟以内，扣 3 分/次。",
            }
        ]
        return result


class RangeSummaryPartialUnknownStubPipeline(GuardedStubPipeline):
    def ask(self, **kwargs):
        result = super().ask(**kwargs)
        result["answer"] = (
            "结论：\n"
            "- 累计扣分 1—10 分：由班主任谈话警告。\n"
            "- 累计扣分 11—20 分：每次扣分后，由学工部教导处谈话。\n"
            "- 累计扣分 40 分及以上：无条件退学，且学费不予退还。\n\n"
            "补充说明：\n"
            "- 参考资料未明确累计扣分 21—39 分的处理方式。"
        )
        result["debug"]["reranked_count"] = 3
        result["debug"]["reranked_chunks"] = [
            {"chunk_id": "rule_doc_3", "parent_id": "rule_doc_3", "parent_text": "累计扣分 1—10 分：由班主任谈话警告。"},
            {"chunk_id": "rule_doc_1", "parent_id": "rule_doc_1", "parent_text": "累计扣分 11—20 分：每次扣分后，由学工部教导处谈话。"},
            {"chunk_id": "rule_doc_0", "parent_id": "rule_doc_0", "parent_text": "累计扣分 40 分及以上：无条件退学，学费不予退还。"},
        ]
        result["debug"]["final_context_chunks"] = list(result["debug"]["reranked_chunks"])
        result["citations"] = [
            {"document_id": "kb_001", "chunk_id": "rule_doc_3-1", "content": "累计扣分 1—10 分：由班主任谈话警告。", "score": 0.9},
            {"document_id": "kb_001", "chunk_id": "rule_doc_1-2", "content": "累计扣分 11—20 分：每次扣分后，由学工部教导处谈话。", "score": 0.8},
            {"document_id": "kb_001", "chunk_id": "rule_doc_0-3", "content": "累计扣分 40 分及以上：无条件退学，学费不予退还。", "score": 0.7},
        ]
        return result


class MismatchShadowStubPipeline(StubPipeline):
    def run_shadow_graph(self, **kwargs):
        self.shadow_called = True
        return {
            "answer": "shadow answer",
            "citations": [
                {
                    "document_id": "kb_001",
                    "chunk_id": "shadow_parent-1",
                    "content": "shadow content",
                    "score": 0.5,
                }
            ],
            "debug": {
                "rewritten_query": "shadow rewritten",
                "retrieved_count": 0,
                "reranked_count": 1,
                "enable_rewrite": kwargs["enable_rewrite"],
                "enable_rerank": kwargs["enable_rerank"],
                "rewrite_ms": 0,
                "retrieve_ms": 0,
                "rerank_ms": 0,
                "generate_ms": 0,
                "latency_ms": 0,
                "llm_finish_reason": "stop",
                "llm_model": None,
                "llm_prompt_tokens": None,
                "llm_completion_tokens": None,
                "llm_total_tokens": None,
                "used_conversation_history": False,
                "history_message_count": 0,
                "history_rewritten_query": None,
                "history_rewrite_ms": 0,
                "final_context_chunks": [
                    {
                        "chunk_id": "shadow_parent",
                        "parent_id": "shadow_parent",
                        "parent_text": "shadow parent",
                    }
                ],
            },
        }


class ForceGraphStubPipeline(GuardedStubPipeline):
    def run_shadow_graph(self, **kwargs):
        self.shadow_called = True
        result = super().ask(**kwargs)
        result["answer"] = "shadow graph answer"
        result["citations"] = [
            {
                "document_id": "kb_001",
                "chunk_id": "graph_parent-1",
                "content": "graph content",
                "score": 0.8,
            }
        ]
        result["debug"]["final_context_chunks"] = [
            {
                "chunk_id": "graph_parent",
                "parent_id": "graph_parent",
                "parent_text": "graph parent",
            }
        ]
        return result


class FakeRouteEmbedder:
    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    def encode(self, text: str, *, is_query: bool = False) -> list[float]:
        return list(self.vectors.get(text, [0.0, 0.0]))


def test_greeting_query_returns_guidance_without_retrieval(monkeypatch):
    pipeline = StubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    service.repository = FakeMetadataRepository()

    payload = ChatQueryRequest(knowledge_base_id="kb_001", query="你好呀", debug=True)
    result = asyncio.run(service.query(payload))

    assert "我是 ReguRAG 助手" in result["answer"]
    assert result["conversation_id"] == "conv_001"
    assert "相关制度、流程、要求、范围或处理规则" in result["answer"]
    assert result["citations"] == []
    assert result["debug"]["llm_finish_reason"] == "greeting_short_circuit"
    assert pipeline.called is False


def test_thanks_query_returns_guidance_without_retrieval(monkeypatch):
    pipeline = StubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    service.repository = FakeMetadataRepository()

    payload = ChatQueryRequest(knowledge_base_id="kb_001", query="谢谢你！", debug=True)
    result = asyncio.run(service.query(payload))

    assert "不客气" in result["answer"]
    assert result["conversation_id"] == "conv_001"
    assert "ReguRAG 助手" in result["answer"]
    assert result["debug"]["llm_finish_reason"] == "thanks_short_circuit"
    assert pipeline.called is False


def test_identity_query_returns_guidance_without_retrieval(monkeypatch):
    pipeline = StubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    service.repository = FakeMetadataRepository()

    payload = ChatQueryRequest(knowledge_base_id="kb_001", query="你是谁？", debug=True)
    result = asyncio.run(service.query(payload))

    assert "我是 ReguRAG 助手" in result["answer"]
    assert result["conversation_id"] == "conv_001"
    assert "和鸣教育管理制度" in result["answer"]
    assert result["debug"]["llm_finish_reason"] == "identity_short_circuit"
    assert pipeline.called is False


def test_greeting_plus_identity_query_short_circuits_without_retrieval(monkeypatch):
    pipeline = StubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    service.repository = FakeMetadataRepository()

    payload = ChatQueryRequest(knowledge_base_id="kb_001", query="你好，你是？", debug=True)
    result = asyncio.run(service.query(payload))

    assert "我是 ReguRAG 助手" in result["answer"]
    assert result["conversation_id"] == "conv_001"
    assert result["debug"]["llm_finish_reason"] == "identity_short_circuit"
    assert pipeline.called is False


def test_capability_query_returns_guidance_without_retrieval(monkeypatch):
    pipeline = StubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    service.repository = FakeMetadataRepository()

    payload = ChatQueryRequest(knowledge_base_id="kb_001", query="你能做什么", debug=True)
    result = asyncio.run(service.query(payload))

    assert "主要帮助你查询和鸣教育管理制度" in result["answer"]
    assert result["conversation_id"] == "conv_001"
    assert "相关流程是什么" in result["answer"]
    assert "这类情况怎么处理" in result["answer"]
    assert result["debug"]["llm_finish_reason"] == "capability_short_circuit"
    assert pipeline.called is False


def test_farewell_query_returns_guidance_without_retrieval(monkeypatch):
    pipeline = StubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    service.repository = FakeMetadataRepository()

    payload = ChatQueryRequest(knowledge_base_id="kb_001", query="再见", debug=True)
    result = asyncio.run(service.query(payload))

    assert "再见" in result["answer"]
    assert result["conversation_id"] == "conv_001"
    assert "ReguRAG 助手" in result["answer"]
    assert result["debug"]["llm_finish_reason"] == "farewell_short_circuit"
    assert pipeline.called is False


def test_off_topic_query_returns_scope_guidance_without_retrieval(monkeypatch):
    pipeline = StubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    service.repository = FakeMetadataRepository()

    payload = ChatQueryRequest(knowledge_base_id="kb_001", query="今天天气怎么样？", debug=True)
    result = asyncio.run(service.query(payload))

    assert "目前主要回答和鸣教育管理制度相关问题" in result["answer"]
    assert result["conversation_id"] == "conv_001"
    assert "相关制度、流程、要求、范围或处理规则" in result["answer"]
    assert result["debug"]["llm_finish_reason"] == "off_topic_short_circuit"
    assert pipeline.called is False


def test_meaningless_laughter_query_short_circuits_without_retrieval(monkeypatch):
    pipeline = StubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    service.repository = FakeMetadataRepository()

    payload = ChatQueryRequest(knowledge_base_id="kb_law", query="哈哈哈", debug=True)
    result = asyncio.run(service.query(payload))

    assert "目前主要回答劳动相关法律相关问题" in result["answer"]
    assert result["conversation_id"] == "conv_001"
    assert result["debug"]["llm_finish_reason"] == "meaningless_input_short_circuit"
    assert result["debug"]["intent_name"] == "meaningless_input"
    assert result["debug"]["intent_classifier_source"] == "heuristic"
    assert result["debug"]["intent_classifier_mode"] == "heuristic"
    assert result["debug"]["intent_classifier_score"] is None
    assert result["debug"]["intent_classifier_margin"] is None
    assert pipeline.called is False


def test_meaningless_interjection_query_short_circuits_without_retrieval(monkeypatch):
    pipeline = StubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    service.repository = FakeMetadataRepository()

    payload = ChatQueryRequest(knowledge_base_id="kb_law", query="嘻嘻", debug=True)
    result = asyncio.run(service.query(payload))

    assert "目前主要回答劳动相关法律相关问题" in result["answer"]
    assert result["conversation_id"] == "conv_001"
    assert result["debug"]["llm_finish_reason"] == "meaningless_input_short_circuit"
    assert result["debug"]["intent_name"] == "meaningless_input"
    assert result["debug"]["intent_classifier_source"] == "heuristic"
    assert result["debug"]["intent_classifier_mode"] == "heuristic"
    assert pipeline.called is False


def test_faq_query_returns_high_confidence_short_circuit_without_retrieval(monkeypatch):
    pipeline = StubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)
    monkeypatch.setattr(
        rag_service,
        "load_faq_shortcut_config",
        lambda: FAQShortcutConfig(
            rules=(
                FAQShortcutRule(
                    name="leave_policy_overview",
                    patterns=("请假制度是什么",),
                    answer_template="如果你想了解“{topic}”，建议直接追问更具体的规则点。你可以继续问：“{example_query_1}”“{example_query_2}”“{example_query_3}”。",
                    finish_reason="faq_short_circuit",
                    topic="请假制度",
                    suggested_queries=("请假流程是什么", "请假需要提前多久", "补假怎么处理"),
                ),
            )
        ),
    )

    service = rag_service.RAGService()
    repository = FakeMetadataRepository()
    service.repository = repository

    payload = ChatQueryRequest(knowledge_base_id="kb_001", query="请假制度是什么", debug=True)
    result = asyncio.run(service.query(payload))

    assert "请假制度" in result["answer"]
    assert "请假流程是什么" in result["answer"]
    assert result["answer_source"] == "faq_short_circuit"
    assert result["conversation_id"] == "conv_001"
    assert result["debug"]["llm_finish_reason"] == "faq_short_circuit"
    assert result["debug"]["faq_short_circuit_applied"] is True
    assert result["debug"]["faq_rule_name"] == "leave_policy_overview"
    assert result["debug"]["faq_topic"] == "请假制度"
    assert len(repository.messages["conv_001"]) == 2
    assert pipeline.called is False


def test_exact_business_question_does_not_fall_into_faq_short_circuit(monkeypatch):
    pipeline = StubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)
    monkeypatch.setattr(
        rag_service,
        "load_faq_shortcut_config",
        lambda: FAQShortcutConfig(
            rules=(
                FAQShortcutRule(
                    name="leave_policy_overview",
                    patterns=("请假制度", "请假制度是什么"),
                    answer_template="如果你想了解“{topic}”，建议直接追问更具体的规则点。",
                    finish_reason="faq_short_circuit",
                    topic="请假制度",
                    suggested_queries=(),
                ),
            )
        ),
    )

    service = rag_service.RAGService()
    service.repository = FakeMetadataRepository()

    payload = ChatQueryRequest(knowledge_base_id="kb_001", query="请假流程是什么", debug=True)
    result = asyncio.run(service.query(payload))

    assert result["answer"] == "pipeline answer"
    assert result["debug"]["faq_short_circuit_applied"] is False
    assert pipeline.called is True


def test_light_intent_response_uses_knowledge_base_subject(monkeypatch):
    pipeline = StubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    repository = FakeMetadataRepository()
    repository.get_knowledge_base = lambda knowledge_base_id: {
        "id": knowledge_base_id,
        "subject": "劳动合同法",
        "status": "ready",
    }
    service.repository = repository

    payload = ChatQueryRequest(knowledge_base_id="kb_001", query="你是谁？", debug=True)
    result = asyncio.run(service.query(payload))

    assert "劳动合同法" in result["answer"]
    assert "和鸣教育管理制度" not in result["answer"]


def test_policy_query_still_uses_pipeline_even_if_it_mentions_weather(monkeypatch):
    pipeline = StubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    service.repository = FakeMetadataRepository()

    payload = ChatQueryRequest(knowledge_base_id="kb_001", query="下雨天上课迟到怎么算考勤", debug=True)
    result = asyncio.run(service.query(payload))

    assert result["answer"] == "pipeline answer"
    assert result["conversation_id"] == "conv_001"
    assert pipeline.called is True


def test_non_greeting_query_still_uses_pipeline(monkeypatch):
    pipeline = StubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    service.repository = FakeMetadataRepository()

    payload = ChatQueryRequest(knowledge_base_id="kb_001", query="请介绍一下请假制度", debug=True)
    result = asyncio.run(service.query(payload))

    assert result["answer"] == "pipeline answer"
    assert result["conversation_id"] == "conv_001"
    assert pipeline.called is True


def test_behavior_question_is_not_treated_as_global_policy_question(monkeypatch):
    pipeline = StubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    service.repository = FakeMetadataRepository()

    payload = ChatQueryRequest(knowledge_base_id="kb_001", query="迟到会怎么样", debug=True)
    result = asyncio.run(service.query(payload))

    assert result["answer"] == "pipeline answer"
    assert result["conversation_id"] == "conv_001"
    assert pipeline.called is True


def test_query_flags_are_passed_to_pipeline(monkeypatch):
    pipeline = StubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    service.repository = FakeMetadataRepository()

    payload = ChatQueryRequest(
        knowledge_base_id="kb_001",
        query="请介绍一下请假制度",
        enable_rewrite=False,
        enable_rerank=False,
        debug=True,
    )
    result = asyncio.run(service.query(payload))

    assert result["answer"] == "pipeline answer"
    assert result["conversation_id"] == "conv_001"
    assert pipeline.called is True
    assert pipeline.last_kwargs["enable_rewrite"] is False
    assert pipeline.last_kwargs["enable_rerank"] is False


def test_debug_mode_uses_structured_answer_style(monkeypatch):
    pipeline = StubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    service.repository = FakeMetadataRepository()

    payload = ChatQueryRequest(
        knowledge_base_id="kb_001",
        query="请介绍一下请假制度",
        debug=True,
    )
    asyncio.run(service.query(payload))

    assert pipeline.last_kwargs["answer_style"] == "structured"


def test_non_debug_mode_uses_concise_answer_style(monkeypatch):
    pipeline = StubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    service.repository = FakeMetadataRepository()

    payload = ChatQueryRequest(
        knowledge_base_id="kb_001",
        query="请介绍一下请假制度",
    )
    asyncio.run(service.query(payload))

    assert pipeline.last_kwargs["answer_style"] == "concise"


def test_debug_answer_style_override_uses_concise(monkeypatch):
    pipeline = StubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    service.repository = FakeMetadataRepository()

    payload = ChatQueryRequest(
        knowledge_base_id="kb_001",
        query="请介绍一下请假制度",
        debug=True,
        debug_answer_style="concise",
    )
    asyncio.run(service.query(payload))

    assert pipeline.last_kwargs["answer_style"] == "concise"


def test_query_marks_legacy_pipeline_as_answer_source(monkeypatch):
    pipeline = StubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    service.repository = FakeMetadataRepository()

    payload = ChatQueryRequest(
        knowledge_base_id="kb_001",
        query="请介绍一下请假制度",
        debug=True,
    )
    result = asyncio.run(service.query(payload))

    assert result["answer_source"] == "legacy_pipeline"


def test_debug_force_graph_response_returns_shadow_graph_answer(monkeypatch):
    pipeline = ForceGraphStubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    service.repository = FakeMetadataRepository()

    payload = ChatQueryRequest(
        knowledge_base_id="kb_001",
        query="请介绍一下请假制度",
        debug=True,
        debug_force_graph_response=True,
    )
    result = asyncio.run(service.query(payload))

    assert pipeline.shadow_called is True
    assert result["answer"] == "shadow graph answer"
    assert result["answer_source"] == "shadow_graph"
    assert result["debug"]["shadow_compare"]["status"] == "forced_graph_response"


def test_debug_shadow_compare_attaches_match_summary(monkeypatch):
    pipeline = GuardedStubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    service.repository = FakeMetadataRepository()

    payload = ChatQueryRequest(
        knowledge_base_id="kb_001",
        query="请介绍一下请假制度",
        debug=True,
        debug_shadow_compare=True,
    )
    result = asyncio.run(service.query(payload))

    assert pipeline.shadow_called is True
    assert result["debug"]["shadow_compare"]["status"] == "match"
    assert result["debug"]["shadow_compare"]["mismatch_fields"] == []
    assert result["debug"]["shadow_compare"]["answer_match"] is True


def test_debug_shadow_compare_attaches_mismatch_summary(monkeypatch):
    pipeline = MismatchShadowStubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    service.repository = FakeMetadataRepository()

    payload = ChatQueryRequest(
        knowledge_base_id="kb_001",
        query="请介绍一下请假制度",
        debug=True,
        debug_shadow_compare=True,
    )
    result = asyncio.run(service.query(payload))

    assert pipeline.shadow_called is True
    assert result["debug"]["shadow_compare"]["status"] == "mismatch"
    assert "answer" in result["debug"]["shadow_compare"]["mismatch_fields"]
    assert "citation_ids" in result["debug"]["shadow_compare"]["mismatch_fields"]
    assert "rewritten_query" in result["debug"]["shadow_compare"]["mismatch_fields"]


def test_shadow_compare_rollout_can_be_enabled_by_settings_sampling(monkeypatch):
    pipeline = GuardedStubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)
    monkeypatch.setattr(
        rag_service,
        "_settings",
        lambda: type(
            "SettingsStub",
            (),
            {
                "chat_auto_route_enabled": True,
                "chat_shadow_compare_enabled": True,
                "chat_shadow_compare_sample_rate": 1.0,
            },
        )(),
    )
    monkeypatch.setattr(rag_service.random, "random", lambda: 0.0)

    service = rag_service.RAGService()
    service.repository = FakeMetadataRepository()

    payload = ChatQueryRequest(
        knowledge_base_id="kb_001",
        query="请介绍一下请假制度",
        debug=True,
    )
    result = asyncio.run(service.query(payload))

    assert pipeline.shadow_called is True
    assert result["debug"]["shadow_compare"]["status"] == "match"


def test_shadow_compare_rollout_does_not_run_when_sampling_disabled(monkeypatch):
    pipeline = GuardedStubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)
    monkeypatch.setattr(
        rag_service,
        "_settings",
        lambda: type(
            "SettingsStub",
            (),
            {
                "chat_auto_route_enabled": True,
                "chat_shadow_compare_enabled": False,
                "chat_shadow_compare_sample_rate": 0.0,
            },
        )(),
    )

    service = rag_service.RAGService()
    service.repository = FakeMetadataRepository()

    payload = ChatQueryRequest(
        knowledge_base_id="kb_001",
        query="请介绍一下请假制度",
        debug=True,
    )
    result = asyncio.run(service.query(payload))

    assert pipeline.shadow_called is False
    assert result["debug"].get("shadow_compare") is None


def test_follow_up_query_uses_recent_history_for_retrieval(monkeypatch):
    pipeline = StubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    repository = FakeMetadataRepository()
    service.repository = repository
    repository.conversations["conv_001"] = {
        "id": "conv_001",
        "default_knowledge_base_id": "kb_001",
        "title": "测试会话",
    }
    repository.messages["conv_001"] = [
        {"role": "user", "content": "迟到会怎么处理", "knowledge_base_id": "kb_001"},
        {"role": "assistant", "content": "会扣分", "knowledge_base_id": "kb_001"},
    ]

    payload = ChatQueryRequest(
        knowledge_base_id="kb_001",
        conversation_id="conv_001",
        query="那如果是第一次呢",
        debug=True,
    )
    result = asyncio.run(service.query(payload))

    assert result["answer"] == "pipeline answer"
    assert pipeline.last_kwargs["history_rewritten_query"] == "改写后:那如果是第一次呢"
    assert result["debug"]["used_conversation_history"] is True
    assert result["debug"]["history_message_count"] == 2


def test_meaningless_laughter_follow_up_does_not_reuse_history(monkeypatch):
    pipeline = StubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    repository = FakeMetadataRepository()
    service.repository = repository

    asyncio.run(
        service.query(
            ChatQueryRequest(
                knowledge_base_id="kb_law",
                query="试用期不签劳动合同，这样的公司能去入职吗？",
                debug=True,
            )
        )
    )

    pipeline.called = False
    pipeline.last_kwargs = None

    result = asyncio.run(
        service.query(
            ChatQueryRequest(
                knowledge_base_id="kb_law",
                query="哈哈哈",
                conversation_id="conv_001",
                debug=True,
            )
        )
    )

    assert result["debug"]["llm_finish_reason"] == "meaningless_input_short_circuit"
    assert result["debug"]["intent_name"] == "meaningless_input"
    assert result["conversation_id"] == "conv_001"
    assert pipeline.called is False
    assert pipeline.last_kwargs is None


def test_query_without_history_keeps_original_retrieval_query(monkeypatch):
    pipeline = StubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    service.repository = FakeMetadataRepository()

    payload = ChatQueryRequest(
        knowledge_base_id="kb_001",
        query="请介绍一下请假制度",
        debug=True,
    )
    result = asyncio.run(service.query(payload))

    assert result["answer"] == "pipeline answer"
    assert result["debug"]["intent_classifier_source"] == "default"
    assert result["debug"]["intent_classifier_mode"] == "default"
    assert result["debug"]["intent_classifier_score"] is None
    assert result["debug"]["intent_classifier_margin"] is None
    assert pipeline.last_kwargs["history_rewritten_query"] is None
    assert result["debug"]["used_conversation_history"] is False


def test_query_persists_failure_messages_when_pipeline_raises(monkeypatch):
    pipeline = ErrorStubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    repository = FakeMetadataRepository()
    service.repository = repository

    with pytest.raises(RuntimeError, match="Incorrect API key provided"):
        asyncio.run(
            service.query(
                ChatQueryRequest(
                    knowledge_base_id="kb_law",
                    query="试用期老板无故辞退我可以吗？",
                    debug=True,
                )
            )
        )

    messages = repository.list_messages("conv_001")

    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "试用期老板无故辞退我可以吗？"
    assert messages[1]["content"] == "LLM 服务鉴权失败，请检查 OPENAI_API_KEY / REWRITE_API_KEY 配置。"


def test_stream_query_persists_failure_messages_when_pipeline_raises(monkeypatch):
    pipeline = ErrorStubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    repository = FakeMetadataRepository()
    service.repository = repository

    stream = service.stream_query(
        ChatQueryRequest(
            knowledge_base_id="kb_law",
            query="试用期老板无故辞退我可以吗？",
            debug=True,
        )
    )

    first_event = next(stream)

    assert first_event["event"] == "start"

    with pytest.raises(RuntimeError, match="Incorrect API key provided"):
        next(stream)

    messages = repository.list_messages("conv_001")

    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "试用期老板无故辞退我可以吗？"
    assert messages[1]["content"] == "LLM 服务鉴权失败，请检查 OPENAI_API_KEY / REWRITE_API_KEY 配置。"


def test_build_source_name_map_uses_official_labor_law_names() -> None:
    source_map = rag_service._build_source_name_map(
        [
            {"id": "doc_1", "filename": "劳动法.pdf"},
            {"id": "doc_2", "filename": "劳动合同法_整理版.docx"},
        ]
    )

    assert source_map["doc_1"] == "《中华人民共和国劳动法》"
    assert source_map["doc_2"] == "《中华人民共和国劳动合同法》"


def test_query_passes_source_name_map_to_pipeline(monkeypatch):
    pipeline = StubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    service.repository = FakeMetadataRepository()

    payload = ChatQueryRequest(
        knowledge_base_id="kb_law",
        query="用人单位辞退员工必须要满足什么条件才能辞退？",
        debug=True,
    )
    asyncio.run(service.query(payload))

    assert pipeline.last_kwargs["source_name_by_document_id"] == {
        "doc_labor_law": "《中华人民共和国劳动法》",
        "doc_labor_contract_law": "《中华人民共和国劳动合同法》",
    }


def test_auto_route_switches_to_labor_law_knowledge_base(monkeypatch):
    pipeline = StubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    service.repository = FakeMetadataRepository()

    payload = ChatQueryRequest(
        knowledge_base_id="kb_001",
        query="劳动法规定试用期最长多久？",
        debug=True,
    )
    result = asyncio.run(service.query(payload))

    assert result["answer"] == "pipeline answer"
    assert result["knowledge_base_id"] == "kb_law"
    assert result["knowledge_base_name"] == "劳动&劳动合同"
    assert result["auto_routed"] is True
    assert pipeline.last_kwargs["knowledge_base_id"] == "kb_law"
    assert result["debug"]["resolved_knowledge_base_id"] == "kb_law"


def test_auto_route_switches_back_to_labor_law_for_trial_salary_question(monkeypatch):
    pipeline = StubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    repository = FakeMetadataRepository()
    service.repository = repository
    repository.conversations["conv_001"] = {
        "id": "conv_001",
        "default_knowledge_base_id": "kb_001",
        "title": "测试会话",
    }

    payload = ChatQueryRequest(
        knowledge_base_id="kb_001",
        conversation_id="conv_001",
        query="试用期可以不发工资吗？",
        debug=True,
    )
    result = asyncio.run(service.query(payload))

    assert result["knowledge_base_id"] == "kb_law"
    assert result["auto_routed"] is True
    assert pipeline.last_kwargs["knowledge_base_id"] == "kb_law"


def test_router_can_be_driven_by_external_config():
    router = KnowledgeBaseRouter(
        FakeMetadataRepository(),
        config=KnowledgeBaseRoutingConfig(
            category_keywords={
                "legal": ("劳动法", "劳动合同法"),
                "regulation": ("规章", "学生管理"),
            },
            knowledge_base_rules=(),
            scores=RoutingScoreConfig(
                keyword_match=8,
                metadata_term_min=2,
                metadata_term_max=6,
                preferred_knowledge_base=2,
                category_match=10,
                keep_requested_margin=3,
                knowledge_base_alias_match=12,
                knowledge_base_priority_match=6,
                knowledge_base_exclude_penalty=12,
            ),
        ),
    )

    routed = router.route(
        query="劳动法规定试用期最长多久？",
        requested_knowledge_base_id="kb_001",
    )

    assert routed is not None
    assert routed.knowledge_base_id == "kb_law"
    assert "命中领域:legal" in routed.reason


def test_router_supports_per_knowledge_base_alias_and_exclude_rules():
    router = KnowledgeBaseRouter(
        FakeMetadataRepository(),
        config=KnowledgeBaseRoutingConfig(
            category_keywords={
                "legal": ("劳动法", "劳动合同法"),
                "regulation": ("规章", "学生管理"),
            },
            knowledge_base_rules=(),
            scores=RoutingScoreConfig(
                keyword_match=0,
                metadata_term_min=0,
                metadata_term_max=0,
                preferred_knowledge_base=0,
                category_match=0,
                keep_requested_margin=1,
                knowledge_base_alias_match=10,
                knowledge_base_priority_match=5,
                knowledge_base_exclude_penalty=20,
            ),
        ),
    )

    router.config = KnowledgeBaseRoutingConfig(
        category_keywords=router.config.category_keywords,
        knowledge_base_rules=(
            KnowledgeBaseRuleConfig(
                name="labor_law",
                match_any=("劳动相关法律",),
                category="legal",
                aliases=("劳动法库",),
                priority_terms=("试用期",),
                exclude_terms=("宿舍",),
                query_rules=(),
            ),
            KnowledgeBaseRuleConfig(
                name="heming_regulation",
                match_any=("和鸣教育管理制度",),
                category="regulation",
                aliases=("和鸣制度",),
                priority_terms=("宿舍",),
                exclude_terms=("劳动法库",),
                query_rules=(),
            ),
        ),
        scores=router.config.scores,
    )

    routed = router.route(
        query="劳动法库里试用期怎么规定？",
        requested_knowledge_base_id="kb_001",
    )

    assert routed is not None
    assert routed.knowledge_base_id == "kb_law"
    assert "命中知识库别名:劳动法库" in routed.reason


def test_router_supports_boolean_query_matching_rules():
    router = KnowledgeBaseRouter(
        FakeMetadataRepository(),
        config=KnowledgeBaseRoutingConfig(
            category_keywords={
                "legal": ("劳动法", "劳动合同法"),
                "regulation": ("规章", "学生管理"),
            },
            knowledge_base_rules=(
                KnowledgeBaseRuleConfig(
                    name="labor_law",
                    match_any=("劳动相关法律",),
                    category="legal",
                    aliases=(),
                    priority_terms=(),
                    exclude_terms=(),
                    query_rules=(
                        KnowledgeBaseQueryRuleConfig(
                            name="trial_salary",
                            all_of=("试用期",),
                            any_of=("工资", "薪资"),
                            none_of=("宿舍",),
                            boost=15,
                        ),
                    ),
                ),
                KnowledgeBaseRuleConfig(
                    name="heming_regulation",
                    match_any=("和鸣教育管理制度",),
                    category="regulation",
                    aliases=(),
                    priority_terms=(),
                    exclude_terms=(),
                    query_rules=(
                        KnowledgeBaseQueryRuleConfig(
                            name="trial_dormitory",
                            all_of=("试用期",),
                            any_of=("宿舍",),
                            none_of=(),
                            boost=15,
                        ),
                    ),
                ),
            ),
            scores=RoutingScoreConfig(
                keyword_match=0,
                metadata_term_min=0,
                metadata_term_max=0,
                preferred_knowledge_base=0,
                category_match=0,
                keep_requested_margin=1,
                knowledge_base_alias_match=0,
                knowledge_base_priority_match=0,
                knowledge_base_exclude_penalty=0,
            ),
        ),
    )

    routed = router.route(
        query="试用期可以不发工资吗？",
        requested_knowledge_base_id="kb_001",
    )

    assert routed is not None
    assert routed.knowledge_base_id == "kb_law"
    assert "命中布尔规则:trial_salary" in routed.reason


def test_router_supports_multiple_boolean_query_rules_for_one_knowledge_base():
    router = KnowledgeBaseRouter(
        FakeMetadataRepository(),
        config=KnowledgeBaseRoutingConfig(
            category_keywords={
                "legal": ("劳动法", "劳动合同法"),
                "regulation": ("规章", "学生管理"),
            },
            knowledge_base_rules=(
                KnowledgeBaseRuleConfig(
                    name="labor_law",
                    match_any=("劳动相关法律",),
                    category="legal",
                    aliases=(),
                    priority_terms=(),
                    exclude_terms=(),
                    query_rules=(
                        KnowledgeBaseQueryRuleConfig(
                            name="trial_salary",
                            all_of=("试用期",),
                            any_of=("工资",),
                            none_of=("宿舍",),
                            boost=10,
                        ),
                        KnowledgeBaseQueryRuleConfig(
                            name="dismissal_compensation",
                            all_of=(),
                            any_of=("辞退", "经济补偿"),
                            none_of=("宿舍",),
                            boost=12,
                        ),
                    ),
                ),
                KnowledgeBaseRuleConfig(
                    name="heming_regulation",
                    match_any=("和鸣教育管理制度",),
                    category="regulation",
                    aliases=(),
                    priority_terms=(),
                    exclude_terms=(),
                    query_rules=(),
                ),
            ),
            scores=RoutingScoreConfig(
                keyword_match=0,
                metadata_term_min=0,
                metadata_term_max=0,
                preferred_knowledge_base=0,
                category_match=0,
                keep_requested_margin=1,
                knowledge_base_alias_match=0,
                knowledge_base_priority_match=0,
                knowledge_base_exclude_penalty=0,
            ),
        ),
    )

    routed = router.route(
        query="辞退员工需要支付经济补偿吗？",
        requested_knowledge_base_id="kb_001",
    )

    assert routed is not None
    assert routed.knowledge_base_id == "kb_law"
    assert "命中布尔规则:dismissal_compensation" in routed.reason


def test_router_prefers_embedding_route_when_similarity_is_confident():
    repository = FakeMetadataRepository()
    router = KnowledgeBaseRouter(
        repository,
        config=KnowledgeBaseRoutingConfig(
            category_keywords={
                "legal": ("劳动法",),
                "regulation": ("规章",),
            },
            knowledge_base_rules=(),
            scores=RoutingScoreConfig(
                keyword_match=0,
                metadata_term_min=0,
                metadata_term_max=0,
                preferred_knowledge_base=0,
                category_match=0,
                keep_requested_margin=1,
                knowledge_base_alias_match=0,
                knowledge_base_priority_match=0,
                knowledge_base_exclude_penalty=0,
            ),
            embedding=RoutingEmbeddingConfig(
                enabled=True,
                min_similarity=0.45,
                min_margin=0.2,
                profile_document_limit=8,
            ),
        ),
        embedder=FakeRouteEmbedder(
            {
                "劳动法试用期工资问题": [1.0, 0.0],
            }
        ),
    )
    router._encode_profile = lambda knowledge_base_id, profile_text: [1.0, 0.0] if knowledge_base_id == "kb_law" else [0.0, 1.0]

    routed = router.route(
        query="劳动法试用期工资问题",
        requested_knowledge_base_id="kb_001",
    )

    assert routed is not None
    assert routed.knowledge_base_id == "kb_law"
    assert "命中向量路由:" in routed.reason


def test_router_filters_candidates_by_business_domain_when_domain_is_clear():
    router = KnowledgeBaseRouter(FakeMetadataRepository())

    routed = router.route(
        query="离职后社保怎么处理？",
        requested_knowledge_base_id="kb_001",
    )

    assert routed is not None
    assert routed.knowledge_base_id == "kb_law"
    assert "命中业务域:labor_law" in routed.reason


def test_router_falls_back_to_rule_route_when_embedding_margin_is_low():
    repository = FakeMetadataRepository()
    router = KnowledgeBaseRouter(
        repository,
        config=KnowledgeBaseRoutingConfig(
            category_keywords={
                "legal": ("劳动法", "劳动合同法"),
                "regulation": ("规章", "学生管理"),
            },
            knowledge_base_rules=(
                KnowledgeBaseRuleConfig(
                    name="labor_law",
                    match_any=("劳动相关法律",),
                    category="legal",
                    aliases=(),
                    priority_terms=(),
                    exclude_terms=(),
                    query_rules=(
                        KnowledgeBaseQueryRuleConfig(
                            name="trial_salary",
                            all_of=("试用期",),
                            any_of=("工资",),
                            none_of=(),
                            boost=15,
                        ),
                    ),
                ),
                KnowledgeBaseRuleConfig(
                    name="heming_regulation",
                    match_any=("和鸣教育管理制度",),
                    category="regulation",
                    aliases=(),
                    priority_terms=(),
                    exclude_terms=(),
                    query_rules=(),
                ),
            ),
            scores=RoutingScoreConfig(
                keyword_match=0,
                metadata_term_min=0,
                metadata_term_max=0,
                preferred_knowledge_base=0,
                category_match=0,
                keep_requested_margin=1,
                knowledge_base_alias_match=0,
                knowledge_base_priority_match=0,
                knowledge_base_exclude_penalty=0,
            ),
            embedding=RoutingEmbeddingConfig(
                enabled=True,
                min_similarity=0.1,
                min_margin=0.2,
                profile_document_limit=8,
            ),
        ),
        embedder=FakeRouteEmbedder(
            {
                "试用期可以不发工资吗？": [1.0, 0.0],
            }
        ),
    )
    router._encode_profile = lambda knowledge_base_id, profile_text: [1.0, 0.0] if knowledge_base_id == "kb_law" else [0.98, 0.02]

    routed = router.route(
        query="试用期可以不发工资吗？",
        requested_knowledge_base_id="kb_001",
    )

    assert routed is not None
    assert routed.knowledge_base_id == "kb_law"
    assert "命中布尔规则:trial_salary" in routed.reason
    assert "命中向量路由:" not in routed.reason


def test_query_with_no_answer_response_clears_final_context_and_citations(monkeypatch):
    pipeline = NoAnswerStubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    service.repository = FakeMetadataRepository()

    payload = ChatQueryRequest(
        knowledge_base_id="kb_001",
        query="校园里可以带宠物进宿舍吗？",
        debug=True,
        enable_auto_route=False,
    )
    result = asyncio.run(service.query(payload))

    assert "无法确定" in result["answer"]
    assert result["citations"] == []
    assert result["debug"]["final_context_chunks"] == []
    assert result["debug"]["reranked_chunks"] == []
    assert result["debug"]["reranked_count"] == 0
    assert result["debug"]["answer_guard_applied"] is False



def test_query_with_partial_unknown_clause_keeps_final_context_and_citations(monkeypatch):
    pipeline = PartialUnknownClauseStubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    service.repository = FakeMetadataRepository()

    payload = ChatQueryRequest(
        knowledge_base_id="kb_001",
        query="累计扣分一般怎么处理？",
        debug=True,
        enable_auto_route=False,
    )
    result = asyncio.run(service.query(payload))

    assert "无法确定" in result["answer"]
    assert result["citations"] != []
    assert result["debug"]["final_context_chunks"] != []
    assert result["debug"]["reranked_chunks"] != []
    assert result["debug"]["reranked_count"] == 1
    assert result["debug"]["answer_guard_applied"] is False


def test_query_with_cited_partial_unknown_clause_keeps_final_context_and_citations(monkeypatch):
    pipeline = CitedPartialUnknownClauseStubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    service.repository = FakeMetadataRepository()

    payload = ChatQueryRequest(
        knowledge_base_id="kb_001",
        query="早退会怎么处理？",
        debug=True,
        enable_auto_route=False,
    )
    result = asyncio.run(service.query(payload))

    assert "参考资料未明确" in result["answer"]
    assert result["citations"] != []
    assert result["debug"]["final_context_chunks"] != []
    assert result["debug"]["reranked_chunks"] != []
    assert result["debug"]["reranked_count"] == 1
    assert result["debug"]["answer_guard_applied"] is False

def test_query_with_unsupported_qualifier_uses_conservative_answer_guard(monkeypatch):
    pipeline = GuardedStubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    service.repository = FakeMetadataRepository()

    payload = ChatQueryRequest(
        knowledge_base_id="kb_001",
        query="实习期间宿舍违规会有什么后果",
        debug=True,
    )
    result = asyncio.run(service.query(payload))

    assert "现有资料中没有明确覆盖“实习期间”这一限定场景" in result["answer"]
    assert result["citations"] == []
    assert result["debug"]["answer_guard_applied"] is True
    assert result["debug"]["unsupported_qualifiers"] == ["实习期间"]


def test_cross_domain_compound_query_returns_split_prompt(monkeypatch):
    pipeline = StubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    service.repository = FakeMetadataRepository()

    payload = ChatQueryRequest(
        knowledge_base_id="kb_001",
        query="离职和退学分别怎么处理？",
        debug=True,
    )
    result = asyncio.run(service.query(payload))

    assert "这个问题同时涉及“劳动法”、“培训管理”多个业务域" in result["answer"]
    assert "你是想先问“劳动法”、“培训管理”中的哪一类问题？" in result["answer"]
    assert "离职怎么处理？" in result["answer"]
    assert "退学怎么处理？" in result["answer"]
    assert result["citations"] == []
    assert result["debug"]["cross_domain_guard_applied"] is True
    assert result["debug"]["detected_domains"] == ["labor_law", "training_management"]
    assert result["debug"]["cross_domain_guard_reason"] == "检测到跨业务域复合问题，转为拆问提示"
    assert pipeline.called is False



def test_query_with_overview_partial_answer_keeps_final_context_and_citations(monkeypatch):
    pipeline = OverviewStubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    service.repository = FakeMetadataRepository()

    payload = ChatQueryRequest(
        knowledge_base_id="kb_001",
        query="课堂违纪一般会怎么处理？",
        debug=True,
        enable_auto_route=False,
    )
    result = asyncio.run(service.query(payload))

    assert "需要区分具体情形" in result["answer"]
    assert result["citations"] != []
    assert result["debug"]["final_context_chunks"] != []
    assert result["debug"]["reranked_count"] == 1
    assert result["debug"]["answer_guard_applied"] is False


def test_query_with_range_summary_partial_unknown_keeps_final_context_and_citations(monkeypatch):
    pipeline = RangeSummaryPartialUnknownStubPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = rag_service.RAGService()
    service.repository = FakeMetadataRepository()

    payload = ChatQueryRequest(
        knowledge_base_id="kb_001",
        query="累计扣分达到一定程度会怎么样？",
        debug=True,
        enable_auto_route=False,
    )
    result = asyncio.run(service.query(payload))

    assert "未明确累计扣分 21—39 分" in result["answer"]
    assert result["citations"] != []
    assert result["debug"]["final_context_chunks"] != []
    assert result["debug"]["reranked_chunks"] != []
    assert result["debug"]["reranked_count"] == 3
    assert result["debug"]["answer_guard_applied"] is False

