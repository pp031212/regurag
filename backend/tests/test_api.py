from datetime import UTC, datetime

from app.api.deps import (
    get_conversation_service,
    get_document_service,
    get_ingest_service,
    get_knowledge_base_service,
    get_rag_service,
    get_task_service,
)
from app.core.exceptions import (
    DocumentFileTooLargeError,
    DuplicateDocumentError,
    KnowledgeBaseNotFoundError,
    KnowledgeBaseNotReadyError,
    TaskNotFoundError,
    UnsupportedDocumentTypeError,
)


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class FakeKnowledgeBaseService:
    def get_domain_options(self):
        return {
            "items": [
                {
                    "value": "general",
                    "label": "通用知识",
                    "description": "默认域",
                },
                {
                    "value": "training_management",
                    "label": "培训管理",
                    "description": "培训管理域",
                },
            ],
            "default_domain": "general",
        }

    def list_knowledge_bases(self):
        return [
            {
                "id": "kb_test1234",
                "name": "测试知识库",
                "description": "测试描述",
                "subject": "测试主题",
                "domain": "general",
                "status": "ready",
                "created_at": _now(),
                "updated_at": _now(),
            }
        ]

    def get_knowledge_base(self, knowledge_base_id):
        return {
            "id": knowledge_base_id,
            "name": "测试知识库",
            "description": "测试描述",
            "subject": "测试主题",
            "domain": "general",
            "status": "ready",
            "created_at": _now(),
            "updated_at": _now(),
        }

    def create_knowledge_base(self, name, description, subject, domain):
        return {
            "id": "kb_test1234",
            "name": name,
            "description": description,
            "subject": subject,
            "domain": domain,
            "status": "empty",
            "created_at": _now(),
            "updated_at": _now(),
        }

    def delete_knowledge_base(self, knowledge_base_id):
        return {"id": knowledge_base_id, "deleted": True}


class FakeRAGService:
    def __init__(self):
        self.last_payload = None

    async def query(self, payload):
        self.last_payload = payload
        debug_payload = {
            "rewritten_query": payload.query,
            "retrieved_count": 3,
            "reranked_count": 1,
            "enable_rewrite": payload.enable_rewrite,
            "enable_rerank": payload.enable_rerank,
            "rewrite_ms": 10,
            "retrieve_ms": 20,
            "rerank_ms": 30,
            "context_build_ms": 15,
            "generate_ms": 40,
            "latency_ms": 100,
            "llm_finish_reason": "stop",
            "llm_model": "gpt-test",
            "llm_prompt_tokens": 100,
            "llm_completion_tokens": 50,
            "llm_total_tokens": 150,
            "stage_timings_ms": {
                "history_rewrite_ms": 0,
                "rewrite_ms": 10,
                "retrieve_ms": 20,
                "rerank_ms": 30,
                "context_build_ms": 15,
                "generate_ms": 40,
                "llm_first_token_ms": None,
                "llm_after_first_token_ms": None,
                "service_overhead_ms": 5,
            },
        }
        if payload.debug_shadow_compare:
            debug_payload["shadow_compare"] = {
                "status": "match",
                "compared_stage": "pipeline_core",
                "answer_match": True,
                "citation_count_match": True,
                "citation_ids_match": True,
                "final_context_ids_match": True,
                "rewritten_query_match": True,
                "mismatch_fields": [],
                "legacy_answer_hash": "legacy",
                "shadow_answer_hash": "shadow",
                "legacy_citation_count": 1,
                "shadow_citation_count": 1,
                "legacy_final_context_count": 1,
                "shadow_final_context_count": 1,
                "shadow_latency_ms": 12,
                "error": None,
            }
        return {
            "answer": "测试回答",
            "answer_source": "legacy_pipeline",
            "conversation_id": "conv_test1234",
            "knowledge_base_id": payload.knowledge_base_id or "kb_001",
            "knowledge_base_name": "测试知识库",
            "auto_routed": False,
            "citations": [
                {
                    "document_id": payload.knowledge_base_id,
                    "chunk_id": "rule_doc_1-1",
                    "content": "测试引用",
                    "score": 0.95,
                    "source_type": "table",
                    "page_number": 1,
                    "block_index": 0,
                }
            ],
            "debug": debug_payload,
        }

    def stream_query(self, payload):
        self.last_payload = payload
        yield {
            "event": "start",
            "data": {
                "conversation_id": "conv_test1234",
                "knowledge_base_id": payload.knowledge_base_id or "kb_001",
                "knowledge_base_name": "测试知识库",
                "auto_routed": False,
            },
        }
        yield {"event": "token", "data": {"delta": "测试"}}
        yield {"event": "token", "data": {"delta": "回答"}}
        yield {
            "event": "end",
            "data": {
                "answer": "测试回答",
                "answer_source": "legacy_pipeline",
                "conversation_id": "conv_test1234",
                "knowledge_base_id": payload.knowledge_base_id or "kb_001",
                "knowledge_base_name": "测试知识库",
                "auto_routed": False,
                "citations": [
                    {
                        "document_id": payload.knowledge_base_id or "kb_001",
                        "chunk_id": "rule_doc_1-1",
                        "content": "测试引用",
                        "score": 0.95,
                        "source_type": "table",
                        "page_number": 1,
                        "block_index": 0,
                    }
                ],
                "debug": {
                    "rewritten_query": payload.query,
                    "retrieved_count": 3,
                    "reranked_count": 1,
                    "enable_rewrite": payload.enable_rewrite,
                    "enable_rerank": payload.enable_rerank,
                    "rewrite_ms": 10,
                    "retrieve_ms": 20,
                    "rerank_ms": 30,
                    "context_build_ms": 15,
                    "generate_ms": 40,
                    "latency_ms": 100,
                    "llm_finish_reason": "stop",
                    "llm_model": "gpt-test",
                    "llm_prompt_tokens": 100,
                    "llm_completion_tokens": 50,
                    "llm_total_tokens": 150,
                    "llm_first_token_ms": 12,
                    "stage_timings_ms": {
                        "history_rewrite_ms": 0,
                        "rewrite_ms": 10,
                        "retrieve_ms": 20,
                        "rerank_ms": 30,
                        "context_build_ms": 15,
                        "generate_ms": 40,
                        "llm_first_token_ms": 12,
                        "llm_after_first_token_ms": 28,
                        "service_overhead_ms": 5,
                    },
                },
            },
        }


class MissingKnowledgeBaseRAGService:
    async def query(self, payload):
        raise KnowledgeBaseNotFoundError()


class NotReadyKnowledgeBaseRAGService:
    async def query(self, payload):
        raise KnowledgeBaseNotReadyError()


class FakeDocumentService:
    def __init__(self):
        self.deleted_document_id = None
        self.requested_document_id = None
        self.raise_duplicate = False
        self.raise_unsupported = False
        self.raise_too_large = False

    async def upload_document(self, knowledge_base_id, upload_file):
        if self.raise_unsupported:
            raise UnsupportedDocumentTypeError(
                message="unsupported document type: .pptx",
                details={"allowed_extensions": [".docx", ".md", ".pdf", ".txt", ".xlsx"]},
            )
        if self.raise_too_large:
            raise DocumentFileTooLargeError(
                message="文件超过当前类型限制，最大允许 10MB。",
                details={"max_size_bytes": 10485760, "max_size": "10MB", "actual_size_bytes": 10485761},
            )
        if self.raise_duplicate:
            raise DuplicateDocumentError(
                message="该文件已存在于当前知识库，无需重复上传。",
                details={"document_id": "doc_existing", "filename": upload_file.filename, "file_size": 5},
            )
        return {
            "id": "doc_test1234",
            "knowledge_base_id": knowledge_base_id,
            "filename": upload_file.filename,
            "content_type": upload_file.content_type or "text/plain",
            "file_size": 5,
            "status": "uploaded",
            "created_at": _now(),
            "updated_at": _now(),
        }

    def list_documents(self, knowledge_base_id):
        return [
            {
                "id": "doc_test1234",
                "knowledge_base_id": knowledge_base_id,
                "filename": "sample.txt",
                "content_type": "text/plain",
                "file_size": 5,
                "status": "ready",
                "created_at": _now(),
                "updated_at": _now(),
            }
        ]

    def delete_document(self, document_id):
        self.deleted_document_id = document_id
        return {"id": document_id, "deleted": True}

    def get_document(self, document_id):
        self.requested_document_id = document_id
        return {
            "id": document_id,
            "knowledge_base_id": "kb_001",
            "filename": "sample.txt",
            "content_type": "text/plain",
            "file_size": 5,
            "file_path": "D:/workspace/regurag/backend/data/uploads/kb_001/sample.txt",
            "status": "ready",
            "created_at": _now(),
            "updated_at": _now(),
        }


class FakeConversationService:
    def __init__(self):
        self.deleted_conversation_id = None

    def list_conversations(self, knowledge_base_id=None):
        return [
            {
                "id": "conv_test1234",
                "default_knowledge_base_id": knowledge_base_id,
                "title": "测试会话",
                "created_at": _now(),
                "updated_at": _now(),
            }
        ]

    def create_conversation(self, default_knowledge_base_id=None, title=None, fallback_query=None):
        return {
            "id": "conv_test1234",
            "default_knowledge_base_id": default_knowledge_base_id,
            "title": title or fallback_query or "测试会话",
            "created_at": _now(),
            "updated_at": _now(),
        }

    def list_messages(self, conversation_id):
        return [
            {
                "id": "msg_test1234",
                "conversation_id": conversation_id,
                "knowledge_base_id": "kb_001",
                "sequence": 1,
                "role": "assistant",
                "content": "历史回答",
                "citations": [],
                "debug": None,
                "created_at": _now(),
            }
        ]

    def delete_conversation(self, conversation_id):
        self.deleted_conversation_id = conversation_id
        return {"id": conversation_id, "deleted": True}


class FakeIngestService:
    def __init__(self):
        self.calls = []
        self.last_created_task_type = None

    def create_ingest_task(self, knowledge_base_id, document_ids, task_type="ingest", message="ingest task created"):
        self.last_created_task_type = task_type
        return {
            "id": "task_test1234" if message == "ingest task created" else "task_rebuild1234",
            "knowledge_base_id": knowledge_base_id,
            "task_type": task_type,
            "document_ids": document_ids or ["doc_001"],
            "status": "pending",
            "message": message,
            "attempt_count": 0,
            "last_error": None,
            "started_at": None,
            "finished_at": None,
            "locked_at": None,
            "locked_by": None,
            "created_at": _now(),
            "updated_at": _now(),
        }

    def run_ingest_task(self, task_id, rebuild=False):
        self.calls.append((task_id, rebuild))


class MissingKnowledgeBaseIngestService:
    def create_ingest_task(self, knowledge_base_id, document_ids, message="ingest task created"):
        raise KnowledgeBaseNotFoundError()


class FakeTaskService:
    def get_task(self, task_id):
        return {
            "id": task_id,
            "knowledge_base_id": "kb_001",
            "task_type": "ingest",
            "document_ids": ["doc_001"],
            "status": "running",
            "message": "processing doc_001",
            "attempt_count": 2,
            "last_error": "previous transient error",
            "started_at": _now(),
            "finished_at": None,
            "locked_at": _now(),
            "locked_by": "worker-test",
            "created_at": _now(),
            "updated_at": _now(),
        }

    def get_task_events(self, task_id, limit=100):
        return {
            "items": [
                {
                    "id": "evt_test1234",
                    "task_id": task_id,
                    "event_type": "claimed",
                    "message": "claimed by worker-test",
                    "payload": {"worker_id": "worker-test", "attempt_count": 2},
                    "created_at": _now(),
                }
            ],
            "total": 1,
        }

    def list_tasks(self, knowledge_base_id=None, status=None, limit=50):
        return {
            "items": [
                {
                    "id": "task_test1234",
                    "knowledge_base_id": knowledge_base_id or "kb_001",
                    "task_type": "ingest",
                    "document_ids": ["doc_001"],
                    "status": status or "running",
                    "message": "processing doc_001",
                    "attempt_count": 2,
                    "last_error": "previous transient error",
                    "started_at": _now(),
                    "finished_at": None,
                    "locked_at": _now(),
                    "locked_by": "worker-test",
                    "created_at": _now(),
                    "updated_at": _now(),
                }
            ],
            "total": 1,
        }

    def get_task_stats(self, knowledge_base_id=None):
        return {
            "total": 4,
            "pending": 1,
            "running": 1,
            "completed": 1,
            "failed": 1,
            "retrying": 2,
            "stale_running": 1,
        }

    def get_task_overview(self, knowledge_base_id=None):
        return {
            "total": 6,
            "pending": 2,
            "running": 1,
            "completed": 1,
            "failed": 2,
            "retrying": 3,
            "stale_running": 1,
            "active_workers": 0,
            "oldest_pending_age_seconds": 120,
            "long_running": 1,
            "recent_failed": 2,
            "recent_retried": 4,
            "knowledge_bases_with_recent_failures": [
                {"knowledge_base_id": knowledge_base_id or "kb_001", "task_count": 2}
            ],
        }

    def get_task_alerts(self, knowledge_base_id=None):
        return {
            "items": [
                {
                    "code": "PENDING_WITHOUT_ACTIVE_WORKERS",
                    "severity": "critical",
                    "message": "pending tasks exist but no active workers were detected",
                    "count": 2,
                    "details": {"pending": 2, "active_workers": 0},
                },
                {
                    "code": "STALE_RUNNING_TASKS",
                    "severity": "warning",
                    "message": "stale running tasks detected",
                    "count": 1,
                    "details": {"stale_running": 1, "lease_seconds": 1800},
                },
            ],
            "total": 2,
        }

    def get_task_trends(self, limit=20):
        return {
            "items": [
                {
                    "knowledge_base_id": "kb_001",
                    "knowledge_base_name": "测试知识库 A",
                    "pending": 2,
                    "running": 1,
                    "recent_failed": 3,
                    "previous_failed": 1,
                    "failed_delta": 2,
                    "recent_retried": 4,
                    "previous_retried": 2,
                    "retried_delta": 2,
                    "recent_completed": 5,
                    "previous_completed": 3,
                    "completed_delta": 2,
                    "updated_at": _now(),
                },
                {
                    "knowledge_base_id": "kb_002",
                    "knowledge_base_name": "测试知识库 B",
                    "pending": 0,
                    "running": 0,
                    "recent_failed": 0,
                    "previous_failed": 2,
                    "failed_delta": -2,
                    "recent_retried": 0,
                    "previous_retried": 1,
                    "retried_delta": -1,
                    "recent_completed": 2,
                    "previous_completed": 1,
                    "completed_delta": 1,
                    "updated_at": _now(),
                },
            ][:limit],
            "total": min(limit, 2),
        }


class MissingTaskService:
    def get_task(self, task_id):
        raise TaskNotFoundError()

    def get_task_events(self, task_id, limit=100):
        raise TaskNotFoundError()


def test_health_check_returns_ok(client):
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "app_name" in body
    assert "environment" in body



def test_create_knowledge_base_returns_created_record(client):
    client.app.dependency_overrides[get_knowledge_base_service] = lambda: FakeKnowledgeBaseService()

    response = client.post(
        "/api/v1/knowledge-bases",
        json={
            "name": "测试知识库",
            "description": "测试描述",
            "subject": "测试主题",
            "domain": "training_management",
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["id"] == "kb_test1234"
    assert body["name"] == "测试知识库"
    assert body["domain"] == "training_management"
    assert body["status"] == "empty"


def test_list_knowledge_base_domains_returns_configured_options(client):
    client.app.dependency_overrides[get_knowledge_base_service] = lambda: FakeKnowledgeBaseService()

    response = client.get("/api/v1/knowledge-bases/domains")

    assert response.status_code == 200
    body = response.json()
    assert body["default_domain"] == "general"
    assert body["items"] == [
        {
            "value": "general",
            "label": "通用知识",
            "description": "默认域",
        },
        {
            "value": "training_management",
            "label": "培训管理",
            "description": "培训管理域",
        },
    ]



def test_get_knowledge_base_returns_record(client):
    client.app.dependency_overrides[get_knowledge_base_service] = lambda: FakeKnowledgeBaseService()

    response = client.get("/api/v1/knowledge-bases/kb_test1234")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "kb_test1234"
    assert body["status"] == "ready"



def test_delete_knowledge_base_returns_deleted_flag(client):
    client.app.dependency_overrides[get_knowledge_base_service] = lambda: FakeKnowledgeBaseService()

    response = client.delete("/api/v1/knowledge-bases/kb_test1234")

    assert response.status_code == 200
    body = response.json()
    assert body == {"id": "kb_test1234", "deleted": True}



def test_rebuild_returns_pending_task(client):
    service = FakeIngestService()
    client.app.dependency_overrides[get_ingest_service] = lambda: service

    response = client.post(
        "/api/v1/knowledge-bases/kb_001/rebuild",
        json={"document_ids": []},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["id"] == "task_rebuild1234"
    assert body["message"] == "rebuild task created"
    assert service.last_created_task_type == "rebuild"



def test_chat_query_returns_answer_and_debug(client):
    service = FakeRAGService()
    client.app.dependency_overrides[get_rag_service] = lambda: service

    response = client.post(
        "/api/v1/chat/query",
        json={
            "knowledge_base_id": "kb_001",
            "query": "测试问题",
            "top_k_retrieve": 5,
            "top_k_rerank": 2,
            "enable_rewrite": False,
            "enable_rerank": False,
            "debug": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "测试回答"
    assert body["answer_source"] == "legacy_pipeline"
    assert body["conversation_id"] == "conv_test1234"
    assert body["citations"][0]["document_id"] == "kb_001"
    assert body["debug"]["llm_finish_reason"] == "stop"
    assert body["debug"]["enable_rewrite"] is False
    assert body["debug"]["enable_rerank"] is False
    assert body["debug"]["context_build_ms"] == 15
    assert body["debug"]["stage_timings_ms"]["context_build_ms"] == 15
    assert service.last_payload.enable_rewrite is False
    assert service.last_payload.enable_rerank is False


def test_chat_query_returns_shadow_compare_when_requested(client):
    service = FakeRAGService()
    client.app.dependency_overrides[get_rag_service] = lambda: service

    response = client.post(
        "/api/v1/chat/query",
        json={
            "knowledge_base_id": "kb_001",
            "query": "测试问题",
            "debug": True,
            "debug_shadow_compare": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["debug"]["shadow_compare"]["status"] == "match"
    assert body["debug"]["shadow_compare"]["shadow_latency_ms"] == 12
    assert service.last_payload.debug_shadow_compare is True


def test_chat_query_returns_answer_source_field(client):
    service = FakeRAGService()
    client.app.dependency_overrides[get_rag_service] = lambda: service

    response = client.post(
        "/api/v1/chat/query",
        json={
            "knowledge_base_id": "kb_001",
            "query": "测试问题",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer_source"] == "legacy_pipeline"


def test_chat_stream_returns_sse_events(client):
    service = FakeRAGService()
    client.app.dependency_overrides[get_rag_service] = lambda: service

    with client.stream(
        "POST",
        "/api/v1/chat/stream",
        json={
            "knowledge_base_id": "kb_001",
            "query": "测试问题",
            "debug": True,
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: start" in body
    assert "event: token" in body
    assert "event: end" in body
    assert '"conversation_id": "conv_test1234"' in body
    assert '"answer": "测试回答"' in body
    assert '"llm_first_token_ms": 12' in body
    assert '"context_build_ms": 15' in body


def test_chat_query_returns_faq_short_circuit_debug_fields(client):
    class FakeFAQRAGService(FakeRAGService):
        async def query(self, payload):
            result = await super().query(payload)
            result["answer"] = "如果你想了解“请假制度”，建议继续问更具体的规则点。"
            result["answer_source"] = "faq_short_circuit"
            result["debug"]["llm_finish_reason"] = "faq_short_circuit"
            result["debug"]["faq_short_circuit_applied"] = True
            result["debug"]["faq_rule_name"] = "leave_policy_overview"
            result["debug"]["faq_topic"] = "请假制度"
            return result

    client.app.dependency_overrides[get_rag_service] = lambda: FakeFAQRAGService()

    response = client.post(
        "/api/v1/chat/query",
        json={
            "knowledge_base_id": "kb_001",
            "query": "请假制度是什么",
            "debug": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer_source"] == "faq_short_circuit"
    assert body["debug"]["llm_finish_reason"] == "faq_short_circuit"
    assert body["debug"]["faq_short_circuit_applied"] is True
    assert body["debug"]["faq_rule_name"] == "leave_policy_overview"
    assert body["debug"]["faq_topic"] == "请假制度"


def test_chat_query_returns_structured_error_when_kb_not_found(client):
    client.app.dependency_overrides[get_rag_service] = lambda: MissingKnowledgeBaseRAGService()

    response = client.post(
        "/api/v1/chat/query",
        json={
            "knowledge_base_id": "kb_missing",
            "query": "测试问题",
            "top_k_retrieve": 5,
            "top_k_rerank": 2,
            "debug": True,
        },
    )

    assert response.status_code == 404
    body = response.json()
    assert body == {
        "code": "KNOWLEDGE_BASE_NOT_FOUND",
        "message": "knowledge base not found",
        "details": {},
    }



def test_chat_query_returns_structured_error_when_kb_not_ready(client):
    client.app.dependency_overrides[get_rag_service] = lambda: NotReadyKnowledgeBaseRAGService()

    response = client.post(
        "/api/v1/chat/query",
        json={
            "knowledge_base_id": "kb_pending",
            "query": "测试问题",
            "top_k_retrieve": 5,
            "top_k_rerank": 2,
            "debug": True,
        },
    )

    assert response.status_code == 409
    body = response.json()
    assert body == {
        "code": "KNOWLEDGE_BASE_NOT_READY",
        "message": "knowledge base is not ready",
        "details": {},
    }



def test_upload_document_returns_created_record(client):
    client.app.dependency_overrides[get_document_service] = lambda: FakeDocumentService()

    response = client.post(
        "/api/v1/documents/upload",
        data={"knowledge_base_id": "kb_001"},
        files={"file": ("sample.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["id"] == "doc_test1234"
    assert body["filename"] == "sample.txt"
    assert body["status"] == "uploaded"



def test_upload_docx_document_returns_created_record(client):
    client.app.dependency_overrides[get_document_service] = lambda: FakeDocumentService()

    response = client.post(
        "/api/v1/documents/upload",
        data={"knowledge_base_id": "kb_001"},
        files={
            "file": (
                "sample.docx",
                b"fake-docx-content",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "sample.docx"
    assert body["content_type"] == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    assert body["status"] == "uploaded"


def test_upload_markdown_document_returns_created_record(client):
    client.app.dependency_overrides[get_document_service] = lambda: FakeDocumentService()

    response = client.post(
        "/api/v1/documents/upload",
        data={"knowledge_base_id": "kb_001"},
        files={
            "file": (
                "sample.md",
                b"# markdown",
                "text/markdown",
            )
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "sample.md"
    assert body["content_type"] == "text/markdown"
    assert body["status"] == "uploaded"


def test_upload_xlsx_document_returns_created_record(client):
    client.app.dependency_overrides[get_document_service] = lambda: FakeDocumentService()

    response = client.post(
        "/api/v1/documents/upload",
        data={"knowledge_base_id": "kb_001"},
        files={
            "file": (
                "sample.xlsx",
                b"fake-xlsx-content",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "sample.xlsx"
    assert body["content_type"] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    assert body["status"] == "uploaded"


def test_upload_image_document_returns_created_record(client):
    client.app.dependency_overrides[get_document_service] = lambda: FakeDocumentService()

    response = client.post(
        "/api/v1/documents/upload",
        data={"knowledge_base_id": "kb_001"},
        files={
            "file": (
                "sample.png",
                b"fake-png-content",
                "image/png",
            )
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "sample.png"
    assert body["content_type"] == "image/png"
    assert body["status"] == "uploaded"


def test_upload_duplicate_document_returns_structured_error(client):
    service = FakeDocumentService()
    service.raise_duplicate = True
    client.app.dependency_overrides[get_document_service] = lambda: service

    response = client.post(
        "/api/v1/documents/upload",
        data={"knowledge_base_id": "kb_001"},
        files={"file": ("sample.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 409
    body = response.json()
    assert body == {
        "code": "DUPLICATE_DOCUMENT",
        "message": "该文件已存在于当前知识库，无需重复上传。",
        "details": {
            "document_id": "doc_existing",
            "filename": "sample.txt",
            "file_size": 5,
        },
    }


def test_upload_unsupported_document_returns_structured_error(client):
    service = FakeDocumentService()
    service.raise_unsupported = True
    client.app.dependency_overrides[get_document_service] = lambda: service

    response = client.post(
        "/api/v1/documents/upload",
        data={"knowledge_base_id": "kb_001"},
        files={
            "file": (
                "sample.pptx",
                b"slides",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            )
        },
    )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "UNSUPPORTED_DOCUMENT_TYPE"
    assert body["details"]["allowed_extensions"] == [".docx", ".md", ".pdf", ".txt", ".xlsx"]


def test_upload_too_large_document_returns_structured_error(client):
    service = FakeDocumentService()
    service.raise_too_large = True
    client.app.dependency_overrides[get_document_service] = lambda: service

    response = client.post(
        "/api/v1/documents/upload",
        data={"knowledge_base_id": "kb_001"},
        files={"file": ("sample.txt", b"too-large", "text/plain")},
    )

    assert response.status_code == 413
    body = response.json()
    assert body == {
        "code": "DOCUMENT_FILE_TOO_LARGE",
        "message": "文件超过当前类型限制，最大允许 10MB。",
        "details": {
            "max_size_bytes": 10485760,
            "max_size": "10MB",
            "actual_size_bytes": 10485761,
        },
    }


def test_delete_document_returns_deleted_flag(client):
    service = FakeDocumentService()
    client.app.dependency_overrides[get_document_service] = lambda: service

    response = client.delete("/api/v1/documents/doc_test1234")

    assert response.status_code == 200
    body = response.json()
    assert body == {"id": "doc_test1234", "deleted": True}
    assert service.deleted_document_id == "doc_test1234"


def test_rebuild_document_returns_pending_task(client):
    document_service = FakeDocumentService()
    ingest_service = FakeIngestService()
    client.app.dependency_overrides[get_document_service] = lambda: document_service
    client.app.dependency_overrides[get_ingest_service] = lambda: ingest_service

    response = client.post("/api/v1/documents/doc_test1234/rebuild")

    assert response.status_code == 202
    body = response.json()
    assert body["id"] == "task_rebuild1234"
    assert body["message"] == "document rebuild task created"
    assert body["document_ids"] == ["doc_test1234"]
    assert document_service.requested_document_id == "doc_test1234"
    assert ingest_service.last_created_task_type == "rebuild"


def test_ingest_returns_pending_task(client):
    service = FakeIngestService()
    client.app.dependency_overrides[get_ingest_service] = lambda: service

    response = client.post(
        "/api/v1/knowledge-bases/kb_001/ingest",
        json={"document_ids": []},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["id"] == "task_test1234"
    assert body["status"] == "pending"
    assert service.last_created_task_type == "ingest"



def test_ingest_returns_structured_error_when_kb_not_found(client):
    client.app.dependency_overrides[get_ingest_service] = lambda: MissingKnowledgeBaseIngestService()

    response = client.post(
        "/api/v1/knowledge-bases/kb_missing/ingest",
        json={"document_ids": []},
    )

    assert response.status_code == 404
    body = response.json()
    assert body == {
        "code": "KNOWLEDGE_BASE_NOT_FOUND",
        "message": "knowledge base not found",
        "details": {},
    }



def test_get_task_returns_observability_fields(client):
    client.app.dependency_overrides[get_task_service] = lambda: FakeTaskService()

    response = client.get("/api/v1/tasks/task_test1234")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "task_test1234"
    assert body["task_type"] == "ingest"
    assert body["attempt_count"] == 2
    assert body["last_error"] == "previous transient error"
    assert body["locked_by"] == "worker-test"


def test_list_tasks_returns_filtered_items(client):
    client.app.dependency_overrides[get_task_service] = lambda: FakeTaskService()

    response = client.get(
        "/api/v1/tasks",
        params={"knowledge_base_id": "kb_001", "status": "running", "limit": 20},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["items"][0]["knowledge_base_id"] == "kb_001"
    assert body["items"][0]["status"] == "running"
    assert body["items"][0]["attempt_count"] == 2


def test_get_task_stats_returns_status_counts(client):
    client.app.dependency_overrides[get_task_service] = lambda: FakeTaskService()

    response = client.get("/api/v1/tasks/stats", params={"knowledge_base_id": "kb_001"})

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "total": 4,
        "pending": 1,
        "running": 1,
        "completed": 1,
        "failed": 1,
        "retrying": 2,
        "stale_running": 1,
    }


def test_get_task_overview_returns_monitoring_summary(client):
    client.app.dependency_overrides[get_task_service] = lambda: FakeTaskService()

    response = client.get("/api/v1/tasks/overview", params={"knowledge_base_id": "kb_001"})

    assert response.status_code == 200
    body = response.json()
    assert body["pending"] == 2
    assert body["active_workers"] == 0
    assert body["oldest_pending_age_seconds"] == 120
    assert body["long_running"] == 1
    assert body["knowledge_bases_with_recent_failures"][0]["knowledge_base_id"] == "kb_001"


def test_get_task_alerts_returns_alert_items(client):
    client.app.dependency_overrides[get_task_service] = lambda: FakeTaskService()

    response = client.get("/api/v1/tasks/alerts", params={"knowledge_base_id": "kb_001"})

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert body["items"][0]["code"] == "PENDING_WITHOUT_ACTIVE_WORKERS"
    assert body["items"][0]["severity"] == "critical"
    assert body["items"][1]["code"] == "STALE_RUNNING_TASKS"


def test_get_task_trends_returns_knowledge_base_rows(client):
    client.app.dependency_overrides[get_task_service] = lambda: FakeTaskService()

    response = client.get("/api/v1/tasks/trends", params={"limit": 10})

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert body["items"][0]["knowledge_base_id"] == "kb_001"
    assert body["items"][0]["failed_delta"] == 2
    assert body["items"][1]["retried_delta"] == -1


def test_list_task_events_returns_event_items(client):
    client.app.dependency_overrides[get_task_service] = lambda: FakeTaskService()

    response = client.get("/api/v1/tasks/task_test1234/events", params={"limit": 20})

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["items"][0]["task_id"] == "task_test1234"
    assert body["items"][0]["event_type"] == "claimed"
    assert body["items"][0]["payload"]["worker_id"] == "worker-test"


def test_get_task_returns_structured_error_when_missing(client):
    client.app.dependency_overrides[get_task_service] = lambda: MissingTaskService()

    response = client.get("/api/v1/tasks/task_missing")

    assert response.status_code == 404
    body = response.json()
    assert body == {
        "code": "TASK_NOT_FOUND",
        "message": "task not found",
        "details": {},
    }


def test_list_conversations_returns_items(client):
    client.app.dependency_overrides[get_conversation_service] = lambda: FakeConversationService()

    response = client.get("/api/v1/conversations", params={"knowledge_base_id": "kb_001"})

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["items"][0]["id"] == "conv_test1234"


def test_list_conversation_messages_returns_items(client):
    client.app.dependency_overrides[get_conversation_service] = lambda: FakeConversationService()

    response = client.get("/api/v1/conversations/conv_test1234/messages")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["items"][0]["id"] == "msg_test1234"
    assert body["items"][0]["content"] == "历史回答"


def test_delete_conversation_returns_deleted_flag(client):
    service = FakeConversationService()
    client.app.dependency_overrides[get_conversation_service] = lambda: service

    response = client.delete("/api/v1/conversations/conv_test1234")

    assert response.status_code == 200
    body = response.json()
    assert body == {"id": "conv_test1234", "deleted": True}
    assert service.deleted_conversation_id == "conv_test1234"
