from app.core.exceptions import InvalidKnowledgeBaseDomainError
from app.services import knowledge_base_service


class FakeMetadataRepository:
    def __init__(self) -> None:
        self.created_payloads: list[dict[str, object]] = []

    def create_knowledge_base(self, *, name: str, description: str, subject: str, domain: str) -> dict[str, object]:
        payload = {
            "id": "kb_test1234",
            "name": name,
            "description": description,
            "subject": subject,
            "domain": domain,
            "status": "empty",
            "created_at": "2026-04-16T00:00:00Z",
            "updated_at": "2026-04-16T00:00:00Z",
        }
        self.created_payloads.append(payload)
        return payload


def test_create_knowledge_base_uses_configured_default_domain(monkeypatch) -> None:
    repository = FakeMetadataRepository()
    monkeypatch.setattr(knowledge_base_service, "get_metadata_repository", lambda: repository)

    service = knowledge_base_service.KnowledgeBaseService()
    record = service.create_knowledge_base(
        name="测试知识库",
        description="",
        subject="测试主题",
        domain=None,
    )

    assert record["domain"] == "general"
    assert repository.created_payloads[0]["domain"] == "general"


def test_create_knowledge_base_rejects_unknown_domain(monkeypatch) -> None:
    repository = FakeMetadataRepository()
    monkeypatch.setattr(knowledge_base_service, "get_metadata_repository", lambda: repository)

    service = knowledge_base_service.KnowledgeBaseService()

    try:
        service.create_knowledge_base(
            name="测试知识库",
            description="",
            subject="测试主题",
            domain="finance_policy",
        )
    except InvalidKnowledgeBaseDomainError as exc:
        assert exc.code == "INVALID_KNOWLEDGE_BASE_DOMAIN"
        assert exc.details["domain"] == "finance_policy"
        assert "general" in exc.details["allowed_domains"]
        return

    raise AssertionError("expected InvalidKnowledgeBaseDomainError to be raised")
