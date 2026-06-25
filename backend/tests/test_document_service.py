from pathlib import Path

import pytest

from app.core.exceptions import DocumentNotFoundError
from app.services.knowledge_base_service import DocumentService


class StubRepository:
    def __init__(self, document: dict | None, knowledge_base: dict | None) -> None:
        self.document = document
        self.knowledge_base = knowledge_base
        self.deleted_document_id: str | None = None

    def get_document(self, document_id: str) -> dict | None:
        if self.document and self.document["id"] == document_id:
            return dict(self.document)
        return None

    def get_knowledge_base(self, knowledge_base_id: str) -> dict | None:
        if self.knowledge_base and self.knowledge_base["id"] == knowledge_base_id:
            return dict(self.knowledge_base)
        return None

    def delete_document(self, document_id: str) -> None:
        self.deleted_document_id = document_id


def test_delete_document_removes_source_artifacts_and_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    document_id = "doc_001"
    knowledge_base_id = "kb_001"
    source_path = tmp_path / knowledge_base_id / "sample.pdf"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"pdf")
    artifact_dir = source_path.parent / "_artifacts" / document_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "structured.json").write_text("{}", encoding="utf-8")

    repository = StubRepository(
        document={
            "id": document_id,
            "knowledge_base_id": knowledge_base_id,
            "file_path": str(source_path),
        },
        knowledge_base={
            "id": knowledge_base_id,
            "subject": "测试知识库",
        },
    )
    deleted_index_calls: list[tuple[str, str, str]] = []

    monkeypatch.setattr(
        "app.services.knowledge_base_service.delete_document_index",
        lambda knowledge_base_id, subject, document_id: deleted_index_calls.append(
            (knowledge_base_id, subject, document_id)
        ),
    )

    service = DocumentService()
    service.repository = repository

    result = service.delete_document(document_id)

    assert result == {"id": document_id, "deleted": True}
    assert repository.deleted_document_id == document_id
    assert deleted_index_calls == [(knowledge_base_id, "测试知识库", document_id)]
    assert not source_path.exists()
    assert not artifact_dir.exists()


def test_delete_document_raises_when_document_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    service = DocumentService()
    service.repository = StubRepository(document=None, knowledge_base=None)

    monkeypatch.setattr(
        "app.services.knowledge_base_service.delete_document_index",
        lambda knowledge_base_id, subject, document_id: None,
    )

    with pytest.raises(DocumentNotFoundError):
        service.delete_document("doc_missing")
