from io import BytesIO

import pytest
from fastapi import UploadFile

from app.core.exceptions import DocumentFileTooLargeError, DuplicateDocumentError, UnsupportedDocumentTypeError
from app.services.knowledge_base_service import DocumentService


class StubUploadRepository:
    def __init__(self, duplicate_document: dict | None = None) -> None:
        self.duplicate_document = duplicate_document
        self.created_payload: dict | None = None

    def get_knowledge_base(self, knowledge_base_id: str) -> dict | None:
        return {"id": knowledge_base_id, "subject": "测试知识库"}

    def find_duplicate_document(
        self,
        knowledge_base_id: str,
        filename: str,
        file_size: int,
        content_hash: str,
    ) -> dict | None:
        return self.duplicate_document

    def create_document(self, **payload: object) -> dict:
        self.created_payload = dict(payload)
        return {
            "id": "doc_001",
            "knowledge_base_id": payload["knowledge_base_id"],
            "filename": payload["filename"],
            "content_type": payload["content_type"],
            "file_size": payload["file_size"],
            "content_hash": payload["content_hash"],
            "file_path": payload["file_path"],
            "status": "uploaded",
            "created_at": "2026-04-03T00:00:00Z",
            "updated_at": "2026-04-03T00:00:00Z",
        }


@pytest.mark.anyio
async def test_upload_document_rejects_duplicate_file(tmp_path) -> None:
    service = DocumentService()
    service.uploads_dir = tmp_path
    service.repository = StubUploadRepository(
        duplicate_document={
            "id": "doc_existing",
            "knowledge_base_id": "kb_001",
            "filename": "sample.txt",
            "file_size": 5,
            "content_hash": "hash",
        }
    )
    upload_file = UploadFile(filename="sample.txt", file=BytesIO(b"hello"), headers={"content-type": "text/plain"})

    with pytest.raises(DuplicateDocumentError):
        await service.upload_document("kb_001", upload_file)


@pytest.mark.anyio
async def test_upload_document_rejects_unsupported_extension(tmp_path) -> None:
    service = DocumentService()
    service.uploads_dir = tmp_path
    service.repository = StubUploadRepository()
    upload_file = UploadFile(
        filename="slides.pptx",
        file=BytesIO(b"fake-pptx-content"),
        headers={"content-type": "application/vnd.openxmlformats-officedocument.presentationml.presentation"},
    )

    with pytest.raises(UnsupportedDocumentTypeError) as exc_info:
        await service.upload_document("kb_001", upload_file)

    assert exc_info.value.details["allowed_extensions"] == sorted(service.ALLOWED_EXTENSIONS)


@pytest.mark.anyio
async def test_upload_document_rejects_file_over_type_limit(tmp_path) -> None:
    service = DocumentService()
    service.uploads_dir = tmp_path
    service.repository = StubUploadRepository()
    service.MAX_FILE_SIZE_BYTES_BY_EXTENSION = {
        **service.MAX_FILE_SIZE_BYTES_BY_EXTENSION,
        ".txt": 4,
    }
    upload_file = UploadFile(filename="sample.txt", file=BytesIO(b"hello"), headers={"content-type": "text/plain"})

    with pytest.raises(DocumentFileTooLargeError) as exc_info:
        await service.upload_document("kb_001", upload_file)

    assert exc_info.value.details["max_size_bytes"] == 4
    assert exc_info.value.details["actual_size_bytes"] == 5


@pytest.mark.anyio
async def test_upload_document_rejects_mismatched_pdf_signature(tmp_path) -> None:
    service = DocumentService()
    service.uploads_dir = tmp_path
    service.repository = StubUploadRepository()
    upload_file = UploadFile(filename="sample.pdf", file=BytesIO(b"not a pdf"), headers={"content-type": "application/pdf"})

    with pytest.raises(UnsupportedDocumentTypeError) as exc_info:
        await service.upload_document("kb_001", upload_file)

    assert exc_info.value.details == {"extension": ".pdf", "reason": "PDF 文件头无效"}


@pytest.mark.anyio
async def test_upload_document_persists_hash_and_size(tmp_path) -> None:
    service = DocumentService()
    service.uploads_dir = tmp_path
    repository = StubUploadRepository()
    service.repository = repository
    upload_file = UploadFile(filename="sample.txt", file=BytesIO(b"hello"), headers={"content-type": "text/plain"})

    result = await service.upload_document("kb_001", upload_file)

    assert result["file_size"] == 5
    assert repository.created_payload is not None
    assert repository.created_payload["file_size"] == 5
    assert len(str(repository.created_payload["content_hash"])) == 64


@pytest.mark.anyio
async def test_upload_markdown_document_uses_markdown_content_type(tmp_path) -> None:
    service = DocumentService()
    service.uploads_dir = tmp_path
    repository = StubUploadRepository()
    service.repository = repository
    upload_file = UploadFile(filename="sample.md", file=BytesIO(b"# title"), headers={"content-type": "text/plain"})

    result = await service.upload_document("kb_001", upload_file)

    assert result["filename"] == "sample.md"
    assert result["content_type"] == "text/markdown"
    assert repository.created_payload is not None
    assert repository.created_payload["content_type"] == "text/markdown"
