from pathlib import Path

import pytest

from app.document_processing.pdf.service import PDFStructuringService
from app.services.ingest_service import IngestService


class StubPDFStructuringService:
    def __init__(self, structured_json_path: Path, used_full_page_ocr_fallback: bool = False) -> None:
        self.structured_json_path = structured_json_path
        self.used_full_page_ocr_fallback = used_full_page_ocr_fallback
        self.calls: list[tuple[Path, Path]] = []

    def preprocess(self, pdf_path: Path, output_dir: Path):
        self.calls.append((pdf_path, output_dir))
        return type(
            "Artifacts",
            (),
            {
                "structured_json_path": self.structured_json_path,
                "used_full_page_ocr_fallback": self.used_full_page_ocr_fallback,
            },
        )()


class StubDOCXTextExtractionService:
    def __init__(self, extracted_txt_path: Path) -> None:
        self.extracted_txt_path = extracted_txt_path
        self.calls: list[tuple[Path, Path]] = []

    def preprocess(self, docx_path: Path, output_dir: Path):
        self.calls.append((docx_path, output_dir))
        return type(
            "Artifacts",
            (),
            {
                "extracted_txt_path": self.extracted_txt_path,
            },
        )()


class StubXLSXTextExtractionService:
    def __init__(self, extracted_txt_path: Path) -> None:
        self.extracted_txt_path = extracted_txt_path
        self.calls: list[tuple[Path, Path]] = []

    def preprocess(self, xlsx_path: Path, output_dir: Path):
        self.calls.append((xlsx_path, output_dir))
        return type(
            "Artifacts",
            (),
            {
                "extracted_txt_path": self.extracted_txt_path,
            },
        )()


class StubImageOCRService:
    def __init__(self, structured_json_path: Path) -> None:
        self.structured_json_path = structured_json_path
        self.calls: list[tuple[Path, Path]] = []

    def preprocess(self, image_path: Path, output_dir: Path):
        self.calls.append((image_path, output_dir))
        return type(
            "Artifacts",
            (),
            {
                "structured_json_path": self.structured_json_path,
            },
        )()


def test_prepare_ingest_path_returns_original_path_for_non_pdf(tmp_path: Path) -> None:
    service = IngestService()
    text_path = tmp_path / "sample.txt"

    assert service._prepare_ingest_path("doc_001", text_path) == text_path


def test_prepare_ingest_path_runs_pdf_structuring_for_pdf(tmp_path: Path) -> None:
    service = IngestService()
    pdf_path = tmp_path / "sample.pdf"
    structured_path = tmp_path / "_artifacts" / "doc_001" / "structured.json"
    stub_service = StubPDFStructuringService(structured_path)
    service.pdf_structuring_service = stub_service

    ingest_path = service._prepare_ingest_path("doc_001", pdf_path)

    assert ingest_path == structured_path
    assert stub_service.calls == [
        (
            pdf_path,
            pdf_path.parent / "_artifacts" / "doc_001",
        )
    ]


def test_prepare_ingest_target_returns_notice_for_full_page_pdf_ocr_fallback(tmp_path: Path) -> None:
    service = IngestService()
    pdf_path = tmp_path / "vector.pdf"
    structured_path = tmp_path / "_artifacts" / "doc_001" / "structured.json"
    stub_service = StubPDFStructuringService(structured_path, used_full_page_ocr_fallback=True)
    service.pdf_structuring_service = stub_service

    ingest_path, ingest_notice = service._prepare_ingest_target("doc_001", pdf_path)

    assert ingest_path == structured_path
    assert ingest_notice == IngestService.PDF_FALLBACK_NOTICE


def test_prepare_ingest_path_extracts_text_for_docx(tmp_path: Path) -> None:
    service = IngestService()
    docx_path = tmp_path / "sample.docx"
    extracted_path = tmp_path / "_artifacts" / "doc_001" / "extracted.txt"
    stub_service = StubDOCXTextExtractionService(extracted_path)
    service.docx_text_extraction_service = stub_service

    ingest_path = service._prepare_ingest_path("doc_001", docx_path)

    assert ingest_path == extracted_path
    assert stub_service.calls == [
        (
            docx_path,
            docx_path.parent / "_artifacts" / "doc_001",
        )
    ]


def test_prepare_ingest_path_extracts_text_for_xlsx(tmp_path: Path) -> None:
    service = IngestService()
    xlsx_path = tmp_path / "sample.xlsx"
    extracted_path = tmp_path / "_artifacts" / "doc_001" / "extracted.txt"
    stub_service = StubXLSXTextExtractionService(extracted_path)
    service.xlsx_text_extraction_service = stub_service

    ingest_path = service._prepare_ingest_path("doc_001", xlsx_path)

    assert ingest_path == extracted_path
    assert stub_service.calls == [
        (
            xlsx_path,
            xlsx_path.parent / "_artifacts" / "doc_001",
        )
    ]


def test_prepare_ingest_path_runs_image_ocr_for_image(tmp_path: Path) -> None:
    service = IngestService()
    image_path = tmp_path / "sample.png"
    structured_path = tmp_path / "_artifacts" / "doc_001" / "structured.json"
    stub_service = StubImageOCRService(structured_path)
    service.image_ocr_service = stub_service

    ingest_path = service._prepare_ingest_path("doc_001", image_path)

    assert ingest_path == structured_path
    assert stub_service.calls == [
        (
            image_path,
            image_path.parent / "_artifacts" / "doc_001",
        )
    ]


class StubRepository:
    def __init__(self, tmp_path: Path) -> None:
        self.knowledge_base = {
            "id": "kb_test",
            "subject": "测试知识库",
            "status": "empty",
        }
        self.documents = {
            "doc_001": {
                "id": "doc_001",
                "knowledge_base_id": "kb_test",
                "file_path": str(tmp_path / "sample.txt"),
                "status": "uploaded",
            }
        }
        self.task = {
            "id": "task_001",
            "knowledge_base_id": "kb_test",
            "document_ids": ["doc_001"],
            "status": "pending",
            "message": "",
        }

    def get_task(self, task_id: str):
        return self.task if task_id == self.task["id"] else None

    def get_knowledge_base(self, knowledge_base_id: str):
        return self.knowledge_base if knowledge_base_id == self.knowledge_base["id"] else None

    def list_documents(self, knowledge_base_id: str | None = None):
        docs = list(self.documents.values())
        if knowledge_base_id is None:
            return docs
        return [doc for doc in docs if doc["knowledge_base_id"] == knowledge_base_id]

    def update_task(self, task_id: str, **changes):
        assert task_id == self.task["id"]
        self.task.update(changes)
        return dict(self.task)

    def update_knowledge_base(self, knowledge_base_id: str, status: str):
        assert knowledge_base_id == self.knowledge_base["id"]
        self.knowledge_base["status"] = status
        return dict(self.knowledge_base)

    def update_document(self, document_id: str, status: str):
        self.documents[document_id]["status"] = status
        return dict(self.documents[document_id])


class StubTaskQueue:
    def __init__(self, repository: StubRepository) -> None:
        self.repository = repository
        self.events: list[tuple[str, str, str, dict | None]] = []

    def enqueue_task(self, *, knowledge_base_id: str, document_ids: list[str], task_type: str, message: str) -> dict:
        self.repository.task = {
            "id": "task_001",
            "knowledge_base_id": knowledge_base_id,
            "task_type": task_type,
            "document_ids": document_ids,
            "status": "pending",
            "message": message,
        }
        return dict(self.repository.task)

    def get_task(self, task_id: str):
        return self.repository.get_task(task_id)

    def create_task_event(self, task_id: str, *, event_type: str, message: str, payload: dict | None = None):
        self.events.append((task_id, event_type, message, payload))
        return {"task_id": task_id, "event_type": event_type, "message": message, "payload": payload}

    def update_task(self, task_id: str, **changes):
        return self.repository.update_task(task_id, **changes)

    def claim_next_task(self, *, worker_id: str, lease_seconds: int, max_attempts: int):
        task = self.repository.get_task("task_001")
        if task is None:
            return None
        task["locked_by"] = worker_id
        return task

    def heartbeat_task(self, task_id: str, *, worker_id: str, message: str | None = None):
        changes = {"locked_by": worker_id}
        if message is not None:
            changes["message"] = message
        return self.repository.update_task(task_id, **changes)


class StubPipeline:
    def __init__(self, chunks_to_return: int) -> None:
        self.chunks_to_return = chunks_to_return
        self.ingested_paths: list[tuple[Path, str | None]] = []

    def ingest_file(self, ingest_path: Path, document_id: str | None = None) -> int:
        self.ingested_paths.append((ingest_path, document_id))
        return self.chunks_to_return


def _event_types(events: list[tuple[str, str, str, dict | None]]) -> list[str]:
    return [event[1] for event in events]


def _stage_names(events: list[tuple[str, str, str, dict | None]]) -> list[str]:
    return [
        str((event[3] or {}).get("stage"))
        for event in events
        if event[1] == "stage"
    ]


def test_run_ingest_task_marks_zero_chunk_document_and_task_as_failed(tmp_path: Path, monkeypatch) -> None:
    source_path = tmp_path / "sample.txt"
    source_path.write_text("empty", encoding="utf-8")

    repository = StubRepository(tmp_path)
    pipeline = StubPipeline(chunks_to_return=0)
    task_queue = StubTaskQueue(repository)
    service = IngestService(task_queue=task_queue)
    service.repository = repository

    monkeypatch.setattr("app.services.ingest_service.get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    result = service.run_ingest_task("task_001")

    assert result["status"] == "failed"
    assert result["message"] == "ingest failed, total chunks: 0"
    assert repository.documents["doc_001"]["status"] == "failed"
    assert repository.knowledge_base["status"] == "failed"
    assert pipeline.ingested_paths == [(source_path, "doc_001")]
    assert [event_type for event_type in _event_types(task_queue.events) if event_type != "stage"] == [
        "started",
        "document_started",
        "document_failed",
        "failed",
    ]
    assert _stage_names(task_queue.events) == [
        "knowledge_base_indexing_started",
        "pipeline_get_started",
        "pipeline_get_completed",
        "prepare_ingest_target",
        "prepare_ingest_target_completed",
        "pipeline_ingest_started",
        "pipeline_ingest_completed",
    ]


def test_run_ingest_task_rebuilds_single_document_without_resetting_whole_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "sample.txt"
    source_path.write_text("content", encoding="utf-8")

    repository = StubRepository(tmp_path)
    repository.documents["doc_002"] = {
        "id": "doc_002",
        "knowledge_base_id": "kb_test",
        "file_path": str(tmp_path / "other.txt"),
        "status": "ready",
    }
    repository.task["document_ids"] = ["doc_001"]
    pipeline = StubPipeline(chunks_to_return=2)
    task_queue = StubTaskQueue(repository)
    service = IngestService(task_queue=task_queue)
    service.repository = repository
    deleted_document_ids: list[tuple[str, str, str]] = []
    reset_calls: list[tuple[str, str]] = []

    monkeypatch.setattr("app.services.ingest_service.get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)
    monkeypatch.setattr(
        "app.services.ingest_service.delete_document_index",
        lambda knowledge_base_id, subject, document_id: deleted_document_ids.append(
            (knowledge_base_id, subject, document_id)
        ),
    )
    monkeypatch.setattr(
        "app.services.ingest_service.reset_knowledge_base_index",
        lambda knowledge_base_id, subject: reset_calls.append((knowledge_base_id, subject)),
    )

    result = service.run_ingest_task("task_001", rebuild=True)

    assert result["status"] == "completed"
    assert result["message"] == "rebuild completed, total chunks: 2"
    assert deleted_document_ids == [("kb_test", "测试知识库", "doc_001")]
    assert reset_calls == []
    assert pipeline.ingested_paths == [(source_path, "doc_001")]
    assert [event_type for event_type in _event_types(task_queue.events) if event_type != "stage"] == [
        "started",
        "document_rebuild_started",
        "document_started",
        "document_completed",
        "completed",
    ]
    assert _stage_names(task_queue.events) == [
        "knowledge_base_indexing_started",
        "pipeline_get_started",
        "pipeline_get_completed",
        "prepare_ingest_target",
        "prepare_ingest_target_completed",
        "pipeline_ingest_started",
        "pipeline_ingest_completed",
    ]


def test_pdf_structuring_service_rejects_payload_without_extractable_content(tmp_path: Path) -> None:
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    payload = {
        "file_name": pdf_path.name,
        "file_path": str(pdf_path),
        "pdf_file_id": "empty_pdf",
        "total_pages": 2,
        "pages": [
            {"page_number": 1, "text": "", "images": [], "tables": []},
            {"page_number": 2, "text": "", "images": [], "tables": []},
        ],
    }

    with pytest.raises(RuntimeError, match="未提取到可用文本、表格或 OCR 内容"):
        PDFStructuringService._ensure_extractable_payload(payload, pdf_path)


def test_pdf_structuring_service_accepts_payload_with_full_page_ocr_fallback(tmp_path: Path) -> None:
    pdf_path = tmp_path / "vector.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    payload = {
        "file_name": pdf_path.name,
        "file_path": str(pdf_path),
        "pdf_file_id": "vector_pdf",
        "total_pages": 1,
        "pages": [
            {
                "page_number": 1,
                "text": "",
                "tables": [],
                "images": [
                    {
                        "index": "full_page_ocr",
                        "ocr_text": "劳动法 第一条",
                        "translated_text": "劳动法 第一条",
                        "translated_source_type": "image_ocr",
                    }
                ],
            }
        ],
    }

    PDFStructuringService._ensure_extractable_payload(payload, pdf_path)
