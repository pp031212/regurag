"""文档入库和任务监控服务。

入库流程负责把用户上传的原始文件转成 RAG pipeline 可消费的文本或结构化 JSON，
再写入对应知识库的向量索引。这里也封装后台任务的创建、执行、心跳和监控统计。
"""

import logging
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from ..core.config import get_settings
from ..document_processing.docx import DOCXTextExtractionService
from ..document_processing.image import ImageOCRService
from ..document_processing.pdf import PDFStructuringService
from ..document_processing.xlsx import XLSXTextExtractionService
from ..core.exceptions import (
    DocumentsNotFoundError,
    KnowledgeBaseNotFoundError,
    NoMatchingDocumentsError,
    TaskNotFoundError,
)
from ..repositories.metadata_repository import MetadataRepository, get_metadata_repository
from ..task_queue import TaskQueueBackend, get_task_queue_backend
from .rag_service import (
    RAGPipelineRegistry,
    delete_document_index,
    get_default_pipeline_registry,
    get_rag_pipeline,
    reset_knowledge_base_index,
)

logger = logging.getLogger(__name__)


class IngestService:
    """执行知识库文档入库/重建任务的应用服务。"""

    PDF_FALLBACK_NOTICE = (
        "检测到当前 PDF 已进入整页 OCR 降级处理，耗时可能更长。"
        "若条件允许，建议优先使用文本型 PDF、DOCX 或 XLSX 等更常规文件。"
    )

    def __init__(
        self,
        *,
        repository: MetadataRepository | None = None,
        task_queue: TaskQueueBackend | None = None,
        pipeline_registry: RAGPipelineRegistry | None = None,
    ) -> None:
        self.settings = get_settings()
        self.repository = repository or get_metadata_repository()
        self.task_queue = task_queue or get_task_queue_backend(repository=self.repository)
        self.pipeline_registry = pipeline_registry or get_default_pipeline_registry()
        self.pdf_structuring_service = PDFStructuringService()
        self.docx_text_extraction_service = DOCXTextExtractionService()
        self.xlsx_text_extraction_service = XLSXTextExtractionService()
        self.image_ocr_service = ImageOCRService()

    def _get_pipeline(self, knowledge_base_id: str, subject: str):
        if self.pipeline_registry is get_default_pipeline_registry():
            return get_rag_pipeline(knowledge_base_id, subject)
        return self.pipeline_registry.get(knowledge_base_id, subject)

    def _reset_knowledge_base_index(self, knowledge_base_id: str, subject: str) -> None:
        if self.pipeline_registry is get_default_pipeline_registry():
            reset_knowledge_base_index(knowledge_base_id, subject)
            return
        self.pipeline_registry.reset_index(knowledge_base_id, subject)

    def _delete_document_index(self, knowledge_base_id: str, subject: str, document_id: str) -> None:
        if self.pipeline_registry is get_default_pipeline_registry():
            delete_document_index(knowledge_base_id, subject, document_id)
            return
        self.pipeline_registry.delete_document_index(knowledge_base_id, subject, document_id)

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC).replace(tzinfo=None)

    def create_ingest_task(
        self,
        knowledge_base_id: str,
        document_ids: list[str],
        *,
        task_type: str = "ingest",
        message: str = "ingest task created",
    ) -> dict:
        """创建异步入库任务；空 document_ids 表示处理该知识库全部文档。"""
        knowledge_base = self.repository.get_knowledge_base(knowledge_base_id)
        if knowledge_base is None:
            raise KnowledgeBaseNotFoundError()

        documents = self.repository.list_documents(knowledge_base_id=knowledge_base_id)
        if not documents:
            raise DocumentsNotFoundError()

        if document_ids:
            selected_documents = [item for item in documents if item["id"] in document_ids]
        else:
            selected_documents = documents

        if not selected_documents:
            raise NoMatchingDocumentsError()

        return self.task_queue.enqueue_task(
            knowledge_base_id=knowledge_base_id,
            document_ids=[item["id"] for item in selected_documents],
            message=message,
            task_type=task_type,
        )

    def run_ingest_task(self, task_id: str, rebuild: bool | None = None) -> dict:
        """执行一个已被 worker 认领的入库任务。"""
        task = self.task_queue.get_task(task_id)
        if task is None:
            raise TaskNotFoundError()
        if rebuild is None:
            rebuild = task.get("task_type") == "rebuild"

        knowledge_base_id = task["knowledge_base_id"]
        knowledge_base = self.repository.get_knowledge_base(knowledge_base_id)
        if knowledge_base is None:
            self._record_task_event(task_id, event_type="failed", message="knowledge base not found")
            return self.task_queue.update_task(
                task_id,
                status="failed",
                message="knowledge base not found",
                last_error="knowledge base not found",
                finished_at=self._now(),
                locked_at=None,
                locked_by=None,
            )

        all_documents = self.repository.list_documents(knowledge_base_id=knowledge_base_id)
        documents = [
            item
            for item in all_documents
            if item["id"] in task["document_ids"]
        ]
        if not documents:
            self._record_task_event(task_id, event_type="failed", message="no matching documents found")
            return self.task_queue.update_task(
                task_id,
                status="failed",
                message="no matching documents found",
                last_error="no matching documents found",
                finished_at=self._now(),
                locked_at=None,
                locked_by=None,
            )

        self._heartbeat_task(task, message="rebuild running" if rebuild else "ingest running")
        self.task_queue.update_task(task_id, status="running", message="rebuild running" if rebuild else "ingest running")
        self._record_task_event(
            task_id,
            event_type="started",
            message="rebuild running" if rebuild else "ingest running",
            payload={"task_type": task.get("task_type")},
        )
        self._record_stage(
            task_id,
            stage="knowledge_base_indexing_started",
            message=f"marking knowledge base {knowledge_base_id} as indexing",
            payload={"knowledge_base_id": knowledge_base_id},
            heartbeat_task=task,
        )
        self.repository.update_knowledge_base(knowledge_base_id, status="indexing")
        rebuild_all_documents = rebuild and len(documents) == len(all_documents)
        if rebuild_all_documents:
            # 全量 rebuild 直接重置整个 collection；部分 rebuild 只删对应文档的旧索引。
            self._reset_knowledge_base_index(knowledge_base_id, knowledge_base["subject"])
        self._record_stage(
            task_id,
            stage="pipeline_get_started",
            message=f"resolving pipeline for knowledge base {knowledge_base_id}",
            payload={"knowledge_base_id": knowledge_base_id},
            heartbeat_task=task,
        )
        pipeline_started = perf_counter()
        pipeline = self._get_pipeline(knowledge_base_id, knowledge_base["subject"])
        self._record_stage(
            task_id,
            stage="pipeline_get_completed",
            message=f"resolved pipeline for knowledge base {knowledge_base_id}",
            payload={
                "knowledge_base_id": knowledge_base_id,
                "elapsed_ms": int((perf_counter() - pipeline_started) * 1000),
            },
            heartbeat_task=task,
        )

        total_chunks = 0
        try:
            if rebuild and not rebuild_all_documents:
                for document in documents:
                    # 部分重建先删除旧 chunk，再按当前文件内容重新写入，避免重复命中。
                    self._heartbeat_task(task, message=f"rebuilding {document['id']}")
                    self._record_task_event(
                        task_id,
                        event_type="document_rebuild_started",
                        message=f"rebuilding {document['id']}",
                        payload={"document_id": document["id"]},
                    )
                    self._delete_document_index(
                        knowledge_base_id=knowledge_base_id,
                        subject=knowledge_base["subject"],
                        document_id=document["id"],
                    )
            for document in documents:
                document_started = perf_counter()
                self._heartbeat_task(task, message=f"processing {document['id']}")
                self._record_task_event(
                    task_id,
                    event_type="document_started",
                    message=f"processing {document['id']}",
                    payload={"document_id": document["id"]},
                )
                self.repository.update_document(document["id"], status="indexing")
                file_path = Path(document["file_path"])
                self._record_stage(
                    task_id,
                    stage="prepare_ingest_target",
                    message=f"preparing ingest target for {document['id']}",
                    payload={"document_id": document["id"], "file_path": str(file_path)},
                    heartbeat_task=task,
                )
                prepare_started = perf_counter()
                # 入库前先把不同文件格式统一成 pipeline 支持的 .txt/.md/.json。
                ingest_path, ingest_notice = self._prepare_ingest_target(document["id"], file_path)
                self._record_stage(
                    task_id,
                    stage="prepare_ingest_target_completed",
                    message=f"prepared ingest target for {document['id']}",
                    payload={
                        "document_id": document["id"],
                        "ingest_path": str(ingest_path),
                        "elapsed_ms": int((perf_counter() - prepare_started) * 1000),
                    },
                    heartbeat_task=task,
                )
                if ingest_notice:
                    self._heartbeat_task(task, message=ingest_notice)
                    self.task_queue.update_task(task_id, status="running", message=ingest_notice)
                    self._record_task_event(
                        task_id,
                        event_type="notice",
                        message=ingest_notice,
                        payload={"document_id": document["id"]},
                    )
                self._record_stage(
                    task_id,
                    stage="pipeline_ingest_started",
                    message=f"pipeline ingest started for {document['id']}",
                    payload={"document_id": document["id"], "ingest_path": str(ingest_path)},
                    heartbeat_task=task,
                )
                ingest_started = perf_counter()
                chunks = pipeline.ingest_file(ingest_path, document_id=document["id"])
                self._record_stage(
                    task_id,
                    stage="pipeline_ingest_completed",
                    message=f"pipeline ingest completed for {document['id']}",
                    payload={
                        "document_id": document["id"],
                        "chunks": chunks,
                        "elapsed_ms": int((perf_counter() - ingest_started) * 1000),
                        "document_elapsed_ms": int((perf_counter() - document_started) * 1000),
                    },
                    heartbeat_task=task,
                )
                total_chunks += chunks

                if chunks <= 0:
                    self.repository.update_document(document["id"], status="failed")
                    self._record_task_event(
                        task_id,
                        event_type="document_failed",
                        message=f"document {document['id']} produced no chunks",
                        payload={"document_id": document["id"], "chunks": chunks},
                    )
                    continue

                self.repository.update_document(document["id"], status="ready")
                self._record_task_event(
                    task_id,
                    event_type="document_completed",
                    message=f"document {document['id']} indexed",
                    payload={"document_id": document["id"], "chunks": chunks},
                )

            if total_chunks <= 0:
                self.repository.update_knowledge_base(knowledge_base_id, status="failed")
                final_message = "rebuild failed" if rebuild else "ingest failed"
                self._record_task_event(
                    task_id,
                    event_type="failed",
                    message=f"{final_message}, total chunks: 0",
                    payload={"total_chunks": total_chunks},
                )
                return self.task_queue.update_task(
                    task_id,
                    status="failed",
                    message=f"{final_message}, total chunks: 0",
                    last_error=f"{final_message}, total chunks: 0",
                    finished_at=self._now(),
                    locked_at=None,
                    locked_by=None,
                )

            self.repository.update_knowledge_base(knowledge_base_id, status="ready")
            final_message = "rebuild completed" if rebuild else "ingest completed"
            self._record_task_event(
                task_id,
                event_type="completed",
                message=f"{final_message}, total chunks: {total_chunks}",
                payload={"total_chunks": total_chunks},
            )
            return self.task_queue.update_task(
                task_id,
                status="completed",
                message=f"{final_message}, total chunks: {total_chunks}",
                last_error=None,
                finished_at=self._now(),
                locked_at=None,
                locked_by=None,
            )
        except Exception as exc:
            # 任务级异常说明本轮入库无法可靠继续，统一标记涉及文档和 KB 为 failed。
            for document in documents:
                self.repository.update_document(document["id"], status="failed")
            self.repository.update_knowledge_base(knowledge_base_id, status="failed")
            self._record_task_event(
                task_id,
                event_type="failed",
                message=str(exc),
                payload={"error": str(exc)},
            )
            return self.task_queue.update_task(
                task_id,
                status="failed",
                message=str(exc),
                last_error=str(exc),
                finished_at=self._now(),
                locked_at=None,
                locked_by=None,
            )

    def claim_next_task(self, worker_id: str) -> dict | None:
        return self.task_queue.claim_next_task(
            worker_id=worker_id,
            lease_seconds=self.settings.task_worker_lease_seconds,
            max_attempts=self.settings.task_worker_max_attempts,
        )

    def _heartbeat_task(self, task: dict, *, message: str | None = None) -> None:
        """刷新 worker 租约，避免长文档处理期间任务被其他 worker 抢走。"""
        worker_id = task.get("locked_by")
        if not worker_id:
            return
        self.task_queue.heartbeat_task(task["id"], worker_id=worker_id, message=message)

    def _record_task_event(
        self,
        task_id: str,
        *,
        event_type: str,
        message: str,
        payload: dict | None = None,
    ) -> None:
        self.task_queue.create_task_event(
            task_id,
            event_type=event_type,
            message=message,
            payload=payload,
        )

    def _record_stage(
        self,
        task_id: str,
        *,
        stage: str,
        message: str,
        payload: dict | None = None,
        heartbeat_task: dict | None = None,
    ) -> None:
        """记录可展示在任务时间线里的阶段事件，并按需刷新租约。"""
        stage_payload = {"stage": stage}
        if payload:
            stage_payload.update(payload)
        self._record_task_event(
            task_id,
            event_type="stage",
            message=message,
            payload=stage_payload,
        )
        if heartbeat_task is not None:
            self._heartbeat_task(heartbeat_task, message=message)
        logger.info(
            "ingest_stage task_id=%s stage=%s message=%s payload=%s",
            task_id,
            stage,
            message,
            stage_payload,
        )

    def _prepare_ingest_path(self, document_id: str, file_path: Path) -> Path:
        ingest_path, _ = self._prepare_ingest_target(document_id, file_path)
        return ingest_path

    def _prepare_ingest_target(self, document_id: str, file_path: Path) -> tuple[Path, str | None]:
        """把原始上传文件预处理成 pipeline 的统一入库目标。"""
        if file_path.suffix.lower() == ".docx":
            artifact_dir = file_path.parent / "_artifacts" / document_id
            artifacts = self.docx_text_extraction_service.preprocess(file_path, artifact_dir)
            return artifacts.extracted_txt_path, None

        if file_path.suffix.lower() == ".xlsx":
            artifact_dir = file_path.parent / "_artifacts" / document_id
            artifacts = self.xlsx_text_extraction_service.preprocess(file_path, artifact_dir)
            return artifacts.extracted_txt_path, None

        if file_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            artifact_dir = file_path.parent / "_artifacts" / document_id
            artifacts = self.image_ocr_service.preprocess(file_path, artifact_dir)
            return artifacts.structured_json_path, None

        if file_path.suffix.lower() != ".pdf":
            return file_path, None

        artifact_dir = file_path.parent / "_artifacts" / document_id
        # PDF 可能走文本抽取、表格抽取或 OCR；结构化 JSON 用于保留页码和块类型。
        artifacts = self.pdf_structuring_service.preprocess(file_path, artifact_dir)
        notice = self.PDF_FALLBACK_NOTICE if artifacts.used_full_page_ocr_fallback else None
        return artifacts.structured_json_path, notice


class TaskService:
    """任务查询和监控页聚合服务。"""

    def __init__(self, *, task_queue: TaskQueueBackend | None = None) -> None:
        self.settings = get_settings()
        self.task_queue = task_queue or get_task_queue_backend()

    def get_task(self, task_id: str) -> dict:
        record = self.task_queue.get_task(task_id)
        if record is None:
            raise TaskNotFoundError()
        return record

    def get_task_events(self, task_id: str, *, limit: int = 100) -> dict[str, object]:
        task = self.task_queue.get_task(task_id)
        if task is None:
            raise TaskNotFoundError()
        items = self.task_queue.list_task_events(task_id, limit=limit)
        return {
            "items": items,
            "total": len(items),
        }

    def list_tasks(
        self,
        *,
        knowledge_base_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> dict[str, object]:
        items = self.task_queue.list_tasks(
            knowledge_base_id=knowledge_base_id,
            status=status,
            limit=limit,
        )
        return {
            "items": items,
            "total": len(items),
        }

    def get_task_stats(self, *, knowledge_base_id: str | None = None) -> dict[str, int]:
        return self.task_queue.get_task_stats(
            knowledge_base_id=knowledge_base_id,
            lease_seconds=self.settings.task_worker_lease_seconds,
        )

    def get_task_overview(self, *, knowledge_base_id: str | None = None) -> dict[str, object]:
        return self.task_queue.get_task_overview(
            knowledge_base_id=knowledge_base_id,
            lease_seconds=self.settings.task_worker_lease_seconds,
            monitoring_window_hours=self.settings.task_monitor_window_hours,
            long_running_seconds=self.settings.task_monitor_long_running_seconds,
        )

    def get_task_alerts(self, *, knowledge_base_id: str | None = None) -> dict[str, object]:
        """把底层统计转换成前端可直接展示的告警项。"""
        overview = self.get_task_overview(knowledge_base_id=knowledge_base_id)
        items: list[dict[str, object]] = []

        if int(overview["pending"]) > 0 and int(overview["active_workers"]) == 0:
            items.append(
                {
                    "code": "PENDING_WITHOUT_ACTIVE_WORKERS",
                    "severity": "critical",
                    "message": "pending tasks exist but no active workers were detected",
                    "count": int(overview["pending"]),
                    "details": {
                        "pending": int(overview["pending"]),
                        "active_workers": int(overview["active_workers"]),
                    },
                }
            )

        if int(overview["stale_running"]) > 0:
            items.append(
                {
                    "code": "STALE_RUNNING_TASKS",
                    "severity": "warning",
                    "message": "stale running tasks detected",
                    "count": int(overview["stale_running"]),
                    "details": {
                        "stale_running": int(overview["stale_running"]),
                        "lease_seconds": self.settings.task_worker_lease_seconds,
                    },
                }
            )

        if int(overview["long_running"]) > 0:
            items.append(
                {
                    "code": "LONG_RUNNING_TASKS",
                    "severity": "warning",
                    "message": "long-running tasks detected",
                    "count": int(overview["long_running"]),
                    "details": {
                        "long_running": int(overview["long_running"]),
                        "threshold_seconds": self.settings.task_monitor_long_running_seconds,
                    },
                }
            )

        if int(overview["recent_failed"]) >= self.settings.task_monitor_recent_failure_threshold:
            items.append(
                {
                    "code": "RECENT_FAILURE_SPIKE",
                    "severity": "warning",
                    "message": "recent failed task count exceeded threshold",
                    "count": int(overview["recent_failed"]),
                    "details": {
                        "recent_failed": int(overview["recent_failed"]),
                        "threshold": self.settings.task_monitor_recent_failure_threshold,
                        "window_hours": self.settings.task_monitor_window_hours,
                        "knowledge_bases": list(overview["knowledge_bases_with_recent_failures"]),
                    },
                }
            )

        if int(overview["recent_retried"]) >= self.settings.task_monitor_recent_retry_threshold:
            items.append(
                {
                    "code": "RECENT_RETRY_SPIKE",
                    "severity": "warning",
                    "message": "recent retry count exceeded threshold",
                    "count": int(overview["recent_retried"]),
                    "details": {
                        "recent_retried": int(overview["recent_retried"]),
                        "threshold": self.settings.task_monitor_recent_retry_threshold,
                        "window_hours": self.settings.task_monitor_window_hours,
                    },
                }
            )

        return {
            "items": items,
            "total": len(items),
        }

    def get_task_trends(self, *, limit: int = 20) -> dict[str, object]:
        items = self.task_queue.get_task_trends(
            monitoring_window_hours=self.settings.task_monitor_window_hours,
            limit=limit,
        )
        return {
            "items": items,
            "total": len(items),
        }
