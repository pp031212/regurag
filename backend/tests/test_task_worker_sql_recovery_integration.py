"""SQL 任务队列和 worker 恢复能力集成测试。

这组测试依赖真实 SQL 元数据后端。环境不可用时会自动 skip，避免本地轻量测试被外部服务阻塞。
"""

from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.core.exceptions import KnowledgeBaseNotFoundError
from app.db.session import engine
from app.repositories.metadata_repository import MetadataRepository, get_metadata_repository
from app.services.ingest_service import IngestService
from app.task_queue import MySQLTaskQueueBackend
from app.workers.task_worker import TaskWorker


def _now() -> datetime:
    """返回数据库字段使用的无时区 UTC 时间。"""
    return datetime.now(UTC).replace(tzinfo=None)


@pytest.fixture()
def sql_recovery_context() -> tuple[MySQLTaskQueueBackend, MetadataRepository, dict]:
    """准备临时知识库和 SQL 任务队列，测试结束后清理知识库。"""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"SQL metadata backend unavailable for integration test: {exc}")

    repository = get_metadata_repository()
    backend = MySQLTaskQueueBackend(repository=repository)
    knowledge_base = repository.create_knowledge_base(
        name=f"Recovery KB {uuid4().hex[:6]}",
        description="task worker recovery integration",
        subject="task worker recovery subject",
        domain="general",
    )
    try:
        yield backend, repository, knowledge_base
    finally:
        with suppress(KnowledgeBaseNotFoundError):
            repository.delete_knowledge_base(knowledge_base["id"])


class FailingThenSucceedingIngestService:
    """可控失败次数的入库服务，用来模拟 worker 重试恢复。"""

    def __init__(
        self,
        *,
        task_queue: MySQLTaskQueueBackend,
        repository: MetadataRepository,
        fail_runs: int,
        lease_seconds: int = 60,
        max_attempts: int = 3,
    ) -> None:
        self.task_queue = task_queue
        self.repository = repository
        self.remaining_failures = fail_runs
        self.completed_runs = 0
        self.settings = SimpleNamespace(
            task_worker_lease_seconds=lease_seconds,
            task_worker_max_attempts=max_attempts,
        )

    def claim_next_task(self, worker_id: str) -> dict | None:
        return self.task_queue.claim_next_task(
            worker_id=worker_id,
            lease_seconds=self.settings.task_worker_lease_seconds,
            max_attempts=self.settings.task_worker_max_attempts,
        )

    def run_ingest_task(self, task_id: str) -> dict:
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            raise RuntimeError("transient worker failure")

        self.completed_runs += 1
        return self.task_queue.update_task(
            task_id,
            status="completed",
            message="recovered after retry",
            last_error=None,
            finished_at=_now(),
            locked_at=None,
            locked_by=None,
        )


class SelectiveChunkPipeline:
    """按 document_id 控制某个文档入库失败的测试 pipeline。"""

    def __init__(self, failed_document_id: str) -> None:
        self.failed_document_id = failed_document_id
        self.calls: list[tuple[Path, str | None]] = []

    def ingest_file(self, ingest_path: Path, document_id: str | None = None) -> int:
        self.calls.append((ingest_path, document_id))
        if document_id == self.failed_document_id:
            return 0
        return 2


def test_worker_recovers_task_after_retry_with_sql_queue(sql_recovery_context) -> None:
    """第一次 worker 失败后，第二个 worker 应能重新认领并完成任务。"""
    backend, repository, knowledge_base = sql_recovery_context
    task = backend.enqueue_task(
        knowledge_base_id=knowledge_base["id"],
        document_ids=["doc_recovery_001"],
        task_type="ingest",
        message="queued for retry recovery",
    )
    service = FailingThenSucceedingIngestService(
        task_queue=backend,
        repository=repository,
        fail_runs=1,
    )

    first_worker = TaskWorker(worker_id="worker-retry-1", ingest_service=service, poll_interval_seconds=0.01)
    second_worker = TaskWorker(worker_id="worker-retry-2", ingest_service=service, poll_interval_seconds=0.01)

    assert first_worker.run_once() is True
    after_first_failure = backend.get_task(task["id"])
    assert after_first_failure is not None
    assert after_first_failure["status"] == "pending"
    assert after_first_failure["attempt_count"] == 1
    assert after_first_failure["last_error"] == "transient worker failure"
    assert after_first_failure["locked_by"] is None

    assert second_worker.run_once() is True
    recovered_task = backend.get_task(task["id"])
    assert recovered_task is not None
    assert recovered_task["status"] == "completed"
    assert recovered_task["attempt_count"] == 2
    assert recovered_task["last_error"] is None
    assert recovered_task["locked_by"] is None
    assert recovered_task["finished_at"] is not None
    assert service.completed_runs == 1


def test_worker_claims_stale_task_after_lease_expires_with_sql_queue(sql_recovery_context) -> None:
    """租约过期的 running 任务应能被新 worker 接管。"""
    backend, repository, knowledge_base = sql_recovery_context
    task = backend.enqueue_task(
        knowledge_base_id=knowledge_base["id"],
        document_ids=["doc_recovery_002"],
        task_type="rebuild",
        message="queued for stale lease recovery",
    )
    stale_time = _now() - timedelta(minutes=10)
    backend.update_task(
        task["id"],
        status="running",
        attempt_count=1,
        locked_at=stale_time,
        locked_by="worker-stale",
        started_at=stale_time,
        finished_at=None,
    )

    service = FailingThenSucceedingIngestService(
        task_queue=backend,
        repository=repository,
        fail_runs=0,
    )
    recovery_worker = TaskWorker(worker_id="worker-fresh", ingest_service=service, poll_interval_seconds=0.01)

    assert recovery_worker.run_once() is True
    recovered_task = backend.get_task(task["id"])
    assert recovered_task is not None
    assert recovered_task["status"] == "completed"
    assert recovered_task["attempt_count"] == 2
    assert recovered_task["locked_by"] is None
    assert recovered_task["finished_at"] is not None
    assert service.completed_runs == 1


def test_ingest_task_completes_when_only_some_documents_fail(sql_recovery_context, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """部分文档失败时，任务整体完成，但失败文档应被标记为 failed。"""
    backend, repository, knowledge_base = sql_recovery_context
    first_path = tmp_path / "doc_ready.md"
    second_path = tmp_path / "doc_failed.md"
    first_path.write_text("# Ready\n\nalpha content", encoding="utf-8")
    second_path.write_text("# Failed\n\nbeta content", encoding="utf-8")

    ready_document = repository.create_document(
        knowledge_base_id=knowledge_base["id"],
        filename=first_path.name,
        content_type="text/markdown",
        file_size=first_path.stat().st_size,
        content_hash=f"hash-{uuid4().hex}",
        file_path=str(first_path),
    )
    failed_document = repository.create_document(
        knowledge_base_id=knowledge_base["id"],
        filename=second_path.name,
        content_type="text/markdown",
        file_size=second_path.stat().st_size,
        content_hash=f"hash-{uuid4().hex}",
        file_path=str(second_path),
    )

    pipeline = SelectiveChunkPipeline(failed_document_id=failed_document["id"])
    monkeypatch.setattr("app.services.ingest_service.get_rag_pipeline", lambda knowledge_base_id, subject: pipeline)

    service = IngestService(task_queue=backend)
    service.repository = repository
    task = service.create_ingest_task(
        knowledge_base["id"],
        [ready_document["id"], failed_document["id"]],
    )

    result = service.run_ingest_task(task["id"])
    ready_after = repository.get_document(ready_document["id"])
    failed_after = repository.get_document(failed_document["id"])
    knowledge_base_after = repository.get_knowledge_base(knowledge_base["id"])
    task_after = backend.get_task(task["id"])

    assert result["status"] == "completed"
    assert result["message"] == "ingest completed, total chunks: 2"
    assert ready_after is not None
    assert ready_after["status"] == "ready"
    assert failed_after is not None
    assert failed_after["status"] == "failed"
    assert knowledge_base_after is not None
    assert knowledge_base_after["status"] == "ready"
    assert task_after is not None
    assert task_after["status"] == "completed"
    assert {document_id for _, document_id in pipeline.calls} == {ready_document["id"], failed_document["id"]}
