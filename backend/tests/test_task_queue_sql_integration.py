from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.db.session import engine
from app.db.session import SessionLocal
from app.models.task import Task
from app.repositories.metadata_repository import get_metadata_repository
from app.task_queue import MySQLTaskQueueBackend


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


@pytest.fixture()
def sql_task_queue_backend() -> MySQLTaskQueueBackend:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"SQL metadata backend unavailable for integration test: {exc}")

    repository = get_metadata_repository()
    backend = MySQLTaskQueueBackend(repository=repository)
    knowledge_base = repository.create_knowledge_base(
        name=f"Integration Queue KB {uuid4().hex[:6]}",
        description="integration test knowledge base",
        subject="integration queue subject",
        domain="general",
    )
    try:
        yield backend, repository, knowledge_base["id"]
    finally:
        repository.delete_knowledge_base(knowledge_base["id"])


def test_sql_task_queue_backend_enqueues_and_claims_pending_task(sql_task_queue_backend) -> None:
    backend, _, knowledge_base_id = sql_task_queue_backend

    task = backend.enqueue_task(
        knowledge_base_id=knowledge_base_id,
        document_ids=["doc_integration_001"],
        task_type="ingest",
        message="integration task queued",
    )

    claimed = backend.claim_next_task(worker_id="worker-integration", lease_seconds=60, max_attempts=3)

    assert claimed is not None
    assert claimed["id"] == task["id"]
    assert claimed["status"] == "running"
    assert claimed["task_type"] == "ingest"
    assert claimed["attempt_count"] == 1
    assert claimed["locked_by"] == "worker-integration"
    assert claimed["started_at"] is not None
    assert claimed["locked_at"] is not None


def test_sql_task_queue_backend_reclaims_stale_running_task(sql_task_queue_backend) -> None:
    backend, _, knowledge_base_id = sql_task_queue_backend
    task = backend.enqueue_task(
        knowledge_base_id=knowledge_base_id,
        document_ids=["doc_integration_002"],
        task_type="rebuild",
        message="integration rebuild queued",
    )

    stale_time = _now() - timedelta(minutes=10)
    backend.update_task(
        task["id"],
        status="running",
        attempt_count=1,
        locked_at=stale_time,
        locked_by="worker-stale",
        started_at=stale_time,
    )

    claimed = backend.claim_next_task(worker_id="worker-fresh", lease_seconds=60, max_attempts=3)

    assert claimed is not None
    assert claimed["id"] == task["id"]
    assert claimed["status"] == "running"
    assert claimed["task_type"] == "rebuild"
    assert claimed["attempt_count"] == 2
    assert claimed["locked_by"] == "worker-fresh"


def test_sql_task_queue_backend_updates_heartbeat_for_current_worker(sql_task_queue_backend) -> None:
    backend, _, knowledge_base_id = sql_task_queue_backend
    task = backend.enqueue_task(
        knowledge_base_id=knowledge_base_id,
        document_ids=["doc_integration_003"],
        task_type="ingest",
        message="heartbeat queued",
    )
    claimed = backend.claim_next_task(worker_id="worker-heartbeat", lease_seconds=60, max_attempts=3)
    assert claimed is not None

    updated = backend.heartbeat_task(
        task["id"],
        worker_id="worker-heartbeat",
        message="heartbeat refreshed",
    )

    assert updated["id"] == task["id"]
    assert updated["message"] == "heartbeat refreshed"
    assert updated["locked_by"] == "worker-heartbeat"
    assert updated["locked_at"] is not None


def test_sql_task_queue_backend_does_not_claim_task_past_max_attempts(sql_task_queue_backend) -> None:
    backend, _, knowledge_base_id = sql_task_queue_backend
    task = backend.enqueue_task(
        knowledge_base_id=knowledge_base_id,
        document_ids=["doc_integration_004"],
        task_type="ingest",
        message="max attempts queued",
    )
    backend.update_task(
        task["id"],
        status="pending",
        attempt_count=3,
        locked_at=None,
        locked_by=None,
    )

    claimed = backend.claim_next_task(worker_id="worker-limit", lease_seconds=60, max_attempts=3)

    assert claimed is None


def test_sql_task_queue_backend_lists_tasks_and_reports_stats(sql_task_queue_backend) -> None:
    backend, _, knowledge_base_id = sql_task_queue_backend
    pending_task = backend.enqueue_task(
        knowledge_base_id=knowledge_base_id,
        document_ids=["doc_integration_005"],
        task_type="ingest",
        message="pending task",
    )
    stale_task = backend.enqueue_task(
        knowledge_base_id=knowledge_base_id,
        document_ids=["doc_integration_006"],
        task_type="rebuild",
        message="stale running task",
    )
    completed_task = backend.enqueue_task(
        knowledge_base_id=knowledge_base_id,
        document_ids=["doc_integration_007"],
        task_type="ingest",
        message="completed task",
    )

    stale_time = _now() - timedelta(minutes=10)
    backend.update_task(
        stale_task["id"],
        status="running",
        attempt_count=2,
        locked_at=stale_time,
        locked_by="worker-stale",
        started_at=stale_time,
    )
    backend.update_task(
        completed_task["id"],
        status="completed",
        attempt_count=1,
        finished_at=_now(),
        locked_at=None,
        locked_by=None,
    )

    listed = backend.list_tasks(knowledge_base_id=knowledge_base_id, limit=10)
    stats = backend.get_task_stats(knowledge_base_id=knowledge_base_id, lease_seconds=60)

    listed_ids = {item["id"] for item in listed}
    assert {pending_task["id"], stale_task["id"], completed_task["id"]}.issubset(listed_ids)
    assert stats == {
        "total": 3,
        "pending": 1,
        "running": 1,
        "completed": 1,
        "failed": 0,
        "retrying": 1,
        "stale_running": 1,
    }


def test_sql_task_queue_backend_persists_task_events(sql_task_queue_backend, monkeypatch: pytest.MonkeyPatch) -> None:
    backend, repository, knowledge_base_id = sql_task_queue_backend
    fixed_now = datetime(2026, 4, 12, 0, 0, 0)
    monkeypatch.setattr(repository, "_now", lambda: fixed_now)
    task = backend.enqueue_task(
        knowledge_base_id=knowledge_base_id,
        document_ids=["doc_integration_008"],
        task_type="ingest",
        message="event task queued",
    )

    claimed = backend.claim_next_task(worker_id="worker-events", lease_seconds=60, max_attempts=3)
    assert claimed is not None
    backend.heartbeat_task(task["id"], worker_id="worker-events", message="processing doc_integration_008")
    backend.create_task_event(
        task["id"],
        event_type="document_completed",
        message="document doc_integration_008 indexed",
        payload={"document_id": "doc_integration_008", "chunks": 2},
    )

    events = backend.list_task_events(task["id"], limit=10)

    assert [event["event_type"] for event in events] == [
        "created",
        "claimed",
        "heartbeat",
        "document_completed",
    ]
    assert events[1]["payload"] == {
        "worker_id": "worker-events",
        "attempt_count": 1,
        "stale_reclaimed": False,
    }
    assert events[2]["message"] == "processing doc_integration_008"


def test_sql_task_queue_backend_reports_monitoring_overview(sql_task_queue_backend) -> None:
    backend, _, knowledge_base_id = sql_task_queue_backend
    pending_task = backend.enqueue_task(
        knowledge_base_id=knowledge_base_id,
        document_ids=["doc_integration_009"],
        task_type="ingest",
        message="pending task for overview",
    )
    failed_task = backend.enqueue_task(
        knowledge_base_id=knowledge_base_id,
        document_ids=["doc_integration_010"],
        task_type="ingest",
        message="failed task for overview",
    )
    retry_task = backend.enqueue_task(
        knowledge_base_id=knowledge_base_id,
        document_ids=["doc_integration_011"],
        task_type="ingest",
        message="retry task for overview",
    )
    long_running_task = backend.enqueue_task(
        knowledge_base_id=knowledge_base_id,
        document_ids=["doc_integration_012"],
        task_type="rebuild",
        message="long running task for overview",
    )

    now = _now()
    old_started_at = now - timedelta(hours=2)
    backend.update_task(
        failed_task["id"],
        status="failed",
        message="failed recently",
        last_error="failed recently",
        finished_at=now,
        locked_at=None,
        locked_by=None,
    )
    backend.update_task(
        retry_task["id"],
        status="pending",
        attempt_count=2,
        locked_at=None,
        locked_by=None,
    )
    backend.create_task_event(
        retry_task["id"],
        event_type="retrying",
        message="retry scheduled after worker failure: transient",
        payload={"worker_id": "worker-retry", "attempt_count": 2},
    )
    backend.update_task(
        long_running_task["id"],
        status="running",
        attempt_count=1,
        started_at=old_started_at,
        locked_at=now,
        locked_by="worker-active",
        finished_at=None,
    )

    overview = backend.get_task_overview(
        knowledge_base_id=knowledge_base_id,
        lease_seconds=60,
        monitoring_window_hours=24,
        long_running_seconds=3600,
    )

    assert overview["total"] == 4
    assert overview["pending"] == 2
    assert overview["running"] == 1
    assert overview["failed"] == 1
    assert overview["retrying"] == 1
    assert overview["active_workers"] == 1
    assert overview["oldest_pending_age_seconds"] is not None
    assert overview["long_running"] == 1
    assert overview["recent_failed"] == 1
    assert overview["recent_retried"] == 1
    assert overview["knowledge_bases_with_recent_failures"] == [
        {"knowledge_base_id": knowledge_base_id, "task_count": 1}
    ]


def test_sql_task_queue_backend_reports_knowledge_base_trends(sql_task_queue_backend) -> None:
    backend, repository, knowledge_base_id = sql_task_queue_backend
    second_knowledge_base = repository.create_knowledge_base(
        name=f"Trend KB {uuid4().hex[:6]}",
        description="trend test knowledge base",
        subject="trend subject",
        domain="general",
    )
    try:
        current_failed = backend.enqueue_task(
            knowledge_base_id=knowledge_base_id,
            document_ids=["doc_trend_001"],
            task_type="ingest",
            message="current failed task",
        )
        previous_failed = backend.enqueue_task(
            knowledge_base_id=knowledge_base_id,
            document_ids=["doc_trend_002"],
            task_type="ingest",
            message="previous failed task",
        )
        current_retry = backend.enqueue_task(
            knowledge_base_id=second_knowledge_base["id"],
            document_ids=["doc_trend_003"],
            task_type="rebuild",
            message="current retry task",
        )

        now = _now()
        current_time = now - timedelta(hours=2)
        previous_time = now - timedelta(hours=30)
        backend.update_task(
            current_failed["id"],
            status="failed",
            last_error="current failure",
            updated_at=current_time,
            finished_at=current_time,
        )
        backend.update_task(
            previous_failed["id"],
            status="failed",
            last_error="previous failure",
            updated_at=previous_time,
            finished_at=previous_time,
        )
        backend.update_task(
            current_retry["id"],
            status="completed",
            updated_at=current_time,
            finished_at=current_time,
        )
        with SessionLocal() as session:
            current_failed_record = session.get(Task, current_failed["id"])
            previous_failed_record = session.get(Task, previous_failed["id"])
            current_retry_record = session.get(Task, current_retry["id"])
            assert current_failed_record is not None
            assert previous_failed_record is not None
            assert current_retry_record is not None
            current_failed_record.updated_at = current_time
            current_failed_record.finished_at = current_time
            previous_failed_record.updated_at = previous_time
            previous_failed_record.finished_at = previous_time
            current_retry_record.updated_at = current_time
            current_retry_record.finished_at = current_time
            session.commit()
        repository.create_task_event(
            current_retry["id"],
            event_type="retrying",
            message="retry scheduled",
            payload={"worker_id": "worker-trend"},
        )

        trends = backend.get_task_trends(monitoring_window_hours=24, limit=10)

        by_kb = {item["knowledge_base_id"]: item for item in trends}
        assert by_kb[knowledge_base_id]["recent_failed"] == 1
        assert by_kb[knowledge_base_id]["previous_failed"] == 1
        assert by_kb[knowledge_base_id]["failed_delta"] == 0
        assert by_kb[second_knowledge_base["id"]]["recent_retried"] == 1
        assert by_kb[second_knowledge_base["id"]]["retried_delta"] == 1
    finally:
        repository.delete_knowledge_base(second_knowledge_base["id"])
