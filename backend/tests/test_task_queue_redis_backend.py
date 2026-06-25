from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from app.task_queue import RedisTaskQueueBackend, SqlTaskQueueBackend, get_task_queue_backend


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.lists: dict[str, list[str]] = {}

    def set(self, key: str, value: str, nx: bool = False):
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    def rpush(self, key: str, value: str) -> int:
        self.lists.setdefault(key, []).append(value)
        return len(self.lists[key])

    def lpop(self, key: str):
        items = self.lists.get(key) or []
        if not items:
            return None
        value = items.pop(0)
        if not items:
            self.lists.pop(key, None)
        return value

    def delete(self, key: str) -> int:
        removed = 0
        if key in self.values:
            self.values.pop(key, None)
            removed += 1
        if key in self.lists:
            self.lists.pop(key, None)
            removed += 1
        return removed


class FakeRepository:
    def __init__(self) -> None:
        self.tasks: dict[str, dict] = {}
        self.created_events: list[dict] = []
        self.claimed_from_fallback: list[dict] = []
        self.create_counter = 0

    def create_task(self, knowledge_base_id: str, document_ids: list[str], message: str, task_type: str = "ingest") -> dict:
        self.create_counter += 1
        task_id = f"task_{self.create_counter:03d}"
        task = {
            "id": task_id,
            "knowledge_base_id": knowledge_base_id,
            "task_type": task_type,
            "document_ids": list(document_ids),
            "status": "pending",
            "message": message,
            "attempt_count": 0,
            "last_error": None,
            "started_at": None,
            "finished_at": None,
            "locked_at": None,
            "locked_by": None,
        }
        self.tasks[task_id] = task
        return dict(task)

    def get_task(self, task_id: str) -> dict | None:
        task = self.tasks.get(task_id)
        return dict(task) if task is not None else None

    def create_task_event(self, task_id: str, *, event_type: str, message: str, payload: dict | None = None) -> dict:
        event = {
            "task_id": task_id,
            "event_type": event_type,
            "message": message,
            "payload": payload or {},
        }
        self.created_events.append(event)
        return dict(event)

    def list_task_events(self, task_id: str, *, limit: int = 100) -> list[dict]:
        return [item for item in self.created_events if item["task_id"] == task_id][:limit]

    def list_tasks(self, *, knowledge_base_id: str | None = None, status: str | None = None, limit: int = 50) -> list[dict]:
        items = list(self.tasks.values())
        if knowledge_base_id is not None:
            items = [item for item in items if item["knowledge_base_id"] == knowledge_base_id]
        if status is not None:
            items = [item for item in items if item["status"] == status]
        return [dict(item) for item in items[:limit]]

    def get_task_stats(self, *, knowledge_base_id: str | None = None, lease_seconds: int) -> dict[str, int]:
        return {"pending": 0, "running": 0, "completed": 0, "failed": 0, "stale_running": 0}

    def get_task_overview(
        self,
        *,
        knowledge_base_id: str | None = None,
        lease_seconds: int,
        monitoring_window_hours: int,
        long_running_seconds: int,
    ) -> dict[str, object]:
        return {"pending": 0, "running": 0, "stale_running": 0, "active_workers": 0, "long_running": 0}

    def get_task_trends(self, *, monitoring_window_hours: int, limit: int = 20) -> list[dict]:
        return []

    def update_task(self, task_id: str, **changes: object) -> dict:
        task = self.tasks[task_id]
        task.update(changes)
        return dict(task)

    def claim_next_task(self, *, worker_id: str, lease_seconds: int, max_attempts: int) -> dict | None:
        for task in self.tasks.values():
            if task["status"] == "pending" and int(task["attempt_count"] or 0) < max_attempts:
                task["status"] = "running"
                task["attempt_count"] = int(task["attempt_count"] or 0) + 1
                task["locked_by"] = worker_id
                self.claimed_from_fallback.append(dict(task))
                return dict(task)
        return None

    def heartbeat_task(self, task_id: str, *, worker_id: str, message: str | None = None) -> dict:
        task = self.tasks[task_id]
        task["locked_by"] = worker_id
        if message is not None:
            task["message"] = message
        return dict(task)


def test_redis_task_queue_backend_enqueues_and_claims_task() -> None:
    repository = FakeRepository()
    redis_client = FakeRedis()
    backend = RedisTaskQueueBackend(
        repository=repository,
        redis_url="redis://example",
        pending_list_key="queue:pending",
        marker_prefix="queue:marker",
        redis_client=redis_client,
    )

    task = backend.enqueue_task(
        knowledge_base_id="kb_001",
        document_ids=["doc_1"],
        task_type="ingest",
        message="ingest task created",
    )

    claimed = backend.claim_next_task(worker_id="worker-1", lease_seconds=1800, max_attempts=3)

    assert task["id"] == "task_001"
    assert claimed is not None
    assert claimed["id"] == task["id"]
    assert claimed["status"] == "running"
    assert claimed["attempt_count"] == 1
    assert claimed["locked_by"] == "worker-1"
    assert repository.created_events[-1]["event_type"] == "claimed"
    assert redis_client.lists == {}


def test_redis_task_queue_backend_requeues_pending_updates_without_duplicates() -> None:
    repository = FakeRepository()
    redis_client = FakeRedis()
    backend = RedisTaskQueueBackend(
        repository=repository,
        redis_url="redis://example",
        pending_list_key="queue:pending",
        marker_prefix="queue:marker",
        redis_client=redis_client,
    )
    task = backend.enqueue_task(
        knowledge_base_id="kb_001",
        document_ids=["doc_1"],
        task_type="ingest",
        message="ingest task created",
    )

    backend.update_task(task["id"], status="pending", message="retry")
    backend.update_task(task["id"], status="pending", message="retry again")

    assert redis_client.lists["queue:pending"] == [task["id"]]


def test_redis_task_queue_backend_falls_back_to_repository_claim_for_orphan_pending_task() -> None:
    repository = FakeRepository()
    task = repository.create_task("kb_001", ["doc_1"], "created")
    redis_client = FakeRedis()
    backend = RedisTaskQueueBackend(
        repository=repository,
        redis_url="redis://example",
        pending_list_key="queue:pending",
        marker_prefix="queue:marker",
        redis_client=redis_client,
    )

    claimed = backend.claim_next_task(worker_id="worker-2", lease_seconds=1800, max_attempts=3)

    assert claimed is not None
    assert claimed["id"] == task["id"]
    assert repository.claimed_from_fallback[-1]["id"] == task["id"]


def test_redis_task_queue_backend_reclaims_stale_running_task_from_queue() -> None:
    repository = FakeRepository()
    redis_client = FakeRedis()
    backend = RedisTaskQueueBackend(
        repository=repository,
        redis_url="redis://example",
        pending_list_key="queue:pending",
        marker_prefix="queue:marker",
        redis_client=redis_client,
    )
    task = backend.enqueue_task(
        knowledge_base_id="kb_001",
        document_ids=["doc_1"],
        task_type="ingest",
        message="ingest task created",
    )
    stale_time = (datetime.now(UTC) - timedelta(hours=2)).replace(tzinfo=None).isoformat() + "Z"
    repository.update_task(
        task["id"],
        status="running",
        attempt_count=1,
        locked_at=stale_time,
        locked_by="old-worker",
    )
    backend.update_task(task["id"], status="pending", locked_at=stale_time, locked_by=None)

    claimed = backend.claim_next_task(worker_id="worker-3", lease_seconds=60, max_attempts=3)

    assert claimed is not None
    assert claimed["locked_by"] == "worker-3"
    assert claimed["attempt_count"] == 2


def test_redis_task_queue_backend_reclaim_parses_string_started_at() -> None:
    repository = FakeRepository()
    redis_client = FakeRedis()
    backend = RedisTaskQueueBackend(
        repository=repository,
        redis_url="redis://example",
        pending_list_key="queue:pending",
        marker_prefix="queue:marker",
        redis_client=redis_client,
    )
    task = backend.enqueue_task(
        knowledge_base_id="kb_001",
        document_ids=["doc_1"],
        task_type="ingest",
        message="task with string started_at",
    )
    stale_time = (datetime.now(UTC) - timedelta(hours=2)).replace(tzinfo=None).isoformat() + "Z"
    repository.update_task(
        task["id"],
        status="running",
        attempt_count=1,
        started_at=stale_time,
        locked_at=stale_time,
        locked_by="old-worker",
    )
    backend.update_task(task["id"], status="pending", started_at=stale_time, locked_at=stale_time, locked_by=None)

    claimed = backend.claim_next_task(worker_id="worker-4", lease_seconds=60, max_attempts=3)

    assert claimed is not None
    assert claimed["attempt_count"] == 2
    assert claimed["started_at"] == datetime.fromisoformat(stale_time.removesuffix("Z"))


def test_get_task_queue_backend_selects_redis_backend() -> None:
    backend = get_task_queue_backend(
        repository=FakeRepository(),
        settings=SimpleNamespace(
            normalized_task_queue_backend="redis",
            task_queue_backend="redis",
            resolved_redis_url="redis://example",
            task_queue_redis_pending_list_key="queue:pending",
            task_queue_redis_marker_prefix="queue:marker",
        ),
        redis_client=FakeRedis(),
    )

    assert isinstance(backend, RedisTaskQueueBackend)


def test_get_task_queue_backend_selects_sql_backend_for_neutral_alias() -> None:
    backend = get_task_queue_backend(
        repository=FakeRepository(),
        settings=SimpleNamespace(
            normalized_task_queue_backend="sql",
            task_queue_backend="sql",
            resolved_redis_url="redis://example",
            task_queue_redis_pending_list_key="queue:pending",
            task_queue_redis_marker_prefix="queue:marker",
        ),
    )

    assert isinstance(backend, SqlTaskQueueBackend)


def test_get_task_queue_backend_keeps_mysql_alias_compatible() -> None:
    backend = get_task_queue_backend(
        repository=FakeRepository(),
        settings=SimpleNamespace(
            normalized_task_queue_backend="mysql",
            task_queue_backend="mysql",
            resolved_redis_url="redis://example",
            task_queue_redis_pending_list_key="queue:pending",
            task_queue_redis_marker_prefix="queue:marker",
        ),
    )

    assert isinstance(backend, SqlTaskQueueBackend)
