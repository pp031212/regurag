from types import SimpleNamespace

from app.workers.task_worker import TaskWorker


class StubRepository:
    def __init__(self) -> None:
        self.updated: list[tuple[str, dict]] = []

    def _now(self):  # pragma: no cover - simple stub shape
        return None

    def update_task(self, task_id: str, **changes):
        self.updated.append((task_id, changes))
        return {"id": task_id, **changes}


class StubTaskQueue:
    def __init__(self) -> None:
        self.updated: list[tuple[str, dict]] = []
        self.events: list[tuple[str, str, str, dict | None]] = []

    def update_task(self, task_id: str, **changes):
        self.updated.append((task_id, changes))
        return {"id": task_id, **changes}

    def create_task_event(self, task_id: str, *, event_type: str, message: str, payload: dict | None = None):
        self.events.append((task_id, event_type, message, payload))
        return {"task_id": task_id, "event_type": event_type, "message": message, "payload": payload}


class StubIngestService:
    def __init__(self, claimed_task=None, *, should_raise: bool = False, settings=None) -> None:
        self.claimed_task = claimed_task
        self.should_raise = should_raise
        self.claim_worker_ids: list[str] = []
        self.run_calls: list[str] = []
        self.repository = StubRepository()
        self.task_queue = StubTaskQueue()
        self.settings = settings

    def claim_next_task(self, worker_id: str):
        self.claim_worker_ids.append(worker_id)
        return self.claimed_task

    def run_ingest_task(self, task_id: str):
        self.run_calls.append(task_id)
        if self.should_raise:
            raise RuntimeError("boom")
        return {"id": task_id, "status": "completed"}


def test_worker_run_once_returns_false_when_no_task() -> None:
    service = StubIngestService(claimed_task=None)
    worker = TaskWorker(worker_id="worker-test", ingest_service=service, poll_interval_seconds=0.01)

    result = worker.run_once()

    assert result is False
    assert service.claim_worker_ids == ["worker-test"]
    assert service.run_calls == []


def test_worker_run_once_processes_claimed_task() -> None:
    service = StubIngestService(claimed_task={"id": "task_001", "attempt_count": 1, "task_type": "ingest"})
    worker = TaskWorker(worker_id="worker-test", ingest_service=service, poll_interval_seconds=0.01)

    result = worker.run_once()

    assert result is True
    assert service.run_calls == ["task_001"]


def test_worker_requeues_task_when_execution_raises_before_max_attempts() -> None:
    service = StubIngestService(
        claimed_task={"id": "task_001", "attempt_count": 1, "task_type": "ingest"},
        should_raise=True,
    )
    worker = TaskWorker(worker_id="worker-test", ingest_service=service, poll_interval_seconds=0.01)

    result = worker.run_once()

    assert result is True
    assert service.run_calls == ["task_001"]
    assert service.task_queue.events[0][1] == "retrying"
    assert service.task_queue.updated[0][0] == "task_001"
    assert service.task_queue.updated[0][1]["status"] == "pending"
    assert service.task_queue.updated[0][1]["finished_at"] is None
    assert service.task_queue.updated[0][1]["locked_by"] is None


def test_worker_marks_task_failed_after_max_attempts() -> None:
    service = StubIngestService(
        claimed_task={"id": "task_001", "attempt_count": 3, "task_type": "ingest"},
        should_raise=True,
    )
    worker = TaskWorker(worker_id="worker-test", ingest_service=service, poll_interval_seconds=0.01)

    result = worker.run_once()

    assert result is True
    assert service.run_calls == ["task_001"]
    assert service.task_queue.events[0][1] == "failed"
    assert service.task_queue.updated[0][0] == "task_001"
    assert service.task_queue.updated[0][1]["status"] == "failed"
    assert service.task_queue.updated[0][1]["locked_by"] is None


def test_worker_injects_first_attempt_failure_for_configured_document_ids() -> None:
    service = StubIngestService(
        claimed_task={
            "id": "task_001",
            "attempt_count": 1,
            "task_type": "ingest",
            "document_ids": ["doc_001"],
        },
        settings=SimpleNamespace(
            task_worker_max_attempts=3,
            parsed_task_worker_inject_fail_on_first_attempt_document_ids=["doc_001"],
        ),
    )
    worker = TaskWorker(worker_id="worker-test", ingest_service=service, poll_interval_seconds=0.01)

    result = worker.run_once()

    assert result is True
    assert service.run_calls == []
    assert service.task_queue.events[0][1] == "retrying"
    assert "injected worker failure on first attempt" in service.task_queue.events[0][2]
    assert service.task_queue.updated[0][1]["status"] == "pending"
