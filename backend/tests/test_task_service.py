from types import SimpleNamespace

from app.services.ingest_service import TaskService


class StubTaskQueue:
    def __init__(self, overview: dict[str, object]) -> None:
        self.overview = overview
        self.calls: list[tuple[str | None, int, int, int]] = []

    def get_task_overview(
        self,
        *,
        knowledge_base_id: str | None = None,
        lease_seconds: int,
        monitoring_window_hours: int,
        long_running_seconds: int,
    ) -> dict[str, object]:
        self.calls.append((knowledge_base_id, lease_seconds, monitoring_window_hours, long_running_seconds))
        return dict(self.overview)


def test_task_service_builds_alerts_from_overview() -> None:
    task_queue = StubTaskQueue(
        overview={
            "total": 6,
            "pending": 2,
            "running": 2,
            "completed": 1,
            "failed": 1,
            "retrying": 4,
            "stale_running": 1,
            "active_workers": 0,
            "oldest_pending_age_seconds": 600,
            "long_running": 1,
            "recent_failed": 3,
            "recent_retried": 5,
            "knowledge_bases_with_recent_failures": [
                {"knowledge_base_id": "kb_001", "task_count": 2},
                {"knowledge_base_id": "kb_002", "task_count": 1},
            ],
        }
    )
    service = TaskService(task_queue=task_queue)
    service.settings = SimpleNamespace(
        task_worker_lease_seconds=1800,
        task_monitor_window_hours=24,
        task_monitor_long_running_seconds=3600,
        task_monitor_recent_failure_threshold=3,
        task_monitor_recent_retry_threshold=5,
    )

    alerts = service.get_task_alerts()

    assert task_queue.calls == [(None, 1800, 24, 3600)]
    assert alerts["total"] == 5
    codes = [item["code"] for item in alerts["items"]]
    assert codes == [
        "PENDING_WITHOUT_ACTIVE_WORKERS",
        "STALE_RUNNING_TASKS",
        "LONG_RUNNING_TASKS",
        "RECENT_FAILURE_SPIKE",
        "RECENT_RETRY_SPIKE",
    ]
    assert alerts["items"][3]["details"]["knowledge_bases"] == [
        {"knowledge_base_id": "kb_001", "task_count": 2},
        {"knowledge_base_id": "kb_002", "task_count": 1},
    ]
