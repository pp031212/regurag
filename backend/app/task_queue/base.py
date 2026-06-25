from typing import Protocol


class TaskQueueBackend(Protocol):
    def enqueue_task(
        self,
        *,
        knowledge_base_id: str,
        document_ids: list[str],
        task_type: str,
        message: str,
    ) -> dict: ...

    def get_task(self, task_id: str) -> dict | None: ...

    def create_task_event(
        self,
        task_id: str,
        *,
        event_type: str,
        message: str,
        payload: dict | None = None,
    ) -> dict: ...

    def list_task_events(self, task_id: str, *, limit: int = 100) -> list[dict]: ...

    def list_tasks(self, *, knowledge_base_id: str | None = None, status: str | None = None, limit: int = 50) -> list[dict]: ...

    def get_task_stats(self, *, knowledge_base_id: str | None = None, lease_seconds: int) -> dict[str, int]: ...

    def get_task_overview(
        self,
        *,
        knowledge_base_id: str | None = None,
        lease_seconds: int,
        monitoring_window_hours: int,
        long_running_seconds: int,
    ) -> dict[str, object]: ...

    def get_task_trends(
        self,
        *,
        monitoring_window_hours: int,
        limit: int = 20,
    ) -> list[dict]: ...

    def update_task(self, task_id: str, **changes: object) -> dict: ...

    def claim_next_task(self, *, worker_id: str, lease_seconds: int, max_attempts: int) -> dict | None: ...

    def heartbeat_task(self, task_id: str, *, worker_id: str, message: str | None = None) -> dict: ...
