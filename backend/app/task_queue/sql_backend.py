from ..repositories.metadata_repository import MetadataRepository, get_metadata_repository


class SqlTaskQueueBackend:
    """基于 SQL 元数据库的任务队列封装，MySQL/PostgreSQL 共用。"""

    def __init__(self, repository: MetadataRepository | None = None) -> None:
        self.repository = repository or get_metadata_repository()

    def enqueue_task(
        self,
        *,
        knowledge_base_id: str,
        document_ids: list[str],
        task_type: str,
        message: str,
    ) -> dict:
        return self.repository.create_task(
            knowledge_base_id=knowledge_base_id,
            document_ids=document_ids,
            message=message,
            task_type=task_type,
        )

    def get_task(self, task_id: str) -> dict | None:
        return self.repository.get_task(task_id)

    def create_task_event(
        self,
        task_id: str,
        *,
        event_type: str,
        message: str,
        payload: dict | None = None,
    ) -> dict:
        return self.repository.create_task_event(
            task_id,
            event_type=event_type,
            message=message,
            payload=payload,
        )

    def list_task_events(self, task_id: str, *, limit: int = 100) -> list[dict]:
        return self.repository.list_task_events(task_id, limit=limit)

    def list_tasks(
        self,
        *,
        knowledge_base_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        return self.repository.list_tasks(
            knowledge_base_id=knowledge_base_id,
            status=status,
            limit=limit,
        )

    def get_task_stats(self, *, knowledge_base_id: str | None = None, lease_seconds: int) -> dict[str, int]:
        return self.repository.get_task_stats(
            knowledge_base_id=knowledge_base_id,
            lease_seconds=lease_seconds,
        )

    def get_task_overview(
        self,
        *,
        knowledge_base_id: str | None = None,
        lease_seconds: int,
        monitoring_window_hours: int,
        long_running_seconds: int,
    ) -> dict[str, object]:
        return self.repository.get_task_overview(
            knowledge_base_id=knowledge_base_id,
            lease_seconds=lease_seconds,
            monitoring_window_hours=monitoring_window_hours,
            long_running_seconds=long_running_seconds,
        )

    def get_task_trends(
        self,
        *,
        monitoring_window_hours: int,
        limit: int = 20,
    ) -> list[dict]:
        return self.repository.get_task_trends(
            monitoring_window_hours=monitoring_window_hours,
            limit=limit,
        )

    def update_task(self, task_id: str, **changes: object) -> dict:
        return self.repository.update_task(task_id, **changes)

    def claim_next_task(self, *, worker_id: str, lease_seconds: int, max_attempts: int) -> dict | None:
        return self.repository.claim_next_task(
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            max_attempts=max_attempts,
        )

    def heartbeat_task(self, task_id: str, *, worker_id: str, message: str | None = None) -> dict:
        return self.repository.heartbeat_task(task_id, worker_id=worker_id, message=message)


def get_task_queue_backend(repository: MetadataRepository | None = None) -> SqlTaskQueueBackend:
    return SqlTaskQueueBackend(repository=repository)
