from __future__ import annotations

from ..core.config import Settings, get_settings
from ..repositories.metadata_repository import MetadataRepository
from .base import TaskQueueBackend
from .redis_backend import RedisTaskQueueBackend
from .sql_backend import SqlTaskQueueBackend


def get_task_queue_backend(
    *,
    repository: MetadataRepository | None = None,
    settings: Settings | None = None,
    redis_client=None,
) -> TaskQueueBackend:
    resolved_settings = settings or get_settings()
    backend = resolved_settings.normalized_task_queue_backend
    if backend in {"mysql", "sql"}:
        return SqlTaskQueueBackend(repository=repository)
    if backend == "redis":
        return RedisTaskQueueBackend(
            repository=repository,
            redis_url=resolved_settings.resolved_redis_url,
            pending_list_key=resolved_settings.task_queue_redis_pending_list_key,
            marker_prefix=resolved_settings.task_queue_redis_marker_prefix,
            redis_client=redis_client,
        )
    raise ValueError(f"Unsupported task queue backend: {resolved_settings.task_queue_backend}")
