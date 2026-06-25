from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from redis import Redis

from ..repositories.metadata_repository import MetadataRepository, get_metadata_repository


class RedisTaskQueueBackend:
    def __init__(
        self,
        *,
        repository: MetadataRepository | None = None,
        redis_url: str,
        pending_list_key: str,
        marker_prefix: str,
        redis_client: Redis | None = None,
    ) -> None:
        self.repository = repository or get_metadata_repository()
        self.redis = redis_client or Redis.from_url(redis_url, decode_responses=True)
        self.pending_list_key = pending_list_key
        self.marker_prefix = marker_prefix.rstrip(":")

    def enqueue_task(
        self,
        *,
        knowledge_base_id: str,
        document_ids: list[str],
        task_type: str,
        message: str,
    ) -> dict:
        task = self.repository.create_task(
            knowledge_base_id=knowledge_base_id,
            document_ids=document_ids,
            message=message,
            task_type=task_type,
        )
        self._queue_task(task["id"])
        return task

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
        task = self.repository.update_task(task_id, **changes)
        status = str(task.get("status") or "")
        if status == "pending":
            self._queue_task(task_id)
        else:
            self._clear_task_marker(task_id)
        return task

    def claim_next_task(self, *, worker_id: str, lease_seconds: int, max_attempts: int) -> dict | None:
        while True:
            task_id = self.redis.lpop(self.pending_list_key)
            if task_id is None:
                break
            self._clear_task_marker(str(task_id))
            claimed = self._claim_task_by_id(
                str(task_id),
                worker_id=worker_id,
                lease_seconds=lease_seconds,
                max_attempts=max_attempts,
            )
            if claimed is not None:
                return claimed

        claimed = self.repository.claim_next_task(
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            max_attempts=max_attempts,
        )
        if claimed is not None:
            self._clear_task_marker(str(claimed["id"]))
        return claimed

    def heartbeat_task(self, task_id: str, *, worker_id: str, message: str | None = None) -> dict:
        return self.repository.heartbeat_task(task_id, worker_id=worker_id, message=message)

    def _queue_task(self, task_id: str) -> None:
        marker_key = self._marker_key(task_id)
        queued = self.redis.set(marker_key, "1", nx=True)
        if queued:
            self.redis.rpush(self.pending_list_key, task_id)

    def _clear_task_marker(self, task_id: str) -> None:
        self.redis.delete(self._marker_key(task_id))

    def _marker_key(self, task_id: str) -> str:
        return f"{self.marker_prefix}:{task_id}"

    def _claim_task_by_id(
        self,
        task_id: str,
        *,
        worker_id: str,
        lease_seconds: int,
        max_attempts: int,
    ) -> dict | None:
        task = self.repository.get_task(task_id)
        if task is None:
            return None

        attempt_count = int(task.get("attempt_count") or 0)
        if attempt_count >= max_attempts:
            return None

        now = self._now()
        stale_before = (now.replace(tzinfo=UTC) - timedelta(seconds=lease_seconds)).replace(tzinfo=None)
        status = str(task.get("status") or "")
        locked_at = self._parse_iso_datetime(task.get("locked_at"))
        started_at = self._parse_task_datetime(task.get("started_at")) or now
        stale_reclaimed = status == "running" and locked_at is not None and locked_at < stale_before
        if status != "pending" and not stale_reclaimed:
            return None

        claimed = self.repository.update_task(
            task_id,
            status="running",
            attempt_count=attempt_count + 1,
            started_at=started_at,
            finished_at=None,
            locked_at=now,
            locked_by=worker_id,
            last_error=None,
        )
        self.repository.create_task_event(
            task_id,
            event_type="claimed",
            message=f"claimed by {worker_id}",
            payload={
                "worker_id": worker_id,
                "attempt_count": int(claimed.get("attempt_count") or 0),
                "stale_reclaimed": stale_reclaimed,
                "queue_backend": "redis",
            },
        )
        return claimed

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC).replace(tzinfo=None)

    @staticmethod
    def _parse_iso_datetime(value: object) -> datetime | None:
        if not isinstance(value, str) or not value:
            return None
        return datetime.fromisoformat(value.removesuffix("Z"))

    @classmethod
    def _parse_task_datetime(cls, value: object) -> datetime | None:
        if isinstance(value, datetime):
            return value
        return cls._parse_iso_datetime(value)
