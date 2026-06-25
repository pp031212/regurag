"""元数据仓储层。

本模块封装知识库、文档、会话、消息和后台任务的数据库访问。上层服务只拿
dict 结构，避免 API/service 层直接依赖 SQLAlchemy model，也方便后续从 MySQL
切到 PostgreSQL 时保持业务层调用不变。
"""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from ..core.exceptions import ConversationNotFoundError, DocumentNotFoundError, KnowledgeBaseNotFoundError, TaskNotFoundError
from ..models.conversation import Conversation
from ..db.session import SessionLocal
from ..models.document import Document
from ..models.knowledge_base import KnowledgeBase
from ..models.message import Message
from ..models.message_context import MessageContext
from ..models.task import Task
from ..models.task_event import TaskEvent


class MetadataRepository:
    """面向业务服务的元数据读写入口。"""

    def _now(self) -> datetime:
        # 数据库字段统一存 naive UTC，序列化时再补 Z，避免不同时区混进同一列。
        return datetime.now(UTC).replace(tzinfo=None)

    @staticmethod
    def _iso_or_none(value: datetime | None) -> str | None:
        if value is None:
            return None
        return value.isoformat() + "Z"

    @staticmethod
    def _kb_to_dict(record: KnowledgeBase) -> dict:
        return {
            "id": record.id,
            "name": record.name,
            "description": record.description,
            "subject": record.subject,
            "domain": record.domain,
            "status": record.status,
            "created_at": record.created_at.isoformat() + "Z",
            "updated_at": record.updated_at.isoformat() + "Z",
        }

    @staticmethod
    def _document_to_dict(record: Document) -> dict:
        return {
            "id": record.id,
            "knowledge_base_id": record.knowledge_base_id,
            "filename": record.filename,
            "content_type": record.content_type,
            "file_size": record.file_size,
            "content_hash": record.content_hash,
            "file_path": record.file_path,
            "status": record.status,
            "created_at": record.created_at.isoformat() + "Z",
            "updated_at": record.updated_at.isoformat() + "Z",
        }

    @staticmethod
    def _task_to_dict(record: Task) -> dict:
        return {
            "id": record.id,
            "knowledge_base_id": record.knowledge_base_id,
            "task_type": record.task_type,
            "document_ids": list(record.document_ids or []),
            "status": record.status,
            "message": record.message,
            "attempt_count": record.attempt_count,
            "last_error": record.last_error,
            "started_at": MetadataRepository._iso_or_none(record.started_at),
            "finished_at": MetadataRepository._iso_or_none(record.finished_at),
            "locked_at": MetadataRepository._iso_or_none(record.locked_at),
            "locked_by": record.locked_by,
            "created_at": record.created_at.isoformat() + "Z",
            "updated_at": record.updated_at.isoformat() + "Z",
        }

    @staticmethod
    def _task_event_to_dict(record: TaskEvent) -> dict:
        return {
            "id": record.id,
            "task_id": record.task_id,
            "event_type": record.event_type,
            "message": record.message,
            "payload": record.payload,
            "created_at": record.created_at.isoformat() + "Z",
        }

    @staticmethod
    def _new_task_event(
        *,
        task_id: str,
        event_type: str,
        message: str,
        payload: dict | None,
        created_at: datetime,
    ) -> TaskEvent:
        event_id = f"evt_{created_at.strftime('%Y%m%d%H%M%S%f')}_{uuid4().hex[:6]}"
        return TaskEvent(
            id=event_id,
            task_id=task_id,
            event_type=event_type,
            message=message,
            payload=payload,
            created_at=created_at,
        )

    def _next_task_event_created_at(self, session: Session, *, task_id: str, candidate: datetime) -> datetime:
        """为同一任务生成单调递增的事件时间，保证时间线展示稳定。"""
        normalized_candidate = candidate.replace(microsecond=0)
        latest_created_at = session.scalar(
            select(TaskEvent.created_at)
            .where(TaskEvent.task_id == task_id)
            .order_by(TaskEvent.created_at.desc(), TaskEvent.id.desc())
            .limit(1)
        )
        if latest_created_at is None or latest_created_at < normalized_candidate:
            return normalized_candidate
        return latest_created_at + timedelta(seconds=1)

    @staticmethod
    def _conversation_to_dict(record: Conversation) -> dict:
        return {
            "id": record.id,
            "default_knowledge_base_id": record.default_knowledge_base_id,
            "title": record.title,
            "created_at": record.created_at.isoformat() + "Z",
            "updated_at": record.updated_at.isoformat() + "Z",
        }

    @staticmethod
    def _message_to_dict(record: Message, context: MessageContext | None = None) -> dict:
        return {
            "id": record.id,
            "conversation_id": record.conversation_id,
            "knowledge_base_id": context.knowledge_base_id if context else None,
            "sequence": record.sequence,
            "role": record.role,
            "content": record.content,
            "citations": list(context.citations or []) if context else [],
            "debug": context.debug if context else None,
            "created_at": record.created_at.isoformat() + "Z",
        }

    def list_knowledge_bases(self) -> list[dict]:
        with SessionLocal() as session:
            records = session.scalars(select(KnowledgeBase).order_by(KnowledgeBase.created_at)).all()
            return [self._kb_to_dict(item) for item in records]

    def get_knowledge_base(self, knowledge_base_id: str) -> dict | None:
        with SessionLocal() as session:
            record = session.get(KnowledgeBase, knowledge_base_id)
            return self._kb_to_dict(record) if record else None

    def create_knowledge_base(self, name: str, description: str, subject: str, domain: str) -> dict:
        with SessionLocal() as session:
            now = self._now()
            record = KnowledgeBase(
                id=f"kb_{uuid4().hex[:8]}",
                name=name,
                description=description,
                subject=subject,
                domain=domain,
                status="empty",
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return self._kb_to_dict(record)

    def upsert_knowledge_base(self, record: dict) -> dict:
        with SessionLocal() as session:
            existing = session.get(KnowledgeBase, record["id"])
            if existing is None:
                existing = KnowledgeBase(
                    id=record["id"],
                    name=record["name"],
                    description=record["description"],
                    subject=record["subject"],
                    domain=str(record.get("domain") or "general"),
                    status=record["status"],
                    created_at=datetime.fromisoformat(record["created_at"].replace("Z", "")),
                    updated_at=datetime.fromisoformat(record["updated_at"].replace("Z", "")),
                )
                session.add(existing)
            else:
                existing.name = record["name"]
                existing.description = record["description"]
                existing.subject = record["subject"]
                existing.domain = str(record.get("domain") or "general")
                existing.status = record["status"]
                existing.created_at = datetime.fromisoformat(record["created_at"].replace("Z", ""))
                existing.updated_at = datetime.fromisoformat(record["updated_at"].replace("Z", ""))
            session.commit()
            session.refresh(existing)
            return self._kb_to_dict(existing)

    def update_knowledge_base(self, knowledge_base_id: str, **changes: object) -> dict:
        with SessionLocal() as session:
            record = session.get(KnowledgeBase, knowledge_base_id)
            if record is None:
                raise KnowledgeBaseNotFoundError()
            for key, value in changes.items():
                setattr(record, key, value)
            record.updated_at = self._now()
            session.commit()
            session.refresh(record)
            return self._kb_to_dict(record)

    def delete_knowledge_base(self, knowledge_base_id: str) -> None:
        with SessionLocal() as session:
            record = session.get(KnowledgeBase, knowledge_base_id)
            if record is None:
                raise KnowledgeBaseNotFoundError()
            # 会话历史可能仍要保留，所以删除 KB 时只解除引用，不级联删除消息。
            session.query(MessageContext).filter(MessageContext.knowledge_base_id == knowledge_base_id).update(
                {MessageContext.knowledge_base_id: None}
            )
            session.query(Conversation).filter(Conversation.default_knowledge_base_id == knowledge_base_id).update(
                {Conversation.default_knowledge_base_id: None}
            )
            session.query(Document).filter(Document.knowledge_base_id == knowledge_base_id).delete()
            session.query(Task).filter(Task.knowledge_base_id == knowledge_base_id).delete()
            session.delete(record)
            session.commit()

    def list_documents(self, knowledge_base_id: str | None = None) -> list[dict]:
        with SessionLocal() as session:
            stmt = select(Document)
            if knowledge_base_id is not None:
                stmt = stmt.where(Document.knowledge_base_id == knowledge_base_id)
            records = session.scalars(stmt.order_by(Document.created_at)).all()
            return [self._document_to_dict(item) for item in records]

    def get_document(self, document_id: str) -> dict | None:
        with SessionLocal() as session:
            record = session.get(Document, document_id)
            return self._document_to_dict(record) if record else None

    def find_duplicate_document(
        self,
        knowledge_base_id: str,
        filename: str,
        file_size: int,
        content_hash: str,
    ) -> dict | None:
        with SessionLocal() as session:
            stmt = (
                select(Document)
                .where(Document.knowledge_base_id == knowledge_base_id)
                .where(Document.filename == filename)
                .where(Document.file_size == file_size)
                .where(Document.content_hash == content_hash)
                .order_by(Document.created_at.desc())
            )
            record = session.scalars(stmt).first()
            return self._document_to_dict(record) if record else None

    def create_document(
        self,
        knowledge_base_id: str,
        filename: str,
        content_type: str,
        file_size: int,
        content_hash: str,
        file_path: str,
    ) -> dict:
        with SessionLocal() as session:
            now = self._now()
            record = Document(
                id=f"doc_{uuid4().hex[:8]}",
                knowledge_base_id=knowledge_base_id,
                filename=filename,
                content_type=content_type,
                file_size=file_size,
                content_hash=content_hash,
                file_path=file_path,
                status="uploaded",
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return self._document_to_dict(record)

    def update_document(self, document_id: str, **changes: object) -> dict:
        with SessionLocal() as session:
            record = session.get(Document, document_id)
            if record is None:
                raise DocumentNotFoundError()
            for key, value in changes.items():
                setattr(record, key, value)
            record.updated_at = self._now()
            session.commit()
            session.refresh(record)
            return self._document_to_dict(record)

    def delete_document(self, document_id: str) -> None:
        with SessionLocal() as session:
            record = session.get(Document, document_id)
            if record is None:
                raise DocumentNotFoundError()
            session.delete(record)
            session.commit()

    def create_task(self, knowledge_base_id: str, document_ids: list[str], message: str, task_type: str = "ingest") -> dict:
        with SessionLocal() as session:
            now = self._now()
            record = Task(
                id=f"task_{uuid4().hex[:8]}",
                knowledge_base_id=knowledge_base_id,
                task_type=task_type,
                document_ids=document_ids,
                status="pending",
                message=message,
                attempt_count=0,
                last_error=None,
                started_at=None,
                finished_at=None,
                locked_at=None,
                locked_by=None,
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            event_created_at = self._next_task_event_created_at(session, task_id=record.id, candidate=now)
            session.add(
                self._new_task_event(
                    task_id=record.id,
                    event_type="created",
                    message=message,
                    payload={
                        "task_type": task_type,
                        "knowledge_base_id": knowledge_base_id,
                        "document_ids": list(document_ids),
                    },
                    created_at=event_created_at,
                )
            )
            session.commit()
            session.refresh(record)
            return self._task_to_dict(record)

    def update_task(self, task_id: str, **changes: object) -> dict:
        with SessionLocal() as session:
            record = session.get(Task, task_id)
            if record is None:
                raise TaskNotFoundError()
            for key, value in changes.items():
                setattr(record, key, value)
            record.updated_at = self._now()
            session.commit()
            session.refresh(record)
            return self._task_to_dict(record)

    def get_task(self, task_id: str) -> dict | None:
        with SessionLocal() as session:
            record = session.get(Task, task_id)
            return self._task_to_dict(record) if record else None

    def create_task_event(
        self,
        task_id: str,
        *,
        event_type: str,
        message: str,
        payload: dict | None = None,
    ) -> dict:
        with SessionLocal() as session:
            task = session.get(Task, task_id)
            if task is None:
                raise TaskNotFoundError()
            # 事件时间要按任务内顺序推进，前端时间线才能稳定排序。
            event_created_at = self._next_task_event_created_at(session, task_id=task_id, candidate=self._now())
            event = self._new_task_event(
                task_id=task_id,
                event_type=event_type,
                message=message,
                payload=payload,
                created_at=event_created_at,
            )
            session.add(event)
            session.commit()
            session.refresh(event)
            return self._task_event_to_dict(event)

    def list_task_events(self, task_id: str, *, limit: int = 100) -> list[dict]:
        with SessionLocal() as session:
            stmt = (
                select(TaskEvent)
                .where(TaskEvent.task_id == task_id)
                .order_by(TaskEvent.created_at.asc(), TaskEvent.id.asc())
                .limit(limit)
            )
            records = session.scalars(stmt).all()
            return [self._task_event_to_dict(item) for item in records]

    def list_tasks(
        self,
        *,
        knowledge_base_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        with SessionLocal() as session:
            stmt = select(Task)
            if knowledge_base_id is not None:
                stmt = stmt.where(Task.knowledge_base_id == knowledge_base_id)
            if status is not None:
                stmt = stmt.where(Task.status == status)
            stmt = stmt.order_by(Task.updated_at.desc(), Task.created_at.desc()).limit(limit)
            records = session.scalars(stmt).all()
            return [self._task_to_dict(item) for item in records]

    def get_task_stats(self, *, knowledge_base_id: str | None = None, lease_seconds: int) -> dict[str, int]:
        now = self._now()
        # locked_at 早于租约窗口的 running 任务视为 stale，可被 worker 重新认领。
        stale_before = (now.replace(tzinfo=UTC) - timedelta(seconds=lease_seconds)).replace(tzinfo=None)
        with SessionLocal() as session:
            filters = []
            if knowledge_base_id is not None:
                filters.append(Task.knowledge_base_id == knowledge_base_id)

            def _count(*extra_filters) -> int:
                stmt = select(func.count()).select_from(Task)
                for clause in [*filters, *extra_filters]:
                    stmt = stmt.where(clause)
                return int(session.scalar(stmt) or 0)

            return {
                "total": _count(),
                "pending": _count(Task.status == "pending"),
                "running": _count(Task.status == "running"),
                "completed": _count(Task.status == "completed"),
                "failed": _count(Task.status == "failed"),
                "retrying": _count(Task.attempt_count > 1),
                "stale_running": _count(Task.status == "running", Task.locked_at.is_not(None), Task.locked_at < stale_before),
            }

    def get_task_overview(
        self,
        *,
        knowledge_base_id: str | None = None,
        lease_seconds: int,
        monitoring_window_hours: int,
        long_running_seconds: int,
    ) -> dict[str, object]:
        now = self._now()
        # 监控页同时看当前租约状态、近期失败/重试和长耗时任务，便于判断 worker 是否卡住。
        stale_before = (now.replace(tzinfo=UTC) - timedelta(seconds=lease_seconds)).replace(tzinfo=None)
        recent_window_start = (now.replace(tzinfo=UTC) - timedelta(hours=monitoring_window_hours)).replace(tzinfo=None)
        long_running_before = (now.replace(tzinfo=UTC) - timedelta(seconds=long_running_seconds)).replace(tzinfo=None)
        with SessionLocal() as session:
            filters = []
            if knowledge_base_id is not None:
                filters.append(Task.knowledge_base_id == knowledge_base_id)

            def _count(*extra_filters: object) -> int:
                stmt = select(func.count()).select_from(Task)
                for clause in [*filters, *extra_filters]:
                    stmt = stmt.where(clause)
                return int(session.scalar(stmt) or 0)

            oldest_pending_created_at = session.scalar(
                select(func.min(Task.created_at)).where(*filters, Task.status == "pending")
            )
            active_workers = int(
                session.scalar(
                    select(func.count(func.distinct(Task.locked_by))).select_from(Task).where(
                        *filters,
                        Task.status == "running",
                        Task.locked_by.is_not(None),
                        Task.locked_at.is_not(None),
                        Task.locked_at >= stale_before,
                    )
                )
                or 0
            )
            recent_retried = int(
                session.scalar(
                    select(func.count())
                    .select_from(TaskEvent)
                    .join(Task, Task.id == TaskEvent.task_id)
                    .where(
                        *filters,
                        TaskEvent.event_type == "retrying",
                        TaskEvent.created_at >= recent_window_start,
                    )
                )
                or 0
            )
            failure_rows = session.execute(
                select(Task.knowledge_base_id, func.count().label("task_count"))
                .where(
                    *filters,
                    Task.status == "failed",
                    Task.updated_at >= recent_window_start,
                )
                .group_by(Task.knowledge_base_id)
                .order_by(func.count().desc(), Task.knowledge_base_id.asc())
                .limit(5)
            ).all()
            oldest_pending_age_seconds = (
                max(0, int((now - oldest_pending_created_at).total_seconds()))
                if oldest_pending_created_at is not None
                else None
            )

            return {
                "total": _count(),
                "pending": _count(Task.status == "pending"),
                "running": _count(Task.status == "running"),
                "completed": _count(Task.status == "completed"),
                "failed": _count(Task.status == "failed"),
                "retrying": _count(Task.attempt_count > 1),
                "stale_running": _count(Task.status == "running", Task.locked_at.is_not(None), Task.locked_at < stale_before),
                "active_workers": active_workers,
                "oldest_pending_age_seconds": oldest_pending_age_seconds,
                "long_running": _count(
                    Task.status == "running",
                    Task.started_at.is_not(None),
                    Task.started_at < long_running_before,
                    or_(Task.locked_at.is_(None), Task.locked_at >= stale_before),
                ),
                "recent_failed": _count(Task.status == "failed", Task.updated_at >= recent_window_start),
                "recent_retried": recent_retried,
                "knowledge_bases_with_recent_failures": [
                    {"knowledge_base_id": row.knowledge_base_id, "task_count": int(row.task_count)}
                    for row in failure_rows
                ],
            }

    def get_task_trends(
        self,
        *,
        monitoring_window_hours: int,
        limit: int = 20,
    ) -> list[dict]:
        now = self._now()
        current_window_start = (now.replace(tzinfo=UTC) - timedelta(hours=monitoring_window_hours)).replace(tzinfo=None)
        previous_window_start = (
            now.replace(tzinfo=UTC) - timedelta(hours=monitoring_window_hours * 2)
        ).replace(tzinfo=None)

        with SessionLocal() as session:
            knowledge_bases = session.scalars(
                select(KnowledgeBase).order_by(KnowledgeBase.updated_at.desc(), KnowledgeBase.created_at.desc())
            ).all()
            items: list[dict] = []

            for knowledge_base in knowledge_bases:
                knowledge_base_id = knowledge_base.id

                def _task_count(*extra_filters: object) -> int:
                    stmt = select(func.count()).select_from(Task).where(Task.knowledge_base_id == knowledge_base_id)
                    for clause in extra_filters:
                        stmt = stmt.where(clause)
                    return int(session.scalar(stmt) or 0)

                def _event_count(*extra_filters: object) -> int:
                    stmt = (
                        select(func.count())
                        .select_from(TaskEvent)
                        .join(Task, Task.id == TaskEvent.task_id)
                        .where(Task.knowledge_base_id == knowledge_base_id)
                    )
                    for clause in extra_filters:
                        stmt = stmt.where(clause)
                    return int(session.scalar(stmt) or 0)

                recent_failed = _task_count(Task.status == "failed", Task.updated_at >= current_window_start)
                previous_failed = _task_count(
                    Task.status == "failed",
                    Task.updated_at >= previous_window_start,
                    Task.updated_at < current_window_start,
                )
                recent_completed = _task_count(Task.status == "completed", Task.updated_at >= current_window_start)
                previous_completed = _task_count(
                    Task.status == "completed",
                    Task.updated_at >= previous_window_start,
                    Task.updated_at < current_window_start,
                )
                recent_retried = _event_count(
                    TaskEvent.event_type == "retrying",
                    TaskEvent.created_at >= current_window_start,
                )
                previous_retried = _event_count(
                    TaskEvent.event_type == "retrying",
                    TaskEvent.created_at >= previous_window_start,
                    TaskEvent.created_at < current_window_start,
                )
                updated_at = session.scalar(
                    select(func.max(Task.updated_at)).where(Task.knowledge_base_id == knowledge_base_id)
                )

                items.append(
                    {
                        "knowledge_base_id": knowledge_base_id,
                        "knowledge_base_name": knowledge_base.name,
                        "pending": _task_count(Task.status == "pending"),
                        "running": _task_count(Task.status == "running"),
                        "recent_failed": recent_failed,
                        "previous_failed": previous_failed,
                        "failed_delta": recent_failed - previous_failed,
                        "recent_retried": recent_retried,
                        "previous_retried": previous_retried,
                        "retried_delta": recent_retried - previous_retried,
                        "recent_completed": recent_completed,
                        "previous_completed": previous_completed,
                        "completed_delta": recent_completed - previous_completed,
                        "updated_at": self._iso_or_none(updated_at),
                    }
                )

            items.sort(
                key=lambda item: (
                    item["failed_delta"],
                    item["retried_delta"],
                    item["pending"] + item["running"],
                    item["recent_failed"],
                    item["recent_retried"],
                ),
                reverse=True,
            )
            return items[:limit]

    def claim_next_task(self, *, worker_id: str, lease_seconds: int, max_attempts: int) -> dict | None:
        now = self._now()
        stale_before = (now.replace(tzinfo=UTC) - timedelta(seconds=lease_seconds)).replace(tzinfo=None)
        with SessionLocal() as session:
            # pending 任务和租约过期的 running 任务都可以被认领；skip_locked 支持多 worker 并发抢任务。
            stmt = (
                select(Task)
                .where(Task.attempt_count < max_attempts)
                .where(
                    or_(
                        Task.status == "pending",
                        and_(Task.status == "running", Task.locked_at.is_not(None), Task.locked_at < stale_before),
                    )
                )
                .order_by(Task.created_at)
                .with_for_update(skip_locked=True)
            )
            with session.begin():
                record = session.scalars(stmt).first()
                if record is None:
                    return None
                stale_reclaimed = record.status == "running"
                # 认领本身和 claimed 事件在同一个事务里提交，避免任务状态和事件时间线不一致。
                record.status = "running"
                record.attempt_count += 1
                record.started_at = record.started_at or now
                record.finished_at = None
                record.locked_at = now
                record.locked_by = worker_id
                record.last_error = None
                record.updated_at = now
                event_created_at = self._next_task_event_created_at(session, task_id=record.id, candidate=now)
                session.add(
                    self._new_task_event(
                        task_id=record.id,
                        event_type="claimed",
                        message=f"claimed by {worker_id}",
                        payload={
                            "worker_id": worker_id,
                            "attempt_count": record.attempt_count,
                            "stale_reclaimed": stale_reclaimed,
                        },
                        created_at=event_created_at,
                    )
                )
                session.flush()
                return self._task_to_dict(record)

    def heartbeat_task(self, task_id: str, *, worker_id: str, message: str | None = None) -> dict:
        with SessionLocal() as session:
            record = session.get(Task, task_id)
            if record is None:
                raise TaskNotFoundError()
            if record.locked_by != worker_id:
                # 非持有租约的 worker 不能续租；这里复用 NotFound，避免暴露任务锁状态。
                raise TaskNotFoundError()
            record.locked_at = self._now()
            record.updated_at = record.locked_at
            if message is not None:
                record.message = message
                event_created_at = self._next_task_event_created_at(
                    session,
                    task_id=record.id,
                    candidate=record.locked_at,
                )
                session.add(
                    self._new_task_event(
                        task_id=record.id,
                        event_type="heartbeat",
                        message=message,
                        payload={"worker_id": worker_id},
                        created_at=event_created_at,
                    )
                )
            session.commit()
            session.refresh(record)
            return self._task_to_dict(record)

    def list_conversations(self, default_knowledge_base_id: str | None = None) -> list[dict]:
        with SessionLocal() as session:
            stmt = select(Conversation)
            if default_knowledge_base_id is not None:
                stmt = stmt.where(Conversation.default_knowledge_base_id == default_knowledge_base_id)
            stmt = stmt.order_by(Conversation.updated_at.desc(), Conversation.created_at.desc())
            records = session.scalars(stmt).all()
            return [self._conversation_to_dict(item) for item in records]

    def get_conversation(self, conversation_id: str) -> dict | None:
        with SessionLocal() as session:
            record = session.get(Conversation, conversation_id)
            return self._conversation_to_dict(record) if record else None

    def create_conversation(self, title: str, default_knowledge_base_id: str | None = None) -> dict:
        with SessionLocal() as session:
            now = self._now()
            record = Conversation(
                id=f"conv_{uuid4().hex[:8]}",
                default_knowledge_base_id=default_knowledge_base_id,
                title=title,
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return self._conversation_to_dict(record)

    def update_conversation(self, conversation_id: str, **changes: object) -> dict:
        with SessionLocal() as session:
            record = session.get(Conversation, conversation_id)
            if record is None:
                raise ConversationNotFoundError()
            for key, value in changes.items():
                setattr(record, key, value)
            record.updated_at = self._now()
            session.commit()
            session.refresh(record)
            return self._conversation_to_dict(record)

    def delete_conversation(self, conversation_id: str) -> None:
        with SessionLocal() as session:
            record = session.get(Conversation, conversation_id)
            if record is None:
                raise ConversationNotFoundError()
            message_ids = session.scalars(
                select(Message.id).where(Message.conversation_id == conversation_id)
            ).all()
            if message_ids:
                session.query(MessageContext).filter(MessageContext.message_id.in_(message_ids)).delete(
                    synchronize_session=False
                )
            session.query(Message).filter(Message.conversation_id == conversation_id).delete()
            session.delete(record)
            session.commit()

    def list_messages(self, conversation_id: str) -> list[dict]:
        with SessionLocal() as session:
            records = session.execute(
                select(Message, MessageContext)
                .outerjoin(MessageContext, MessageContext.message_id == Message.id)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.sequence, Message.created_at)
            ).all()
            return [self._message_to_dict(message, context) for message, context in records]

    def create_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> dict:
        with SessionLocal() as session:
            now = self._now()
            next_sequence = (
                session.scalar(
                    select(func.max(Message.sequence)).where(Message.conversation_id == conversation_id)
                )
                or 0
            ) + 1
            record = Message(
                id=f"msg_{uuid4().hex[:8]}",
                conversation_id=conversation_id,
                sequence=next_sequence,
                role=role,
                content=content,
                created_at=now,
            )
            session.add(record)
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                raise ConversationNotFoundError()
            conversation.updated_at = now
            session.commit()
            session.refresh(record)
            return self._message_to_dict(record)

    def create_message_context(
        self,
        message_id: str,
        knowledge_base_id: str | None = None,
        citations: list[dict] | None = None,
        debug: dict | None = None,
    ) -> dict:
        with SessionLocal() as session:
            now = self._now()
            record = MessageContext(
                id=f"ctx_{uuid4().hex[:8]}",
                message_id=message_id,
                knowledge_base_id=knowledge_base_id,
                citations=citations or [],
                debug=debug,
                created_at=now,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return {
                "id": record.id,
                "message_id": record.message_id,
                "knowledge_base_id": record.knowledge_base_id,
                "citations": list(record.citations or []),
                "debug": record.debug,
                "created_at": record.created_at.isoformat() + "Z",
            }


_repository: MetadataRepository | None = None


def get_metadata_repository() -> MetadataRepository:
    global _repository
    if _repository is None:
        _repository = MetadataRepository()
    return _repository
