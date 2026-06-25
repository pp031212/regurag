from datetime import datetime

from pydantic import BaseModel, Field


class IngestTaskCreateRequest(BaseModel):
    document_ids: list[str] = Field(default_factory=list)


class IngestTaskResponse(BaseModel):
    id: str
    knowledge_base_id: str
    task_type: str
    document_ids: list[str]
    status: str
    message: str
    attempt_count: int = 0
    last_error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    locked_at: datetime | None = None
    locked_by: str | None = None
    created_at: datetime
    updated_at: datetime


class TaskEventResponse(BaseModel):
    id: str
    task_id: str
    event_type: str
    message: str
    payload: dict | None = None
    created_at: datetime


class TaskEventListResponse(BaseModel):
    items: list[TaskEventResponse]
    total: int


class IngestTaskListResponse(BaseModel):
    items: list[IngestTaskResponse]
    total: int


class TaskStatsResponse(BaseModel):
    total: int
    pending: int
    running: int
    completed: int
    failed: int
    retrying: int
    stale_running: int


class KnowledgeBaseFailureSummaryResponse(BaseModel):
    knowledge_base_id: str
    task_count: int


class TaskOverviewResponse(BaseModel):
    total: int
    pending: int
    running: int
    completed: int
    failed: int
    retrying: int
    stale_running: int
    active_workers: int
    oldest_pending_age_seconds: int | None = None
    long_running: int
    recent_failed: int
    recent_retried: int
    knowledge_bases_with_recent_failures: list[KnowledgeBaseFailureSummaryResponse]


class TaskAlertResponse(BaseModel):
    code: str
    severity: str
    message: str
    count: int
    details: dict


class TaskAlertListResponse(BaseModel):
    items: list[TaskAlertResponse]
    total: int


class KnowledgeBaseTaskTrendResponse(BaseModel):
    knowledge_base_id: str
    knowledge_base_name: str
    pending: int
    running: int
    recent_failed: int
    previous_failed: int
    failed_delta: int
    recent_retried: int
    previous_retried: int
    retried_delta: int
    recent_completed: int
    previous_completed: int
    completed_delta: int
    updated_at: datetime | None = None


class KnowledgeBaseTaskTrendListResponse(BaseModel):
    items: list[KnowledgeBaseTaskTrendResponse]
    total: int
