"""后台任务观测接口。

这些接口服务于任务观测页和文档页告警：列表看当前任务，overview/alerts/trends 看 worker
是否积压、卡住或近期失败增多。
"""

from fastapi import APIRouter, Depends, Query

from ..deps import get_task_service
from ...schemas.task import (
    IngestTaskListResponse,
    IngestTaskResponse,
    KnowledgeBaseTaskTrendListResponse,
    TaskAlertListResponse,
    TaskEventListResponse,
    TaskOverviewResponse,
    TaskStatsResponse,
)
from ...services.ingest_service import TaskService

router = APIRouter(prefix="/tasks")


@router.get("", response_model=IngestTaskListResponse)
async def list_tasks(
    knowledge_base_id: str | None = None,
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    service: TaskService = Depends(get_task_service),
) -> IngestTaskListResponse:
    payload = service.list_tasks(
        knowledge_base_id=knowledge_base_id,
        status=status,
        limit=limit,
    )
    return IngestTaskListResponse(**payload)


@router.get("/stats", response_model=TaskStatsResponse)
async def get_task_stats(
    knowledge_base_id: str | None = None,
    service: TaskService = Depends(get_task_service),
) -> TaskStatsResponse:
    payload = service.get_task_stats(knowledge_base_id=knowledge_base_id)
    return TaskStatsResponse(**payload)


@router.get("/overview", response_model=TaskOverviewResponse)
async def get_task_overview(
    knowledge_base_id: str | None = None,
    service: TaskService = Depends(get_task_service),
) -> TaskOverviewResponse:
    """返回任务总览，包含 stale running、active workers 和近期失败等监控指标。"""
    payload = service.get_task_overview(knowledge_base_id=knowledge_base_id)
    return TaskOverviewResponse(**payload)


@router.get("/alerts", response_model=TaskAlertListResponse)
async def get_task_alerts(
    knowledge_base_id: str | None = None,
    service: TaskService = Depends(get_task_service),
) -> TaskAlertListResponse:
    """把总览指标转换成前端可直接展示的告警项。"""
    payload = service.get_task_alerts(knowledge_base_id=knowledge_base_id)
    return TaskAlertListResponse(**payload)


@router.get("/trends", response_model=KnowledgeBaseTaskTrendListResponse)
async def get_task_trends(
    limit: int = Query(default=20, ge=1, le=100),
    service: TaskService = Depends(get_task_service),
) -> KnowledgeBaseTaskTrendListResponse:
    payload = service.get_task_trends(limit=limit)
    return KnowledgeBaseTaskTrendListResponse(**payload)


@router.get("/{task_id}", response_model=IngestTaskResponse)
async def get_task(
    task_id: str,
    service: TaskService = Depends(get_task_service),
) -> IngestTaskResponse:
    record = service.get_task(task_id)
    return IngestTaskResponse(**record)


@router.get("/{task_id}/events", response_model=TaskEventListResponse)
async def list_task_events(
    task_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    service: TaskService = Depends(get_task_service),
) -> TaskEventListResponse:
    """返回单任务事件时间线，用于定位任务卡在哪个阶段。"""
    payload = service.get_task_events(task_id, limit=limit)
    return TaskEventListResponse(**payload)
