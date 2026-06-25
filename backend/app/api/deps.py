"""FastAPI 依赖注入入口。

这里把运行时对象、仓储、服务和任务队列统一组装起来。测试或应用启动时如果挂载了
AppRuntime，就优先复用 runtime 中的单例；否则退回默认工厂。
"""

from fastapi import Depends, Request

from ..services.conversation_service import ConversationService
from ..services.ingest_service import IngestService, TaskService
from ..services.knowledge_base_service import DocumentService, KnowledgeBaseService
from ..services.rag_service import RAGPipelineRegistry, RAGService, get_default_pipeline_registry
from ..repositories.metadata_repository import MetadataRepository, get_metadata_repository
from ..runtime import AppRuntime
from ..task_queue import TaskQueueBackend, get_task_queue_backend


def get_app_runtime_dep(request: Request) -> AppRuntime | None:
    return getattr(request.app.state, "runtime", None)


def get_metadata_repository_dep(
    runtime: AppRuntime | None = Depends(get_app_runtime_dep),
) -> MetadataRepository:
    """获取元数据仓储；runtime 存在时复用同一个实例。"""
    if runtime is not None:
        return runtime.repository
    return get_metadata_repository()


def get_pipeline_registry_dep(
    runtime: AppRuntime | None = Depends(get_app_runtime_dep),
) -> RAGPipelineRegistry:
    if runtime is not None:
        return runtime.pipeline_registry
    return get_default_pipeline_registry()


def get_knowledge_base_service(
    repository: MetadataRepository = Depends(get_metadata_repository_dep),
    pipeline_registry: RAGPipelineRegistry = Depends(get_pipeline_registry_dep),
) -> KnowledgeBaseService:
    return KnowledgeBaseService(repository=repository, pipeline_registry=pipeline_registry)


def get_document_service(
    repository: MetadataRepository = Depends(get_metadata_repository_dep),
    pipeline_registry: RAGPipelineRegistry = Depends(get_pipeline_registry_dep),
) -> DocumentService:
    return DocumentService(repository=repository, pipeline_registry=pipeline_registry)


def get_task_queue(
    repository: MetadataRepository = Depends(get_metadata_repository_dep),
    runtime: AppRuntime | None = Depends(get_app_runtime_dep),
) -> TaskQueueBackend:
    """获取任务队列后端，确保 API 和 worker 使用同一套队列配置。"""
    if runtime is not None:
        return runtime.task_queue
    return get_task_queue_backend(repository=repository)


def get_ingest_service(
    repository: MetadataRepository = Depends(get_metadata_repository_dep),
    task_queue: TaskQueueBackend = Depends(get_task_queue),
    pipeline_registry: RAGPipelineRegistry = Depends(get_pipeline_registry_dep),
) -> IngestService:
    return IngestService(repository=repository, task_queue=task_queue, pipeline_registry=pipeline_registry)


def get_task_service(
    task_queue: TaskQueueBackend = Depends(get_task_queue),
) -> TaskService:
    return TaskService(task_queue=task_queue)


def get_rag_service(
    repository: MetadataRepository = Depends(get_metadata_repository_dep),
    pipeline_registry: RAGPipelineRegistry = Depends(get_pipeline_registry_dep),
) -> RAGService:
    return RAGService(repository=repository, pipeline_registry=pipeline_registry)


def get_conversation_service(
    repository: MetadataRepository = Depends(get_metadata_repository_dep),
) -> ConversationService:
    return ConversationService(repository=repository)
