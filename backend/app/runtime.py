from dataclasses import dataclass

from .repositories.metadata_repository import MetadataRepository
from .services.rag_service import RAGPipelineRegistry
from .task_queue import TaskQueueBackend


@dataclass(frozen=True)
class AppRuntime:
    repository: MetadataRepository
    task_queue: TaskQueueBackend
    pipeline_registry: RAGPipelineRegistry
