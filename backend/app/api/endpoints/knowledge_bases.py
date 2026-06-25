"""知识库接口。

endpoint 层只处理 HTTP schema 转换；知识库删除会同时清理索引，导入/重建会创建后台任务。
"""

from fastapi import APIRouter, Depends, status

from ..deps import get_ingest_service, get_knowledge_base_service
from ...schemas.knowledge_base import (
    KnowledgeBaseCreateRequest,
    KnowledgeBaseDeleteResponse,
    KnowledgeBaseDomainOptionResponse,
    KnowledgeBaseDomainOptionsResponse,
    KnowledgeBaseListResponse,
    KnowledgeBaseResponse,
)
from ...schemas.task import IngestTaskCreateRequest, IngestTaskResponse
from ...services.ingest_service import IngestService
from ...services.knowledge_base_service import KnowledgeBaseService

router = APIRouter(prefix="/knowledge-bases")


@router.get("", response_model=KnowledgeBaseListResponse)
async def list_knowledge_bases(
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> KnowledgeBaseListResponse:
    items = service.list_knowledge_bases()
    return KnowledgeBaseListResponse(items=[KnowledgeBaseResponse(**item) for item in items], total=len(items))


@router.get("/domains", response_model=KnowledgeBaseDomainOptionsResponse)
async def list_knowledge_base_domains(
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> KnowledgeBaseDomainOptionsResponse:
    payload = service.get_domain_options()
    return KnowledgeBaseDomainOptionsResponse(
        items=[KnowledgeBaseDomainOptionResponse(**item) for item in list(payload["items"])],
        default_domain=str(payload["default_domain"]),
    )


@router.get("/{knowledge_base_id}", response_model=KnowledgeBaseResponse)
async def get_knowledge_base(
    knowledge_base_id: str,
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> KnowledgeBaseResponse:
    record = service.get_knowledge_base(knowledge_base_id)
    return KnowledgeBaseResponse(**record)


@router.post("", response_model=KnowledgeBaseResponse, status_code=status.HTTP_201_CREATED)
async def create_knowledge_base(
    payload: KnowledgeBaseCreateRequest,
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> KnowledgeBaseResponse:
    record = service.create_knowledge_base(payload.name, payload.description, payload.subject, payload.domain)
    return KnowledgeBaseResponse(**record)


@router.delete("/{knowledge_base_id}", response_model=KnowledgeBaseDeleteResponse)
async def delete_knowledge_base(
    knowledge_base_id: str,
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> KnowledgeBaseDeleteResponse:
    """删除知识库元数据和索引；历史会话会保留，但解除对该知识库的引用。"""
    record = service.delete_knowledge_base(knowledge_base_id)
    return KnowledgeBaseDeleteResponse(**record)


@router.post("/{knowledge_base_id}/ingest", response_model=IngestTaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_knowledge_base(
    knowledge_base_id: str,
    payload: IngestTaskCreateRequest,
    service: IngestService = Depends(get_ingest_service),
) -> IngestTaskResponse:
    """创建知识库入库任务，实际处理由后台 worker 完成。"""
    record = service.create_ingest_task(knowledge_base_id, payload.document_ids)
    return IngestTaskResponse(**record)


@router.post("/{knowledge_base_id}/rebuild", response_model=IngestTaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def rebuild_knowledge_base(
    knowledge_base_id: str,
    payload: IngestTaskCreateRequest,
    service: IngestService = Depends(get_ingest_service),
) -> IngestTaskResponse:
    """创建知识库重建任务；document_ids 为空时表示全量重建。"""
    record = service.create_ingest_task(
        knowledge_base_id,
        payload.document_ids,
        task_type="rebuild",
        message="rebuild task created",
    )
    return IngestTaskResponse(**record)
