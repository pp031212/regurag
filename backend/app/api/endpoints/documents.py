"""文档资源接口。

上传/删除由 DocumentService 处理；导入和重建只创建后台任务，不在 HTTP 请求里同步跑重活。
"""

from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from ..deps import get_document_service, get_ingest_service
from ...schemas.document import DocumentDeleteResponse, DocumentListResponse, DocumentResponse
from ...schemas.task import IngestTaskResponse
from ...services.ingest_service import IngestService
from ...services.knowledge_base_service import DocumentService

router = APIRouter()


@router.post("/documents/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    knowledge_base_id: str = Form(...),
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    """保存上传文件和元数据，文件此时还未进入向量索引。"""
    record = await service.upload_document(knowledge_base_id, file)
    return DocumentResponse(**record)


@router.get("/knowledge-bases/{knowledge_base_id}/documents", response_model=DocumentListResponse)
async def list_documents(
    knowledge_base_id: str,
    service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    items = service.list_documents(knowledge_base_id)
    return DocumentListResponse(items=[DocumentResponse(**item) for item in items], total=len(items))


@router.delete("/documents/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
) -> DocumentDeleteResponse:
    record = service.delete_document(document_id)
    return DocumentDeleteResponse(**record)


@router.post("/documents/{document_id}/rebuild", response_model=IngestTaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def rebuild_document(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service),
    ingest_service: IngestService = Depends(get_ingest_service),
) -> IngestTaskResponse:
    """为单个文档创建重建任务，实际删除旧索引和重新入库由 worker 完成。"""
    document = document_service.get_document(document_id)
    record = ingest_service.create_ingest_task(
        document["knowledge_base_id"],
        [document_id],
        task_type="rebuild",
        message="document rebuild task created",
    )
    return IngestTaskResponse(**record)
