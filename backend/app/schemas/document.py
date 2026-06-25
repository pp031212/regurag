from datetime import datetime

from pydantic import BaseModel


class DocumentResponse(BaseModel):
    id: str
    knowledge_base_id: str
    filename: str
    content_type: str
    file_size: int
    status: str
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]
    total: int


class DocumentDeleteResponse(BaseModel):
    id: str
    deleted: bool
