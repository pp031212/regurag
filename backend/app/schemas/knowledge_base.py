from datetime import datetime

from pydantic import BaseModel, Field


class KnowledgeBaseCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)
    subject: str | None = Field(default=None, max_length=100)
    domain: str | None = Field(default=None, max_length=64)


class KnowledgeBaseDomainOptionResponse(BaseModel):
    value: str
    label: str
    description: str


class KnowledgeBaseDomainOptionsResponse(BaseModel):
    items: list[KnowledgeBaseDomainOptionResponse]
    default_domain: str


class KnowledgeBaseResponse(BaseModel):
    id: str
    name: str
    description: str
    subject: str
    domain: str
    status: str
    is_default: bool = False
    created_at: datetime
    updated_at: datetime


class KnowledgeBaseListResponse(BaseModel):
    items: list[KnowledgeBaseResponse]
    total: int


class KnowledgeBaseDeleteResponse(BaseModel):
    id: str
    deleted: bool
