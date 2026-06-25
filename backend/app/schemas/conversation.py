from datetime import datetime

from pydantic import BaseModel, Field

from .chat import ChatCitation, ChatDebug


class ConversationCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    default_knowledge_base_id: str | None = Field(default=None, min_length=1)


class ConversationResponse(BaseModel):
    id: str
    default_knowledge_base_id: str | None = None
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationListResponse(BaseModel):
    items: list[ConversationResponse]
    total: int


class ConversationDeleteResponse(BaseModel):
    id: str
    deleted: bool


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    knowledge_base_id: str | None = None
    sequence: int
    role: str
    content: str
    citations: list[ChatCitation] = Field(default_factory=list)
    debug: ChatDebug | None = None
    created_at: datetime


class MessageListResponse(BaseModel):
    items: list[MessageResponse]
    total: int
