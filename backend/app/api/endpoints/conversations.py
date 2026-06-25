"""会话接口。

会话独立于当前前端选中的知识库；每条 assistant 消息的 message_context 会记录当轮实际使用的知识库。
"""

from fastapi import APIRouter, Depends, status

from ..deps import get_conversation_service
from ...schemas.chat import ChatCitation, ChatDebug
from ...schemas.conversation import (
    ConversationCreateRequest,
    ConversationDeleteResponse,
    ConversationListResponse,
    ConversationResponse,
    MessageListResponse,
    MessageResponse,
)
from ...services.conversation_service import ConversationService

router = APIRouter()


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    knowledge_base_id: str | None = None,
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationListResponse:
    items = service.list_conversations(knowledge_base_id)
    return ConversationListResponse(items=[ConversationResponse(**item) for item in items], total=len(items))


@router.get("/knowledge-bases/{knowledge_base_id}/conversations", response_model=ConversationListResponse)
async def list_conversations_by_knowledge_base(
    knowledge_base_id: str,
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationListResponse:
    items = service.list_conversations(knowledge_base_id)
    return ConversationListResponse(items=[ConversationResponse(**item) for item in items], total=len(items))


@router.post("/conversations", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    payload: ConversationCreateRequest,
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationResponse:
    record = service.create_conversation(
        default_knowledge_base_id=payload.default_knowledge_base_id,
        title=payload.title,
    )
    return ConversationResponse(**record)


@router.post("/knowledge-bases/{knowledge_base_id}/conversations", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation_by_knowledge_base(
    knowledge_base_id: str,
    payload: ConversationCreateRequest,
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationResponse:
    record = service.create_conversation(
        default_knowledge_base_id=knowledge_base_id,
        title=payload.title,
    )
    return ConversationResponse(**record)


@router.delete("/conversations/{conversation_id}", response_model=ConversationDeleteResponse)
async def delete_conversation(
    conversation_id: str,
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationDeleteResponse:
    record = service.delete_conversation(conversation_id)
    return ConversationDeleteResponse(**record)


@router.get("/conversations/{conversation_id}/messages", response_model=MessageListResponse)
async def list_messages(
    conversation_id: str,
    service: ConversationService = Depends(get_conversation_service),
) -> MessageListResponse:
    """返回会话消息，并把 citations/debug 从存储结构转换成前端响应模型。"""
    items = service.list_messages(conversation_id)
    return MessageListResponse(
        items=[
            MessageResponse(
                **{
                    **item,
                    "citations": [ChatCitation(**citation) for citation in item.get("citations", [])],
                    "debug": ChatDebug(**item["debug"]) if item.get("debug") else None,
                }
            )
            for item in items
        ],
        total=len(items),
    )
