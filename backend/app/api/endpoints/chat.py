"""聊天接口。

endpoint 层只负责把 service 返回的 dict 转成 Pydantic response，并处理 SSE 编码；
路由、短路、检索和持久化都在 RAGService 中完成。
"""

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ..deps import get_rag_service
from ...schemas.chat import ChatCitation, ChatDebug, ChatQueryRequest, ChatQueryResponse
from ...services.rag_service import RAGService

router = APIRouter()

_CHUNK_DEBUG_KEYS = (
    "retrieved_chunks",
    "mmr_selected_chunks",
    "reranked_chunks",
    "final_context_chunks",
)


def _build_debug_payload(raw_debug: dict[str, object], include_chunks: bool) -> ChatDebug:
    """按前端开关裁剪 debug chunk，避免默认响应过大。"""
    debug_payload = dict(raw_debug)
    if not include_chunks:
        for key in _CHUNK_DEBUG_KEYS:
            debug_payload.pop(key, None)
    return ChatDebug(**debug_payload)


def _encode_sse_event(event: str, data: dict[str, object]) -> str:
    """把事件名和 JSON 数据编码成浏览器 EventSource/fetch 可解析的 SSE block。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _build_stream_end_payload(result: dict[str, object], include_chunks: bool) -> dict[str, object]:
    debug_payload = result.get("debug")
    response = ChatQueryResponse(
        answer=result["answer"],
        answer_source=str(result.get("answer_source") or "legacy_pipeline"),
        conversation_id=result["conversation_id"],
        knowledge_base_id=result["knowledge_base_id"],
        knowledge_base_name=result.get("knowledge_base_name"),
        auto_routed=bool(result.get("auto_routed")),
        citations=[ChatCitation(**citation) for citation in result["citations"]],
        debug=_build_debug_payload(debug_payload, include_chunks) if isinstance(debug_payload, dict) else None,
    )
    return response.model_dump()


@router.post("/query", response_model=ChatQueryResponse)
async def query_chat(
    payload: ChatQueryRequest,
    service: RAGService = Depends(get_rag_service),
) -> ChatQueryResponse:
    """非流式问答入口。"""
    result = await service.query(payload)
    return ChatQueryResponse(
        answer=result["answer"],
        answer_source=str(result.get("answer_source") or "legacy_pipeline"),
        conversation_id=result["conversation_id"],
        knowledge_base_id=result["knowledge_base_id"],
        knowledge_base_name=result.get("knowledge_base_name"),
        auto_routed=bool(result.get("auto_routed")),
        citations=[ChatCitation(**citation) for citation in result["citations"]],
        debug=_build_debug_payload(result["debug"], payload.debug_chunks) if payload.debug else None,
    )


@router.post("/stream")
async def stream_chat(
    payload: ChatQueryRequest,
    service: RAGService = Depends(get_rag_service),
) -> StreamingResponse:
    def event_stream():
        try:
            for item in service.stream_query(payload):
                # service 输出统一事件结构，endpoint 只在 end 事件上做 response model 转换。
                event = str(item["event"])
                data = dict(item["data"])
                if event == "end":
                    data = _build_stream_end_payload(data, payload.debug_chunks)
                yield _encode_sse_event(event, data)
        except Exception as exc:
            yield _encode_sse_event(
                "error",
                {
                    "code": getattr(exc, "code", "STREAM_ERROR"),
                    "message": getattr(exc, "message", str(exc)),
                    "details": getattr(exc, "details", {}),
                },
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")
