from fastapi import APIRouter

from .endpoints import chat, conversations, documents, health, knowledge_bases, tasks

# 所有业务路由在这里集中注册，main.py 只需要挂载一个 api_router。
api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(conversations.router, tags=["conversations"])
api_router.include_router(knowledge_bases.router, tags=["knowledge-bases"])
api_router.include_router(documents.router, tags=["documents"])
api_router.include_router(tasks.router, tags=["tasks"])
