"""会话服务。

会话本身只是聊天容器，真正每轮问答使用的知识库记录在 message_context 中。
因此切换前端当前知识库不会自动切换或删除已有会话。
"""

from ..core.exceptions import ConversationNotFoundError, KnowledgeBaseNotFoundError
from ..repositories.metadata_repository import MetadataRepository, get_metadata_repository


class ConversationService:
    """会话列表、创建、删除和消息读取。"""

    def __init__(self, *, repository: MetadataRepository | None = None) -> None:
        self.repository = repository or get_metadata_repository()

    def _build_title(self, title: str | None, fallback_query: str | None = None) -> str:
        """会话标题最多保留 50 字，避免侧边栏被长问题撑开。"""
        raw = (title or fallback_query or "新对话").strip()
        return raw[:50] or "新对话"

    def list_conversations(self, default_knowledge_base_id: str | None = None) -> list[dict]:
        """可按默认知识库筛选会话，但会话并不强绑定某个知识库。"""
        if default_knowledge_base_id is not None:
            knowledge_base = self.repository.get_knowledge_base(default_knowledge_base_id)
            if knowledge_base is None:
                raise KnowledgeBaseNotFoundError()
        return self.repository.list_conversations(default_knowledge_base_id)

    def create_conversation(
        self,
        default_knowledge_base_id: str | None = None,
        title: str | None = None,
        fallback_query: str | None = None,
    ) -> dict:
        """创建会话；默认知识库只是偏好设置，不代表历史消息都会使用它。"""
        if default_knowledge_base_id is not None:
            knowledge_base = self.repository.get_knowledge_base(default_knowledge_base_id)
            if knowledge_base is None:
                raise KnowledgeBaseNotFoundError()
        return self.repository.create_conversation(
            title=self._build_title(title, fallback_query),
            default_knowledge_base_id=default_knowledge_base_id,
        )

    def get_conversation(self, conversation_id: str) -> dict:
        record = self.repository.get_conversation(conversation_id)
        if record is None:
            raise ConversationNotFoundError()
        return record

    def delete_conversation(self, conversation_id: str) -> dict:
        """删除会话和该会话下的消息/上下文。"""
        self.get_conversation(conversation_id)
        self.repository.delete_conversation(conversation_id)
        return {"id": conversation_id, "deleted": True}

    def list_messages(self, conversation_id: str) -> list[dict]:
        """读取会话消息前先确认会话存在，避免返回孤儿消息。"""
        conversation = self.repository.get_conversation(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError()
        return self.repository.list_messages(conversation_id)
