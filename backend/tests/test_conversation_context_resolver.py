from app.services.conversation_context_resolver import ConversationContextResolver


class FakeRepository:
    def __init__(self, messages):
        self.messages = messages

    def list_messages(self, conversation_id: str):
        return list(self.messages.get(conversation_id, []))


class FakeRewriter:
    def rewrite_with_history(self, user_query: str, history_messages: list[dict[str, str]]) -> str:
        return f"改写:{user_query}" if history_messages else user_query


def test_context_resolver_only_reads_current_conversation_and_kb_messages() -> None:
    repository = FakeRepository(
        {
            "conv_001": [
                {"id": "m1", "role": "user", "content": "试用期多久", "knowledge_base_id": "kb_001"},
                {"id": "m2", "role": "assistant", "content": "最长六个月", "knowledge_base_id": "kb_001"},
                {"id": "m3", "role": "user", "content": "别的知识库消息", "knowledge_base_id": "kb_999"},
            ],
            "conv_002": [
                {"id": "m4", "role": "user", "content": "其他会话消息", "knowledge_base_id": "kb_001"},
            ],
        }
    )
    resolver = ConversationContextResolver(repository, FakeRewriter())

    result = resolver.resolve(
        conversation_id="conv_001",
        knowledge_base_id="kb_001",
        query="那如果是一年合同呢",
        history_rewrite_ms=12,
    )

    assert result.used_history is True
    assert result.standalone_query == "改写:那如果是一年合同呢"
    assert result.history_message_count == 2
    assert result.history_message_ids == ["m1", "m2"]
    assert result.history_rewrite_ms == 12
