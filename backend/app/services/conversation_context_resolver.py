from dataclasses import dataclass

from ..rag.query_rewriter import QueryRewriter


@dataclass(slots=True)
class ConversationContextResolution:
    standalone_query: str
    used_history: bool
    history_message_count: int
    history_rewrite_ms: int
    history_message_ids: list[str]


class ConversationContextResolver:
    MAX_HISTORY_MESSAGES = 6
    MAX_HISTORY_CHARS_PER_MESSAGE = 200

    def __init__(self, repository, rewriter: QueryRewriter) -> None:
        self.repository = repository
        self.rewriter = rewriter

    def _build_recent_history(self, conversation_id: str, knowledge_base_id: str) -> list[dict[str, str]]:
        messages = self.repository.list_messages(conversation_id)
        filtered_messages: list[dict[str, str]] = []
        for message in messages:
            if message.get("knowledge_base_id") != knowledge_base_id:
                continue

            role = str(message.get("role") or "").strip()
            content = str(message.get("content") or "").strip()
            if role not in {"user", "assistant"} or not content:
                continue

            filtered_messages.append(
                {
                    "id": str(message.get("id") or ""),
                    "role": role,
                    "content": content[: self.MAX_HISTORY_CHARS_PER_MESSAGE],
                }
            )

        return filtered_messages[-self.MAX_HISTORY_MESSAGES :]

    def resolve(
        self,
        *,
        conversation_id: str,
        knowledge_base_id: str,
        query: str,
        history_rewrite_ms: int = 0,
    ) -> ConversationContextResolution:
        history_messages = self._build_recent_history(conversation_id, knowledge_base_id)
        if not history_messages:
            return ConversationContextResolution(
                standalone_query=query,
                used_history=False,
                history_message_count=0,
                history_rewrite_ms=history_rewrite_ms,
                history_message_ids=[],
            )

        rewritten_query = self.rewriter.rewrite_with_history(query, history_messages).strip()
        if not rewritten_query:
            rewritten_query = query

        return ConversationContextResolution(
            standalone_query=rewritten_query,
            used_history=rewritten_query != query,
            history_message_count=len(history_messages),
            history_rewrite_ms=history_rewrite_ms,
            history_message_ids=[message["id"] for message in history_messages if message["id"]],
        )
