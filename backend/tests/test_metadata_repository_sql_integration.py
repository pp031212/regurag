from contextlib import suppress
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.core.exceptions import ConversationNotFoundError, KnowledgeBaseNotFoundError
from app.db.session import engine
from app.repositories.metadata_repository import MetadataRepository, get_metadata_repository


@pytest.fixture()
def sql_metadata_repository() -> MetadataRepository:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"SQL metadata backend unavailable for integration test: {exc}")
    return get_metadata_repository()


def _cleanup(
    repository: MetadataRepository,
    *,
    conversation_ids: list[str] | None = None,
    knowledge_base_ids: list[str] | None = None,
) -> None:
    for conversation_id in conversation_ids or []:
        with suppress(ConversationNotFoundError):
            repository.delete_conversation(conversation_id)
    for knowledge_base_id in knowledge_base_ids or []:
        with suppress(KnowledgeBaseNotFoundError):
            repository.delete_knowledge_base(knowledge_base_id)


def test_metadata_repository_persists_and_updates_knowledge_base(sql_metadata_repository: MetadataRepository) -> None:
    repository = sql_metadata_repository
    knowledge_base = repository.create_knowledge_base(
        name=f"Integration KB {uuid4().hex[:6]}",
        description="integration description",
        subject="integration subject",
        domain="general",
    )

    try:
        fetched = repository.get_knowledge_base(knowledge_base["id"])
        assert fetched is not None
        assert fetched["name"] == knowledge_base["name"]

        listed_ids = {item["id"] for item in repository.list_knowledge_bases()}
        assert knowledge_base["id"] in listed_ids

        updated = repository.update_knowledge_base(
            knowledge_base["id"],
            status="ready",
            description="updated integration description",
        )
        assert updated["status"] == "ready"
        assert updated["description"] == "updated integration description"
    finally:
        _cleanup(repository, knowledge_base_ids=[knowledge_base["id"]])


def test_metadata_repository_persists_documents_and_detects_duplicates(sql_metadata_repository: MetadataRepository) -> None:
    repository = sql_metadata_repository
    knowledge_base = repository.create_knowledge_base(
        name=f"Document KB {uuid4().hex[:6]}",
        description="document integration description",
        subject="document subject",
        domain="general",
    )

    try:
        document = repository.create_document(
            knowledge_base_id=knowledge_base["id"],
            filename="rules.md",
            content_type="text/markdown",
            file_size=128,
            content_hash="hash-doc-001",
            file_path="D:/workspace/regurag/backend/data/uploads/integration/rules.md",
        )

        listed_documents = repository.list_documents(knowledge_base_id=knowledge_base["id"])
        assert {item["id"] for item in listed_documents} == {document["id"]}

        duplicate = repository.find_duplicate_document(
            knowledge_base_id=knowledge_base["id"],
            filename="rules.md",
            file_size=128,
            content_hash="hash-doc-001",
        )
        assert duplicate is not None
        assert duplicate["id"] == document["id"]

        repository.delete_document(document["id"])
        assert repository.get_document(document["id"]) is None
    finally:
        _cleanup(repository, knowledge_base_ids=[knowledge_base["id"]])


def test_delete_knowledge_base_clears_conversation_and_message_context_references(
    sql_metadata_repository: MetadataRepository,
) -> None:
    repository = sql_metadata_repository
    knowledge_base = repository.create_knowledge_base(
        name=f"Reference KB {uuid4().hex[:6]}",
        description="reference integration description",
        subject="reference subject",
        domain="general",
    )
    conversation = repository.create_conversation(
        title="Integration Conversation",
        default_knowledge_base_id=knowledge_base["id"],
    )

    try:
        user_message = repository.create_message(
            conversation_id=conversation["id"],
            role="user",
            content="integration question",
        )
        repository.create_message_context(
            message_id=user_message["id"],
            knowledge_base_id=knowledge_base["id"],
            citations=[{"chunk_id": "chunk-001"}],
            debug={"stage": "integration"},
        )

        repository.delete_knowledge_base(knowledge_base["id"])

        updated_conversation = repository.get_conversation(conversation["id"])
        assert updated_conversation is not None
        assert updated_conversation["default_knowledge_base_id"] is None

        messages = repository.list_messages(conversation["id"])
        assert len(messages) == 1
        assert messages[0]["knowledge_base_id"] is None
        assert messages[0]["citations"] == [{"chunk_id": "chunk-001"}]
        assert messages[0]["debug"] == {"stage": "integration"}
    finally:
        _cleanup(repository, conversation_ids=[conversation["id"]], knowledge_base_ids=[knowledge_base["id"]])


def test_delete_conversation_cascades_messages_and_message_contexts(sql_metadata_repository: MetadataRepository) -> None:
    repository = sql_metadata_repository
    knowledge_base = repository.create_knowledge_base(
        name=f"Conversation KB {uuid4().hex[:6]}",
        description="conversation integration description",
        subject="conversation subject",
        domain="general",
    )
    conversation = repository.create_conversation(
        title="Conversation Cascade",
        default_knowledge_base_id=knowledge_base["id"],
    )

    try:
        first_message = repository.create_message(
            conversation_id=conversation["id"],
            role="user",
            content="first message",
        )
        second_message = repository.create_message(
            conversation_id=conversation["id"],
            role="assistant",
            content="second message",
        )
        repository.create_message_context(
            message_id=second_message["id"],
            knowledge_base_id=knowledge_base["id"],
            citations=[{"chunk_id": "chunk-002"}],
            debug={"answer_source": "integration"},
        )

        messages_before_delete = repository.list_messages(conversation["id"])
        assert [item["sequence"] for item in messages_before_delete] == [1, 2]
        assert messages_before_delete[1]["knowledge_base_id"] == knowledge_base["id"]

        repository.delete_conversation(conversation["id"])

        assert repository.get_conversation(conversation["id"]) is None
        assert repository.list_messages(conversation["id"]) == []
    finally:
        _cleanup(repository, conversation_ids=[conversation["id"]], knowledge_base_ids=[knowledge_base["id"]])
