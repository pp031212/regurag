from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from ..db.base import Base


class MessageContext(Base):
    __tablename__ = "message_contexts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    message_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("messages.id"),
        nullable=False,
        unique=True,
        index=True,
    )
    knowledge_base_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("knowledge_bases.id"),
        nullable=True,
        index=True,
    )
    citations: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)
    debug: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
