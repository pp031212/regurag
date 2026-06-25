from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ..core.config import get_settings
from .base import Base


settings = get_settings()
engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)



def init_db() -> None:
    from ..models.conversation import Conversation
    from ..models.document import Document
    from ..models.knowledge_base import KnowledgeBase
    from ..models.message import Message
    from ..models.message_context import MessageContext
    from ..models.task import Task
    from ..models.task_event import TaskEvent

    Base.metadata.create_all(bind=engine)
