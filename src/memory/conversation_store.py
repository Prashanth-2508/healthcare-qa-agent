"""Session memory backed by SQLite via SQLAlchemy."""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.utils.logger import get_logger

_log = get_logger(__name__)

_DB_URL = os.getenv("DATABASE_URL", "sqlite:///./healthcare_qa.db")

engine = create_engine(
    _DB_URL,
    connect_args={"check_same_thread": False} if "sqlite" in _DB_URL else {},
    echo=False,
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(128), index=True, nullable=False)
    role = Column(String(32), nullable=False)  # "user" | "assistant" | "tool"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class AgentTrace(Base):
    __tablename__ = "agent_traces"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(128), index=True, nullable=False)
    iteration = Column(Integer, default=0)
    step = Column(String(64), nullable=False)
    content = Column(Text, nullable=False)
    metadata_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


def init_db() -> None:
    """Create all tables. Call once at application startup."""
    Base.metadata.create_all(bind=engine)
    _log.info("db_initialized", url=_DB_URL)


class ConversationStore:
    """CRUD operations for session-scoped conversation history."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id

    def _db(self) -> Session:
        return SessionLocal()

    def add_message(self, role: str, content: str) -> None:
        with self._db() as db:
            msg = ConversationMessage(
                session_id=self.session_id,
                role=role,
                content=content,
            )
            db.add(msg)
            db.commit()
        _log.debug("memory_write", session_id=self.session_id, role=role)

    def get_history(self, limit: int = 20) -> list[dict[str, str]]:
        with self._db() as db:
            stmt = (
                select(ConversationMessage)
                .where(ConversationMessage.session_id == self.session_id)
                .order_by(ConversationMessage.created_at.asc())
                .limit(limit)
            )
            rows = db.execute(stmt).scalars().all()
        return [{"role": r.role, "content": r.content} for r in rows]

    def clear_history(self) -> None:
        with self._db() as db:
            db.query(ConversationMessage).filter(
                ConversationMessage.session_id == self.session_id
            ).delete()
            db.commit()

    def save_trace_entry(
        self,
        step: str,
        content: str,
        iteration: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._db() as db:
            trace = AgentTrace(
                session_id=self.session_id,
                iteration=iteration,
                step=step,
                content=content,
                metadata_json=json.dumps(metadata or {}),
            )
            db.add(trace)
            db.commit()

    def get_trace(self) -> list[dict[str, Any]]:
        with self._db() as db:
            stmt = (
                select(AgentTrace)
                .where(AgentTrace.session_id == self.session_id)
                .order_by(AgentTrace.created_at.asc())
            )
            rows = db.execute(stmt).scalars().all()
        return [
            {
                "iteration": r.iteration,
                "step": r.step,
                "content": r.content,
                "metadata": json.loads(r.metadata_json),
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]

    def token_count_estimate(self) -> int:
        """Rough token estimate (4 chars ≈ 1 token) for context management."""
        history = self.get_history(limit=100)
        total_chars = sum(len(m["content"]) for m in history)
        return total_chars // 4

    def trim_if_needed(self, max_tokens: int = 6000) -> int:
        """Remove oldest messages to stay under token budget. Returns removed count."""
        if self.token_count_estimate() <= max_tokens:
            return 0

        removed = 0
        with self._db() as db:
            while True:
                oldest = (
                    db.query(ConversationMessage)
                    .filter(ConversationMessage.session_id == self.session_id)
                    .order_by(ConversationMessage.created_at.asc())
                    .first()
                )
                if oldest is None:
                    break
                db.delete(oldest)
                db.commit()
                removed += 1
                if self.token_count_estimate() <= max_tokens:
                    break

        _log.info("memory_trimmed", session_id=self.session_id, removed=removed)
        return removed
