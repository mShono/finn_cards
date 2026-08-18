"""SQLAlchemy 2.0 models mirroring cards/schema.json.

`notes.meta` keeps the schema's free-form `note.meta` object (principal_forms,
cognates, forms_source, ...) as a single JSON blob rather than normalizing
every field - it is already schema-validated on import (see import_cards.py),
and phase 1 has no query that needs to filter on its contents.
"""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, Enum, ForeignKey, Integer, String, TypeDecorator
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


class UTCDateTime(TypeDecorator):
    """An aware UTC datetime, stored as an epoch-second integer.

    SQLite's DateTime(timezone=True) silently drops tzinfo on read-back
    (round-trips as a naive datetime) - storing an unambiguous epoch integer
    instead is what "due as a UTC timestamp" actually requires, and it
    can't be misread as local time by accident.
    """

    impl = Integer
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> int | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("UTCDateTime requires a timezone-aware datetime")
        return int(value.astimezone(UTC).timestamp())

    def process_result_value(self, value: int | None, dialect) -> datetime | None:
        if value is None:
            return None
        return datetime.fromtimestamp(value, tz=UTC)


class NoteKind(str, enum.Enum):
    word = "word"
    pattern = "pattern"


class SourceType(str, enum.Enum):
    video = "video"
    audio = "audio"
    article = "article"
    conversation = "conversation"
    book = "book"
    other = "other"


class CardType(str, enum.Enum):
    recognition = "recognition"
    production = "production"
    inflection = "inflection"
    usage = "usage"


class CardState(str, enum.Enum):
    learning = "learning"
    review = "review"
    relearning = "relearning"


class User(Base):
    __tablename__ = "users"

    # Telegram user id - the app has one user in phase 1, but every other
    # table still carries user_id so multi-user is a non-event later.
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=lambda: datetime.now(UTC)
    )


class Source(Base):
    __tablename__ = "sources"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    type: Mapped[SourceType] = mapped_column(Enum(SourceType, native_enum=False))
    ref: Mapped[str] = mapped_column(String)
    context_fi: Mapped[str | None] = mapped_column(String, nullable=True)


class Note(Base):
    __tablename__ = "notes"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    lemma: Mapped[str] = mapped_column(String)
    pos: Mapped[str | None] = mapped_column(String, nullable=True)
    translation_ru: Mapped[str] = mapped_column(String)
    example_fi: Mapped[str] = mapped_column(String)
    example_ru: Mapped[str] = mapped_column(String)
    kind: Mapped[NoteKind] = mapped_column(Enum(NoteKind, native_enum=False))
    source_id: Mapped[str | None] = mapped_column(
        ForeignKey("sources.id"), nullable=True
    )
    meta: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=lambda: datetime.now(UTC)
    )

    cards: Mapped[list[Card]] = relationship(back_populates="note")


class Card(Base):
    __tablename__ = "cards"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    note_id: Mapped[str] = mapped_column(ForeignKey("notes.id"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    type: Mapped[CardType] = mapped_column(Enum(CardType, native_enum=False))

    # SRS state - schema's card.srs, flattened. Written by phase 2 (py-fsrs);
    # phase 1 only needs the columns to exist.
    state: Mapped[CardState] = mapped_column(
        Enum(CardState, native_enum=False), default=CardState.learning
    )
    due: Mapped[datetime] = mapped_column(
        UTCDateTime, default=lambda: datetime.now(UTC)
    )
    stability: Mapped[float] = mapped_column(default=0.0)
    difficulty: Mapped[float] = mapped_column(default=0.0)
    reps: Mapped[int] = mapped_column(default=0)
    lapses: Mapped[int] = mapped_column(default=0)

    note: Mapped[Note] = relationship(back_populates="cards")


class Review(Base):
    __tablename__ = "reviews"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    card_id: Mapped[str] = mapped_column(ForeignKey("cards.id"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    rating: Mapped[int] = mapped_column()
    reviewed_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=lambda: datetime.now(UTC)
    )
