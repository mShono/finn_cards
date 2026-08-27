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

from sqlalchemy import JSON, Boolean, Enum, ForeignKey, Integer, String, TypeDecorator
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
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=lambda: datetime.now(UTC))
    # The deck new notes are filed into by default - see db/decks.py's
    # active_deck(). None until the user's first save picks (and thereby
    # creates) one. Deliberately a plain column, not ForeignKey("decks.id") -
    # decks.user_id already points at users.id, and a FK back the other way
    # makes the two tables mutually dependent (SQLAlchemy can't topologically
    # sort them for create_all/migrations - confirmed by a SAWarning during
    # `alembic revision --autogenerate`). SQLite doesn't enforce FKs in this
    # setup anyway (see db/decks.py's set_active_deck comment).
    last_deck_id: Mapped[str | None] = mapped_column(String, nullable=True)


class Deck(Base):
    """A user-named grouping of notes (plan: "колоды", chosen deliberately
    manual - the learner decides what goes where, not an automatic
    new-vs-mature split). Purely organizational: it narrows /learn's queue
    and /add's save target, it does not change FSRS scheduling itself.
    """

    __tablename__ = "decks"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    name: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=lambda: datetime.now(UTC))

    notes: Mapped[list[Note]] = relationship(back_populates="deck")


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
    source_id: Mapped[str | None] = mapped_column(ForeignKey("sources.id"), nullable=True)
    # Nullable at the DB level only for old rows predating decks (backfilled
    # to a "Общая" deck by the migration that added this column) - app code
    # always resolves one via db.decks.active_deck() before insert.
    deck_id: Mapped[str | None] = mapped_column(ForeignKey("decks.id"), nullable=True)
    meta: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=lambda: datetime.now(UTC))

    cards: Mapped[list[Card]] = relationship(back_populates="note")
    deck: Mapped[Deck | None] = relationship(back_populates="notes")


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
    due: Mapped[datetime] = mapped_column(UTCDateTime, default=lambda: datetime.now(UTC))
    # None means "never reviewed" - py-fsrs uses that as its own sentinel for
    # a brand new card and picks the initial value itself. Storing 0.0 here
    # instead would be read back as an existing (and nonsensical, FSRS
    # stability is always positive) card state and corrupt the very first
    # review's math.
    stability: Mapped[float | None] = mapped_column(nullable=True, default=None)
    difficulty: Mapped[float | None] = mapped_column(nullable=True, default=None)
    reps: Mapped[int] = mapped_column(default=0)
    lapses: Mapped[int] = mapped_column(default=0)
    # py-fsrs's sub-step index within its short learning/relearning steps
    # (e.g. 1min, 10min) - must round-trip through the DB or a restart loses
    # where a card was mid-step and effectively restarts its learning phase.
    step: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)

    note: Mapped[Note] = relationship(back_populates="cards")


class IngestCache(Base):
    """Cached /add candidate-generation result, keyed by the input text's hash.

    Saves a re-paid LLM call if the same text comes through /add twice - a
    crashed confirmation flow, a resent paragraph - see plan 3.8/3.11.
    """

    __tablename__ = "ingest_cache"

    text_hash: Mapped[str] = mapped_column(String, primary_key=True)
    model: Mapped[str] = mapped_column(String)
    candidates: Mapped[list] = mapped_column(JSON)
    # Chat reply text (plan 3.11 v2: conversational /add) - nullable because
    # rows written before this column existed have none; get_cached_chat()
    # treats those as a cache miss rather than crashing or returning empty text.
    reply_ru: Mapped[str | None] = mapped_column(String, nullable=True)
    # Whether reply_ru is a clarifying question rather than a finished answer
    # (plan: /add redesign). Nullable for rows written before this column
    # existed - get_cached_chat() reads NULL as False, matching those rows'
    # actual behavior (they always acted immediately, never asked back).
    needs_clarification: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=lambda: datetime.now(UTC))


class Review(Base):
    __tablename__ = "reviews"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    card_id: Mapped[str] = mapped_column(ForeignKey("cards.id"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    rating: Mapped[int] = mapped_column()
    reviewed_at: Mapped[datetime] = mapped_column(UTCDateTime, default=lambda: datetime.now(UTC))
