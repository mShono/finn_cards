"""SQLAlchemy 2.0 models mirroring cards/schema.json.

`notes.meta` keeps the schema's free-form `note.meta` object (principal_forms,
rektio, forms_source, ...) as a single JSON blob rather than normalizing
every field - it is already schema-validated on import (see import_cards.py),
and phase 1 has no query that needs to filter on its contents.
"""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    TypeDecorator,
    text,
)
from sqlalchemy.exc import IntegrityError
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


class CardType(str, enum.Enum):
    recognition = "recognition"
    production = "production"
    inflection = "inflection"


class CardState(str, enum.Enum):
    learning = "learning"
    review = "review"
    relearning = "relearning"


class CardStatus(str, enum.Enum):
    """Whether a card has entered the learner's rotation at all.

    A separate axis from CardState: CardState belongs to py-fsrs and every
    card carries one from birth, so it cannot express "exists but has never
    been shown". Existing is cheap (a noun opens 12 form cards at once);
    being introduced is what costs the learner attention, and only
    introduced cards are due, count as debt, or reach FSRS:

        not_introduced             created, waiting for the curriculum
        introduced + learning      in the FSRS learning steps
        introduced + review        scheduled by FSRS
        suspended                  parked by hand, never shown
    """

    not_introduced = "not_introduced"
    introduced = "introduced"
    suspended = "suspended"


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


NOTE_UNIQUE_INDEX = "uq_notes_user_deck_lemma_pos"


def is_note_duplicate_error(error: IntegrityError) -> bool:
    """True only for a NOTE_UNIQUE_INDEX violation, not for any IntegrityError.

    A bare `except IntegrityError` would also swallow a broken FK or a NOT
    NULL miss - real bugs that must keep surfacing. SQLite names the offending
    index in the message for an expression index ("UNIQUE constraint failed:
    index 'uq_notes_user_deck_lemma_pos'"), which is what identifies this one.

    Matched against `error.orig` - the driver's own message - not str(error),
    which appends the statement and its bound parameters. A lemma comes from
    the LLM, and a parameter echo would let one containing this index name turn
    an unrelated IntegrityError into a silent "already added".
    """
    return NOTE_UNIQUE_INDEX in str(error.orig)


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
    # Nullable at the DB level only for old rows predating decks (backfilled
    # to a "Общая" deck by the migration that added this column) - app code
    # always resolves one via db.decks.active_deck() before insert.
    deck_id: Mapped[str | None] = mapped_column(ForeignKey("decks.id"), nullable=True)
    meta: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=lambda: datetime.now(UTC))

    cards: Mapped[list[Card]] = relationship(back_populates="note")
    deck: Mapped[Deck | None] = relationship(back_populates="notes")

    # One note per (user, deck, lemma, pos) - the DB's own copy of the dedup
    # contract /add enforces in Python (ingest.existing_note_keys), so two
    # concurrent /add turns that both pass the pre-insert lookup can't both
    # land. Deck-scoped on purpose: the same word in a different deck is
    # allowed (requested 03.09.2026, see existing_note_keys).
    #
    # COALESCE, not a plain UNIQUE(...), because SQLite treats NULLs as
    # distinct in a unique index: `pos` is legitimately NULL for every
    # kind="pattern" note (cards/schema.json requires pos only when
    # kind="word") and `deck_id` is NULL for rows predating decks and for
    # import_cards.py's CLI imports - a plain constraint would let exactly
    # those duplicate freely.
    #
    # Case is already canonical by the time a lemma reaches here:
    # ingest.canonical_key() runs it through the FST, which folds
    # capitalization for known words ("Hakea" -> "hakea") while keeping
    # proper nouns ("Helsingissä" -> "Helsinki"). A lower()-based index
    # would be both a second normalization and a wrong one - SQLite's
    # lower() only folds ASCII, so 'Äiti' would never match 'äiti'.
    __table_args__ = (
        Index(
            NOTE_UNIQUE_INDEX,
            "user_id",
            "lemma",
            text("coalesce(deck_id, '')"),
            text("coalesce(pos, '')"),
            unique=True,
        ),
    )


class Card(Base):
    __tablename__ = "cards"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    note_id: Mapped[str] = mapped_column(ForeignKey("notes.id"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    type: Mapped[CardType] = mapped_column(Enum(CardType, native_enum=False))
    # Which principal form an inflection card quizzes (a key of
    # kielikaveri.grammar.FORM_TASKS); NULL on every other type. One card
    # per form, because FSRS schedules whatever it rates: with a single
    # card drawing a random form each time, "easy" on the illative would
    # push the translative out by the same interval.
    form: Mapped[str | None] = mapped_column(String, nullable=True)
    # Defaults to `introduced` so a card created without a thought about the
    # curriculum behaves the way every card did before this column existed;
    # inflection cards are the ones created not_introduced on purpose.
    status: Mapped[CardStatus] = mapped_column(
        Enum(CardStatus, native_enum=False), default=CardStatus.introduced, index=True
    )
    # When the curriculum let this card in. NULL for cards that never were -
    # used to count today's introductions against the daily form budget.
    introduced_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True, default=None)

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

    # SQLite indexes no foreign key on its own, so every "this note's cards"
    # lookup was a full scan of `cards`. /learn asks for exactly that on the
    # hot path - once per user in graduation.cards_by_note (joined to notes)
    # and once per rating in learn_rate's ensure_card_types.
    __table_args__ = (Index("ix_cards_note_id", "note_id"),)


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
