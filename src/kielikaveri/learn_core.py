"""Review sessions without Telegram: the database side of learning in the
reader (plan 2026-11-20-kirjakaveri-reader, "П5"; reader/learn.py runs the
session on top of it).

No aiogram or HTTP here and no UI text beyond the few sentences the reader
shows as is. The caller owns the session state (the reader's in-memory
LearnSession) and the logs of its own steps; this module owns preparing a
session, building its queue, applying one rating, and the card as data.

Logs: card ids and numbers only, never a card's text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from kielikaveri.config import Settings
from kielikaveri.db.models import Card, CardState, CardType, Note, Review
from kielikaveri.grammar import FORM_TASKS
from kielikaveri.srs.curriculum import introduce_due_forms
from kielikaveri.srs.graduation import ensure_card_types, sync_user_card_types
from kielikaveri.srs.queue import build_session_queue, overdue_count
from kielikaveri.srs.scheduler import Rating, apply_review

logger = logging.getLogger(__name__)

EMPTY_ALL_TEXT = "Нечего повторять - все карточки выучены на сегодня."
EMPTY_SIDE_TEXT = "На этой стороне сейчас нечего повторять."
NO_PRODUCTION_TEXT = (
    "Русская сторона открывается для слова, когда его финская уже хорошо запомнилась."
)


class LearnSide(StrEnum):
    """Which side of the cards a session shows.

    A side is a filter, not a flip: recognition and production are separate
    cards with FSRS schedules of their own, so showing a recognition card
    Russian-first would record a production answer into recognition's
    history. Inflection cards ask a form of a Finnish lemma, so they go with
    Finnish.
    """

    fi = "fi"
    ru = "ru"
    mix = "mix"

    @property
    def card_types(self) -> tuple[CardType, ...] | None:
        """The card types this side shows; None - every type, unfiltered."""
        if self is LearnSide.fi:
            return (CardType.recognition, CardType.inflection)
        if self is LearnSide.ru:
            return (CardType.production,)
        return None

    @property
    def shows_forms(self) -> bool:
        return self.card_types is None or CardType.inflection in self.card_types


async def prepare_session(
    session: AsyncSession,
    settings: Settings,
    user_id: int,
    now: datetime,
    deck_id: str | None,
    side: LearnSide,
) -> int:
    """Bring the user's cards up to date and return how many reviewed cards
    of this side are overdue (the debt). Commits."""
    await sync_user_card_types(session, user_id, now)
    # Cards first, then the curriculum decides which of the new forms
    # the learner actually meets today - see srs/curriculum.py. Not for
    # a session that shows no forms: opening them there would spend the
    # day's form budget on cards nobody sees.
    if side.shows_forms:
        await introduce_due_forms(
            session,
            user_id,
            now,
            daily_new_forms=settings.daily_new_forms,
            boundary_hour=settings.day_boundary_hour,
            deck_id=deck_id,
        )
    await session.commit()
    return await overdue_count(
        session,
        user_id,
        now,
        deck_id=deck_id,
        reviewed_only=True,
        card_types=side.card_types,
    )


@dataclass(frozen=True)
class SessionQueue:
    queue: list[str]
    side: LearnSide
    # The Russian side is empty because no word has opened it yet (only
    # ever True for an empty queue of LearnSide.ru).
    no_production: bool

    @property
    def empty_text(self) -> str:
        return empty_session_text(self.side, self.no_production)


async def build_queue(
    session: AsyncSession,
    settings: Settings,
    user_id: int,
    now: datetime,
    deck_id: str | None,
    side: LearnSide,
) -> SessionQueue:
    queue = await build_session_queue(
        session,
        user_id,
        now,
        session_max_cards=settings.session_max_cards,
        daily_new_limit=settings.daily_new_limit,
        boundary_hour=settings.day_boundary_hour,
        deck_id=deck_id,
        card_types=side.card_types,
    )
    no_production = (
        side is LearnSide.ru and not queue and not await has_production(session, user_id, deck_id)
    )
    return SessionQueue(queue, side, no_production)


async def has_production(session: AsyncSession, user_id: int, deck_id: str | None) -> bool:
    stmt = select(Card.id).where(Card.user_id == user_id, Card.type == CardType.production)
    if deck_id is not None:
        stmt = stmt.join(Note, Card.note_id == Note.id).where(Note.deck_id == deck_id)
    return await session.scalar(stmt.limit(1)) is not None


def empty_session_text(side: LearnSide, no_production: bool) -> str:
    if side is LearnSide.mix:
        return EMPTY_ALL_TEXT
    # Only this side is done - the other may still have cards waiting.
    text = EMPTY_SIDE_TEXT
    if no_production:
        # Production cards open only once the Finnish side is known well
        # (graduation.py) - until the first one does, Russian is empty.
        text += "\n" + NO_PRODUCTION_TEXT
    return text


def session_end_text(reviewed: int, remaining: int, side: LearnSide, continue_hint: str) -> str:
    """The end-of-session line; `continue_hint` follows the count of cards
    left (the reader's end screen has its own "start again" button)."""
    text = f"Сессия окончена: {reviewed} карточек пройдено"
    if remaining:
        text += f", осталось {remaining}{continue_hint}"
    elif side == LearnSide.mix:
        text += " - всё на сегодня!"
    else:
        # Only this side is done - see empty_session_text.
        text += " - на этой стороне всё на сегодня!"
    return text


async def drop_deleted_head(
    session: AsyncSession, queue: list[str], source: str | None = None
) -> list[str]:
    """The queue without deleted cards at its head. A word deleted
    mid-session (/delete, 🗑 in a deck) leaves its card ids in the session's
    queue - deleting knows nothing about sessions. `source` goes into the
    log line (the reader's "reader"; None leaves it out)."""
    suffix = "" if source is None else f" source={source}"
    while queue and await session.get(Card, queue[0]) is None:
        logger.info("event=learn.skip_deleted card_id=%s%s", queue[0], suffix)
        queue = queue[1:]
    return queue


@dataclass(frozen=True)
class SrsColumns:
    """A card's SRS columns, as they were - what an undo puts back."""

    state: CardState
    due: datetime
    stability: float | None
    difficulty: float | None
    reps: int
    lapses: int
    step: int | None

    @classmethod
    def of(cls, card: Card) -> SrsColumns:
        return cls(
            state=card.state,
            due=card.due,
            stability=card.stability,
            difficulty=card.difficulty,
            reps=card.reps,
            lapses=card.lapses,
            step=card.step,
        )

    def restore(self, card: Card) -> None:
        card.state = self.state
        card.due = self.due
        card.stability = self.stability
        card.difficulty = self.difficulty
        card.reps = self.reps
        card.lapses = self.lapses
        card.step = self.step


@dataclass(frozen=True)
class ReviewRecord:
    """One applied rating: enough to take it back."""

    card_id: str
    before: SrsColumns
    review_id: int
    # Cards ensure_card_types created because of this rating.
    created_card_ids: list[str] = field(default_factory=list)


async def record_review(
    session: AsyncSession, card: Card, rating: Rating, now: datetime
) -> ReviewRecord:
    """Apply `rating` to `card` (loaded in `session`), add its `reviews`
    row, open the card types the note has become eligible for. Commits."""
    before = SrsColumns.of(card)
    # When this card was last answered - FSRS schedules off the gap since
    # then. Read *before* the new row below is added to the session:
    # adding it first would let autoflush include this very answer in the
    # max(), making last_review == now, and a zero gap reads as a perfect
    # recall. None here means "never answered", which is what FSRS wants
    # for a card's first review.
    last_review = await session.scalar(
        select(func.max(Review.reviewed_at)).where(Review.card_id == card.id)
    )
    # Moves every SRS column on the card; the `reviews` row below is ours.
    apply_review(card, rating, now, last_review)

    review = Review(card_id=card.id, user_id=card.user_id, rating=rating.value, reviewed_at=now)
    session.add(review)

    note = await session.get(Note, card.note_id)
    created = await ensure_card_types(session, note, now)
    # Ids are known after the flush the commit would make anyway.
    await session.flush()
    record = ReviewRecord(card.id, before, review.id, [c.id for c in created])
    await session.commit()
    return record


async def undo_review(session: AsyncSession, record: ReviewRecord) -> bool:
    """Take `record` back: the card's SRS columns, its `reviews` row, and
    the cards that rating created - those that have no reviews of their
    own yet. Commits. False, changing nothing, when the card is gone
    (deleted since) or was rated again after `record` (somewhere else):
    putting the old columns back would erase that newer answer."""
    card = await session.get(Card, record.card_id)
    if card is None or await _rated_since(session, card, record):
        return False
    record.before.restore(card)
    review = await session.get(Review, record.review_id)
    if review is not None:
        await session.delete(review)
    for card_id in record.created_card_ids:
        created = await session.get(Card, card_id)
        if created is None:
            continue
        reviewed = await session.scalar(select(Review.id).where(Review.card_id == card_id).limit(1))
        if reviewed is None:
            await session.delete(created)
    await session.commit()
    return True


async def _rated_since(session: AsyncSession, card: Card, record: ReviewRecord) -> bool:
    if card.reps != record.before.reps + 1:
        return True
    newer = await session.scalar(
        select(Review.id).where(Review.card_id == card.id, Review.id > record.review_id).limit(1)
    )
    return newer is not None


@dataclass(frozen=True)
class CardView:
    """A card's content by its type - the data behind the reader's card
    JSON."""

    id: str
    type: CardType
    lemma: str
    translation_ru: str
    example_fi: str
    example_ru: str
    # Inflection only. A form with no FORM_TASKS entry has no cue: it shows
    # the lemma on both sides (form_answer is the lemma, no label).
    cue: str | None = None
    form_answer: str | None = None
    form_label: str | None = None

    def as_json(self) -> dict:
        body: dict = {
            "id": self.id,
            "type": self.type.value,
            "lemma": self.lemma,
            "translation_ru": self.translation_ru,
            "example_fi": self.example_fi,
            "example_ru": self.example_ru,
        }
        if self.type == CardType.inflection:
            body |= {
                "cue": self.cue,
                "form_answer": self.form_answer,
                "form_label": self.form_label,
            }
        return body


def card_view(card: Card, note: Note) -> CardView:
    view = CardView(
        id=card.id,
        type=card.type,
        lemma=note.lemma,
        translation_ru=note.translation_ru,
        example_fi=note.example_fi,
        example_ru=note.example_ru,
    )
    if card.type != CardType.inflection:
        return view
    forms: dict = note.meta.get("principal_forms") or {}
    # The card names its own form (one card per form), so the question
    # is stable across reviews and its FSRS interval means something.
    # A form with no FORM_TASKS entry is never asked - the front must
    # not carry a bare key like "nut_partisiippi".
    task = FORM_TASKS.get(card.form)
    if task is not None and card.form in forms:
        return replace(view, cue=task.cue, form_answer=forms[card.form], form_label=task.label)
    return replace(view, form_answer=note.lemma)
