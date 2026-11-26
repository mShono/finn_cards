"""📚 Learning in the Mini App (plan 2026-11-20-kirjakaveri-reader, "П5"):
the review session behind /api/learn/*.

The queue lives here, on the server; the client gets one card at a time.
One session per user, in this process's memory (LEARN_SESSIONS in
reader/app.py): a new start replaces the old one, and after a restart there
is none - the client is told `session_gone` and starts again. The ratings
are in the database already, only the place in the queue is lost.

Every call takes the user's lock (KeyedLocks), so a double swipe rates the
card once: the second call finds another card at the head and is `stale`.

Only the last rating can be undone, while the session lives: the card's SRS
columns, its `reviews` row and the cards that rating opened go back
(learn_core.undo_review); the card is the head of the queue again.

Logs: event=learn.* source=reader, with card ids and numbers - never a
card's text.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kielikaveri.config import Settings
from kielikaveri.db.decks import list_decks
from kielikaveri.db.models import Card, Deck, Note
from kielikaveri.learn_core import (
    LearnSide,
    ReviewRecord,
    build_queue,
    card_view,
    drop_deleted_head,
    prepare_session,
    record_review,
    session_end_text,
    undo_review,
)
from kielikaveri.reader.lookup import KeyedLocks
from kielikaveri.srs.queue import defer_overdue_tail
from kielikaveri.srs.scheduler import Rating

logger = logging.getLogger(__name__)

# After "осталось N": the end screen has its own "start again" button.
CONTINUE_HINT = "."


class SessionGoneError(Exception):
    """No session of this id: replaced by a newer start, or the process restarted."""


class NotFoundError(Exception):
    """A deck or card that isn't the user's - it looks like a missing one."""


class NothingToUndoError(Exception):
    pass


class StaleError(Exception):
    """The request is about a card that isn't the head any more; `payload`
    is the session as it is now, for the client to redraw from."""

    def __init__(self, payload: dict) -> None:
        super().__init__("stale")
        self.payload = payload


@dataclass(frozen=True)
class PendingDebt:
    """The debt question is open: the start's time and the overdue count."""

    now: datetime
    overdue: int


@dataclass
class LearnSession:
    id: str
    deck_id: str | None
    side: LearnSide
    started_at: datetime
    max_minutes: int
    queue: list[str] = field(default_factory=list)
    reviewed_count: int = 0
    # The last rating, for undo.
    last: ReviewRecord | None = None
    pending_debt: PendingDebt | None = None
    # Over: the queue ran dry or the time is up. Kept for undo.
    ended: bool = False


Sessions = dict[int, LearnSession]


async def decks(session_factory: async_sessionmaker[AsyncSession], user_id: int) -> dict:
    async with session_factory() as db:
        found = await list_decks(db, user_id)
    return {"decks": [{"id": deck.id, "name": deck.name} for deck in found]}


def _key(user_id: int) -> str:
    return str(user_id)


def _live(sessions: Sessions, user_id: int, session_id: str) -> LearnSession:
    current = sessions.get(user_id)
    if current is None or current.id != session_id:
        raise SessionGoneError
    return current


def _progress(current: LearnSession) -> dict:
    return {
        "done": current.reviewed_count,
        "total": current.reviewed_count + len(current.queue),
    }


def _debt_payload(current: LearnSession, settings: Settings) -> dict:
    return {
        "state": "debt",
        "session": current.id,
        "overdue": current.pending_debt.overdue,
        "max_cards": settings.session_max_cards,
        "postpone_days": settings.debt_postpone_days,
    }


async def _state(db: AsyncSession, current: LearnSession, *, check_time: bool) -> dict:
    """The session as the client should show it: its head card, or the end.

    `check_time` - the session_max_minutes limit, applied after a rating
    (as the bot's _show_next_card does), not when the state is only
    re-read."""
    current.queue = await drop_deleted_head(db, current.queue)
    if not current.ended:
        elapsed = (datetime.now(UTC) - current.started_at).total_seconds() / 60
        timed_out = check_time and elapsed >= current.max_minutes
        if not current.queue or timed_out:
            current.ended = True
            logger.info(
                "event=learn.session_end session=%s reason=%s reviewed=%d remaining=%d "
                "source=reader",
                current.id,
                "queue_empty" if not current.queue else "max_minutes",
                current.reviewed_count,
                len(current.queue),
            )
    can_undo = current.last is not None
    if current.ended:
        return {
            "state": "end",
            "reviewed": current.reviewed_count,
            "remaining": len(current.queue),
            "text": session_end_text(
                current.reviewed_count, len(current.queue), current.side, CONTINUE_HINT
            ),
            "can_undo": can_undo,
        }
    card = await db.get(Card, current.queue[0])
    note = await db.get(Note, card.note_id)
    return {
        "state": "card",
        "card": card_view(card, note).as_json(),
        "progress": _progress(current),
        "can_undo": can_undo,
    }


async def _begin(
    db: AsyncSession,
    settings: Settings,
    sessions: Sessions,
    user_id: int,
    current: LearnSession,
    now: datetime,
) -> dict:
    """Build the queue of `current` (as of `now`) and show its first card."""
    built = await build_queue(db, settings, user_id, now, current.deck_id, current.side)
    if not built.queue:
        if sessions.get(user_id) is current:
            del sessions[user_id]
        logger.info(
            "event=learn.session_empty deck_id=%s side=%s source=reader",
            current.deck_id,
            current.side,
        )
        return {"state": "empty", "text": built.empty_text}
    current.queue = built.queue
    current.started_at = datetime.now(UTC)
    logger.info(
        "event=learn.session_start session=%s deck_id=%s side=%s queue=%d source=reader",
        current.id,
        current.deck_id,
        current.side,
        len(built.queue),
    )
    return {"session": current.id, **await _state(db, current, check_time=False)}


async def start(
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    sessions: Sessions,
    locks: KeyedLocks,
    user_id: int,
    deck_id: str | None,
    side: LearnSide,
) -> dict:
    async with locks.hold(_key(user_id)), session_factory() as db:
        if deck_id is not None:
            deck = await db.get(Deck, deck_id)
            if deck is None or deck.user_id != user_id:
                raise NotFoundError
        # A new start replaces the old session, whatever comes of it.
        sessions.pop(user_id, None)
        now = datetime.now(UTC)
        overdue = await prepare_session(db, settings, user_id, now, deck_id, side)
        current = LearnSession(
            id=uuid.uuid4().hex,
            deck_id=deck_id,
            side=side,
            started_at=now,
            max_minutes=settings.session_max_minutes,
        )
        sessions[user_id] = current
        if overdue > settings.debt_threshold:
            current.pending_debt = PendingDebt(now, overdue)
            logger.info(
                "event=learn.debt_prompt session=%s deck_id=%s side=%s overdue=%d threshold=%d "
                "source=reader",
                current.id,
                deck_id,
                side,
                overdue,
                settings.debt_threshold,
            )
            return _debt_payload(current, settings)
        return await _begin(db, settings, sessions, user_id, current, now)


async def debt(
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    sessions: Sessions,
    locks: KeyedLocks,
    user_id: int,
    session_id: str,
    defer: bool,
) -> dict:
    async with locks.hold(_key(user_id)), session_factory() as db:
        current = _live(sessions, user_id, session_id)
        pending = current.pending_debt
        if pending is None:
            # Answered already (a double tap): the session as it is.
            logger.info("event=learn.stale session=%s source=reader", current.id)
            raise StaleError(await _state(db, current, check_time=False))
        if defer:
            postponed = await defer_overdue_tail(
                db,
                user_id,
                pending.now,
                keep_n=settings.session_max_cards,
                postpone_days=settings.debt_postpone_days,
                deck_id=current.deck_id,
                card_types=current.side.card_types,
            )
            await db.commit()
            logger.info(
                "event=learn.debt_choice session=%s action=defer postponed=%d source=reader",
                current.id,
                postponed,
            )
        else:
            logger.info("event=learn.debt_choice session=%s action=batch source=reader", current.id)
        current.pending_debt = None
        return await _begin(db, settings, sessions, user_id, current, pending.now)


async def _own_card(db: AsyncSession, user_id: int, card_id: str) -> Card:
    card = await db.get(Card, card_id)
    if card is None or card.user_id != user_id:
        raise NotFoundError
    return card


async def review(
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    sessions: Sessions,
    locks: KeyedLocks,
    user_id: int,
    session_id: str,
    card_id: str,
    rating: Rating,
) -> dict:
    async with locks.hold(_key(user_id)), session_factory() as db:
        current = _live(sessions, user_id, session_id)
        if current.pending_debt is not None:
            logger.info(
                "event=learn.stale session=%s card_id=%s source=reader", current.id, card_id
            )
            raise StaleError(_debt_payload(current, settings))
        if current.ended or not current.queue or current.queue[0] != card_id:
            # Someone else's card or one that never was: 404, as everywhere.
            await _own_card(db, user_id, card_id)
            logger.info(
                "event=learn.stale session=%s card_id=%s source=reader", current.id, card_id
            )
            raise StaleError(await _state(db, current, check_time=False))
        card = await db.get(Card, card_id)
        if card is None:
            # Deleted mid-session: skipped, nothing is rated.
            logger.info("event=learn.skip_deleted card_id=%s source=reader", card_id)
            current.queue = current.queue[1:]
            return await _state(db, current, check_time=True)
        # The session moves on only once the rating is in the database.
        current.last = await record_review(db, card, rating, datetime.now(UTC))
        current.queue = current.queue[1:]
        current.reviewed_count += 1
        logger.info(
            "event=learn.review session=%s card_id=%s rating=%d created=%d source=reader",
            current.id,
            card_id,
            rating.value,
            len(current.last.created_card_ids),
        )
        return await _state(db, current, check_time=True)


async def undo(
    session_factory: async_sessionmaker[AsyncSession],
    sessions: Sessions,
    locks: KeyedLocks,
    user_id: int,
    session_id: str,
) -> dict:
    async with locks.hold(_key(user_id)), session_factory() as db:
        current = _live(sessions, user_id, session_id)
        record = current.last
        if record is None:
            raise NothingToUndoError
        restored = await undo_review(db, record)
        current.last = None
        if not restored:
            # The card was deleted since: nothing left to take back.
            raise NothingToUndoError
        current.queue = [record.card_id, *current.queue]
        current.reviewed_count -= 1
        current.ended = False
        logger.info(
            "event=learn.undo session=%s card_id=%s created=%d source=reader",
            current.id,
            record.card_id,
            len(record.created_card_ids),
        )
        return await _state(db, current, check_time=False)


async def example_to_speak(
    session_factory: async_sessionmaker[AsyncSession],
    sessions: Sessions,
    locks: KeyedLocks,
    user_id: int,
    session_id: str,
    card_id: str,
) -> str:
    """The card's example_fi, for 🔊. The lock is let go before the
    speaking: a slow OpenAI must not hold the next rating up."""
    async with locks.hold(_key(user_id)), session_factory() as db:
        _live(sessions, user_id, session_id)
        card = await _own_card(db, user_id, card_id)
        note = await db.get(Note, card.note_id)
        return note.example_fi
