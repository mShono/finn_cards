"""🗂 Decks in the Mini App (plan 2026-11-20-kirjakaveri-reader, "П6"): the
deck list, a deck's words, a word's screen and its edits - behind
/api/decks and /api/notes, on top of decks_core.

Someone else's deck or note looks exactly like a missing one
(NotFoundError, 404). A double tap acts once: edits, moves and deletions of
a note queue on one lock per note, renaming and deleting a deck on one per
deck (KeyedLocks), and the loser of a double 🗑 finds the note gone - 404,
which the client shows as "already deleted".

A new lemma needs its forms again (ingest.resolve_note_forms, an LLM call
for some words): it goes through the reader's client and breaker, not the
bot's, and a failure changes nothing - unlike ✍️ Слово in the bot, which
saves the lemma without forms. A pattern or a word with no part of speech
needs no LLM, nor a key.

Logs: event=decks.* source=reader, with ids and numbers - never a lemma, a
translation or a deck's name (that is the user's text too). The lemma's
path runs with RedactBookText on, so the shared code's lines (ingest,
morphology) are cut to their event name.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kielikaveri import decks_core
from kielikaveri.db.models import Card, Deck, Note
from kielikaveri.decks_core import ChapterRef, DuplicateNoteError
from kielikaveri.ingest import resolve_note_forms
from kielikaveri.llm.breaker import CallBreaker
from kielikaveri.reader.lookup import KeyedLocks, _book_text_in_flight

logger = logging.getLogger(__name__)

MAX_NAME_LENGTH = 64
# A deck's page of words: the client asks for 50 at a time.
DEFAULT_PAGE = 50
MAX_PAGE = 100


class NotFoundError(Exception):
    """A deck or note that isn't the user's - it looks like a missing one."""


class BadNameError(Exception):
    """A deck's name empty after strip, or longer than MAX_NAME_LENGTH."""


class ConflictError(Exception):
    """409 with `reason`: "duplicate" (a lemma edit, with the settled
    `lemma` in `payload` - what the bot's clash reply names),
    "duplicate_in_target" (a move)."""

    def __init__(self, reason: str, payload: dict | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.payload = payload


class NoOpenAIError(Exception):
    """The edit needs the LLM and there is no OPENAI_API_KEY."""


def clean_name(value: str) -> str:
    name = value.strip()
    if not 1 <= len(name) <= MAX_NAME_LENGTH:
        raise BadNameError
    return name


def _note_key(note_id: str) -> str:
    return f"note/{note_id}"


def deck_lock_key(deck_id: str) -> str:
    """A deck's key in DECK_LOCKS; ➕ (reader/cards.py) holds it too."""
    return f"deck/{deck_id}"


# --- JSON ----------------------------------------------------------------------------


def _chapter_json(chapter: ChapterRef | None) -> dict | None:
    if chapter is None:
        return None
    return {"id": chapter.id, "book_title": chapter.book_title, "number": chapter.number}


async def _deck_json(db: AsyncSession, user_id: int, deck: Deck, now: datetime) -> dict:
    summary = await decks_core.deck_summary(db, user_id, deck, now)
    return {
        "id": deck.id,
        "name": deck.name,
        "notes": summary.notes,
        "due": summary.counters.due,
        "introduced": summary.counters.introduced,
        "not_introduced": summary.counters.not_introduced,
        "chapter": _chapter_json(await decks_core.deck_chapter(db, deck)),
    }


def _card_json(card: Card) -> dict:
    return {
        "type": card.type.value,
        "form": card.form,
        "status": card.status.value,
        "state": card.state.value,
        "due": card.due.isoformat(),
        "reps": card.reps,
        "lapses": card.lapses,
    }


async def _note_json(db: AsyncSession, user_id: int, note_id: str) -> dict:
    detail = await decks_core.note_detail(db, user_id, note_id)
    if detail is None:
        raise NotFoundError
    note = detail.note
    forms = note.meta.get("principal_forms") or None
    deck = None
    if detail.deck is not None:
        deck = {
            "id": detail.deck.id,
            "name": detail.deck.name,
            "chapter": _chapter_json(detail.chapter),
        }
    return {
        "id": note.id,
        "lemma": note.lemma,
        "pos": note.pos,
        "translation_ru": note.translation_ru,
        "example_fi": note.example_fi,
        "example_ru": note.example_ru,
        "puhekieli": note.meta.get("puhekieli"),
        "forms": forms,
        "forms_verified": note.meta.get("forms_verified") is True,
        "deck": deck,
        "cards": [_card_json(card) for card in detail.cards],
    }


# --- decks ---------------------------------------------------------------------------


async def list_decks(session_factory: async_sessionmaker[AsyncSession], user_id: int) -> dict:
    """Every deck with its counters; "Общая" is made when there are none."""
    now = datetime.now(UTC)
    async with session_factory() as db:
        summaries = await decks_core.deck_summaries(db, user_id, now)
        decks = [await _deck_json(db, user_id, s.deck, now) for s in summaries]
    logger.info("event=decks.list count=%d source=reader", len(decks))
    return {"decks": decks}


async def create_deck(
    session_factory: async_sessionmaker[AsyncSession], user_id: int, name: str
) -> dict:
    name = clean_name(name)
    async with session_factory() as db:
        deck = await decks_core.new_deck(db, user_id, name)
        body = await _deck_json(db, user_id, deck, datetime.now(UTC))
    logger.info("event=decks.create deck_id=%s source=reader", deck.id)
    return {"deck": body}


async def rename_deck(
    session_factory: async_sessionmaker[AsyncSession],
    locks: KeyedLocks,
    user_id: int,
    deck_id: str,
    name: str,
) -> dict:
    name = clean_name(name)
    async with locks.hold(deck_lock_key(deck_id)), session_factory() as db:
        deck = await decks_core.rename_deck(db, user_id, deck_id, name)
        if deck is None:
            raise NotFoundError
        body = await _deck_json(db, user_id, deck, datetime.now(UTC))
    logger.info("event=decks.rename deck_id=%s source=reader", deck_id)
    return {"deck": body}


async def delete_deck(
    session_factory: async_sessionmaker[AsyncSession],
    locks: KeyedLocks,
    user_id: int,
    deck_id: str,
) -> dict:
    async with locks.hold(deck_lock_key(deck_id)), session_factory() as db:
        deleted = await decks_core.delete_deck(db, user_id, deck_id)
    if deleted is None:
        raise NotFoundError
    logger.info("event=decks.delete deck_id=%s notes=%d source=reader", deck_id, deleted)
    return {"deleted_notes": deleted}


async def deck_notes(
    session_factory: async_sessionmaker[AsyncSession],
    user_id: int,
    deck_id: str,
    offset: int,
    limit: int,
) -> dict:
    async with session_factory() as db:
        if await decks_core.own_deck(db, user_id, deck_id) is None:
            raise NotFoundError
        total = await decks_core.count_deck_notes(db, user_id, deck_id)
        notes = await decks_core.deck_notes(db, user_id, deck_id, offset, limit)
    logger.info(
        "event=decks.notes deck_id=%s offset=%d limit=%d count=%d total=%d source=reader",
        deck_id,
        offset,
        limit,
        len(notes),
        total,
    )
    return {
        "notes": [
            {"id": n.id, "lemma": n.lemma, "translation_ru": n.translation_ru, "pos": n.pos}
            for n in notes
        ],
        "total": total,
    }


# --- notes ---------------------------------------------------------------------------


async def search(
    session_factory: async_sessionmaker[AsyncSession], user_id: int, query: str
) -> dict:
    async with session_factory() as db:
        found = await decks_core.search_notes(db, user_id, query)
    logger.info("event=decks.search query_length=%d count=%d source=reader", len(query), len(found))
    return {
        "notes": [
            {
                "id": f.note.id,
                "lemma": f.note.lemma,
                "translation_ru": f.note.translation_ru,
                "deck": None if f.deck is None else {"id": f.deck.id, "name": f.deck.name},
            }
            for f in found
        ]
    }


async def note(
    session_factory: async_sessionmaker[AsyncSession], user_id: int, note_id: str
) -> dict:
    async with session_factory() as db:
        body = await _note_json(db, user_id, note_id)
    logger.info("event=decks.note note_id=%s cards=%d source=reader", note_id, len(body["cards"]))
    return body


async def set_translation(
    session_factory: async_sessionmaker[AsyncSession],
    locks: KeyedLocks,
    user_id: int,
    note_id: str,
    value: str,
) -> dict:
    async with locks.hold(_note_key(note_id)), session_factory() as db:
        if await decks_core.set_translation(db, user_id, note_id, value) is None:
            raise NotFoundError
        body = await _note_json(db, user_id, note_id)
    logger.info("event=decks.note_edit note_id=%s field=translation_ru source=reader", note_id)
    return {"note": body}


async def set_lemma(
    session_factory: async_sessionmaker[AsyncSession],
    client: AsyncOpenAI,
    breaker: CallBreaker,
    model: str,
    has_key: bool,
    locks: KeyedLocks,
    user_id: int,
    note_id: str,
    value: str,
) -> dict:
    """✍️ Слово: the new lemma, its forms again, the inflection cards to
    match. Raises NotFoundError, ConflictError("duplicate"), NoOpenAIError
    and what resolve_note_forms raises (CircuitOpenError, OpenAI's errors) -
    the note is left as it was on all of them."""
    flag = _book_text_in_flight.set(True)
    try:
        async with locks.hold(_note_key(note_id)):
            return await _set_lemma(
                session_factory, client, breaker, model, has_key, user_id, note_id, value
            )
    finally:
        _book_text_in_flight.reset(flag)


async def _set_lemma(
    session_factory: async_sessionmaker[AsyncSession],
    client: AsyncOpenAI,
    breaker: CallBreaker,
    model: str,
    has_key: bool,
    user_id: int,
    note_id: str,
    value: str,
) -> dict:
    async with session_factory() as db:
        current = await decks_core.own_note(db, user_id, note_id)
        if current is None:
            raise NotFoundError
        # The FST off the event loop, as the lookup's: a word's first
        # analysis can take a while. The thread gets a copy of this
        # context, so its log lines are cut as well.
        new_lemma = await asyncio.to_thread(decks_core.settle_lemma, current, value)
        clash = await decks_core.find_lemma_clash(db, current, new_lemma)
    if clash is not None:
        logger.info(
            "event=decks.lemma_clash note_id=%s clash_note_id=%s source=reader", note_id, clash.id
        )
        raise ConflictError("duplicate", {"lemma": new_lemma})

    resolved = None
    if decks_core.needs_forms(current):
        if not has_key:
            raise NoOpenAIError
        resolved, _usage = await resolve_note_forms(
            client,
            breaker,
            model,
            new_lemma,
            current.pos,
            datetime.now(UTC),
            context=current.example_fi,
        )

    async with session_factory() as db:
        try:
            saved = await decks_core.save_lemma(db, note_id, new_lemma, resolved, datetime.now(UTC))
        except DuplicateNoteError:
            logger.info("event=decks.lemma_clash_race note_id=%s source=reader", note_id)
            raise ConflictError("duplicate", {"lemma": new_lemma}) from None
        if saved is None:
            raise NotFoundError
        body = await _note_json(db, user_id, note_id)
    logger.info(
        "event=decks.note_edit note_id=%s field=lemma forms_source=%s cards_deleted=%d "
        "cards_created=%d source=reader",
        note_id,
        resolved.forms_source if resolved is not None else "unchanged",
        saved.cards_deleted,
        saved.cards_created,
    )
    return {"note": body}


async def move_note(
    session_factory: async_sessionmaker[AsyncSession],
    locks: KeyedLocks,
    user_id: int,
    note_id: str,
    deck_id: str,
) -> dict:
    # The note's lock, then the target deck's: a deck deleted meanwhile
    # must not get a note moved into it. Deleting a deck takes the deck's
    # lock only, so the order can't deadlock.
    async with (
        locks.hold(_note_key(note_id)),
        locks.hold(deck_lock_key(deck_id)),
        session_factory() as db,
    ):
        before = await decks_core.own_note(db, user_id, note_id)
        from_deck = before.deck_id if before is not None else None
        try:
            moved = await decks_core.move_note(db, user_id, note_id, deck_id)
        except DuplicateNoteError:
            logger.info(
                "event=decks.move_conflict note_id=%s deck_id=%s source=reader",
                note_id,
                deck_id,
            )
            raise ConflictError("duplicate_in_target") from None
        if moved is None:
            raise NotFoundError
        body = await _note_json(db, user_id, note_id)
    logger.info(
        "event=decks.move note_id=%s from_deck_id=%s deck_id=%s source=reader",
        note_id,
        from_deck,
        deck_id,
    )
    return {"note": body}


async def delete_note(
    session_factory: async_sessionmaker[AsyncSession],
    locks: KeyedLocks,
    user_id: int,
    note_id: str,
) -> None:
    async with locks.hold(_note_key(note_id)), session_factory() as db:
        deleted = await decks_core.delete_note(db, user_id, note_id)
    if deleted is None:
        raise NotFoundError
    logger.info(
        "event=decks.note_delete note_id=%s deck_id=%s remaining=%d source=reader",
        note_id,
        deleted.deck_id,
        deleted.remaining,
    )


async def text_to_speak(
    session_factory: async_sessionmaker[AsyncSession], user_id: int, note_id: str, what: str
) -> str:
    """The note's lemma or example_fi, for 🔊."""
    async with session_factory() as db:
        found: Note | None = await decks_core.own_note(db, user_id, note_id)
    if found is None:
        raise NotFoundError
    return found.lemma if what == "lemma" else found.example_fi
