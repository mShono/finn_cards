"""Decks and their words without Telegram (plan 2026-11-20-kirjakaveri-reader,
"П6"): the database side of /decks, a word's edit and its deletion. The bot
(bot/decks.py, bot/edit.py, bot/add.py's delete_confirm) runs on top of it,
and the reader's deck screens (reader/decks.py) do too.

No aiogram or HTTP here and no UI text. The LLM call of a lemma edit
(ingest.resolve_note_forms) stays with the caller: the bot and the reader
each have their own client and breaker, and their own answer when it
fails. This module settles the lemma, finds a clash, and saves.

Logs: none of its own - the callers log their steps. Nothing here may log
a lemma, a translation or a deck's name.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import delete, func, literal_column, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from kielikaveri.db.decks import create_deck, get_or_create_default_deck, list_decks
from kielikaveri.db.models import Card, Deck, Note, NoteKind, Review, is_note_duplicate_error
from kielikaveri.ingest import ResolvedForms, canonical_key
from kielikaveri.srs.graduation import resync_inflection_cards
from kielikaveri.srs.queue import CardCounters, card_counters

# The meta keys of a note's form set: replaced or dropped together.
FORM_META_KEYS = ("principal_forms", "forms_source", "forms_verified")


class DuplicateNoteError(Exception):
    """The note's unique index (NOTE_UNIQUE_INDEX) refused the change: the
    same lemma and part of speech is in that deck already."""


# --- decks ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeckSummary:
    deck: Deck
    notes: int
    counters: CardCounters


async def own_deck(session: AsyncSession, user_id: int, deck_id: str) -> Deck | None:
    """The user's deck; someone else's looks like a missing one (None)."""
    deck = await session.get(Deck, deck_id)
    if deck is None or deck.user_id != user_id:
        return None
    return deck


async def count_deck_notes(session: AsyncSession, user_id: int, deck_id: str) -> int:
    return (
        await session.scalar(
            select(func.count())
            .select_from(Note)
            .where(Note.deck_id == deck_id, Note.user_id == user_id)
        )
        or 0
    )


async def deck_summary(
    session: AsyncSession, user_id: int, deck: Deck, now: datetime
) -> DeckSummary:
    return DeckSummary(
        deck=deck,
        notes=await count_deck_notes(session, user_id, deck.id),
        counters=await card_counters(session, user_id, now, deck_id=deck.id),
    )


async def deck_summaries(session: AsyncSession, user_id: int, now: datetime) -> list[DeckSummary]:
    """Every deck of the user with its counters, in list_decks order. With
    no decks at all, "Общая" is created first (commits)."""
    # Default deck first - it may get created here, and list_decks() must
    # see it (a fresh user with zero decks would otherwise get an empty
    # list).
    await get_or_create_default_deck(session, user_id)
    await session.commit()
    decks = await list_decks(session, user_id)
    return [await deck_summary(session, user_id, deck, now) for deck in decks]


async def deck_notes(
    session: AsyncSession, user_id: int, deck_id: str, offset: int, limit: int
) -> list[Note]:
    """A page of the deck's notes, newest first: a freshly added word is
    the one you most often come back to fix. Ties of a second (created_at
    is whole seconds) go by insertion order, so pages don't overlap."""
    return list(
        (
            await session.scalars(
                select(Note)
                .where(Note.deck_id == deck_id, Note.user_id == user_id)
                .order_by(Note.created_at.desc(), literal_column("notes.rowid").desc())
                .offset(offset)
                .limit(limit)
            )
        ).all()
    )


async def new_deck(session: AsyncSession, user_id: int, name: str) -> Deck:
    """Commits."""
    deck = await create_deck(session, user_id, name)
    await session.commit()
    return deck


# --- a note --------------------------------------------------------------------------


async def own_note(session: AsyncSession, user_id: int, note_id: str) -> Note | None:
    """The user's note; someone else's looks like a missing one (None)."""
    note = await session.get(Note, note_id)
    if note is None or note.user_id != user_id:
        return None
    return note


async def set_translation(
    session: AsyncSession, user_id: int, note_id: str, value: str
) -> tuple[Note, str] | None:
    """The note with its new translation and the old one. Commits. None -
    not the user's note, or gone."""
    note = await own_note(session, user_id, note_id)
    if note is None:
        return None
    old = note.translation_ru
    note.translation_ru = value
    await session.commit()
    return note, old


def settle_lemma(note: Note, raw_value: str) -> str:
    """The lemma a typed value stands for. canonical_key() lemmatizes the
    same way /add's dedup does (kind=pattern has pos=None and is returned
    unchanged - a pattern's "lemma" is the construction string itself, not
    something the FST can resolve)."""
    lemma, _pos = canonical_key(raw_value, note.pos)
    return lemma


def needs_forms(note: Note) -> bool:
    """Whether a new lemma needs its forms again (resolve_note_forms, an
    LLM call). A pattern or a word with no part of speech has none."""
    return note.kind == NoteKind.word and note.pos is not None


async def find_lemma_clash(session: AsyncSession, note: Note, new_lemma: str) -> Note | None:
    """Another note of the user with this lemma in the note's deck.

    Deck-scoped, same as /add's dedup and uq_notes_user_deck_lemma_pos: the
    same word in another deck is allowed (requested 03.09.2026), so an edit
    must not refuse a rename /add would have let through."""
    pos_filter = Note.pos.is_(None) if note.pos is None else Note.pos == note.pos
    deck_filter = Note.deck_id.is_(None) if note.deck_id is None else Note.deck_id == note.deck_id
    return await session.scalar(
        select(Note).where(
            Note.user_id == note.user_id,
            Note.lemma == new_lemma,
            pos_filter,
            deck_filter,
            Note.id != note.id,
        )
    )


@dataclass(frozen=True)
class LemmaSaved:
    note: Note
    old_lemma: str
    cards_deleted: int
    cards_created: int


async def save_lemma(
    session: AsyncSession,
    note_id: str,
    new_lemma: str,
    resolved: ResolvedForms | None,
    now: datetime,
) -> LemmaSaved | None:
    """Write the new lemma and its forms, and match the inflection cards to
    them. Commits. `resolved` - the forms resolve_note_forms gave for the
    new lemma; None when they weren't recomputed (no LLM for this note, or
    the call failed). None back - the note is gone. Raises
    DuplicateNoteError when a concurrent change took the lemma in the
    meantime (the note keeps its old one)."""
    note = await session.get(Note, note_id)
    if note is None:
        return None
    old_lemma = note.lemma
    note.lemma = new_lemma
    if resolved is not None:
        new_meta = dict(note.meta)
        new_meta["principal_forms"] = resolved.principal_forms
        new_meta["forms_source"] = resolved.forms_source
        new_meta["forms_verified"] = resolved.forms_verified
        note.meta = new_meta
        # The new lemma can belong to a different part of speech than the
        # old one did (ingest.resolve_note_pos) - the forms above were
        # generated for resolved.pos, so the note has to agree with them.
        if resolved.pos is not None:
            note.pos = resolved.pos
    elif new_lemma != old_lemma:
        # Forms weren't recomputed (breaker/OpenAI) - the old lemma's forms,
        # still marked verified, would be quizzed as the new lemma's. Drop
        # them: no forms is honest, wrong ones are not.
        note.meta = {key: value for key, value in note.meta.items() if key not in FORM_META_KEYS}
    try:
        # The form set above can be another one than the cards on this
        # note were built from - a different lemma, a different POS, or
        # none at all when the recompute failed. Inside the try, and
        # before the commit, on purpose: the SELECT it starts flushes the
        # lemma written above, so a lost duplicate race still surfaces as
        # the IntegrityError below, and its rollback then takes the card
        # changes with it.
        deleted, created = await resync_inflection_cards(session, note, now)
        await session.commit()
    except IntegrityError as error:
        # The clash lookup said this lemma was free, but a concurrent /add
        # or edit took it in the meantime.
        if not is_note_duplicate_error(error):
            raise
        await session.rollback()
        raise DuplicateNoteError from None
    return LemmaSaved(note, old_lemma, len(deleted), len(created))


@dataclass(frozen=True)
class DeletedNote:
    lemma: str
    deck_id: str | None
    # None - the note was in no deck.
    deck_name: str | None
    # The deck's notes left (0 for no deck).
    remaining: int


async def delete_note(session: AsyncSession, user_id: int, note_id: str) -> DeletedNote | None:
    """Delete the note with its cards and their reviews. Commits. None - not
    the user's note, or already deleted (the loser of a double tap)."""
    note = await own_note(session, user_id, note_id)
    if note is None:
        return None

    lemma = note.lemma
    deck_id = note.deck_id
    deck = await session.get(Deck, deck_id) if deck_id else None
    deck_name = deck.name if deck else None

    # Two requests for one genuine double-tap both pass the None-check
    # above (neither has committed yet) - same race class as ingest_cache.
    # There's nothing to claim ahead of the first await here (the note to
    # delete isn't known until session.get() itself returns), so the note
    # DELETE's rowcount is the guard instead: only the invocation that
    # actually removed the row may report success.
    #
    # Children first: there is no ON DELETE CASCADE (db/models.py) and
    # PRAGMA foreign_keys is on (db/engine.py), so a note with cards, or a
    # card with reviews, cannot be deleted before them. SQLite serializes
    # the writes, so the loser of a double tap runs these after the
    # winner's commit, finds nothing left and rolls back.
    card_ids = select(Card.id).where(Card.note_id == note.id)
    await session.execute(delete(Review).where(Review.card_id.in_(card_ids)))
    await session.execute(delete(Card).where(Card.note_id == note.id))
    result = await session.execute(delete(Note).where(Note.id == note.id))
    if result.rowcount == 0:
        await session.rollback()
        return None
    await session.commit()

    remaining = 0
    if deck_id is not None:
        remaining = await count_deck_notes(session, user_id, deck_id)
    return DeletedNote(lemma, deck_id, deck_name, remaining)
