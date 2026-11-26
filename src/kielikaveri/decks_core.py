"""Decks and their words without Telegram (plan 2026-11-20-kirjakaveri-reader,
"П6"): the database side of /decks, a word's edit and its deletion. The bot
(bot/decks.py, bot/edit.py, bot/add.py's delete_confirm) runs on top of it,
and the reader's deck screens (reader/decks.py) do too.

No aiogram or HTTP here and no UI text. The LLM call of a lemma edit
(ingest.resolve_note_forms) stays with the caller: the bot and the reader
each have their own client and breaker, and their own answer when it
fails. This module settles the lemma, finds a clash, and saves.

What the bot never had and the reader adds: renaming and deleting a deck,
moving a word to another deck (its history goes with it), a search over
every deck and the word's details with its cards.

Logs: none of its own - the callers log their steps. Nothing here may log
a lemma, a translation or a deck's name.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import delete, func, literal_column, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.exc import StaleDataError

from kielikaveri.db.decks import create_deck, get_or_create_default_deck, list_decks
from kielikaveri.db.models import (
    Book,
    BookChapter,
    Card,
    CardType,
    Deck,
    Note,
    NoteKind,
    Review,
    is_note_duplicate_error,
)
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
    try:
        await session.commit()
    except StaleDataError:
        # Deleted meanwhile, with its deck (delete_deck) or alone: the
        # UPDATE found no row.
        await session.rollback()
        return None
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
    the call failed). None back - the note is gone, or went while this ran. Raises
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
    except StaleDataError:
        # Deleted meanwhile (its deck deleted, say): the UPDATE found no row.
        await session.rollback()
        return None
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


# --- what the reader adds (П6) -------------------------------------------------------

# A search answers with this many notes at most.
SEARCH_LIMIT = 50


@dataclass(frozen=True)
class ChapterRef:
    """The chapter a deck was made for (decks.chapter_id), for its label and
    the way back into the book."""

    id: str
    book_title: str
    number: int


async def deck_chapter(session: AsyncSession, deck: Deck) -> ChapterRef | None:
    """None - not a chapter's deck, or its book was deleted since (SET NULL)."""
    if deck.chapter_id is None:
        return None
    chapter = await session.get(BookChapter, deck.chapter_id)
    if chapter is None:
        return None
    book = await session.get(Book, chapter.book_id)
    return ChapterRef(chapter.id, book.title, chapter.number)


async def rename_deck(session: AsyncSession, user_id: int, deck_id: str, name: str) -> Deck | None:
    """Commits. None - not the user's deck."""
    deck = await own_deck(session, user_id, deck_id)
    if deck is None:
        return None
    deck.name = name
    await session.commit()
    return deck


async def delete_deck(session: AsyncSession, user_id: int, deck_id: str) -> int | None:
    """Delete the deck with its notes, their cards and the cards' reviews,
    in one transaction. Commits. The number of notes deleted; None - not the
    user's deck, or deleted already (a double tap: the deck DELETE's
    rowcount decides, as delete_note's does).

    Children first, as in delete_note: no ON DELETE CASCADE, foreign keys
    on. A chapter's deck goes the same way - the next ➕ in the chapter
    makes a new, empty one."""
    if await own_deck(session, user_id, deck_id) is None:
        return None
    note_ids = select(Note.id).where(Note.deck_id == deck_id, Note.user_id == user_id)
    card_ids = select(Card.id).where(Card.note_id.in_(note_ids))
    await session.execute(delete(Review).where(Review.card_id.in_(card_ids)))
    await session.execute(delete(Card).where(Card.note_id.in_(note_ids)))
    notes = await session.execute(
        delete(Note).where(Note.deck_id == deck_id, Note.user_id == user_id)
    )
    result = await session.execute(delete(Deck).where(Deck.id == deck_id, Deck.user_id == user_id))
    if result.rowcount == 0:
        await session.rollback()
        return None
    await session.commit()
    return notes.rowcount


async def move_note(session: AsyncSession, user_id: int, note_id: str, deck_id: str) -> Note | None:
    """Put the note into another deck of the user. Only notes.deck_id
    changes: the cards and their reviews stay with the note, so does its
    history. Commits. None - the note or the deck is not the user's (or the
    note was deleted while this ran). The
    same deck - nothing changes. Raises DuplicateNoteError when that deck
    has the same lemma and part of speech already (NOTE_UNIQUE_INDEX)."""
    note = await own_note(session, user_id, note_id)
    deck = await own_deck(session, user_id, deck_id)
    if note is None or deck is None:
        return None
    if note.deck_id == deck.id:
        return note
    note.deck_id = deck.id
    try:
        await session.commit()
    except StaleDataError:
        # Deleted meanwhile, with its old deck (delete_deck) or alone: the
        # UPDATE found no row.
        await session.rollback()
        return None
    except IntegrityError as error:
        if not is_note_duplicate_error(error):
            raise
        await session.rollback()
        raise DuplicateNoteError from None
    return note


@dataclass(frozen=True)
class FoundNote:
    note: Note
    # None - a note in no deck (old notes, import_cards.py).
    deck: Deck | None


def _match_rank(query: str, value: str | None) -> int | None:
    """0 - the whole value, 1 - the start of a word in it, 2 - anywhere
    else; None - not in it. Both casefolded already."""
    if not value or query not in value:
        return None
    if value == query:
        return 0
    start = value.find(query)
    while start != -1:
        if start == 0 or not value[start - 1].isalnum():
            return 1
        start = value.find(query, start + 1)
    return 2


async def search_notes(
    session: AsyncSession, user_id: int, query: str, limit: int = SEARCH_LIMIT
) -> list[FoundNote]:
    """The user's notes, every deck, whose lemma, translation or colloquial
    spelling (meta.puhekieli) has `query` in it, regardless of case.

    SQLite's lower() folds ASCII only (not ä/ö, not Cyrillic), so the
    matching is Python's casefold() over the user's notes in memory - one
    user, hundreds of notes. Exact matches first, then the start of a word,
    then the rest; by lemma within each."""
    folded = query.strip().casefold()
    if not folded:
        return []
    notes = (await session.scalars(select(Note).where(Note.user_id == user_id))).all()
    ranked = []
    for note in notes:
        puhekieli = note.meta.get("puhekieli")
        values = (note.lemma, note.translation_ru, puhekieli if isinstance(puhekieli, str) else "")
        ranks = [r for v in values if (r := _match_rank(folded, (v or "").casefold())) is not None]
        if ranks:
            ranked.append((min(ranks), note.lemma.casefold(), note.id, note))
    ranked.sort(key=lambda row: row[:3])
    decks = {deck.id: deck for deck in await list_decks(session, user_id)}
    return [FoundNote(row[3], decks.get(row[3].deck_id)) for row in ranked[:limit]]


@dataclass(frozen=True)
class NoteDetail:
    note: Note
    deck: Deck | None
    chapter: ChapterRef | None
    cards: list[Card]


_CARD_TYPE_ORDER = {
    CardType.recognition: 0,
    CardType.production: 1,
    CardType.inflection: 2,
}


async def note_detail(session: AsyncSession, user_id: int, note_id: str) -> NoteDetail | None:
    """The note with its deck, the deck's chapter and every card - for the
    word's screen. None - not the user's note."""
    note = await own_note(session, user_id, note_id)
    if note is None:
        return None
    deck = await session.get(Deck, note.deck_id) if note.deck_id else None
    chapter = await deck_chapter(session, deck) if deck is not None else None
    cards = (await session.scalars(select(Card).where(Card.note_id == note.id))).all()
    forms = list((note.meta.get("principal_forms") or {}).keys())

    def order(card: Card) -> tuple:
        form_index = forms.index(card.form) if card.form in forms else len(forms)
        return (_CARD_TYPE_ORDER[card.type], form_index, card.form or "", card.id)

    return NoteDetail(note, deck, chapter, sorted(cards, key=order))
