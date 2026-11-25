"""➕ in the reader's popup: the tapped word saved as a card (plan
2026-11-20-kirjakaveri-reader, "П4").

The client sends the word's place only, as for a lookup. Everything else is
the server's: the lookup's answer (from lookup_cache, or a fresh lookup on a
miss - same breaker, same errors), the chapter's deck ("<book> - гл.N",
found by decks.chapter_id, created on the first ➕), the example (a piece of
the book, reader/excerpt.py) and /add's own saving (cards_save): the FST's
lemma and forms, the schema, the dedup.

The order saves money on a word already in the deck: the lemma is settled
and checked for a duplicate before the example is asked for.

A double tap on ➕ gives one card: the two requests queue on one lock per
user and word, and the second finds the first's note (duplicate); two
different words of the same lemma are told apart by the unique index
uq_notes_user_deck_lemma_pos, again a duplicate.

Logs: ids, the outcome, tokens and time - never the word, the lemma, the
sentence or the example. The whole path runs with RedactBookText on, so the
shared code's lines (ingest, morphology, cards_save) are cut to their event
name.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kielikaveri.cards_save import (
    DedupState,
    SaveOutcome,
    SaveStatus,
    Usage,
    settle_lemma,
    store_note,
)
from kielikaveri.db.decks import get_or_create_chapter_deck
from kielikaveri.db.models import Book, BookChapter, Deck, Note
from kielikaveri.ingest import SURFACE_FIELD, existing_note_keys
from kielikaveri.llm.breaker import CallBreaker
from kielikaveri.reader.excerpt import make_excerpt
from kielikaveri.reader.lookup import (
    MARK_CLOSE,
    MARK_OPEN,
    KeyedLocks,
    Target,
    _book_text_in_flight,
    lookup,
)
from kielikaveri.reader.warmup import Warmup

logger = logging.getLogger(__name__)


def deck_name(book: Book, chapter: BookChapter) -> str:
    return f"{book.title} - гл.{chapter.number}"


async def in_chapter_deck(
    session_factory: async_sessionmaker[AsyncSession],
    chapter_id: str,
    lemma: str,
    pos: str | None,
) -> bool:
    """Whether the chapter's deck has a note of this lemma and part of
    speech. No deck yet - False."""
    async with session_factory() as session:
        found = await session.scalar(
            select(Note.id)
            .join(Deck, Note.deck_id == Deck.id)
            .where(Deck.chapter_id == chapter_id, Note.lemma == lemma, Note.pos == pos)
            .limit(1)
        )
    return found is not None


@dataclass(frozen=True)
class CardResult:
    status: SaveStatus
    deck: str
    # For a failure: the exception behind it (cards_save.SaveOutcome.error).
    error: Exception | None = None


def _sentence_and_span(target: Target) -> tuple[str, tuple[int, int]]:
    """The plain sentence and the word's offsets in it."""
    marked = target.marked_sentence
    start = marked.index(MARK_OPEN)
    end = marked.index(MARK_CLOSE) - len(MARK_OPEN)
    return marked.replace(MARK_OPEN, "").replace(MARK_CLOSE, ""), (start, end)


async def add_card(
    session_factory: async_sessionmaker[AsyncSession],
    client: AsyncOpenAI,
    breaker: CallBreaker,
    model: str,
    lookup_locks: KeyedLocks,
    card_locks: KeyedLocks,
    warmup: Warmup | None,
    user_id: int,
    chapter: BookChapter,
    book: Book,
    target: Target,
) -> CardResult:
    """Raises what lookup and the excerpt raise (CircuitOpenError,
    LookupFailedError, OpenAI's errors); a save that fails on its own is a
    CardResult with status failed."""
    started = time.monotonic()
    usage = Usage()
    flag = _book_text_in_flight.set(True)
    status = "error"
    try:
        key = f"{user_id}/{target.chapter_id}/{target.paragraph}/{target.token}"
        async with card_locks.hold(key):
            result = await _add_card(
                session_factory,
                client,
                breaker,
                model,
                lookup_locks,
                warmup,
                user_id,
                chapter,
                book,
                target,
                usage,
            )
        status = result.status.value
        return result
    finally:
        _book_text_in_flight.reset(flag)
        logger.info(
            "event=reader.card chapter_id=%s paragraph=%d token=%d status=%s "
            "input_tokens=%d output_tokens=%d duration_ms=%d",
            target.chapter_id,
            target.paragraph,
            target.token,
            status,
            usage.input,
            usage.output,
            (time.monotonic() - started) * 1000,
        )


async def _add_card(
    session_factory: async_sessionmaker[AsyncSession],
    client: AsyncOpenAI,
    breaker: CallBreaker,
    model: str,
    lookup_locks: KeyedLocks,
    warmup: Warmup | None,
    user_id: int,
    chapter: BookChapter,
    book: Book,
    target: Target,
    usage: Usage,
) -> CardResult:
    word = await lookup(
        session_factory, client, breaker, model, lookup_locks, target, warmup, usage=usage
    )
    sentence, span = _sentence_and_span(target)

    async with session_factory() as session:
        deck = await get_or_create_chapter_deck(session, user_id, chapter, deck_name(book, chapter))
        await session.commit()
        deck_id, name = deck.id, deck.name
        dedup = DedupState(await existing_note_keys(session, user_id, deck_id=deck_id))

    candidate: dict = {
        "kind": "word",
        "lemma": word["lemma"],
        "pos": word["pos"],
        "translation_ru": word["translation_ru"],
    }
    # As in the lookup: a colloquial spelling says nothing to the FST, the
    # literary lemma is resolved alone; the spelling goes to meta.puhekieli.
    if word["puhekieli"]:
        candidate["meta"] = {"puhekieli": word["puhekieli"]}
    else:
        candidate[SURFACE_FIELD] = target.surface

    now = datetime.now(UTC)
    before = usage.input
    # The whole sentence is the context of the lemma and the forms, as in
    # the lookup - the example is only a piece of it.
    settled = await settle_lemma(
        client, breaker, model, candidate, dedup, now, context=sentence, usage=usage
    )
    _touch_if_called(warmup, usage, before)
    if isinstance(settled, SaveOutcome):
        return CardResult(settled.status, name, settled.error)

    excerpt = await make_excerpt(
        client, breaker, model, sentence, span, word["sentence_ru"], usage, warmup
    )
    before = usage.input
    settled.candidate = {
        **settled.candidate,
        "example_fi": excerpt.example_fi,
        "example_ru": excerpt.example_ru,
    }
    outcome = await store_note(
        session_factory,
        client,
        breaker,
        model,
        user_id,
        deck_id,
        settled,
        dedup,
        now,
        context=sentence,
        usage=usage,
    )
    _touch_if_called(warmup, usage, before)
    return CardResult(outcome.status, name, outcome.error)


def _touch_if_called(warmup: Warmup | None, usage: Usage, before: int) -> None:
    """The lemma and forms settling call OpenAI only now and then; the
    tokens tell whether it did (reader/warmup.py keeps the time)."""
    if warmup is not None and usage.input > before:
        warmup.touch()
