"""Background translator: takes queued chapters one at a time and runs the
P0 pipeline on them inside the bot's process (plan 2026-11-20-kirjakaveri-reader,
"П1").

In the bot's process on purpose: one SQLite file (two processes would hit
"database is locked"), the FST loaded once, deploy unchanged. A chapter is
minutes of network waits, not CPU; the FST check, the one synchronous part,
runs in a thread (pipeline._fst_flags).

Steps are fixed by the P0 runs: review always on (the draft alone has too
many meaning errors), the Finnish-only editor off. The glossary is built
once per book, from the opening of every chapter, right before its first
chapter - and reused for every chapter after.

No Telegram here: the bot passes `notify` and formats the message itself.
Nothing of the book's text reaches the log - ids, counts and tokens only.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import openai
from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kielikaveri.db.books import claim_next, reset_interrupted
from kielikaveri.db.models import Book, BookChapter, ChapterStatus
from kielikaveri.llm.breaker import CallBreaker, CircuitOpenError
from kielikaveri.translate import chapters as parsed
from kielikaveri.translate.pipeline import (
    BookGuide,
    TranslationError,
    Usage,
    prepare_guide,
    translate_chapter,
)

logger = logging.getLogger(__name__)

_ERROR_CHARS = 500
# After an unexpected failure of the loop itself (not of a chapter) - a DB
# error that repeats must not spin.
_LOOP_ERROR_PAUSE_SECONDS = 60


@dataclass
class ChapterEvent:
    """A chapter finished translating, one way or the other."""

    user_id: int
    book_title: str
    chapter_id: str
    number: int
    status: ChapterStatus
    error: str | None = None


def _error_text(exc: BaseException) -> str:
    # Class name first: the bot tells a refusal from a misalignment by it.
    return f"{type(exc).__name__}: {exc}"[:_ERROR_CHARS]


def _guide(book: Book) -> BookGuide | None:
    if book.glossary is None:
        return None
    return BookGuide(book.glossary["characters"], book.glossary["terms"], book.style_notes or "")


class Translator:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        client_factory: Callable[[], AsyncOpenAI],
        model: str,
        breaker: CallBreaker,
        notify: Callable[[ChapterEvent], Awaitable[None]],
    ) -> None:
        self._session_factory = session_factory
        self._client_factory = client_factory
        self._model = model
        self._breaker = breaker
        self._notify = notify
        self._wake = asyncio.Event()

    def wake(self) -> None:
        """Call after queueing a chapter; the loop otherwise sleeps."""
        self._wake.set()

    async def run(self) -> None:
        async with self._session_factory() as session:
            reset = await reset_interrupted(session)
            await session.commit()
        if reset:
            logger.info("event=translate.requeued chapters=%d", reset)
        while True:
            # Cleared before looking, so a wake() during the claim isn't lost.
            self._wake.clear()
            try:
                async with self._session_factory() as session:
                    chapter_id = await claim_next(session)
                    await session.commit()
            except Exception:
                logger.exception("event=translate.loop_error")
                await asyncio.sleep(_LOOP_ERROR_PAUSE_SECONDS)
                continue
            if chapter_id is None:
                await self._wake.wait()
                continue
            await self.translate_one(chapter_id)

    async def translate_one(self, chapter_id: str) -> None:
        """Translate a chapter already moved to `translating`. Never raises
        (except cancellation): the loop must outlive any one chapter."""
        async with self._session_factory() as session:
            chapter = await session.get(BookChapter, chapter_id)
            if chapter is None:
                return
            book = await session.get(Book, chapter.book_id)
            event = ChapterEvent(
                book.user_id, book.title, chapter.id, chapter.number, ChapterStatus.failed
            )
            guide = _guide(book)
            if guide is None:
                rows = await session.scalars(
                    select(BookChapter)
                    .where(BookChapter.book_id == book.id)
                    .order_by(BookChapter.number)
                )
                whole = parsed.Book(
                    front_matter=book.front_matter,
                    chapters=[
                        parsed.Chapter(c.number, c.part, c.title, c.source_paragraphs)
                        for c in rows.all()
                    ],
                )
            source = parsed.Chapter(
                chapter.number, chapter.part, chapter.title, chapter.source_paragraphs
            )

        logger.info(
            "event=translate.chapter_start book_id=%s chapter=%d paragraphs=%d chars=%d",
            book.id,
            source.number,
            len(source.paragraphs),
            source.chars,
        )
        usage = Usage()
        try:
            client = self._client_factory()
            if guide is None:
                guide = await prepare_guide(
                    client, self._breaker, self._model, usage, whole, full_chapters=[]
                )
                async with self._session_factory() as session:
                    stored = await session.get(Book, book.id)
                    stored.glossary = {"characters": guide.characters, "terms": guide.terms}
                    stored.style_notes = guide.style_notes
                    await session.commit()
                logger.info(
                    "event=translate.guide_ready book_id=%s characters=%d terms=%d",
                    book.id,
                    len(guide.characters),
                    len(guide.terms),
                )
            result = await translate_chapter(
                client, self._breaker, self._model, usage, guide, source, review=True, edit=False
            )
        except asyncio.CancelledError:
            raise
        except (TranslationError, CircuitOpenError, openai.APIError) as exc:
            event.error = _error_text(exc)
            logger.warning(
                "event=translate.chapter_failed book_id=%s chapter=%d error=%s",
                book.id,
                source.number,
                type(exc).__name__,
            )
        except Exception as exc:
            event.error = _error_text(exc)
            logger.exception(
                "event=translate.chapter_failed book_id=%s chapter=%d", book.id, source.number
            )
        else:
            event.status = ChapterStatus.done

        async with self._session_factory() as session:
            stored = await session.get(BookChapter, chapter_id)
            stored.status = event.status
            stored.error = event.error
            if event.status is ChapterStatus.done:
                stored.fi_paragraphs = [result.final[i] for i in range(len(source.paragraphs))]
                # JSON object keys are strings anyway - made explicit here.
                stored.fst_flags = {str(k): v for k, v in result.fst_flags_final.items()}
                stored.review_issues = [asdict(issue) for issue in result.issues]
                stored.translated_at = datetime.now(UTC)
            await session.commit()

        steps = usage.steps.values()
        logger.info(
            "event=translate.chapter_end book_id=%s chapter=%d status=%s calls=%d "
            "input_tokens=%d output_tokens=%d seconds=%d",
            book.id,
            source.number,
            event.status.value,
            sum(s.calls for s in steps),
            sum(s.input_tokens for s in steps),
            sum(s.output_tokens for s in steps),
            int(sum(s.seconds for s in steps)),
        )
        try:
            await self._notify(event)
        except Exception:
            logger.exception("event=translate.notify_failed chapter_id=%s", chapter_id)
