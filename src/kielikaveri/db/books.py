"""Books and chapters in the DB - the boundary between the translator and
the reader (plan 2026-11-20-kirjakaveri-reader, "Архитектура"). The
translator moves chapters through ChapterStatus and writes fi_paragraphs;
the reader only reads them and reports "chapter N opened", which queues
chapter N+1.

Status changes are conditional UPDATEs (`WHERE status IN (...)`), so a
double-tapped button or a restart can't queue or claim a chapter twice.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from kielikaveri.db.models import Book, BookChapter, ChapterStatus
from kielikaveri.translate import chapters as parsed


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def detect_lang(text: str) -> str:
    """ "ru" or "en" - the only two source languages the translator takes."""
    cyrillic = sum(1 for ch in text if "а" <= ch.lower() <= "я" or ch in "ёЁ")
    latin = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    return "ru" if cyrillic >= latin else "en"


async def find_book(session: AsyncSession, user_id: int, source_hash: str) -> Book | None:
    return await session.scalar(
        select(Book).where(Book.user_id == user_id, Book.source_hash == source_hash)
    )


async def store_book(
    session: AsyncSession,
    user_id: int,
    title: str,
    text: str,
    book: parsed.Book,
    *,
    source_lang: str | None = None,
    source_hash: str | None = None,
) -> Book:
    """source_lang and source_hash may come precomputed - the bot works them
    out off the event loop together with the parsing."""
    row = Book(
        user_id=user_id,
        title=title,
        source_lang=source_lang or detect_lang(text),
        source_hash=source_hash or text_hash(text),
        front_matter=book.front_matter,
    )
    session.add(row)
    await session.flush()
    session.add_all(
        BookChapter(
            book_id=row.id,
            number=chapter.number,
            part=chapter.part,
            title=chapter.title,
            source_paragraphs=chapter.paragraphs,
        )
        for chapter in book.chapters
    )
    await session.flush()
    return row


async def delete_book(session: AsyncSession, book_id: str) -> None:
    await session.execute(delete(BookChapter).where(BookChapter.book_id == book_id))
    await session.execute(delete(Book).where(Book.id == book_id))


async def delete_unstarted_book(session: AsyncSession, book_id: str, user_id: int) -> bool:
    """Deletes the book only while every chapter is still pending - the
    check and the delete are the same statements, so a "yes" landing
    in between can't lose a queued chapter. False if nothing was deleted
    (not this user's book, or translation has started).

    Chapters go first (books.id has no ON DELETE CASCADE); decks made from
    a chapter keep their cards, decks.chapter_id is SET NULL by its FK
    (enforced: PRAGMA foreign_keys=ON, db/engine.py)."""
    started = (
        select(BookChapter.id)
        .where(BookChapter.book_id == book_id, BookChapter.status != ChapterStatus.pending)
        .exists()
    )
    owned = select(Book.id).where(Book.id == book_id, Book.user_id == user_id).exists()
    await session.execute(
        delete(BookChapter).where(BookChapter.book_id == book_id, owned, ~started)
    )
    # Gone only if its chapters are: a started book keeps them all.
    left = select(BookChapter.id).where(BookChapter.book_id == book_id).exists()
    result = await session.execute(
        delete(Book).where(Book.id == book_id, Book.user_id == user_id, ~left)
    )
    return result.rowcount == 1


async def chapter_by_number(session: AsyncSession, book_id: str, number: int) -> BookChapter | None:
    return await session.scalar(
        select(BookChapter).where(BookChapter.book_id == book_id, BookChapter.number == number)
    )


async def _move(
    session: AsyncSession, chapter_id: str, from_: set[ChapterStatus], to: ChapterStatus
) -> bool:
    result = await session.execute(
        update(BookChapter)
        .where(BookChapter.id == chapter_id, BookChapter.status.in_(from_))
        .values(status=to)
    )
    return result.rowcount == 1


async def queue_chapter(session: AsyncSession, chapter_id: str) -> bool:
    """pending or failed -> queued. False if it is already queued, running or done."""
    return await _move(
        session, chapter_id, {ChapterStatus.pending, ChapterStatus.failed}, ChapterStatus.queued
    )


async def mark_opened(session: AsyncSession, chapter: BookChapter) -> BookChapter | None:
    """The reader's one signal to the translator: chapter N is being read,
    so N+1 goes into the queue and is ready by the time N is finished.
    Returns the chapter that was queued, if any.

    Only a pending N+1 is queued: a failed one waits for the "retry" button,
    otherwise every reopening of N would pay for another doomed attempt."""
    if chapter.opened_at is None:
        chapter.opened_at = datetime.now(UTC)
    following = await chapter_by_number(session, chapter.book_id, chapter.number + 1)
    if following is not None and await _move(
        session, following.id, {ChapterStatus.pending}, ChapterStatus.queued
    ):
        return following
    return None


async def reset_interrupted(session: AsyncSession) -> int:
    """translating -> queued after a restart: nothing can still be running."""
    result = await session.execute(
        update(BookChapter)
        .where(BookChapter.status == ChapterStatus.translating)
        .values(status=ChapterStatus.queued)
    )
    return result.rowcount


async def claim_next(session: AsyncSession) -> str | None:
    """The oldest queued chapter, moved to translating; its id, or None."""
    candidates = await session.scalars(
        select(BookChapter.id)
        .join(Book)
        .where(BookChapter.status == ChapterStatus.queued)
        .order_by(Book.created_at, BookChapter.number)
    )
    for chapter_id in candidates.all():
        if await _move(session, chapter_id, {ChapterStatus.queued}, ChapterStatus.translating):
            return chapter_id
    return None
