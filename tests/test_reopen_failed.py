"""Opening chapter N queues N+1 only while it is pending; a failed N+1 is
retried by the bot's button alone (plan 2026-11-20-kirjakaveri-reader,
review item И2)."""

from unittest.mock import MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer
from bot_helpers import make_settings
from sqlalchemy import update
from test_reader import BOOK_TEXT, FI, OWNER, TOKEN, header

from kielikaveri.db.books import chapter_by_number, mark_opened, queue_chapter, store_book
from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import BookChapter, ChapterStatus
from kielikaveri.reader.app import make_app
from kielikaveri.translate.chapters import split_book


@pytest.fixture
async def session_factory(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    await create_all(engine)
    yield make_session_factory(engine)
    await engine.dispose()


async def _set_status(session_factory, book_id, number, status, **values):
    async with session_factory() as session:
        await session.execute(
            update(BookChapter)
            .where(BookChapter.book_id == book_id, BookChapter.number == number)
            .values(status=status, **values)
        )
        await session.commit()


@pytest.fixture
async def book_id(session_factory):
    """A three-chapter book of OWNER's: chapter 1 done, chapter 2 failed."""
    async with session_factory() as session:
        book = await store_book(session, OWNER, "Kirja", BOOK_TEXT, split_book(BOOK_TEXT))
        await session.commit()
    await _set_status(session_factory, book.id, 1, ChapterStatus.done, fi_paragraphs=FI)
    await _set_status(session_factory, book.id, 2, ChapterStatus.failed)
    return book.id


@pytest.fixture
def translator():
    return MagicMock()


@pytest.fixture
async def client(session_factory, translator):
    settings = make_settings(bot_token=TOKEN, whitelist_user_ids=str(OWNER))
    async with TestClient(TestServer(make_app(session_factory, settings, translator))) as client:
        yield client


async def _chapter(session_factory, book_id, number) -> BookChapter:
    async with session_factory() as session:
        return await chapter_by_number(session, book_id, number)


async def test_mark_opened_leaves_a_failed_next_chapter_alone(session_factory, book_id):
    async with session_factory() as session:
        first = await chapter_by_number(session, book_id, 1)
        assert await mark_opened(session, first) is None
        assert first.opened_at is not None
        await session.commit()
    assert (await _chapter(session_factory, book_id, 2)).status is ChapterStatus.failed


async def test_retry_button_path_still_requeues_a_failed_chapter(session_factory, book_id):
    second = await _chapter(session_factory, book_id, 2)
    async with session_factory() as session:
        assert await queue_chapter(session, second.id)
        await session.commit()
    assert (await _chapter(session_factory, book_id, 2)).status is ChapterStatus.queued


async def test_reopening_with_a_failed_next_chapter_does_not_wake_the_translator(
    client, session_factory, book_id, translator
):
    first = await _chapter(session_factory, book_id, 1)
    for _ in range(2):
        response = await client.get(f"/api/chapters/{first.id}", headers=header())
        assert response.status == 200
        data = await response.json()
        assert data["next"]["status"] == "failed"
    assert (await _chapter(session_factory, book_id, 2)).status is ChapterStatus.failed
    translator.wake.assert_not_called()


async def test_a_pending_next_chapter_is_still_queued(client, session_factory, book_id, translator):
    await _set_status(session_factory, book_id, 2, ChapterStatus.pending)
    first = await _chapter(session_factory, book_id, 1)
    response = await client.get(f"/api/chapters/{first.id}", headers=header())
    assert response.status == 200
    assert (await response.json())["next"]["status"] == "queued"
    assert (await _chapter(session_factory, book_id, 2)).status is ChapterStatus.queued
    translator.wake.assert_called_once()
