"""Books in the bot (plan 2026-11-20-kirjakaveri-reader, "П1"): storage and
chapter queue (db/books.py), the background translator (translate/worker.py)
and the .txt upload flow through bot/main.py's real Dispatcher."""

import asyncio
import io
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiogram import Bot
from aiogram.fsm.storage.base import StorageKey
from aiogram.methods import SendMessage
from aiogram.types import Chat, Document, Update
from aiogram.types import Message as TgMessage
from bot_helpers import (
    RecordingSession,
    make_breaker,
    make_settings,
    routed_dispatcher,
    tg_callback_update,
    tg_user,
)
from conftest import add_users
from sqlalchemy import func, select

from kielikaveri.bot.add import AddStates
from kielikaveri.bot.books import BUSY_TEXT, NOT_TXT_TEXT, send_chapter_event
from kielikaveri.db.books import (
    chapter_by_number,
    claim_next,
    detect_lang,
    mark_opened,
    queue_chapter,
    reset_interrupted,
    store_book,
)
from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import Book, BookChapter, ChapterStatus
from kielikaveri.llm.breaker import CallBreaker
from kielikaveri.translate import fst_check
from kielikaveri.translate.chapters import split_book
from kielikaveri.translate.worker import ChapterEvent, Translator

PARAGRAPH = "Он ушёл из дома рано утром, и никто не видел, куда он пошёл. " * 6
CHAPTERS = [["1", PARAGRAPH, PARAGRAPH], ["2", PARAGRAPH], ["3", PARAGRAPH, PARAGRAPH]]
BOOK_TEXT = "\n\n".join(["Название"] + [line for chapter in CHAPTERS for line in chapter])
KNOWN = {"hän", "meni", "talo"}


@pytest.fixture
async def session_factory(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    await create_all(engine)
    session_factory = make_session_factory(engine)
    # The users the tests' rows point at (PRAGMA foreign_keys is on).
    await add_users(session_factory, 1, 2)
    yield session_factory
    await engine.dispose()


@pytest.fixture(autouse=True)
def known_words(monkeypatch):
    monkeypatch.setattr(fst_check, "lemmatize", lambda w: [w] if w.lower() in KNOWN else [])


async def _stored(session_factory, user_id=1, text=BOOK_TEXT) -> str:
    async with session_factory() as session:
        book = await store_book(session, user_id, "Kirja", text, split_book(text))
        await session.commit()
        return book.id


async def _chapter(session_factory, book_id, number) -> BookChapter:
    async with session_factory() as session:
        return await chapter_by_number(session, book_id, number)


# --- db/books.py -----------------------------------------------------------------


def test_detect_lang():
    assert detect_lang(BOOK_TEXT) == "ru"
    assert detect_lang("He left the house early.") == "en"


async def test_store_book_keeps_chapters_pending_and_aligned(session_factory):
    book_id = await _stored(session_factory)

    async with session_factory() as session:
        book = await session.get(Book, book_id)
        rows = (await session.scalars(select(BookChapter).order_by(BookChapter.number))).all()
    assert book.source_lang == "ru"
    assert book.front_matter == ["Название"]
    assert book.glossary is None
    assert [(c.number, len(c.source_paragraphs), c.status) for c in rows] == [
        (1, 2, ChapterStatus.pending),
        (2, 1, ChapterStatus.pending),
        (3, 2, ChapterStatus.pending),
    ]


async def test_queue_chapter_is_idempotent(session_factory):
    book_id = await _stored(session_factory)
    first = await _chapter(session_factory, book_id, 1)

    async with session_factory() as session:
        assert await queue_chapter(session, first.id)
        assert not await queue_chapter(session, first.id)
        await session.commit()


async def test_opening_a_chapter_queues_the_next_one_once(session_factory):
    book_id = await _stored(session_factory)

    async with session_factory() as session:
        first = await chapter_by_number(session, book_id, 1)
        queued = await mark_opened(session, first)
        again = await mark_opened(session, first)
        await session.commit()

    assert queued.number == 2
    assert again is None
    assert (await _chapter(session_factory, book_id, 1)).opened_at is not None
    assert (await _chapter(session_factory, book_id, 2)).status is ChapterStatus.queued


async def test_opening_the_last_chapter_queues_nothing(session_factory):
    book_id = await _stored(session_factory)

    async with session_factory() as session:
        last = await chapter_by_number(session, book_id, 3)
        assert await mark_opened(session, last) is None


async def test_claim_next_takes_the_lowest_queued_chapter(session_factory):
    book_id = await _stored(session_factory)
    async with session_factory() as session:
        for number in (3, 2):
            await queue_chapter(session, (await chapter_by_number(session, book_id, number)).id)
        await session.commit()

    async with session_factory() as session:
        claimed = await claim_next(session)
        await session.commit()

    second = await _chapter(session_factory, book_id, 2)
    assert claimed == second.id
    assert second.status is ChapterStatus.translating


async def test_restart_requeues_an_interrupted_chapter(session_factory):
    book_id = await _stored(session_factory)
    first = await _chapter(session_factory, book_id, 1)
    async with session_factory() as session:
        await queue_chapter(session, first.id)
        await claim_next(session)
        await session.commit()

    async with session_factory() as session:
        assert await reset_interrupted(session) == 1
        await session.commit()

    assert (await _chapter(session_factory, book_id, 1)).status is ChapterStatus.queued


# --- translate/worker.py ---------------------------------------------------------


def response(payload=None, text=None):
    return SimpleNamespace(
        output_text=json.dumps(payload, ensure_ascii=False) if text is None else text,
        output=[],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


GUIDE = response(
    {
        "characters": [
            {"name_ru": "Ася", "name_fi": "Asja", "speech": "", "register_fi": "kirjakieli"}
        ],
        "terms": [],
        "style_notes": "lyhyet lauseet",
    }
)
NO_ISSUES = response({"issues": [], "fixed": []})


def translation(*pairs):
    return response({"paragraphs": [{"id": i, "fi": fi} for i, fi in pairs]})


def make_translator(session_factory, *responses):
    client = SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(side_effect=responses)))
    notify = AsyncMock()
    translator = Translator(
        session_factory,
        lambda: client,
        "m",
        CallBreaker(100, timedelta(minutes=10)),
        notify,
    )
    return translator, client, notify


async def _claimed(session_factory, book_id, number) -> str:
    chapter = await _chapter(session_factory, book_id, number)
    async with session_factory() as session:
        await queue_chapter(session, chapter.id)
        await session.commit()
        assert await claim_next(session) == chapter.id
        await session.commit()
    return chapter.id


async def test_first_chapter_builds_the_guide_then_translates(session_factory):
    book_id = await _stored(session_factory)
    chapter_id = await _claimed(session_factory, book_id, 1)
    translator, client, notify = make_translator(
        session_factory, GUIDE, translation((0, "Hän meni."), (1, "Talo.")), NO_ISSUES
    )

    await translator.translate_one(chapter_id)

    chapter = await _chapter(session_factory, book_id, 1)
    assert chapter.status is ChapterStatus.done
    assert chapter.fi_paragraphs == ["Hän meni.", "Talo."]
    assert chapter.fst_flags == {}
    assert chapter.review_issues == []
    assert chapter.translated_at is not None
    async with session_factory() as session:
        book = await session.get(Book, book_id)
    assert book.glossary["characters"][0]["name_fi"] == "Asja"
    assert book.style_notes == "lyhyet lauseet"
    # Guide from chapter openings only - no chapter is sent in full for it.
    guide_call = json.loads(client.responses.create.await_args_list[0].kwargs["input"])
    assert len(guide_call["fragments"]) == 3
    event = notify.await_args.args[0]
    assert (event.number, event.status, event.error) == (1, ChapterStatus.done, None)


async def test_later_chapter_reuses_the_stored_guide(session_factory):
    book_id = await _stored(session_factory)
    first = await _claimed(session_factory, book_id, 1)
    translator, client, _ = make_translator(
        session_factory,
        GUIDE,
        translation((0, "Hän meni."), (1, "Talo.")),
        NO_ISSUES,
        translation((0, "Talo.")),
        NO_ISSUES,
    )
    await translator.translate_one(first)

    await translator.translate_one(await _claimed(session_factory, book_id, 2))

    assert client.responses.create.await_count == 5
    assert (await _chapter(session_factory, book_id, 2)).fi_paragraphs == ["Talo."]


async def test_refusal_fails_the_chapter_with_its_kind(session_factory):
    book_id = await _stored(session_factory)
    chapter_id = await _claimed(session_factory, book_id, 1)
    translator, _, notify = make_translator(session_factory, GUIDE, response(text=""))

    await translator.translate_one(chapter_id)

    chapter = await _chapter(session_factory, book_id, 1)
    assert chapter.status is ChapterStatus.failed
    assert chapter.error.startswith("RefusedError:")
    assert chapter.fi_paragraphs is None
    assert notify.await_args.args[0].status is ChapterStatus.failed


async def test_unexpected_error_fails_the_chapter_instead_of_the_loop(session_factory):
    book_id = await _stored(session_factory)
    chapter_id = await _claimed(session_factory, book_id, 1)
    translator, _, _ = make_translator(session_factory, response(text="not json"))

    await translator.translate_one(chapter_id)

    chapter = await _chapter(session_factory, book_id, 1)
    assert chapter.status is ChapterStatus.failed
    assert chapter.error.startswith("JSONDecodeError:")


async def test_book_text_stays_out_of_worker_logs(session_factory, caplog):
    # Our own loggers only: aiosqlite is held at INFO by setup_logging
    # (tests/test_logging_config.py), not by the code under test here.
    caplog.set_level("DEBUG", logger="kielikaveri")
    book_id = await _stored(session_factory)
    chapter_id = await _claimed(session_factory, book_id, 1)
    translator, _, _ = make_translator(
        session_factory, GUIDE, translation((0, "Hän meni."), (1, "Talo.")), NO_ISSUES
    )

    await translator.translate_one(chapter_id)

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "event=translate.chapter_end" in logged
    for text in ["Он ушёл", "Hän meni", "Asja", "lyhyet"]:
        assert text not in logged


async def test_run_requeues_interrupted_work_and_wakes_on_queue(session_factory):
    book_id = await _stored(session_factory)
    await _claimed(session_factory, book_id, 1)  # left `translating` by a "crash"
    translator, _, notify = make_translator(
        session_factory,
        GUIDE,
        translation((0, "Hän meni."), (1, "Talo.")),
        NO_ISSUES,
        translation((0, "Talo.")),
        NO_ISSUES,
    )
    task = asyncio.create_task(translator.run())
    try:
        await _wait_for(lambda: notify.await_count == 1)
        second = await _chapter(session_factory, book_id, 2)
        async with session_factory() as session:
            await queue_chapter(session, second.id)
            await session.commit()
        translator.wake()
        await _wait_for(lambda: notify.await_count == 2)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert [c.args[0].number for c in notify.await_args_list] == [1, 2]


async def _wait_for(condition, timeout=5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        assert asyncio.get_running_loop().time() < deadline, "timed out"
        await asyncio.sleep(0.01)


async def test_failed_chapter_message_offers_a_retry():
    bot = MagicMock(send_message=AsyncMock())
    event = ChapterEvent(1, "Kirja", "c1", 3, ChapterStatus.failed, "MisalignedError: x")

    await send_chapter_event(bot, event)

    text = bot.send_message.await_args.args[1]
    markup = bot.send_message.await_args.kwargs["reply_markup"]
    assert "главу 3" in text and "потерялись абзацы" in text
    assert markup.inline_keyboard[0][0].callback_data == "book:retry:c1"


# --- upload flow through the Dispatcher ------------------------------------------


def tg_document_update(update_id: int, file_name: str, mime_type: str = "text/plain") -> Update:
    return Update(
        update_id=update_id,
        message=TgMessage(
            message_id=update_id,
            date=datetime.now(UTC),
            chat=Chat(id=1, type="private"),
            from_user=tg_user(),
            document=Document(
                file_id="f",
                file_unique_id="u",
                file_name=file_name,
                mime_type=mime_type,
                file_size=len(BOOK_TEXT.encode()),
            ),
        ),
    )


@pytest.fixture
async def routed(session_factory, monkeypatch):
    content = {"bytes": BOOK_TEXT.encode("cp1251")}
    monkeypatch.setattr(
        Bot, "download", AsyncMock(side_effect=lambda *a, **k: io.BytesIO(content["bytes"]))
    )
    dp = routed_dispatcher()
    telegram = RecordingSession()
    bot = Bot(token="42:TEST", session=telegram)
    key = StorageKey(bot_id=bot.id, chat_id=1, user_id=1)
    await dp.storage.set_state(key, None)
    await dp.storage.set_data(key, {})
    translator = MagicMock()
    context = {
        "session_factory": session_factory,
        "settings": make_settings(),
        "breaker": make_breaker(),
        "translator": translator,
    }

    def _texts(since):
        return [m for m in telegram.sent[since:] if isinstance(m, SendMessage)]

    async def upload(file_name="Kirja_yksi.txt", mime_type="text/plain"):
        since = len(telegram.sent)
        update = tg_document_update(len(telegram.sent) + 1, file_name, mime_type)
        await dp.feed_update(bot, update, **context)
        return _texts(since)

    async def tap(data):
        since = len(telegram.sent)
        await dp.feed_update(bot, tg_callback_update(len(telegram.sent) + 1, data), **context)
        return _texts(since)

    yield SimpleNamespace(
        upload=upload, tap=tap, translator=translator, content=content, dp=dp, key=key
    )
    await dp.storage.set_state(key, None)
    await dp.storage.set_data(key, {})


async def _only_book(session_factory) -> Book:
    async with session_factory() as session:
        return await session.scalar(select(Book))


async def test_txt_upload_stores_the_book_and_asks(routed, session_factory):
    [reply] = await routed.upload()

    book = await _only_book(session_factory)
    assert book.title == "Kirja yksi"
    assert "глав - 3" in reply.text
    buttons = [b.callback_data for b in reply.reply_markup.inline_keyboard[0]]
    assert buttons == [f"book:yes:{book.id}", f"book:no:{book.id}"]
    # Nothing is translated before the answer.
    routed.translator.wake.assert_not_called()


async def test_yes_queues_chapter_one_once(routed, session_factory):
    await routed.upload()
    book = await _only_book(session_factory)

    [reply] = await routed.tap(f"book:yes:{book.id}")
    [again] = await routed.tap(f"book:yes:{book.id}")

    assert "Перевожу главу 1" in reply.text
    assert "уже переводится" in again.text
    assert routed.translator.wake.call_count == 1
    statuses = [(await _chapter(session_factory, book.id, n)).status for n in (1, 2, 3)]
    assert statuses == [ChapterStatus.queued, ChapterStatus.pending, ChapterStatus.pending]


async def test_no_deletes_the_book(routed, session_factory):
    await routed.upload()
    book = await _only_book(session_factory)

    [reply] = await routed.tap(f"book:no:{book.id}")

    assert "удалила" in reply.text
    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(BookChapter)) == 0
    assert await _only_book(session_factory) is None


async def test_no_after_yes_keeps_the_book(routed, session_factory):
    await routed.upload()
    book = await _only_book(session_factory)
    await routed.tap(f"book:yes:{book.id}")

    await routed.tap(f"book:no:{book.id}")

    assert await _only_book(session_factory) is not None


async def test_same_book_twice_is_recognized(routed, session_factory):
    await routed.upload()
    [reply] = await routed.upload("другое имя.txt")

    assert "уже загружена" in reply.text
    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(Book)) == 1


async def test_other_documents_are_declined(routed, session_factory):
    [reply] = await routed.upload("kirja.epub", "application/epub+zip")

    assert reply.text == NOT_TXT_TEXT
    assert await _only_book(session_factory) is None


async def test_txt_during_input_state_is_not_taken(routed, session_factory):
    await routed.dp.storage.set_state(routed.key, AddStates.naming_new_deck.state)

    [reply] = await routed.upload()

    assert reply.text == BUSY_TEXT
    assert await routed.dp.storage.get_state(routed.key) == AddStates.naming_new_deck.state
    assert await _only_book(session_factory) is None


async def test_binary_file_is_reported(routed, session_factory):
    routed.content["bytes"] = b"\x00\x01\x02\xff\xfe" * 400

    [reply] = await routed.upload()

    assert "Не смогла прочитать" in reply.text


async def test_retry_requeues_only_a_failed_chapter(routed, session_factory):
    book_id = await _stored(session_factory)
    first = await _chapter(session_factory, book_id, 1)

    [not_failed] = await routed.tap(f"book:retry:{first.id}")
    async with session_factory() as session:
        (await session.get(BookChapter, first.id)).status = ChapterStatus.failed
        await session.commit()
    [reply] = await routed.tap(f"book:retry:{first.id}")

    assert "уже переводится или готова" in not_failed.text
    assert "ещё раз" in reply.text
    assert (await _chapter(session_factory, book_id, 1)).status is ChapterStatus.queued
    routed.translator.wake.assert_called_once()
