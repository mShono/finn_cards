"""The translator's loop outlives what goes wrong around a chapter: a locked
database at the final write, a book deleted mid-translation, and a task that
dies anyway is logged at once (plan 2026-11-20-kirjakaveri-reader, И1)."""

import asyncio
import logging

import pytest
from sqlalchemy.exc import OperationalError
from test_books import (
    GUIDE,
    KNOWN,
    NO_ISSUES,
    PARAGRAPH,
    _chapter,
    _claimed,
    _stored,
    _wait_for,
    make_translator,
    translation,
)

from kielikaveri.db.books import delete_book, queue_chapter
from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import BookChapter, ChapterStatus
from kielikaveri.translate import fst_check, worker

FIRST = (GUIDE, translation((0, "Hän meni."), (1, "Talo.")), NO_ISSUES)
# Another book, one paragraph per chapter; chapter 1's guide, translation, review.
OTHER_BOOK_TEXT = f"Toinen\n\n1\n\n{PARAGRAPH}\n\n2\n\n{PARAGRAPH}"
OTHER = (GUIDE, translation((0, "Talo.")), NO_ISSUES)
SECRET = "Hän meni."


@pytest.fixture
async def session_factory(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    await create_all(engine)
    yield make_session_factory(engine)
    await engine.dispose()


@pytest.fixture(autouse=True)
def known_words(monkeypatch):
    monkeypatch.setattr(fst_check, "lemmatize", lambda w: [w] if w.lower() in KNOWN else [])


@pytest.fixture(autouse=True)
def no_pauses(monkeypatch):
    monkeypatch.setattr(worker, "_STORE_RETRY_SECONDS", 0)
    monkeypatch.setattr(worker, "_LOOP_ERROR_PAUSE_SECONDS", 0)


def locked() -> OperationalError:
    # Like the real one, its parameters carry the chapter's text.
    return OperationalError("UPDATE book_chapters", {"fi": SECRET}, Exception("database is locked"))


def flaky_store(session_factory, failures):
    """A session factory whose commits of a chapter row (the final write)
    fail with "database is locked" the first `failures` times."""
    left = {"n": failures}

    def factory():
        session = session_factory()
        real_commit = session.commit

        async def commit():
            if left["n"] and any(isinstance(o, BookChapter) for o in session.dirty):
                left["n"] -= 1
                raise locked()
            await real_commit()

        session.commit = commit
        return session

    return factory


async def _queue(session_factory, book_id, number) -> None:
    chapter = await _chapter(session_factory, book_id, number)
    async with session_factory() as session:
        await queue_chapter(session, chapter.id)
        await session.commit()


async def _delete(session_factory, book_id) -> None:
    async with session_factory() as session:
        await delete_book(session, book_id)
        await session.commit()


async def _stop(task) -> None:
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_a_locked_final_write_is_retried(session_factory):
    book_id = await _stored(session_factory)
    chapter_id = await _claimed(session_factory, book_id, 1)
    translator, _, notify = make_translator(flaky_store(session_factory, 2), *FIRST)

    await translator.translate_one(chapter_id)

    chapter = await _chapter(session_factory, book_id, 1)
    assert chapter.status is ChapterStatus.done
    assert chapter.fi_paragraphs == ["Hän meni.", "Talo."]
    assert notify.await_args.args[0].status is ChapterStatus.done


async def test_run_retries_a_failed_final_write_before_the_next_chapter(session_factory, caplog):
    caplog.set_level("DEBUG", logger="kielikaveri")
    book_id = await _stored(session_factory)
    other_id = await _stored(session_factory, user_id=2, text=OTHER_BOOK_TEXT)
    await _queue(session_factory, book_id, 1)
    await _queue(session_factory, other_id, 1)
    translator, _, notify = make_translator(
        flaky_store(session_factory, worker._STORE_ATTEMPTS), *FIRST, *OTHER
    )

    task = asyncio.create_task(translator.run())
    try:
        await _wait_for(lambda: notify.await_count == 2)
    finally:
        await _stop(task)

    # The kept result is written on the loop's next pass, before chapter 2 is taken.
    assert (await _chapter(session_factory, book_id, 1)).fi_paragraphs == ["Hän meni.", "Talo."]
    assert (await _chapter(session_factory, other_id, 1)).fi_paragraphs == ["Talo."]
    assert [c.args[0].user_id for c in notify.await_args_list] == [1, 2]
    failed = [r for r in caplog.records if "event=translate.store_failed" in r.getMessage()]
    assert len(failed) == 1
    chapter_id = (await _chapter(session_factory, book_id, 1)).id
    assert f"chapter_id={chapter_id}" in failed[0].getMessage()
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert SECRET not in logged
    assert not any(r.exc_info for r in caplog.records if r.levelno >= logging.WARNING)


async def test_run_survives_an_error_outside_the_pipeline(session_factory, caplog):
    book_id = await _stored(session_factory)
    other_id = await _stored(session_factory, user_id=2, text=OTHER_BOOK_TEXT)
    await _queue(session_factory, book_id, 1)
    await _queue(session_factory, other_id, 1)
    translator, _, notify = make_translator(session_factory, *OTHER)
    real = translator.translate_one
    calls = []

    async def first_one_breaks(chapter_id):
        calls.append(chapter_id)
        if len(calls) == 1:
            raise locked()
        await real(chapter_id)

    translator.translate_one = first_one_breaks
    task = asyncio.create_task(translator.run())
    try:
        await _wait_for(lambda: notify.await_count == 1)
    finally:
        await _stop(task)

    assert (await _chapter(session_factory, other_id, 1)).status is ChapterStatus.done
    assert "event=translate.loop_error" in caplog.text


async def test_book_deleted_during_translation_does_not_kill_the_loop(session_factory, caplog):
    caplog.set_level("INFO", logger="kielikaveri")
    book_id = await _stored(session_factory)
    other_id = await _stored(session_factory, user_id=2, text=OTHER_BOOK_TEXT)
    await _queue(session_factory, book_id, 1)
    translator, client, notify = make_translator(session_factory)
    answers = iter([*FIRST, *OTHER])

    async def create(**kwargs):
        answer = next(answers)
        if answer is FIRST[1]:  # the user taps "Нет" while chapter 1 translates
            await _delete(session_factory, book_id)
            await _queue(session_factory, other_id, 1)
            translator.wake()
        return answer

    client.responses.create.side_effect = create
    task = asyncio.create_task(translator.run())
    try:
        await _wait_for(lambda: notify.await_count == 1)
    finally:
        await _stop(task)

    assert notify.await_args.args[0].user_id == 2
    assert (await _chapter(session_factory, other_id, 1)).status is ChapterStatus.done
    assert "event=translate.chapter_gone" in caplog.text
    assert "stage=store" in caplog.text
    assert "event=translate.loop_error" not in caplog.text


async def test_book_deleted_before_its_guide_is_stored(session_factory, caplog):
    caplog.set_level("INFO", logger="kielikaveri")
    book_id = await _stored(session_factory)
    chapter_id = await _claimed(session_factory, book_id, 1)
    translator, client, notify = make_translator(session_factory)

    async def create(**kwargs):
        await _delete(session_factory, book_id)
        return GUIDE

    client.responses.create.side_effect = create

    await translator.translate_one(chapter_id)

    assert client.responses.create.await_count == 1
    notify.assert_not_awaited()
    assert "stage=guide" in caplog.text


async def test_vanished_chapter_is_skipped(session_factory, caplog):
    caplog.set_level("INFO", logger="kielikaveri")
    translator, client, notify = make_translator(session_factory)

    await translator.translate_one("no-such-chapter")

    client.responses.create.assert_not_awaited()
    notify.assert_not_awaited()
    assert "event=translate.chapter_gone chapter_id=no-such-chapter stage=start" in caplog.text


async def test_a_dying_task_is_logged_at_once(session_factory, caplog):
    translator, _, _ = make_translator(session_factory)

    async def broken():
        raise RuntimeError("boom")

    translator.run = broken
    task = translator.start()
    await asyncio.wait([task])
    await asyncio.sleep(0)  # done-callbacks run on the next loop step

    died = [r for r in caplog.records if "event=translate.task_died" in r.getMessage()]
    assert len(died) == 1
    assert died[0].levelno == logging.ERROR
    assert "error=RuntimeError" in died[0].getMessage()
    assert "task=translator" in died[0].getMessage()


async def test_a_cancelled_task_is_not_reported(session_factory, caplog):
    translator, _, _ = make_translator(session_factory)
    task = translator.start()
    await asyncio.sleep(0)

    await _stop(task)
    await asyncio.sleep(0)

    assert "event=translate.task_died" not in caplog.text


def test_a_task_that_returns_is_reported(caplog):
    async def returns():
        return None

    async def main():
        task = asyncio.create_task(returns(), name="translator")
        await task
        worker.log_task_death(task)

    asyncio.run(main())

    assert "event=translate.task_died task=translator reason=returned" in caplog.text
