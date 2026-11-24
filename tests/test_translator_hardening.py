"""Translator hardening from the review of 23.11.2026 (plan
2026-11-20-kirjakaveri-reader, "Ревью Л0–П2"): И8 - an exception's text
stays out of the log, traceback included; М15 - the daily cap survives a
restart; М16 - a result whose final write failed is kept and written before
the next chapter is taken. All texts are synthetic."""

import asyncio
import io
import logging
from datetime import UTC, datetime, timedelta

import pytest
from test_books import (
    GUIDE,
    KNOWN,
    NO_ISSUES,
    _chapter,
    _claimed,
    _stored,
    _wait_for,
    make_translator,
    response,
    translation,
)
from test_translator_loop import OTHER, OTHER_BOOK_TEXT, flaky_store

from kielikaveri.db.books import queue_chapter
from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import BookChapter, ChapterStatus, User
from kielikaveri.llm.breaker import BudgetExceededError, CallBreaker
from kielikaveri.logging_config import setup_logging
from kielikaveri.translate import fst_check, worker

# A sentence no real book has, so finding it in the log can only mean a leak.
SECRET = "Лиловый бегемот Евлампий вязал шарф для маяка на краю света."
FIRST = (GUIDE, translation((0, "Hän meni."), (1, "Talo.")), NO_ISSUES)


@pytest.fixture
async def session_factory(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    await create_all(engine)
    factory = make_session_factory(engine)
    async with factory() as session:
        session.add_all([User(id=1), User(id=2)])
        await session.commit()
    yield factory
    await engine.dispose()


@pytest.fixture(autouse=True)
def known_words(monkeypatch):
    monkeypatch.setattr(fst_check, "lemmatize", lambda w: [w] if w.lower() in KNOWN else [])


@pytest.fixture(autouse=True)
def no_pauses(monkeypatch):
    monkeypatch.setattr(worker, "_STORE_RETRY_SECONDS", 0)
    monkeypatch.setattr(worker, "_LOOP_ERROR_PAUSE_SECONDS", 0)


async def _stop(task) -> None:
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# --- И8: no book text in a logged exception ---------------------------------------


@pytest.fixture
def production_log():
    """The worker's records run through the production handler setup (its
    formatter and filters, from setup_logging) into a string."""
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    setup_logging("DEBUG")
    template = root.handlers[0]
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(template.formatter)
    for f in template.filters:
        handler.addFilter(f)
    records: list[logging.LogRecord] = []
    handler.addFilter(lambda r: records.append(r) or True)
    target = logging.getLogger("kielikaveri.translate.worker")
    target.addHandler(handler)
    yield stream, records
    target.removeHandler(handler)


def _assert_clean(production_log, event: str) -> str:
    stream, records = production_log
    text = stream.getvalue()
    [record] = [r for r in records if event in r.getMessage()]
    assert record.exc_info, "the stack is still logged"
    assert SECRET not in record.getMessage()
    assert SECRET not in (record.exc_text or "")
    assert SECRET not in text
    assert "Traceback (most recent call last)" in text
    return text


def _quoting_error() -> KeyError:
    """A KeyError keyed by a sentence, raised from a ValueError that quotes
    it, with a note quoting it too - every place text can hide."""
    try:
        try:
            raise ValueError(f"cannot parse: {SECRET}")
        except ValueError as inner:
            err = KeyError(SECRET)
            err.add_note(f"while reading {SECRET}")
            raise err from inner
    except KeyError as exc:
        return exc


async def test_third_party_exception_with_book_text_is_logged_without_it(
    session_factory, production_log
):
    book_id = await _stored(session_factory)
    chapter_id = await _claimed(session_factory, book_id, 1)
    translator, client, _ = make_translator(session_factory)
    client.responses.create.side_effect = _quoting_error()

    await translator.translate_one(chapter_id)

    assert (await _chapter(session_factory, book_id, 1)).status is ChapterStatus.failed
    text = _assert_clean(production_log, "event=translate.chapter_failed")
    assert "error=KeyError" in text
    assert "KeyError <- ValueError" in text


async def test_loop_error_is_logged_without_book_text(session_factory, production_log):
    book_id = await _stored(session_factory)
    async with session_factory() as session:
        await queue_chapter(session, (await _chapter(session_factory, book_id, 1)).id)
        await session.commit()
    translator, _, _ = make_translator(session_factory)
    seen = asyncio.Event()

    async def breaks(chapter_id):
        seen.set()
        raise _quoting_error()

    translator.translate_one = breaks
    task = asyncio.create_task(translator.run())
    try:
        await asyncio.wait_for(seen.wait(), 5)
        await _wait_for(lambda: "loop_error" in production_log[0].getvalue())
    finally:
        await _stop(task)

    _assert_clean(production_log, "event=translate.loop_error")


async def test_notify_failure_is_logged_without_book_text(session_factory, production_log):
    book_id = await _stored(session_factory)
    chapter_id = await _claimed(session_factory, book_id, 1)
    translator, _, notify = make_translator(session_factory, *FIRST)
    notify.side_effect = _quoting_error()

    await translator.translate_one(chapter_id)

    _assert_clean(production_log, "event=translate.notify_failed")


async def test_task_death_is_logged_without_book_text(session_factory, production_log):
    translator, _, _ = make_translator(session_factory)

    async def broken():
        raise _quoting_error()

    translator.run = broken
    task = translator.start()
    await asyncio.wait([task])
    await asyncio.sleep(0)  # done-callbacks run on the next loop step

    text = _assert_clean(production_log, "event=translate.task_died")
    assert "error=KeyError" in text


def test_chain_names_follow_cause_and_context_but_not_suppressed_context():
    try:
        try:
            raise ValueError(SECRET)
        except ValueError:
            raise TypeError(SECRET)  # the implicit __context__ is the point
    except TypeError as exc:
        assert worker._chain_names(exc) == "TypeError <- ValueError"

    try:
        try:
            raise ValueError(SECRET)
        except ValueError:
            raise TypeError(SECRET) from None
    except TypeError as exc:
        assert worker._chain_names(exc) == "TypeError"


# --- М15: the daily cap survives a restart ----------------------------------------

NOW = datetime(2026, 11, 23, 10, 0, tzinfo=UTC)


def test_seeded_calls_count_against_the_daily_cap():
    breaker = CallBreaker(100, timedelta(minutes=10), daily_max_calls=5)
    seeded = breaker.seed_daily(
        [NOW - timedelta(hours=25)] * 10 + [NOW - timedelta(hours=1)] * 4, NOW
    )

    assert seeded == 4  # the older ones are past the 24 hours
    breaker.check(NOW)
    with pytest.raises(BudgetExceededError):
        breaker.check(NOW + timedelta(minutes=1))
    # Released once the seeded calls age out.
    breaker.check(NOW + timedelta(hours=23, minutes=1))


def test_seeding_without_a_daily_cap_does_nothing():
    breaker = CallBreaker(100, timedelta(minutes=10))

    assert breaker.seed_daily([NOW] * 500, NOW) == 0
    breaker.check(NOW)


def test_passed_counts_only_calls_let_through():
    breaker = CallBreaker(100, timedelta(minutes=10), daily_max_calls=2)
    for i in range(4):
        try:
            breaker.check(NOW + timedelta(minutes=i))
        except BudgetExceededError:
            pass

    assert breaker.passed == 2


async def test_done_chapter_records_its_calls_and_finish_time(session_factory):
    book_id = await _stored(session_factory)
    chapter_id = await _claimed(session_factory, book_id, 1)
    translator, _, _ = make_translator(session_factory, *FIRST)
    before = datetime.now(UTC).replace(microsecond=0)

    await translator.translate_one(chapter_id)

    chapter = await _chapter(session_factory, book_id, 1)
    assert chapter.status is ChapterStatus.done
    assert chapter.calls == 3  # guide, translation, review
    assert before <= chapter.finished_at <= datetime.now(UTC)


async def test_failed_chapter_records_the_calls_it_made(session_factory):
    book_id = await _stored(session_factory)
    chapter_id = await _claimed(session_factory, book_id, 1)
    translator, _, _ = make_translator(session_factory, GUIDE, response(text=""))

    await translator.translate_one(chapter_id)

    chapter = await _chapter(session_factory, book_id, 1)
    assert chapter.status is ChapterStatus.failed
    assert chapter.calls == 2  # the guide and the refused translation
    assert chapter.finished_at is not None


async def test_retry_within_a_day_keeps_the_failed_attempts_calls(session_factory):
    book_id = await _stored(session_factory)
    chapter_id = await _claimed(session_factory, book_id, 1)
    translator, client, _ = make_translator(session_factory, GUIDE, response(text=""))
    await translator.translate_one(chapter_id)
    # The "retry" button: the stored guide is reused, two calls this time.
    client.responses.create.side_effect = [translation((0, "Hän meni."), (1, "Talo.")), NO_ISSUES]

    await _claimed(session_factory, book_id, 1)
    await translator.translate_one(chapter_id)

    chapter = await _chapter(session_factory, book_id, 1)
    assert chapter.status is ChapterStatus.done
    assert chapter.calls == 2 + 2


async def _finished(session_factory, book_id, number, calls, finished_at) -> None:
    async with session_factory() as session:
        chapter = await session.get(
            BookChapter, (await _chapter(session_factory, book_id, number)).id
        )
        chapter.status = ChapterStatus.done
        chapter.calls = calls
        chapter.finished_at = finished_at
        await session.commit()


async def test_run_seeds_the_daily_cap_from_chapters_of_the_last_day(session_factory, caplog):
    caplog.set_level("INFO", logger="kielikaveri")
    book_id = await _stored(session_factory)
    now = datetime.now(UTC)
    await _finished(session_factory, book_id, 1, 4, now - timedelta(hours=2))
    await _finished(session_factory, book_id, 2, 50, now - timedelta(hours=30))
    translator, _, _ = make_translator(session_factory)
    breaker = CallBreaker(100, timedelta(minutes=10), daily_max_calls=5)
    translator._breaker = breaker

    task = asyncio.create_task(translator.run())
    try:
        await _wait_for(lambda: "event=translate.daily_seeded" in caplog.text)
    finally:
        await _stop(task)

    assert "event=translate.daily_seeded calls=4" in caplog.text
    breaker.check(datetime.now(UTC))
    with pytest.raises(BudgetExceededError):
        breaker.check(datetime.now(UTC))


async def test_restart_does_not_refill_the_day(session_factory):
    """Translate, "restart" with a fresh breaker: the chapter's calls are
    still counted."""
    book_id = await _stored(session_factory)
    chapter_id = await _claimed(session_factory, book_id, 1)
    translator, _, _ = make_translator(session_factory, *FIRST)
    await translator.translate_one(chapter_id)

    restarted, _, _ = make_translator(session_factory)
    breaker = CallBreaker(100, timedelta(minutes=10), daily_max_calls=3)
    restarted._breaker = breaker
    task = asyncio.create_task(restarted.run())
    try:
        await _wait_for(lambda: len(breaker._day_calls) == 3)
    finally:
        await _stop(task)

    with pytest.raises(BudgetExceededError):
        breaker.check(datetime.now(UTC))


# --- М16: a failed final write is kept and retried --------------------------------


async def test_a_lost_final_write_keeps_the_result_and_does_not_notify(session_factory):
    book_id = await _stored(session_factory)
    chapter_id = await _claimed(session_factory, book_id, 1)
    translator, _, notify = make_translator(
        flaky_store(session_factory, worker._STORE_ATTEMPTS), *FIRST
    )

    await translator.translate_one(chapter_id)

    notify.assert_not_awaited()
    assert translator._unstored is not None
    assert translator._unstored.chapter_id == chapter_id
    assert (await _chapter(session_factory, book_id, 1)).status is ChapterStatus.translating


async def test_run_writes_the_kept_result_before_taking_the_next_chapter(session_factory, caplog):
    caplog.set_level("INFO", logger="kielikaveri")
    book_id = await _stored(session_factory)
    other_id = await _stored(session_factory, user_id=2, text=OTHER_BOOK_TEXT)
    for b in (book_id, other_id):
        async with session_factory() as session:
            await queue_chapter(session, (await _chapter(session_factory, b, 1)).id)
            await session.commit()
    # Locked through three loop passes, not just one _store's attempts.
    translator, client, notify = make_translator(
        flaky_store(session_factory, worker._STORE_ATTEMPTS * 3)
    )
    answers = iter([*FIRST, *OTHER])
    first_stored_when_other_began = []

    async def create(**kwargs):
        if client.responses.create.await_count == len(FIRST) + 1:
            first = await _chapter(session_factory, book_id, 1)
            first_stored_when_other_began.append(first.status)
        return next(answers)

    client.responses.create.side_effect = create
    task = asyncio.create_task(translator.run())
    try:
        await _wait_for(lambda: notify.await_count == 2)
    finally:
        await _stop(task)

    # Nothing else was translated (paid for) while the write was pending.
    assert first_stored_when_other_began == [ChapterStatus.done]
    assert client.responses.create.await_count == len(FIRST) + len(OTHER)
    assert [c.args[0].user_id for c in notify.await_args_list] == [1, 2]
    assert (await _chapter(session_factory, book_id, 1)).fi_paragraphs == ["Hän meni.", "Talo."]
    failed = [r for r in caplog.records if "event=translate.store_failed" in r.getMessage()]
    assert len(failed) == 3
    assert translator._unstored is None


async def test_a_non_lock_error_on_the_final_write_drops_the_result(session_factory):
    book_id = await _stored(session_factory)
    chapter_id = await _claimed(session_factory, book_id, 1)
    translator, _, notify = make_translator(session_factory, *FIRST)

    async def broken_store(chapter_id, values):
        raise RuntimeError("not a lock")

    translator._store = broken_store

    with pytest.raises(RuntimeError):
        await translator.translate_one(chapter_id)

    # Retrying would fail the same way; the restart requeues the chapter.
    assert translator._unstored is None
    notify.assert_not_awaited()
