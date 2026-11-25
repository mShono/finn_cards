"""The tap made faster (plan 2026-11-20-kirjakaveri-reader, "П3", "Скорость
тапа"): POST /api/lemma (the FST alone), the connection warm-up, the time
breakdown in event=reader.lookup, and in_deck. OpenAI is a fake."""

# The database fixture comes from test_reader_lookup; tests naming it as a
# parameter "redefine" the import for ruff.
# ruff: noqa: F811

import asyncio
import logging
import time

import pytest
from aiohttp.test_utils import TestClient, TestServer
from conftest import log_fields
from sqlalchemy import update
from test_reader import header
from test_reader_cards import FakeOpenAI, build_app
from test_reader_lookup import (
    KAUKANA,
    MA,
    PAVLIVKA,
    TALOSSA,
    make_book,
    session_factory,  # noqa: F401 - the fixture
)

from kielikaveri.db.models import BookChapter, ChapterStatus
from kielikaveri.reader.app import WARMUP
from kielikaveri.reader.warmup import WARMUP_IDLE, Warmup


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
async def book(session_factory):
    return await make_book(session_factory)


@pytest.fixture
def fake():
    return FakeOpenAI()


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
async def app(session_factory, fake, tmp_path, clock):
    app = build_app(session_factory, fake, tmp_path)
    app[WARMUP] = Warmup(clock)
    return app


@pytest.fixture
async def client(app):
    async with TestClient(TestServer(app)) as client:
        yield client


def body(chapter_id, paragraph=0, token=TALOSSA) -> dict:
    return {"chapter_id": chapter_id, "paragraph": paragraph, "token": token}


async def post(client, path, payload):
    return await client.post(path, json=payload, headers=header())


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.DEBUG)
    logging.getLogger("aiosqlite").setLevel(logging.INFO)
    return caplog


def lines(caplog, event: str) -> list[dict]:
    return [
        log_fields(r.getMessage())
        for r in caplog.records
        if r.getMessage().startswith(f"event={event} ")
    ]


# --- /api/lemma ------------------------------------------------------------------------


async def test_lemma_answers_from_the_fst(client, book, fake):
    _, chapter_id, _ = book

    r = await post(client, "/api/lemma", body(chapter_id))

    assert r.status == 200
    assert await r.json() == {"surface": "talossa", "lemmas": ["talo"], "fst_flagged": False}
    assert fake.requests == []


async def test_lemma_of_a_word_the_fst_lacks_is_empty(client, book):
    _, chapter_id, _ = book

    r = await post(client, "/api/lemma", body(chapter_id, token=PAVLIVKA))

    assert await r.json() == {"surface": "Pavlivka", "lemmas": [], "fst_flagged": True}


async def test_lemma_lists_each_lemma_once(client, book):
    _, chapter_id, _ = book

    lemmas = (await (await post(client, "/api/lemma", body(chapter_id, token=MA))).json())["lemmas"]

    assert len(lemmas) == len(set(lemmas))


async def test_lemma_works_with_the_breaker_open(session_factory, book, fake, tmp_path):
    _, chapter_id, _ = book
    app = build_app(session_factory, fake, tmp_path, reader_breaker_max_calls=0)
    async with TestClient(TestServer(app)) as client:
        assert (await post(client, "/api/lookup", body(chapter_id))).status == 503
        r = await post(client, "/api/lemma", body(chapter_id))

    assert r.status == 200
    assert fake.requests == []


@pytest.mark.parametrize(
    ("payload", "status"),
    [({}, 400), ({"chapter_id": "nope", "paragraph": 0, "token": 0}, 404)],
)
async def test_lemma_checks_like_lookup(client, book, payload, status):
    assert (await post(client, "/api/lemma", payload)).status == status


async def test_lemma_needs_init_data(client, book):
    _, chapter_id, _ = book
    assert (await client.post("/api/lemma", json=body(chapter_id))).status == 401


async def test_lemma_log_has_no_text(client, book, logs):
    _, chapter_id, _ = book
    await post(client, "/api/lemma", body(chapter_id))

    [line] = lines(logs, "reader.lemma")
    assert line["chapter_id"] == chapter_id
    assert (line["paragraph"], line["token"], line["count"]) == ("0", str(TALOSSA), "1")
    assert "duration_ms" in line
    for piece in ("talossa", "talo"):
        assert piece not in logs.text


# --- the warm-up -------------------------------------------------------------------------


async def test_opening_a_chapter_warms_the_connection_up(client, app, book, fake, logs):
    _, chapter_id, _ = book

    assert (await client.get(f"/api/chapters/{chapter_id}", headers=header())).status == 200
    await app[WARMUP]._task

    fake.models.retrieve.assert_awaited_once_with("gpt-5.6-terra")
    [line] = lines(logs, "reader.warmup")
    assert line["ok"] == "1"
    assert "ms" in line
    # Not through the breaker.
    assert app[WARMUP].warm()


async def test_lemma_warms_the_connection_up_too(client, app, book, fake):
    _, chapter_id, _ = book

    await post(client, "/api/lemma", body(chapter_id))
    await app[WARMUP]._task

    fake.models.retrieve.assert_awaited_once()


async def test_at_most_one_warmup_a_minute(client, app, book, fake, clock):
    _, chapter_id, _ = book

    async def open_chapter():
        await client.get(f"/api/chapters/{chapter_id}", headers=header())
        task = app[WARMUP]._task
        if task is not None:
            await task

    await open_chapter()
    clock.now += WARMUP_IDLE - 1
    await open_chapter()
    await post(client, "/api/lemma", body(chapter_id))
    assert fake.models.retrieve.await_count == 1

    clock.now += 2
    await open_chapter()
    assert fake.models.retrieve.await_count == 2


async def test_no_warmup_right_after_an_openai_call(client, app, book, fake, clock):
    _, chapter_id, _ = book
    await post(client, "/api/lookup", body(chapter_id))
    clock.now += WARMUP_IDLE + 100  # the start-of-day block is long gone
    app[WARMUP].touch()  # ...but a call just went out

    await client.get(f"/api/chapters/{chapter_id}", headers=header())

    assert app[WARMUP]._task is None
    fake.models.retrieve.assert_not_awaited()


async def test_the_chapter_does_not_wait_for_a_slow_warmup(client, app, book, fake, clock):
    _, chapter_id, _ = book
    started_retrieve = asyncio.Event()

    async def slow(model):
        started_retrieve.set()
        await asyncio.sleep(30)

    fake.models.retrieve.side_effect = slow

    started = time.monotonic()
    r = await client.get(f"/api/chapters/{chapter_id}", headers=header())
    assert r.status == 200
    assert time.monotonic() - started < 5
    await asyncio.wait_for(started_retrieve.wait(), 5)

    # One in flight at most, however long past the minute.
    clock.now += WARMUP_IDLE * 10
    await client.get(f"/api/chapters/{chapter_id}", headers=header())
    await post(client, "/api/lemma", body(chapter_id))
    assert fake.models.retrieve.await_count == 1


async def test_a_warmup_in_flight_is_cancelled_on_shutdown(session_factory, book, tmp_path):
    _, chapter_id, _ = book
    fake = FakeOpenAI()
    cancelled = asyncio.Event()

    async def slow(model):
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    fake.models.retrieve.side_effect = slow
    app = build_app(session_factory, fake, tmp_path)
    app[WARMUP] = Warmup()
    async with TestClient(TestServer(app)) as client:
        await client.get(f"/api/chapters/{chapter_id}", headers=header())
        await asyncio.sleep(0.05)
        task = app[WARMUP]._task
        assert task is not None and not task.done()

    assert cancelled.is_set()
    assert task.done()


async def test_a_failed_warmup_is_one_line_without_a_traceback(client, app, book, fake, logs):
    _, chapter_id, _ = book
    fake.models.retrieve.side_effect = RuntimeError("secret detail")

    await client.get(f"/api/chapters/{chapter_id}", headers=header())
    await app[WARMUP]._task

    [record] = [r for r in logs.records if r.getMessage().startswith("event=reader.warmup ")]
    assert log_fields(record.getMessage())["ok"] == "0"
    assert log_fields(record.getMessage())["error"] == "RuntimeError"
    assert record.exc_info is None
    assert "secret detail" not in logs.text
    assert not app[WARMUP].warm()


# --- the lookup's time breakdown -----------------------------------------------------------


async def test_lookup_log_has_the_breakdown(client, app, book, logs, clock):
    _, chapter_id, _ = book

    await post(client, "/api/lookup", body(chapter_id))
    clock.now += 42
    await post(client, "/api/lookup", body(chapter_id))
    await post(client, "/api/lookup", body(chapter_id, token=KAUKANA))

    first, cached, other = lines(logs, "reader.lookup")
    for line in (first, cached, other):
        for field in ("llm_ms", "resolve_ms", "choose_lemma", "idle_s", "warm", "duration_ms"):
            assert field in line, field
    # Nothing before the first call since start.
    assert first["idle_s"] == "-1"
    assert first["choose_lemma"] == "0"
    assert first["warm"] == "0"
    assert (cached["cached"], cached["llm_ms"], cached["resolve_ms"]) == ("True", "0", "0")
    assert cached["idle_s"] == "42"
    assert other["idle_s"] == "42"


async def test_lookup_log_shows_a_recent_warmup(client, app, book, logs):
    _, chapter_id, _ = book
    await client.get(f"/api/chapters/{chapter_id}", headers=header())
    await app[WARMUP]._task

    await post(client, "/api/lookup", body(chapter_id))

    [line] = lines(logs, "reader.lookup")
    assert line["warm"] == "1"
    assert line["idle_s"] == "0"


async def test_lookup_records_its_call_time(client, app, book, clock):
    _, chapter_id, _ = book
    await post(client, "/api/lookup", body(chapter_id))
    clock.now += 5
    assert app[WARMUP].idle_seconds() == 5


# --- in_deck -------------------------------------------------------------------------------


async def test_in_deck_is_false_without_a_deck(client, book):
    _, chapter_id, _ = book

    r = await post(client, "/api/lookup", body(chapter_id))

    assert (await r.json())["in_deck"] is False


async def test_in_deck_after_plus_even_for_a_cached_lookup(client, book, fake):
    _, chapter_id, _ = book
    before = await (await post(client, "/api/lookup", body(chapter_id))).json()

    assert (await post(client, "/api/cards", body(chapter_id))).status == 200
    after = await (await post(client, "/api/lookup", body(chapter_id))).json()
    other = await (await post(client, "/api/lookup", body(chapter_id, token=KAUKANA))).json()

    assert before["in_deck"] is False
    assert after["cached"] is True
    assert after["in_deck"] is True
    assert other["in_deck"] is False


async def test_in_deck_is_per_chapter(client, book, session_factory):
    _, chapter_id, other_chapter = book
    # Chapter 2, translated with the same text: a deck of its own.
    async with session_factory() as session:
        first = await session.get(BookChapter, chapter_id)
        await session.execute(
            update(BookChapter)
            .where(BookChapter.id == other_chapter)
            .values(
                status=ChapterStatus.done,
                fi_paragraphs=first.fi_paragraphs,
                source_paragraphs=first.source_paragraphs,
            )
        )
        await session.commit()
    await post(client, "/api/cards", body(chapter_id))

    r = await post(client, "/api/lookup", body(other_chapter))

    assert (await r.json())["in_deck"] is False
