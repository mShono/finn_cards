"""POST /api/tts (plan 2026-11-20-kirjakaveri-reader, "П3"): the tapped
word as mp3, cached on disk. OpenAI is a fake: no network."""

# The database and book fixtures come from test_reader_lookup; tests naming
# them as parameters "redefine" the imports for ruff.
# ruff: noqa: F811

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import openai
import pytest
from aiohttp.test_utils import TestClient, TestServer
from bot_helpers import make_settings
from test_reader import OWNER, TOKEN, header
from test_reader_lookup import (  # noqa: F401 - fixtures among them
    BOOK_TEXT_PIECES,
    KAUKANA,
    MA,
    TALOSSA,
    FakeOpenAI,
    body,
    book,
    debug_logs,
    make_book,
    session_factory,
)

from kielikaveri.reader.app import LOOKUP_CLIENT, TTS_CLIENT, TTS_LOCKS, make_app
from kielikaveri.reader.speech import INSTRUCTIONS, cache_path, has_speech

DATA = Path(__file__).parent / "data"
# Real OpenAI answers (25.11.2026, words picked for the test, not book text):
# gpt-4o-mini-tts saying "kirja", and tts-1's empty clip for "talossa" -
# valid mp3, near-silence, the reason for has_speech.
MP3 = (DATA / "tts_speech.mp3").read_bytes()
EMPTY_MP3 = (DATA / "tts_empty.mp3").read_bytes()
MODEL = "gpt-4o-mini-tts"


class FakeTTS:
    def __init__(
        self,
        delay: float = 0.0,
        error: Exception | None = None,
        answers: list[bytes] | None = None,
    ):
        self.delay = delay
        self.error = error
        # Answers in turn; past the list (or without one), MP3.
        self.answers = list(answers or [])
        self.calls: list[dict] = []
        self.audio = SimpleNamespace(speech=SimpleNamespace(create=self._create))
        self.close = AsyncMock()

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        audio = self.answers.pop(0) if self.answers else MP3
        return SimpleNamespace(aread=AsyncMock(return_value=audio))


def build_app(session_factory, tts, cache_dir, **overrides):
    settings = make_settings(
        bot_token=TOKEN,
        whitelist_user_ids=str(OWNER),
        reader_tts_cache_dir=str(cache_dir),
        **overrides,
    )
    app = make_app(session_factory, settings, MagicMock())
    app[TTS_CLIENT] = tts
    app[LOOKUP_CLIENT] = FakeOpenAI()
    return app


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "data" / "tts_cache"


@pytest.fixture
def tts():
    return FakeTTS()


@pytest.fixture
async def client(session_factory, tts, cache_dir):
    async with TestClient(TestServer(build_app(session_factory, tts, cache_dir))) as client:
        yield client


async def post(client, payload, **header_kwargs):
    return await client.post("/api/tts", json=payload, headers=header(**header_kwargs))


async def test_tts_speaks_the_word_as_the_text_spells_it(client, book, tts, cache_dir):
    _, chapter_id, _ = book

    r = await post(client, body(chapter_id, token=TALOSSA))

    assert r.status == 200
    assert r.content_type == "audio/mpeg"
    assert r.headers["Cache-Control"] == "no-store"
    assert await r.read() == MP3
    [call] = tts.calls
    assert call == {
        "model": MODEL,
        "voice": "alloy",
        "input": "talossa",
        "speed": 1.0,
        "instructions": INSTRUCTIONS,
    }
    path = cache_path(cache_dir, MODEL, 1.0, "talossa")
    assert path.read_bytes() == MP3
    # Written atomically: no temporary file left next to it.
    assert [p.name for p in cache_dir.iterdir()] == [path.name]


async def test_second_tts_comes_from_the_disk(client, book, tts):
    _, chapter_id, _ = book

    first = await (await post(client, body(chapter_id))).read()
    second = await (await post(client, body(chapter_id))).read()

    assert first == second == MP3
    assert len(tts.calls) == 1


async def test_an_existing_file_is_served_without_openai(client, book, tts, cache_dir):
    _, chapter_id, _ = book
    cache_dir.mkdir(parents=True)
    cache_path(cache_dir, MODEL, 1.0, "Mä").write_bytes(b"from-disk")

    r = await post(client, body(chapter_id, token=MA))

    assert await r.read() == b"from-disk"
    assert tts.calls == []


def test_file_name_depends_on_model_speed_and_text(tmp_path):
    names = {
        cache_path(tmp_path, "tts-1", 0.8, "talo").name,
        cache_path(tmp_path, "tts-2", 0.8, "talo").name,
        cache_path(tmp_path, "tts-1", 1.0, "talo").name,
        cache_path(tmp_path, "tts-1", 0.8, "Talo").name,
        cache_path(tmp_path, MODEL, 0.8, "talo").name,
    }
    assert len(names) == 5
    assert all(name.endswith(".mp3") and "talo" not in name.lower() for name in names)


async def test_reader_settings_choose_the_model_not_learns(session_factory, book, cache_dir):
    _, chapter_id, _ = book
    tts = FakeTTS()
    app = build_app(
        session_factory,
        tts,
        cache_dir,
        reader_tts_model="tts-1-hd",
        reader_tts_speed=0.9,
        openai_tts_model="tts-1",
    )
    async with TestClient(TestServer(app)) as client:
        assert (await post(client, body(chapter_id, token=TALOSSA))).status == 200

    # tts-1 models take no instructions (the API says they don't work there).
    [call] = tts.calls
    assert call == {"model": "tts-1-hd", "voice": "alloy", "input": "talossa", "speed": 0.9}
    assert cache_path(cache_dir, "tts-1-hd", 0.9, "talossa").read_bytes() == MP3


def test_has_speech_tells_openais_empty_clip_from_a_word():
    assert has_speech(MP3)
    assert not has_speech(EMPTY_MP3)
    # An ID3 tag in front changes nothing.
    tag = b"ID3\x04\x00\x00\x00\x00\x00\x05" + b"\x00" * 5
    assert has_speech(tag + MP3)
    assert not has_speech(tag + EMPTY_MP3)
    # Not an mp3, or no audio at all.
    for junk in (b"", b"ID3fake-mp3", b'{"error": "x"}', MP3[:200], b"\xff" * 64):
        assert not has_speech(junk), junk[:16]


async def test_an_empty_clip_is_retried_and_never_cached(
    session_factory, book, cache_dir, debug_logs
):
    _, chapter_id, _ = book
    tts = FakeTTS(answers=[EMPTY_MP3, MP3])
    async with TestClient(TestServer(build_app(session_factory, tts, cache_dir))) as client:
        r = await post(client, body(chapter_id, token=TALOSSA))
        assert r.status == 200
        assert await r.read() == MP3

    assert len(tts.calls) == 2
    assert cache_path(cache_dir, MODEL, 1.0, "talossa").read_bytes() == MP3
    assert "event=reader.tts_empty " in debug_logs.text
    assert "attempt=1 " in debug_logs.text


async def test_twice_empty_is_502_and_leaves_no_file(session_factory, book, cache_dir, debug_logs):
    _, chapter_id, _ = book
    tts = FakeTTS(answers=[EMPTY_MP3, EMPTY_MP3])
    async with TestClient(TestServer(build_app(session_factory, tts, cache_dir))) as client:
        r = await post(client, body(chapter_id, token=TALOSSA))
        assert r.status == 502
        assert await r.json() == {"reason": "openai"}
        # The next tap asks OpenAI again instead of serving silence.
        again = await post(client, body(chapter_id, token=TALOSSA))
        assert await again.read() == MP3

    assert len(tts.calls) == 3
    assert "error=EmptySpeechError" in debug_logs.text
    for piece in BOOK_TEXT_PIECES:
        assert piece not in debug_logs.text, piece


async def test_each_retry_goes_through_the_breaker(session_factory, book, cache_dir):
    _, chapter_id, _ = book
    tts = FakeTTS(answers=[EMPTY_MP3, MP3])
    app = build_app(session_factory, tts, cache_dir, reader_breaker_max_calls=1)
    async with TestClient(TestServer(app)) as client:
        r = await post(client, body(chapter_id, token=TALOSSA))
        assert r.status == 503
        assert await r.json() == {"reason": "rate"}

    assert len(tts.calls) == 1
    assert not cache_dir.exists() or list(cache_dir.iterdir()) == []


async def test_double_tap_makes_one_tts_call(session_factory, book, cache_dir):
    _, chapter_id, _ = book
    tts = FakeTTS(delay=0.2)
    app = build_app(session_factory, tts, cache_dir)
    async with TestClient(TestServer(app)) as client:
        first, second = await asyncio.gather(
            post(client, body(chapter_id)), post(client, body(chapter_id))
        )
        assert await first.read() == await second.read() == MP3

    assert len(tts.calls) == 1
    assert len(app[TTS_LOCKS]) == 0


async def test_failed_tts_leaves_no_file(session_factory, book, cache_dir):
    _, chapter_id, _ = book
    tts = FakeTTS(error=openai.APIConnectionError(request=SimpleNamespace()))
    async with TestClient(TestServer(build_app(session_factory, tts, cache_dir))) as client:
        r = await post(client, body(chapter_id))

    assert r.status == 502
    assert not cache_dir.exists() or list(cache_dir.iterdir()) == []


async def test_tts_checks_like_lookup(client, book, tts, session_factory):
    _, chapter_id, pending = book
    _, theirs, _ = await make_book(session_factory, user_id=2, title="Toinen")

    assert (await client.post("/api/tts", json=body(chapter_id))).status == 401
    assert (await client.post("/api/tts", data=b"nope", headers=header())).status == 400
    assert (await post(client, body("missing"))).status == 404
    assert (await post(client, body(theirs))).status == 404
    assert (await post(client, body(pending))).status == 409
    assert (await post(client, body(chapter_id, token=99))).status == 422
    assert tts.calls == []


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [({"reader_breaker_max_calls": 1}, "rate"), ({"reader_daily_max_calls": 1}, "daily")],
)
async def test_tts_breaker_is_503(session_factory, book, cache_dir, overrides, reason):
    _, chapter_id, _ = book
    tts = FakeTTS()
    app = build_app(session_factory, tts, cache_dir, **overrides)
    async with TestClient(TestServer(app)) as client:
        assert (await post(client, body(chapter_id, token=TALOSSA))).status == 200
        # A cache hit does not count.
        assert (await post(client, body(chapter_id, token=TALOSSA))).status == 200
        tripped = await post(client, body(chapter_id, token=KAUKANA))
        assert tripped.status == 503
        assert await tripped.json() == {"reason": reason}
    assert len(tts.calls) == 1


async def test_lookup_and_tts_share_the_reader_breaker(session_factory, book, cache_dir):
    _, chapter_id, _ = book
    tts = FakeTTS()
    app = build_app(session_factory, tts, cache_dir, reader_breaker_max_calls=1)
    async with TestClient(TestServer(app)) as client:
        lookup = await client.post("/api/lookup", json=body(chapter_id), headers=header())
        assert lookup.status == 200
        assert (await post(client, body(chapter_id))).status == 503


async def test_tts_logs_never_carry_book_text(session_factory, book, cache_dir, debug_logs):
    _, chapter_id, _ = book
    tts = FakeTTS()
    async with TestClient(TestServer(build_app(session_factory, tts, cache_dir))) as client:
        for token in (TALOSSA, MA, TALOSSA):
            assert (await post(client, body(chapter_id, token=token))).status == 200
    failing = FakeTTS(error=openai.APIConnectionError(request=SimpleNamespace()))
    async with TestClient(TestServer(build_app(session_factory, failing, cache_dir))) as client:
        assert (await post(client, body(chapter_id, token=KAUKANA))).status == 502

    text = debug_logs.text
    assert "event=reader.tts " in text
    assert "event=reader.tts_failed " in text
    for piece in BOOK_TEXT_PIECES:
        assert piece not in text, piece
    hit = [r.getMessage() for r in debug_logs.records if "cached=True" in r.getMessage()]
    assert hit and f"chapter_id={chapter_id} paragraph=0 token={TALOSSA}" in hit[0]
