"""POST /api/cards (plan 2026-11-20-kirjakaveri-reader, "П4"): ➕ saves the
tapped word to the chapter's deck, through a real aiohttp test server. The
FST is real; OpenAI is a fake answering the lookup and the excerpt. The
book is test_reader_lookup's made-up one."""

# The database fixture comes from test_reader_lookup; tests naming it as a
# parameter "redefine" the import for ruff.
# ruff: noqa: F811

import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import jsonschema
import openai
import pytest
from aiohttp.test_utils import TestClient, TestServer
from bot_helpers import make_settings
from conftest import log_fields
from sqlalchemy import func, select
from test_reader import OWNER, STRANGER, TOKEN, header
from test_reader_lookup import (
    ANSWERS,
    KAUKANA,
    MA,
    TALOSSA,
    api_error,
    make_book,
    session_factory,  # noqa: F401 - the fixture
)

from kielikaveri.db.models import Card, Deck, Note
from kielikaveri.reader.app import CARD_LOCKS, LOOKUP_CLIENT, LOOKUP_LOCKS, TTS_CLIENT, make_app
from kielikaveri.reader.excerpt import EXCERPT_SCHEMA_NAME
from kielikaveri.reader.lookup import LOOKUP_SCHEMA_NAME

# Verbatim pieces of test_reader_lookup.FI[0] / SOURCE[0] per word.
EXCERPTS = {
    "talossa": {"fragment_fi": "Mä oon talossa", "fragment_ru": "Я в доме"},
    "Mä": {"fragment_fi": "Mä oon talossa", "fragment_ru": "Я в доме"},
    "kaukana": {"fragment_fi": "Pavlivka on kaukana", "fragment_ru": "Павловка далеко"},
}


def _response(payload: dict | None, text: str | None = None):
    return SimpleNamespace(
        output_text=text if text is not None else json.dumps(payload, ensure_ascii=False),
        output=[],
        usage=SimpleNamespace(input_tokens=300, output_tokens=40, total_tokens=340),
    )


class FakeOpenAI:
    """Answers the lookup and the excerpt by the word they are sent; any
    other call (the lemma or forms settling) is a test failure here."""

    def __init__(self, delay: float = 0.0, error: Exception | None = None, excerpts=None):
        self.delay = delay
        self.error = error
        self.excerpts = excerpts or EXCERPTS
        self.requests: list[dict] = []
        self.responses = SimpleNamespace(create=self._create)
        self.models = SimpleNamespace(retrieve=AsyncMock())
        self.close = AsyncMock()

    def calls(self, name: str) -> int:
        return sum(1 for r in self.requests if r["text"]["format"]["name"] == name)

    async def _create(self, **kwargs):
        self.requests.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        name = kwargs["text"]["format"]["name"]
        payload = json.loads(kwargs["input"])
        if name == LOOKUP_SCHEMA_NAME:
            return _response(ANSWERS[payload["word"]])
        assert name == EXCERPT_SCHEMA_NAME, name
        return _response(self.excerpts[payload["word"]])


def build_app(session_factory, fake, tmp_path, **overrides):
    settings = make_settings(
        bot_token=TOKEN,
        whitelist_user_ids=str(OWNER),
        reader_tts_cache_dir=str(tmp_path / "tts"),
        **overrides,
    )
    app = make_app(session_factory, settings, MagicMock())
    app[LOOKUP_CLIENT] = fake
    tts = MagicMock()
    tts.close = AsyncMock()
    app[TTS_CLIENT] = tts
    return app


@pytest.fixture
async def book(session_factory):
    return await make_book(session_factory)


@pytest.fixture
def fake():
    return FakeOpenAI()


@pytest.fixture
async def client(session_factory, fake, tmp_path):
    async with TestClient(TestServer(build_app(session_factory, fake, tmp_path))) as client:
        yield client


def body(chapter_id, paragraph=0, token=TALOSSA) -> dict:
    return {"chapter_id": chapter_id, "paragraph": paragraph, "token": token}


async def plus(client, payload, **header_kwargs):
    return await client.post("/api/cards", json=payload, headers=header(**header_kwargs))


async def all_notes(session_factory) -> list[Note]:
    async with session_factory() as session:
        return list((await session.scalars(select(Note))).all())


# --- 200 ---------------------------------------------------------------------------


async def test_plus_saves_the_word_to_the_chapter_deck(client, book, fake, session_factory):
    _, chapter_id, _ = book

    r = await plus(client, body(chapter_id))

    assert r.status == 200
    assert await r.json() == {"status": "saved", "deck": "Kirja - гл.1"}
    [note] = await all_notes(session_factory)
    assert (note.lemma, note.pos, note.translation_ru) == ("talo", "substantiivi", "дом")
    assert note.example_fi == "Mä oon talossa."
    assert note.example_ru == "Я в доме."
    assert note.meta["origin"] == "text"
    assert note.meta["forms_source"] == "fst"
    assert "puhekieli" not in note.meta
    async with session_factory() as session:
        deck = await session.get(Deck, note.deck_id)
        assert deck.chapter_id == chapter_id
        assert deck.user_id == OWNER
        # Learning makes the cards (sync_user_card_types), as for /add.
        assert await session.scalar(select(func.count()).select_from(Card)) == 0
    assert fake.calls(LOOKUP_SCHEMA_NAME) == 1
    assert fake.calls(EXCERPT_SCHEMA_NAME) == 1


async def test_plus_uses_the_cached_lookup(client, book, fake):
    _, chapter_id, _ = book
    await client.post("/api/lookup", json=body(chapter_id), headers=header())

    r = await plus(client, body(chapter_id))

    assert r.status == 200
    assert fake.calls(LOOKUP_SCHEMA_NAME) == 1


async def test_the_word_comes_from_the_database_not_the_client(client, book, session_factory):
    _, chapter_id, _ = book
    payload = {**body(chapter_id), "lemma": "huijaus", "translation_ru": "обман"}

    assert (await plus(client, payload)).status == 200

    [note] = await all_notes(session_factory)
    assert (note.lemma, note.translation_ru) == ("talo", "дом")


async def test_a_second_plus_is_a_duplicate_without_another_call(
    client, book, fake, session_factory
):
    _, chapter_id, _ = book
    await plus(client, body(chapter_id))

    r = await plus(client, body(chapter_id))

    assert await r.json() == {"status": "duplicate", "deck": "Kirja - гл.1"}
    assert len(await all_notes(session_factory)) == 1
    # Neither the lookup (cached) nor the excerpt (a duplicate needs none).
    assert fake.calls(LOOKUP_SCHEMA_NAME) == 1
    assert fake.calls(EXCERPT_SCHEMA_NAME) == 1


async def test_double_plus_makes_one_card(session_factory, book, tmp_path):
    _, chapter_id, _ = book
    fake = FakeOpenAI(delay=0.1)
    app = build_app(session_factory, fake, tmp_path)
    async with TestClient(TestServer(app)) as client:
        first, second = await asyncio.gather(
            plus(client, body(chapter_id)), plus(client, body(chapter_id))
        )
        statuses = sorted([(await first.json())["status"], (await second.json())["status"]])

    assert statuses == ["duplicate", "saved"]
    assert len(await all_notes(session_factory)) == 1
    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(Deck)) == 1
    assert fake.calls(LOOKUP_SCHEMA_NAME) == 1
    assert len(app[CARD_LOCKS]) == 0
    assert len(app[LOOKUP_LOCKS]) == 0


async def test_puhekieli_gets_the_literary_lemma_and_meta(client, book, session_factory):
    _, chapter_id, _ = book

    r = await plus(client, body(chapter_id, token=MA))

    assert (await r.json())["status"] == "saved"
    [note] = await all_notes(session_factory)
    assert note.lemma == "minä"
    assert note.meta["puhekieli"] == "Mä"


async def test_a_bad_excerpt_falls_back_to_the_rule(session_factory, book, tmp_path):
    _, chapter_id, _ = book
    fake = FakeOpenAI(excerpts={"talossa": {"fragment_fi": "Olen talossa", "fragment_ru": "x"}})
    async with TestClient(TestServer(build_app(session_factory, fake, tmp_path))) as client:
        assert (await plus(client, body(chapter_id))).status == 200

    [note] = await all_notes(session_factory)
    assert note.example_fi == "Mä oon talossa."
    assert note.example_ru == "Я в доме, — сказал он."


async def test_words_of_one_chapter_share_its_deck(client, book, session_factory):
    _, chapter_id, _ = book
    await plus(client, body(chapter_id, token=TALOSSA))
    await plus(client, body(chapter_id, token=KAUKANA))

    notes = await all_notes(session_factory)
    assert len(notes) == 2
    assert len({n.deck_id for n in notes}) == 1


# --- errors ------------------------------------------------------------------------


async def test_without_init_data_it_is_401(client, book):
    _, chapter_id, _ = book
    assert (await client.post("/api/cards", json=body(chapter_id))).status == 401


@pytest.mark.parametrize(
    "payload",
    [b"not json", b"{}", b'{"chapter_id": "x", "paragraph": 0, "token": true}'],
)
async def test_bad_body_is_400(client, book, fake, payload):
    r = await client.post("/api/cards", data=payload, headers=header())
    assert r.status == 400
    assert fake.requests == []


async def test_someone_elses_chapter_is_404(client, book, fake, session_factory):
    _, theirs, _ = await make_book(session_factory, user_id=STRANGER, title="Toinen")
    assert (await plus(client, body(theirs))).status == 404
    assert fake.requests == []


async def test_untranslated_chapter_is_409(client, book, fake):
    _, _, pending = book
    assert (await plus(client, body(pending))).status == 409


async def test_missing_word_is_422(client, book, fake):
    _, chapter_id, _ = book
    r = await plus(client, body(chapter_id, token=500))
    assert r.status == 422
    assert fake.requests == []


@pytest.mark.parametrize(
    ("limit", "reason"), [("reader_breaker_max_calls", "rate"), ("reader_daily_max_calls", "daily")]
)
async def test_the_breaker_is_503(session_factory, book, tmp_path, limit, reason):
    _, chapter_id, _ = book
    fake = FakeOpenAI()
    # The lookup gets through, the excerpt does not.
    app = build_app(session_factory, fake, tmp_path, **{limit: 1})
    async with TestClient(TestServer(app)) as client:
        r = await plus(client, body(chapter_id))
        assert r.status == 503
        assert await r.json() == {"reason": reason}
    assert await all_notes(session_factory) == []


async def test_openai_error_is_502(session_factory, book, tmp_path):
    _, chapter_id, _ = book
    fake = FakeOpenAI(error=openai.APIConnectionError(request=SimpleNamespace()))
    async with TestClient(TestServer(build_app(session_factory, fake, tmp_path))) as client:
        r = await plus(client, body(chapter_id))
        assert r.status == 502
        assert await r.json() == {"reason": "openai"}


class _Invalid:
    def validate(self, note):
        raise jsonschema.ValidationError(f"{note['lemma']!r} is bad: {note['example_fi']}")


async def test_a_card_that_cannot_be_built_is_422_save_failed(
    client,
    book,
    session_factory,
    monkeypatch,
):
    _, chapter_id, _ = book
    monkeypatch.setattr("kielikaveri.cards_save.load_validator", lambda: _Invalid())

    r = await plus(client, body(chapter_id))

    assert r.status == 422
    assert await r.json() == {"reason": "save_failed"}
    assert await all_notes(session_factory) == []


# --- logs ----------------------------------------------------------------------------

BOOK_TEXT_PIECES = [
    "talossa",
    "talo",
    "Mä",
    "minä",
    "oon",
    "sanoi",
    "kaukana",
    "Pavlivka",
    "доме",
    "сказал",
    "Павловка",
    "дом",
    "далеко",
]


@pytest.fixture
def debug_logs(caplog):
    caplog.set_level(logging.DEBUG)
    logging.getLogger("aiosqlite").setLevel(logging.INFO)
    return caplog


async def test_logs_never_carry_book_text(session_factory, book, tmp_path, debug_logs, monkeypatch):
    _, chapter_id, _ = book
    fake = FakeOpenAI()
    async with TestClient(TestServer(build_app(session_factory, fake, tmp_path))) as client:
        assert (await plus(client, body(chapter_id, token=TALOSSA))).status == 200
        assert (await plus(client, body(chapter_id, token=TALOSSA))).status == 200
        assert (await plus(client, body(chapter_id, token=MA))).status == 200
        # A schema failure: cards_save logs it with a traceback whose
        # message quotes the note.
        monkeypatch.setattr("kielikaveri.cards_save.load_validator", lambda: _Invalid())
        assert (await plus(client, body(chapter_id, token=KAUKANA))).status == 422

        # Something unexpected, its message quoting the word.
        async def explode(*args, **kwargs):
            raise RuntimeError("talossa: '– Mä oon talossa, hän sanoi.' Я в доме")

        monkeypatch.setattr("kielikaveri.reader.cards.store_note", explode)
        monkeypatch.setattr("kielikaveri.cards_save.load_validator", lambda: MagicMock())
        async with session_factory() as session:
            await session.execute(Note.__table__.delete())
            await session.commit()
        assert (await plus(client, body(chapter_id, token=TALOSSA))).status == 500
    # An OpenAI error quoting the request.
    fake.error = api_error("Invalid input: 'kaukana' in 'Pavlivka on kaukana.'")
    async with TestClient(TestServer(build_app(session_factory, fake, tmp_path))) as client:
        async with session_factory() as session:
            await session.execute(Note.__table__.delete())
            await session.commit()
        assert (await plus(client, body(chapter_id, token=1))).status == 502

    text = debug_logs.text
    assert "event=reader.card " in text
    assert "event=reader.card_failed" in text
    for piece in BOOK_TEXT_PIECES:
        assert piece not in text, piece


async def test_card_log_line_has_ids_status_and_tokens(client, book, debug_logs):
    _, chapter_id, _ = book
    await plus(client, body(chapter_id))
    await plus(client, body(chapter_id))

    lines = [
        log_fields(r.getMessage())
        for r in debug_logs.records
        if r.getMessage().startswith("event=reader.card ")
    ]
    assert [line["status"] for line in lines] == ["saved", "duplicate"]
    first, second = lines
    assert first["chapter_id"] == chapter_id
    assert (first["paragraph"], first["token"]) == ("0", str(TALOSSA))
    # The lookup's tokens and the excerpt's.
    assert (first["input_tokens"], first["output_tokens"]) == ("600", "80")
    assert (second["input_tokens"], second["output_tokens"]) == ("0", "0")
    assert "duration_ms" in first
