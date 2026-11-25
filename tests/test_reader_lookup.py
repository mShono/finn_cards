"""POST /api/lookup (plan 2026-11-20-kirjakaveri-reader, "П3"): a tapped
word explained, through a real aiohttp test server. OpenAI is a fake: no
network. The paragraphs are made up here, not taken from a book."""

import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import openai
import pytest
from aiohttp.test_utils import TestClient, TestServer
from bot_helpers import make_settings
from conftest import add_users, delete_book
from sqlalchemy import func, select, update
from test_reader import OWNER, STRANGER, TOKEN, header

from kielikaveri.db.books import chapter_by_number, store_book
from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import BookChapter, ChapterStatus, LookupCache
from kielikaveri.reader.app import LOOKUP_CLIENT, LOOKUP_LOCKS, TTS_CLIENT, make_app
from kielikaveri.reader.lookup import (
    LOOKUP_SCHEMA_NAME,
    MARK_CLOSE,
    MARK_OPEN,
    install_log_redaction,
)
from kielikaveri.translate.chapters import split_book

FI = [
    "– Mä oon talossa, hän sanoi. Pavlivka on kaukana.",
    "Kukaan ei nähnyt häntä.",
    "Ilman alkuperäistä.",
]
SOURCE = [
    "— Я в доме, — сказал он. Павловка далеко.",
    "Никто его не видел.",
    "",
]
FST_FLAGS = {"0": ["Pavlivka"]}
# Indices in words(FI[0]): Mä oon talossa hän sanoi Pavlivka on kaukana
MA, TALOSSA, PAVLIVKA, KAUKANA = 0, 2, 5, 7
BOOK_TEXT = "Kirja\n\n1\n\nТекст.\n\n2\n\nТекст."

ANSWERS = {
    "talossa": {
        "lemma": "talo",
        "pos": "substantiivi",
        "form_ru": "инессив ед. ч., „в доме“",
        "translation_ru": "дом",
        "puhekieli": None,
        "source_sentences": [0],
    },
    "Mä": {
        "lemma": "minä",
        "pos": "pronomini",
        "form_ru": "номинатив ед. ч., „я“",
        "translation_ru": "я",
        "puhekieli": "Mä",
        "source_sentences": [0],
    },
    "Pavlivka": {
        "lemma": "Pavlivka",
        "pos": "substantiivi",
        "form_ru": "номинатив ед. ч.",
        "translation_ru": "Павловка",
        "puhekieli": None,
        "source_sentences": [1],
    },
    "kaukana": {
        "lemma": "kaukana",
        "pos": "adverbi",
        "form_ru": "наречие, „далеко“",
        "translation_ru": "далеко",
        "puhekieli": None,
        "source_sentences": [1],
    },
}


def response(answer: dict | None, text: str | None = None):
    return SimpleNamespace(
        output_text=text if text is not None else json.dumps(answer, ensure_ascii=False),
        output=[],
        usage=SimpleNamespace(input_tokens=321, output_tokens=45, total_tokens=366),
    )


class FakeOpenAI:
    """client.responses.create answering the lookup by the word it is sent."""

    def __init__(self, answers=None, delay: float = 0.0, error: Exception | None = None):
        self.answers = answers or ANSWERS
        self.delay = delay
        self.error = error
        self.requests: list[dict] = []
        self.responses = SimpleNamespace(create=self._create)
        self.close = AsyncMock()

    async def _create(self, **kwargs):
        self.requests.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        assert kwargs["text"]["format"]["name"] == LOOKUP_SCHEMA_NAME
        payload = json.loads(kwargs["input"])
        return response(self.answers[payload["word"]])


@pytest.fixture
async def session_factory(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    await create_all(engine)
    session_factory = make_session_factory(engine)
    await add_users(session_factory, OWNER, STRANGER)
    yield session_factory
    await engine.dispose()


async def make_book(session_factory, user_id=OWNER, title="Kirja") -> tuple[str, str, str]:
    """(book id, id of chapter 1 - done, id of chapter 2 - pending)."""
    async with session_factory() as session:
        book = await store_book(session, user_id, title, BOOK_TEXT, split_book(BOOK_TEXT))
        await session.execute(
            update(BookChapter)
            .where(BookChapter.book_id == book.id, BookChapter.number == 1)
            .values(
                status=ChapterStatus.done,
                fi_paragraphs=FI,
                source_paragraphs=SOURCE,
                fst_flags=FST_FLAGS,
            )
        )
        await session.commit()
        first = await chapter_by_number(session, book.id, 1)
        second = await chapter_by_number(session, book.id, 2)
        return book.id, first.id, second.id


@pytest.fixture
async def book(session_factory):
    return await make_book(session_factory)


@pytest.fixture
def fake():
    return FakeOpenAI()


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
async def client(session_factory, fake, tmp_path):
    app = build_app(session_factory, fake, tmp_path)
    async with TestClient(TestServer(app)) as client:
        yield client


def body(chapter_id, paragraph=0, token=TALOSSA) -> dict:
    return {"chapter_id": chapter_id, "paragraph": paragraph, "token": token}


async def post(client, payload, **header_kwargs):
    return await client.post("/api/lookup", json=payload, headers=header(**header_kwargs))


# --- 200 ---------------------------------------------------------------------------


async def test_lookup_answers_the_contract(client, book, fake):
    _, chapter_id, _ = book

    r = await post(client, body(chapter_id))

    assert r.status == 200
    assert r.headers["Cache-Control"] == "no-store"
    assert await r.json() == {
        "surface": "talossa",
        "lemma": "talo",
        "pos": "substantiivi",
        "in_dictionary": True,
        "fst_flagged": False,
        "form_ru": "инессив ед. ч., „в доме“",
        "translation_ru": "дом",
        "sentence_ru": "— Я в доме, — сказал он.",
        "puhekieli": None,
        "cached": False,
        "in_deck": False,
    }


async def test_the_model_gets_the_marked_sentence_and_the_numbered_original(client, book, fake):
    _, chapter_id, _ = book

    await post(client, body(chapter_id))

    [request] = fake.requests
    assert request["model"] == "gpt-5.6-terra"
    assert request["text"]["format"]["strict"] is True
    assert json.loads(request["input"]) == {
        "sentence_fi": f"– Mä oon {MARK_OPEN}talossa{MARK_CLOSE}, hän sanoi.",
        "word": "talossa",
        "original_sentences": [
            {"n": 0, "ru": "— Я в доме, — сказал он."},
            {"n": 1, "ru": "Павловка далеко."},
        ],
    }


async def test_the_word_comes_from_the_database_not_the_client(client, book, fake):
    _, chapter_id, _ = book
    payload = {**body(chapter_id), "word": "huijaus", "sentence": "Huijaus."}

    r = await post(client, payload)

    assert (await r.json())["surface"] == "talossa"
    assert json.loads(fake.requests[0]["input"])["word"] == "talossa"


async def test_puhekieli_gets_the_literary_lemma(client, book):
    _, chapter_id, _ = book

    r = await post(client, body(chapter_id, token=MA))

    answer = await r.json()
    assert answer["surface"] == "Mä"
    assert answer["lemma"] == "minä"
    assert answer["puhekieli"] == "Mä"
    assert answer["in_dictionary"] is True


async def test_a_flagged_word_is_fst_flagged(client, book):
    _, chapter_id, _ = book

    flagged = await (await post(client, body(chapter_id, token=PAVLIVKA))).json()
    plain = await (await post(client, body(chapter_id, token=KAUKANA))).json()

    assert flagged["fst_flagged"] is True
    assert flagged["in_dictionary"] is False
    assert flagged["sentence_ru"] == "Павловка далеко."
    assert plain["fst_flagged"] is False


@pytest.mark.parametrize("named", [[], [5], [-1], [0, 9]])
async def test_bad_sentence_numbers_fall_back_to_the_whole_original(
    session_factory, book, tmp_path, named
):
    _, chapter_id, _ = book
    fake = FakeOpenAI({"talossa": {**ANSWERS["talossa"], "source_sentences": named}})
    async with TestClient(TestServer(build_app(session_factory, fake, tmp_path))) as client:
        answer = await (await post(client, body(chapter_id))).json()

    assert answer["sentence_ru"] == SOURCE[0]


async def test_two_neighbouring_original_sentences_are_joined(session_factory, book, tmp_path):
    _, chapter_id, _ = book
    fake = FakeOpenAI({"talossa": {**ANSWERS["talossa"], "source_sentences": [1, 0]}})
    async with TestClient(TestServer(build_app(session_factory, fake, tmp_path))) as client:
        answer = await (await post(client, body(chapter_id))).json()

    assert answer["sentence_ru"] == SOURCE[0]


# --- cache -------------------------------------------------------------------------


async def test_second_lookup_comes_from_the_cache(client, book, fake, session_factory):
    _, chapter_id, _ = book

    first = await (await post(client, body(chapter_id))).json()
    second = await (await post(client, body(chapter_id))).json()

    assert len(fake.requests) == 1
    assert first["cached"] is False
    assert second["cached"] is True
    assert {**second, "cached": False} == first
    async with session_factory() as session:
        row = (await session.scalars(select(LookupCache))).one()
    assert row.chapter_id == chapter_id
    assert row.model == "gpt-5.6-terra"


async def test_cache_key_has_the_model(session_factory, book, tmp_path):
    _, chapter_id, _ = book
    fake = FakeOpenAI()
    for model in ("gpt-a", "gpt-b"):
        app = build_app(session_factory, fake, tmp_path, openai_text_model=model)
        async with TestClient(TestServer(app)) as client:
            await post(client, body(chapter_id))

    assert len(fake.requests) == 2


async def test_double_tap_makes_one_call(session_factory, book, tmp_path):
    _, chapter_id, _ = book
    fake = FakeOpenAI(delay=0.2)
    app = build_app(session_factory, fake, tmp_path)
    async with TestClient(TestServer(app)) as client:
        first, second = await asyncio.gather(
            post(client, body(chapter_id)), post(client, body(chapter_id))
        )
        answers = sorted([(await first.json())["cached"], (await second.json())["cached"]])

    assert len(fake.requests) == 1
    assert answers == [False, True]
    assert len(app[LOOKUP_LOCKS]) == 0  # no lock left behind


async def test_deleting_the_book_deletes_its_cache(client, book, session_factory):
    book_id, chapter_id, _ = book
    await post(client, body(chapter_id))

    async with session_factory() as session:
        await delete_book(session, book_id)
        await session.commit()
        left = await session.scalar(select(func.count()).select_from(LookupCache))

    assert left == 0


async def test_cache_hit_does_not_touch_the_breaker(session_factory, book, tmp_path):
    _, chapter_id, _ = book
    fake = FakeOpenAI()
    app = build_app(session_factory, fake, tmp_path, reader_breaker_max_calls=1)
    async with TestClient(TestServer(app)) as client:
        statuses = [(await post(client, body(chapter_id))).status for _ in range(3)]

    assert statuses == [200, 200, 200]


# --- errors ------------------------------------------------------------------------


async def test_without_init_data_it_is_401(client, book):
    _, chapter_id, _ = book
    r = await client.post("/api/lookup", json=body(chapter_id))
    assert r.status == 401


@pytest.mark.parametrize(
    "payload",
    [
        b"not json",
        b"[1, 2]",
        b'"chapter"',
        b"{}",
        b'{"chapter_id": "x", "paragraph": 0}',
        b'{"chapter_id": 5, "paragraph": 0, "token": 0}',
        b'{"chapter_id": "x", "paragraph": "0", "token": 0}',
        b'{"chapter_id": "x", "paragraph": 0, "token": true}',
        b'{"chapter_id": "x", "paragraph": -1, "token": 0}',
        b'{"chapter_id": "x", "paragraph": 0, "token": 1.5}',
        b"\xff\xfe",
    ],
)
async def test_bad_body_is_400(client, book, fake, payload):
    r = await client.post("/api/lookup", data=payload, headers=header())
    assert r.status == 400
    assert fake.requests == []


async def test_a_large_body_is_refused(client, book, fake):
    _, chapter_id, _ = book
    payload = {**body(chapter_id), "padding": "x" * 20_000}

    r = await post(client, payload)

    assert r.status == 413
    assert fake.requests == []


async def test_missing_chapter_is_404(client, book, fake):
    r = await post(client, body("no-such-chapter"))
    assert r.status == 404
    assert fake.requests == []


async def test_someone_elses_chapter_is_404(client, book, fake, session_factory):
    _, theirs, _ = await make_book(session_factory, user_id=STRANGER, title="Toinen")

    r = await post(client, body(theirs))

    assert r.status == 404
    assert fake.requests == []


async def test_untranslated_chapter_is_409(client, book, fake):
    _, _, pending = book
    r = await post(client, body(pending))
    assert r.status == 409
    assert fake.requests == []


@pytest.mark.parametrize(
    ("paragraph", "token"),
    [(3, 0), (99, 0), (0, 8), (0, 500), (1, 4)],
)
async def test_missing_paragraph_or_word_is_422(client, book, fake, paragraph, token):
    _, chapter_id, _ = book
    r = await post(client, body(chapter_id, paragraph=paragraph, token=token))
    assert r.status == 422
    assert fake.requests == []


async def test_paragraph_without_an_original_is_422(client, book, fake):
    _, chapter_id, _ = book
    r = await post(client, body(chapter_id, paragraph=2, token=0))
    assert r.status == 422
    assert fake.requests == []


async def test_rate_breaker_is_503_rate(session_factory, book, tmp_path):
    _, chapter_id, _ = book
    fake = FakeOpenAI()
    app = build_app(session_factory, fake, tmp_path, reader_breaker_max_calls=1)
    async with TestClient(TestServer(app)) as client:
        ok = await post(client, body(chapter_id, token=TALOSSA))
        tripped = await post(client, body(chapter_id, token=KAUKANA))

        assert ok.status == 200
        assert tripped.status == 503
        assert await tripped.json() == {"reason": "rate"}
    assert len(fake.requests) == 1


async def test_daily_cap_is_503_daily(session_factory, book, tmp_path):
    _, chapter_id, _ = book
    fake = FakeOpenAI()
    app = build_app(session_factory, fake, tmp_path, reader_daily_max_calls=1)
    async with TestClient(TestServer(app)) as client:
        ok = await post(client, body(chapter_id, token=TALOSSA))
        tripped = await post(client, body(chapter_id, token=KAUKANA))

        assert ok.status == 200
        assert tripped.status == 503
        assert await tripped.json() == {"reason": "daily"}


def api_error(message: str) -> openai.APIStatusError:
    # Just what openai.APIStatusError reads from a response.
    fake_response = SimpleNamespace(status_code=400, request=SimpleNamespace(), headers={})
    return openai.BadRequestError(message, response=fake_response, body=None)


@pytest.mark.parametrize(
    "error",
    [
        openai.APIConnectionError(request=SimpleNamespace()),
        api_error("bad request"),
    ],
)
async def test_openai_error_is_502(session_factory, book, tmp_path, error):
    _, chapter_id, _ = book
    fake = FakeOpenAI(error=error)
    async with TestClient(TestServer(build_app(session_factory, fake, tmp_path))) as client:
        r = await post(client, body(chapter_id))
    assert r.status == 502


@pytest.mark.parametrize("text", ["", "   ", "{not json", '{"lemma": "talo"}'])
async def test_unusable_model_answer_is_502_and_not_cached(session_factory, book, tmp_path, text):
    _, chapter_id, _ = book
    fake = FakeOpenAI()
    fake.responses = SimpleNamespace(create=AsyncMock(return_value=response(None, text=text)))
    async with TestClient(TestServer(build_app(session_factory, fake, tmp_path))) as client:
        r = await post(client, body(chapter_id))

    assert r.status == 502
    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(LookupCache)) == 0


# --- app wiring ----------------------------------------------------------------------


def test_routes_are_registered():
    app = make_app(MagicMock(), make_settings(bot_token=TOKEN), MagicMock())
    routes = {
        (route.method, route.resource.canonical)
        for route in app.router.routes()
        if route.resource is not None
    }
    assert ("POST", "/api/lookup") in routes
    assert ("POST", "/api/tts") in routes


async def test_csp_allows_blob_media(client):
    r = await client.get("/")
    directives = [d.strip() for d in r.headers["Content-Security-Policy"].split(";")]
    assert "media-src blob:" in directives
    assert "default-src 'none'" in directives


# --- logs ----------------------------------------------------------------------------

# Every piece of the book's text a lookup handles - the word, its sentence,
# the original and the model's answer about them.
BOOK_TEXT_PIECES = [
    "talossa",
    "talo",
    "Mä",
    "minä",
    "oon",
    "sanoi",
    "Pavlivka",
    "доме",
    "сказал",
    "Павловка",
    "дом",
    "инессив",
]


@pytest.fixture
def debug_logs(caplog):
    caplog.set_level(logging.DEBUG)
    # As in production (logging_config._TEXT_LEAKING_LOGGERS): aiosqlite at
    # DEBUG prints every SQL parameter, the cached answer included.
    logging.getLogger("aiosqlite").setLevel(logging.INFO)
    return caplog


async def test_logs_never_carry_book_text(session_factory, book, tmp_path, debug_logs):
    _, chapter_id, _ = book
    fake = FakeOpenAI()
    app = build_app(session_factory, fake, tmp_path, reader_daily_max_calls=3)
    async with TestClient(TestServer(app)) as client:
        for token in (TALOSSA, MA, PAVLIVKA, TALOSSA):
            assert (await post(client, body(chapter_id, token=token))).status == 200
        # Past the daily cap, and an OpenAI error whose message quotes the
        # request (an API error can do that), with its traceback.
        assert (await post(client, body(chapter_id, token=KAUKANA))).status == 503
    fake.error = api_error("Invalid input: 'talossa' in '– Mä oon talossa, hän sanoi.'")
    async with TestClient(TestServer(build_app(session_factory, fake, tmp_path))) as client:
        assert (await post(client, body(chapter_id, token=KAUKANA))).status == 502

    text = debug_logs.text
    assert "event=reader.lookup " in text
    for piece in BOOK_TEXT_PIECES:
        assert piece not in text, piece


async def test_lookup_log_line_has_ids_cache_and_tokens(client, book, debug_logs):
    _, chapter_id, _ = book
    await post(client, body(chapter_id))
    await post(client, body(chapter_id))

    lines = [r.getMessage() for r in debug_logs.records if "event=reader.lookup " in r.getMessage()]
    assert len(lines) == 2
    assert f"chapter_id={chapter_id} paragraph=0 token={TALOSSA} cached=False" in lines[0]
    assert "input_tokens=321 output_tokens=45" in lines[0]
    assert "cached=True input_tokens=0 output_tokens=0" in lines[1]


def test_redaction_leaves_other_tasks_lines_alone(caplog):
    # /add in the bot logs the lemma it resolves: outside a lookup the
    # filter changes nothing.
    install_log_redaction()
    caplog.set_level(logging.INFO)

    logging.getLogger("kielikaveri.ingest").info("event=x lemma=%s", "talo")

    assert "event=x lemma=talo" in caplog.text
