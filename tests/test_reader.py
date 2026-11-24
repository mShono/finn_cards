"""The reader Mini App's server side (plan 2026-11-20-kirjakaveri-reader,
"П2"): initData checks (reader/auth.py) and the /api the static app calls
(reader/app.py), through a real aiohttp test server."""

import hashlib
import hmac
import json
import logging
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock
from urllib.parse import urlencode

import pytest
from aiohttp.test_utils import TestClient, TestServer
from bot_helpers import make_settings
from conftest import add_users
from sqlalchemy import update

from kielikaveri.db.books import chapter_by_number, store_book
from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import BookChapter, ChapterStatus
from kielikaveri.reader.app import make_app
from kielikaveri.reader.auth import AuthError, user_id_from_header
from kielikaveri.translate.chapters import split_book
from kielikaveri.translate.fst_check import split_words, words

TOKEN = "123456:TEST"
OWNER = 1
STRANGER = 2
PARAGRAPH = "Он ушёл из дома рано утром, и никто не видел, куда он пошёл. " * 6
CHAPTERS = [["1", PARAGRAPH], ["2", PARAGRAPH], ["3", PARAGRAPH]]
BOOK_TEXT = "\n\n".join(["Название"] + [line for chapter in CHAPTERS for line in chapter])
FI = ["Hän lähti talosta aikaisin aamulla.", "Kukaan ei nähnyt, minne hän meni."]


def init_data(user_id=OWNER, auth_date=None, token=TOKEN) -> str:
    """initData as Telegram signs it (the algorithm aiogram checks)."""
    auth_date = auth_date or datetime.now(UTC)
    fields = {
        "auth_date": str(int(auth_date.timestamp())),
        "query_id": "AAH",
        "user": json.dumps({"id": user_id, "first_name": "Test"}),
    }
    check = "\n".join(f"{k}={v}" for k, v in sorted(fields.items()))
    secret = hmac.new(b"WebAppData", token.encode(), hashlib.sha256).digest()
    fields["hash"] = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
    return urlencode(fields)


def header(**kwargs) -> dict[str, str]:
    return {"Authorization": "tma " + init_data(**kwargs)}


# --- reader/auth.py ---------------------------------------------------------------


def check(value, **kwargs):
    return user_id_from_header(value, TOKEN, {OWNER}, timedelta(hours=24), **kwargs)


def test_valid_init_data_gives_the_user():
    assert check("tma " + init_data()) == OWNER


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        (None, "missing"),
        (init_data(), "missing"),  # no "tma " scheme
        ("tma " + init_data(token="999:OTHER"), "bad_signature"),
        ("tma " + init_data().replace("Test", "Evil"), "bad_signature"),
        ("tma " + init_data(user_id=STRANGER), "not_whitelisted"),
        ("tma " + init_data(auth_date=datetime.now(UTC) - timedelta(hours=25)), "expired"),
    ],
)
def test_refused_init_data(value, reason):
    with pytest.raises(AuthError, match=reason):
        check(value)


# --- tokenization -----------------------------------------------------------------


def test_split_words_puts_exactly_the_fst_words_at_odd_indices():
    text = "– Mä oon EU:n Moskova-joella, eikö? Hän sanoi: ”Kyllä.”"
    pieces = split_words(text)
    assert pieces[1::2] == words(text)
    assert "".join(pieces) == text


# --- reader/app.py ----------------------------------------------------------------


@pytest.fixture
async def session_factory(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    await create_all(engine)
    session_factory = make_session_factory(engine)
    # The users the tests' rows point at (PRAGMA foreign_keys is on).
    await add_users(session_factory, 1, 2)
    yield session_factory
    await engine.dispose()


@pytest.fixture
async def book_id(session_factory):
    """A three-chapter book of OWNER's with chapter 1 translated."""
    async with session_factory() as session:
        book = await store_book(session, OWNER, "Kirja", BOOK_TEXT, split_book(BOOK_TEXT))
        await session.execute(
            update(BookChapter)
            .where(BookChapter.book_id == book.id, BookChapter.number == 1)
            .values(status=ChapterStatus.done, fi_paragraphs=FI)
        )
        await session.commit()
        return book.id


@pytest.fixture
def translator():
    return MagicMock()


@pytest.fixture
async def client(session_factory, translator):
    settings = make_settings(bot_token=TOKEN, whitelist_user_ids=str(OWNER))
    async with TestClient(TestServer(make_app(session_factory, settings, translator))) as client:
        yield client


async def chapter(session_factory, book_id, number) -> BookChapter:
    async with session_factory() as session:
        return await chapter_by_number(session, book_id, number)


async def test_static_app_is_served_without_auth(client):
    response = await client.get("/")
    assert response.status == 200
    assert "static/app.js" in await response.text()
    assert response.headers["Cache-Control"] == "no-cache"
    assert (await client.get("/static/app.js")).status == 200


@pytest.mark.parametrize("path", ["/api/books", "/api/chapters/x"])
async def test_api_needs_init_data(client, path):
    assert (await client.get(path)).status == 401
    assert (await client.get(path, headers=header(user_id=STRANGER))).status == 401


async def test_books_list_chapter_statuses_without_text(client, book_id):
    response = await client.get("/api/books", headers=header())
    assert response.status == 200
    data = await response.json()
    [book] = data["books"]
    assert book["title"] == "Kirja"
    assert [(c["number"], c["status"]) for c in book["chapters"]] == [
        (1, "done"),
        (2, "pending"),
        (3, "pending"),
    ]
    assert "ушёл" not in json.dumps(data, ensure_ascii=False)


async def test_books_of_other_users_are_not_listed(client, session_factory, book_id):
    async with session_factory() as session:
        await store_book(session, STRANGER, "Muu", BOOK_TEXT + "!", split_book(BOOK_TEXT))
        await session.commit()
    data = await (await client.get("/api/books", headers=header())).json()
    assert [b["title"] for b in data["books"]] == ["Kirja"]


async def test_opening_a_chapter_returns_words_and_queues_the_next(
    client, session_factory, book_id, translator
):
    first = await chapter(session_factory, book_id, 1)
    response = await client.get(f"/api/chapters/{first.id}", headers=header())
    assert response.status == 200
    data = await response.json()
    assert data["paragraphs"] == [split_words(p) for p in FI]
    assert data["book"] == {"id": book_id, "title": "Kirja"}
    assert data["prev"] is None
    assert data["next"]["number"] == 2
    assert data["next"]["status"] == "queued"

    assert (await chapter(session_factory, book_id, 1)).opened_at is not None
    assert (await chapter(session_factory, book_id, 2)).status is ChapterStatus.queued
    translator.wake.assert_called_once()


async def test_reopening_does_not_wake_the_translator_again(
    client, session_factory, book_id, translator
):
    first = await chapter(session_factory, book_id, 1)
    await client.get(f"/api/chapters/{first.id}", headers=header())
    await client.get(f"/api/chapters/{first.id}", headers=header())
    translator.wake.assert_called_once()


async def test_untranslated_chapter_is_refused(client, session_factory, book_id, translator):
    second = await chapter(session_factory, book_id, 2)
    response = await client.get(f"/api/chapters/{second.id}", headers=header())
    assert response.status == 409
    assert (await chapter(session_factory, book_id, 3)).status is ChapterStatus.pending
    translator.wake.assert_not_called()


async def test_someone_elses_chapter_looks_missing(session_factory, book_id, translator):
    settings = make_settings(bot_token=TOKEN, whitelist_user_ids=f"{OWNER},{STRANGER}")
    first = await chapter(session_factory, book_id, 1)
    async with TestClient(TestServer(make_app(session_factory, settings, translator))) as client:
        response = await client.get(f"/api/chapters/{first.id}", headers=header(user_id=STRANGER))
        assert response.status == 404
        missing = await client.get("/api/chapters/nope", headers=header(user_id=STRANGER))
        assert missing.status == 404
    assert (await chapter(session_factory, book_id, 1)).opened_at is None


async def test_logs_carry_neither_init_data_nor_book_text(client, session_factory, book_id, caplog):
    caplog.set_level(logging.DEBUG)
    first = await chapter(session_factory, book_id, 1)
    signed = header()
    await client.get(f"/api/chapters/{first.id}", headers=signed)
    await client.get("/api/books", headers={"Authorization": "tma hash=forged"})

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "event=reader.request" in logged
    assert "event=reader.auth_failed reason=bad_signature" in logged
    assert f"user_id={OWNER}" in logged
    assert signed["Authorization"].split("hash=")[-1] not in logged
    assert "forged" not in logged
    assert "talosta" not in logged
