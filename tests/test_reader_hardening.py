"""Hardening of the reader's web side (plan 2026-11-20-kirjakaveri-reader,
review of Л0-П2, "Мелочи" М1, М2, М3, М13): an empty BOT_TOKEN, the list of
public routes, security headers and auth_date from the future."""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer
from bot_helpers import make_settings
from sqlalchemy import select, update
from test_reader import BOOK_TEXT, FI, OWNER, TOKEN, header, init_data

from kielikaveri.db.books import store_book
from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import BookChapter, ChapterStatus, User
from kielikaveri.reader.app import CONTENT_SECURITY_POLICY, make_app
from kielikaveri.reader.auth import AuthError, user_id_from_header
from kielikaveri.translate.chapters import split_book

TITLE = "Salainen kirja"


@pytest.fixture
async def factory(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    await create_all(engine)
    yield make_session_factory(engine)
    await engine.dispose()


@pytest.fixture
async def chapter_id(factory):
    """OWNER's book with chapter 1 translated; the User row exists."""
    async with factory() as session:
        session.add(User(id=OWNER))
        await session.flush()
        book = await store_book(session, OWNER, TITLE, BOOK_TEXT, split_book(BOOK_TEXT))
        await session.execute(
            update(BookChapter)
            .where(BookChapter.book_id == book.id, BookChapter.number == 1)
            .values(status=ChapterStatus.done, fi_paragraphs=FI)
        )
        await session.commit()
        return await session.scalar(
            select(BookChapter.id).where(BookChapter.book_id == book.id, BookChapter.number == 1)
        )


@pytest.fixture
async def client(factory, chapter_id):
    settings = make_settings(bot_token=TOKEN, whitelist_user_ids=str(OWNER))
    async with TestClient(TestServer(make_app(factory, settings, MagicMock()))) as client:
        yield client


async def raw_get(client, path: str, authorization: str | None = None) -> tuple[int, str]:
    """GET with the request target sent byte for byte: aiohttp's client
    would normalize dot segments before they ever reach the server."""
    reader, writer = await asyncio.open_connection(client.host, client.port)
    lines = [f"GET {path} HTTP/1.1", f"Host: {client.host}", "Connection: close"]
    if authorization:
        lines.append(f"Authorization: {authorization}")
    writer.write(("\r\n".join(lines) + "\r\n\r\n").encode())
    await writer.drain()
    raw = await reader.read()
    writer.close()
    await writer.wait_closed()
    status_line, _, rest = raw.decode(errors="replace").partition("\r\n")
    return int(status_line.split()[1]), rest


# --- М1: empty BOT_TOKEN -----------------------------------------------------------


def test_make_app_refuses_an_empty_bot_token(factory):
    with pytest.raises(ValueError, match="BOT_TOKEN"):
        make_app(factory, make_settings(bot_token=""), MagicMock())


def test_init_data_signed_with_an_empty_token_is_refused_under_an_empty_token():
    # With an empty token the HMAC key is public: such initData is forgeable.
    value = "tma " + init_data(token="")
    with pytest.raises(AuthError, match="no_bot_token"):
        user_id_from_header(value, "", {OWNER}, timedelta(hours=24))


# --- М13: auth_date from the future ------------------------------------------------


def check_at(auth_date: datetime) -> int:
    value = "tma " + init_data(auth_date=auth_date)
    return user_id_from_header(value, TOKEN, {OWNER}, timedelta(hours=24))


def test_auth_date_within_clock_skew_is_accepted():
    assert check_at(datetime.now(UTC) + timedelta(minutes=4)) == OWNER


def test_auth_date_beyond_clock_skew_is_refused():
    with pytest.raises(AuthError, match="from_future"):
        check_at(datetime.now(UTC) + timedelta(minutes=6))


async def test_future_auth_date_is_401_over_http(client):
    future = datetime.now(UTC) + timedelta(hours=1)
    assert (await client.get("/api/books", headers=header(auth_date=future))).status == 401


# --- М2: explicit public routes ----------------------------------------------------


async def test_public_routes_need_no_init_data(client):
    assert (await client.get("/")).status == 200
    assert (await client.head("/")).status == 200
    assert (await client.get("/static/app.js")).status == 200
    assert (await client.get("/static/style.css")).status == 200


@pytest.mark.parametrize("path", ["/nope", "/favicon.ico", "/index.html", "/api", "/robots.txt"])
async def test_unknown_path_without_init_data_is_401(client, path):
    assert (await client.get(path)).status == 401


async def test_unknown_path_with_init_data_is_404(client):
    assert (await client.get("/nope", headers=header())).status == 404


async def test_missing_static_file_is_404_without_init_data(client):
    # The static route is public as a whole; a missing file is just missing.
    assert (await client.get("/static/nope.js")).status == 404


@pytest.mark.parametrize(
    "path",
    [
        "/static/../api/books",
        "/static/%2e%2e/api/books",
        "/static/%2E%2E/api/books",
        "/static/..%2fapi%2fbooks",
        "/static/%2e%2e%2fapi%2fbooks",
        "/static/./../api/books",
        "//api/books",
        "/static/../app.py",
        "/static/%2e%2e/app.py",
        "/static/..%2fapp.py",
        "/static/../../../../etc/passwd",
    ],
)
async def test_no_path_spelling_through_static_gets_past_auth(client, path):
    status, body = await raw_get(client, path)
    assert status in (401, 403, 404)
    assert TITLE not in body
    assert "def make_app" not in body


async def test_dot_segments_to_api_still_need_valid_init_data(client):
    # Whatever the server makes of the path, initData decides: with it the
    # /api answer is the user's own, without it nothing.
    status, body = await raw_get(
        client, "/static/../api/books", authorization="tma " + init_data(user_id=2)
    )
    assert status != 200
    assert TITLE not in body


# --- М3: headers -------------------------------------------------------------------


def assert_security_headers(response):
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["Content-Security-Policy"] == CONTENT_SECURITY_POLICY
    assert "aiohttp" not in response.headers.get("Server", "").lower()
    assert "python" not in response.headers.get("Server", "").lower()


@pytest.mark.parametrize("path", ["/", "/static/app.js", "/static/style.css", "/static/nope.js"])
async def test_public_responses_revalidate_and_carry_security_headers(client, path):
    response = await client.get(path)
    assert response.headers["Cache-Control"] == "no-cache"
    assert_security_headers(response)


async def test_api_answer_is_not_stored(client, chapter_id):
    for path in ("/api/books", f"/api/chapters/{chapter_id}", "/api/chapters/nope"):
        response = await client.get(path, headers=header())
        assert response.status in (200, 404)
        assert response.headers["Cache-Control"] == "no-store"
        assert_security_headers(response)


@pytest.mark.parametrize("path", ["/api/books", "/nope"])
async def test_refusals_carry_the_same_headers(client, path):
    response = await client.get(path)
    assert response.status == 401
    assert response.headers["Cache-Control"] == "no-store"
    assert_security_headers(response)


def test_csp_allows_what_the_page_loads_and_nothing_else():
    directives = dict(d.split(" ", 1) for d in CONTENT_SECURITY_POLICY.split("; "))
    assert directives["default-src"] == "'none'"
    assert directives["script-src"] == "'self' https://telegram.org"
    # No inline script anywhere: only telegram-web-app.js's <style> needs inline.
    assert "unsafe-inline" not in directives["script-src"]
    assert "unsafe-eval" not in CONTENT_SECURITY_POLICY
    assert directives["connect-src"] == "'self'"
    # Breaking the web.telegram.org iframe is worse than leaving framing open.
    assert "frame-ancestors" not in directives


async def test_page_has_no_inline_script_or_style_attribute(client):
    html = await (await client.get("/")).text()
    for tag in html.split("<script")[1:]:
        assert tag.split(">", 1)[0].strip().startswith("src=")
    assert "style=" not in html
    assert "<style" not in html
    js = await (await client.get("/static/app.js")).text()
    assert "innerHTML" not in js
    assert "eval(" not in js
