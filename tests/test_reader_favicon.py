"""/favicon.ico (plan 2026-11-20-kirjakaveri-reader, "Мелочи" М18): browsers
ask for it on their own, without initData. It is public, so it is neither a
401 nor a reader.auth_failed warning on every open of the reader."""

import logging
from unittest.mock import MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer
from bot_helpers import make_settings
from test_reader import OWNER, TOKEN

from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.reader.app import CONTENT_SECURITY_POLICY, STATIC_DIR, make_app


@pytest.fixture
async def client(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    await create_all(engine)
    settings = make_settings(bot_token=TOKEN, whitelist_user_ids=str(OWNER))
    app = make_app(make_session_factory(engine), settings, MagicMock())
    async with TestClient(TestServer(app)) as client:
        yield client
    await engine.dispose()


def logged(caplog) -> str:
    return "\n".join(r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("method", ["GET", "HEAD"])
async def test_favicon_needs_no_init_data_and_logs_no_auth_failure(client, caplog, method):
    caplog.set_level(logging.DEBUG)
    response = await client.request(method, "/favicon.ico")
    assert response.status in (200, 204)
    assert "reader.auth_failed" not in logged(caplog)


async def test_favicon_is_the_svg_icon(client):
    response = await client.get("/favicon.ico")
    assert response.status == 200
    assert response.headers["Content-Type"].startswith("image/svg+xml")
    assert await response.read() == (STATIC_DIR / "favicon.svg").read_bytes()


async def test_favicon_keeps_public_headers(client):
    response = await client.get("/favicon.ico")
    assert response.headers["Cache-Control"] == "no-cache"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["Content-Security-Policy"] == CONTENT_SECURITY_POLICY
    assert "aiohttp" not in response.headers.get("Server", "").lower()


async def test_static_favicon_svg_is_served(client):
    response = await client.get("/static/favicon.svg")
    assert response.status == 200
    assert response.headers["Content-Type"].startswith("image/svg+xml")
    assert b"<svg" in await response.read()


async def test_page_links_the_icon(client):
    html = await (await client.get("/")).text()
    assert '<link rel="icon" href="static/favicon.svg"' in html


@pytest.mark.parametrize(
    "path", ["/api/books", "/api/chapters/nope", "/favicon.png", "/favicon.ico/x", "/nope"]
)
async def test_other_paths_without_init_data_are_still_401(client, caplog, path):
    caplog.set_level(logging.DEBUG)
    assert (await client.get(path)).status == 401
    assert "event=reader.auth_failed reason=missing" in logged(caplog)
