"""The reader's web side (plan 2026-11-20-kirjakaveri-reader, "П2"): an
aiohttp app inside the bot's process that serves the Mini App's static files
and its /api. Caddy terminates HTTPS in front of it; it listens on localhost.

Every request is checked against initData and the whitelist
(reader/auth.py), except the few public routes: the page itself, its
static files and /favicon.ico - they are code, the book only ever comes through /api.

Logs carry ids, statuses and chapter numbers - never book text or initData.

П3 adds the word lookup and 🔊 (reader/lookup.py, reader/speech.py): POST
/api/lookup and /api/tts, both with {"chapter_id", "paragraph", "token"}.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import timedelta
from pathlib import Path

from aiohttp import web
from openai import AsyncOpenAI, OpenAIError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kielikaveri.config import Settings
from kielikaveri.db.books import chapter_by_number, mark_opened
from kielikaveri.db.models import Book, BookChapter, ChapterStatus
from kielikaveri.llm.breaker import BudgetExceededError, CallBreaker, CircuitOpenError
from kielikaveri.llm.client import make_client
from kielikaveri.logging_context import new_trace_id, trace_id_var
from kielikaveri.reader.auth import AuthError, user_id_from_header
from kielikaveri.reader.lookup import (
    KeyedLocks,
    LookupFailedError,
    NoSuchWordError,
    Target,
    install_log_redaction,
    lookup,
    resolve_target,
)
from kielikaveri.reader.speech import EmptySpeechError, speak
from kielikaveri.translate.fst_check import split_words
from kielikaveri.translate.worker import Translator
from kielikaveri.tts import make_tts_client

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

SESSION_FACTORY = web.AppKey("session_factory", async_sessionmaker[AsyncSession])
SETTINGS = web.AppKey("settings", Settings)
TRANSLATOR = web.AppKey("translator", Translator)
# The reader's own breaker (plan 2026-11-20-kirjakaveri-reader, И6): word
# lookup and TTS at reading pace must not trip the bot's shared one.
READER_BREAKER = web.AppKey("reader_breaker", CallBreaker)
# The routes anyone may fetch without initData (the page, its static
# files and /favicon.ico). Matched by the resource routing chose, not by the path string, so
# no spelling of a path (dot segments, percent-encoding) can reach an /api
# handler as "public".
PUBLIC_RESOURCES = web.AppKey("public_resources", frozenset)
# OpenAI clients of the reader, built in make_app from settings.
LOOKUP_CLIENT = web.AppKey("lookup_client", AsyncOpenAI)
TTS_CLIENT = web.AppKey("tts_client", AsyncOpenAI)
# Double-tap guards (reader/lookup.py KeyedLocks), one per endpoint.
LOOKUP_LOCKS = web.AppKey("lookup_locks", KeyedLocks)
TTS_LOCKS = web.AppKey("tts_locks", KeyedLocks)
USER_ID = web.RequestKey("user_id", int)

# Only what the page loads: its own files, Telegram's script and its own
# /api, and the 🔊 mp3: fetched with the Authorization header (an <audio
# src> can't send it) and played from a blob: URL. 'unsafe-inline' in style-src is for telegram-web-app.js: inside
# web.telegram.org's iframe it fills a <style> element with the theme CSS
# the client posts to it (event "set_custom_style"). No frame-ancestors:
# web.telegram.org embeds the app in an iframe, and the full list of
# Telegram's embedding origins is not documented.
CSP_DIRECTIVES = (
    "default-src 'none'",
    "script-src 'self' https://telegram.org",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self'",
    "connect-src 'self'",
    "media-src blob:",
    "base-uri 'none'",
    "form-action 'none'",
)
CONTENT_SECURITY_POLICY = "; ".join(CSP_DIRECTIVES)
# Replaces aiohttp's "Python/3.x aiohttp/3.x": no versions to look up CVEs by.
SERVER_HEADER = "kirjakaveri"
# The lookup and TTS bodies are three short fields; the rest of /api is GET.
CLIENT_MAX_SIZE = 16 * 1024
# A reader tap is a person waiting: one SDK retry, not ingest's three.
LOOKUP_MAX_RETRIES = 1

Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]


@web.middleware
async def log_requests(request: web.Request, handler: Handler) -> web.StreamResponse:
    # Instead of aiohttp's access log (disabled in bot/main.py): one line in
    # the app's own format, with the trace id the rest of the request logs.
    trace_id_var.set(new_trace_id())
    started = time.monotonic()
    status = 500
    try:
        response = await handler(request)
        status = response.status
        return response
    except web.HTTPException as exc:
        status = exc.status
        raise
    finally:
        logger.info(
            "event=reader.request method=%s path=%s status=%d user_id=%s ms=%d",
            request.method,
            request.path,
            status,
            request.get(USER_ID, "-"),
            (time.monotonic() - started) * 1000,
        )


@web.middleware
async def check_init_data(request: web.Request, handler: Handler) -> web.StreamResponse:
    # Runs after routing: an unknown path is refused as 401 before it can
    # answer 404, so the set of routes is not open to probing.
    if _is_public(request):
        return await handler(request)
    settings = request.app[SETTINGS]
    try:
        request[USER_ID] = user_id_from_header(
            request.headers.get("Authorization"),
            settings.bot_token,
            settings.whitelist,
            timedelta(hours=settings.reader_init_data_max_age_hours),
        )
    except AuthError as exc:
        logger.warning("event=reader.auth_failed reason=%s", exc)
        raise web.HTTPUnauthorized() from None
    return await handler(request)


def _is_public(request: web.Request) -> bool:
    return request.match_info.route.resource in request.app[PUBLIC_RESOURCES]


async def set_headers(request: web.Request, response: web.StreamResponse) -> None:
    """on_response_prepare: every response, error pages included."""
    if _is_public(request):
        # Telegram's WebView caches hard: without revalidation a deploy's
        # new app.js may never reach the phone.
        response.headers["Cache-Control"] = "no-cache"
    else:
        # /api answers carry the user's books: no cache keeps a copy.
        response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
    response.headers["Server"] = SERVER_HEADER


async def index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "index.html")


async def favicon(request: web.Request) -> web.FileResponse:
    # Browsers ask for /favicon.ico on their own, without initData: answered
    # here so it is not a 401 and a reader.auth_failed warning on every open.
    return web.FileResponse(STATIC_DIR / "favicon.svg", headers={"Content-Type": "image/svg+xml"})


async def list_books(request: web.Request) -> web.Response:
    """The user's books with their chapters' statuses - no text: the list
    must not load every chapter's paragraphs."""
    user_id = request[USER_ID]
    async with request.app[SESSION_FACTORY]() as session:
        books = (
            await session.scalars(
                select(Book).where(Book.user_id == user_id).order_by(Book.created_at.desc())
            )
        ).all()
        rows = (
            await session.execute(
                select(
                    BookChapter.book_id,
                    BookChapter.id,
                    BookChapter.number,
                    BookChapter.part,
                    BookChapter.title,
                    BookChapter.status,
                )
                .where(BookChapter.book_id.in_([b.id for b in books]))
                .order_by(BookChapter.number)
            )
        ).all()
    chapters: dict[str, list[dict]] = {b.id: [] for b in books}
    for row in rows:
        chapters[row.book_id].append(
            {
                "id": row.id,
                "number": row.number,
                "part": row.part,
                "title": row.title,
                "status": row.status.value,
            }
        )
    return web.json_response(
        {"books": [{"id": b.id, "title": b.title, "chapters": chapters[b.id]} for b in books]}
    )


def _neighbour(chapter: BookChapter | None) -> dict | None:
    if chapter is None:
        return None
    return {"id": chapter.id, "number": chapter.number, "status": chapter.status.value}


async def user_chapter(
    session: AsyncSession, chapter_id: str, user_id: int
) -> tuple[BookChapter, Book]:
    """The user's translated chapter and its book. Someone else's chapter
    looks exactly like a missing one (404); one not translated yet is 409."""
    chapter = await session.get(BookChapter, chapter_id)
    book = await session.get(Book, chapter.book_id) if chapter else None
    if book is None or book.user_id != user_id:
        raise web.HTTPNotFound()
    if chapter.status is not ChapterStatus.done:
        raise web.HTTPConflict()
    return chapter, book


async def get_chapter(request: web.Request) -> web.Response:
    """One translated chapter, each paragraph split as split_words does.
    Opening it is the reader's signal to the translator: chapter N+1 is
    queued and the translator woken (db.books.mark_opened)."""
    chapter_id = request.match_info["chapter_id"]
    async with request.app[SESSION_FACTORY]() as session:
        chapter, book = await user_chapter(session, chapter_id, request[USER_ID])
        queued = await mark_opened(session, chapter)
        await session.commit()
        previous = await chapter_by_number(session, book.id, chapter.number - 1)
        following = await chapter_by_number(session, book.id, chapter.number + 1)
        body = {
            "id": chapter.id,
            "number": chapter.number,
            "part": chapter.part,
            "title": chapter.title,
            "book": {"id": book.id, "title": book.title},
            "paragraphs": [split_words(p) for p in chapter.fi_paragraphs or []],
            "prev": _neighbour(previous),
            "next": _neighbour(following),
        }
    if queued is not None:
        logger.info("event=reader.next_queued book_id=%s chapter=%d", book.id, queued.number)
        request.app[TRANSLATOR].wake()
    return web.json_response(body)


def _is_index(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


async def _word_target(request: web.Request) -> Target:
    """The body's word, checked: 400 for a bad body, 404/409 for the
    chapter (user_chapter), 422 for a paragraph or word that isn't there."""
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise web.HTTPBadRequest() from None
    if (
        not isinstance(body, dict)
        or not isinstance(body.get("chapter_id"), str)
        or not _is_index(body.get("paragraph"))
        or not _is_index(body.get("token"))
    ):
        raise web.HTTPBadRequest()
    async with request.app[SESSION_FACTORY]() as session:
        chapter, _book = await user_chapter(session, body["chapter_id"], request[USER_ID])
        try:
            return resolve_target(chapter, body["paragraph"], body["token"])
        except NoSuchWordError:
            raise web.HTTPUnprocessableEntity() from None


def _call_failed(event: str, target: Target, exc: Exception) -> web.Response:
    """503 with the breaker's reason, 502 for OpenAI. Only the class is
    logged: an API error's message can quote the request."""
    if isinstance(exc, CircuitOpenError):
        reason = "daily" if isinstance(exc, BudgetExceededError) else "rate"
        logger.warning(
            "event=%s chapter_id=%s paragraph=%d token=%d reason=%s",
            event,
            target.chapter_id,
            target.paragraph,
            target.token,
            reason,
        )
        return web.json_response({"reason": reason}, status=503)
    logger.warning(
        "event=%s chapter_id=%s paragraph=%d token=%d error=%s",
        event,
        target.chapter_id,
        target.paragraph,
        target.token,
        type(exc).__name__,
    )
    return web.json_response({"reason": "openai"}, status=502)


async def lookup_word(request: web.Request) -> web.Response:
    target = await _word_target(request)
    app = request.app
    try:
        body = await lookup(
            app[SESSION_FACTORY],
            app[LOOKUP_CLIENT],
            app[READER_BREAKER],
            app[SETTINGS].openai_text_model,
            app[LOOKUP_LOCKS],
            target,
        )
    except (CircuitOpenError, LookupFailedError, OpenAIError) as exc:
        return _call_failed("reader.lookup_failed", target, exc)
    return web.json_response(body)


async def speak_word(request: web.Request) -> web.Response:
    target = await _word_target(request)
    app = request.app
    settings = app[SETTINGS]
    try:
        audio = await speak(
            app[TTS_CLIENT],
            app[READER_BREAKER],
            settings.reader_tts_model,
            settings.reader_tts_speed,
            Path(settings.reader_tts_cache_dir),
            app[TTS_LOCKS],
            target,
        )
    except (CircuitOpenError, OpenAIError, EmptySpeechError) as exc:
        return _call_failed("reader.tts_failed", target, exc)
    return web.Response(body=audio, content_type="audio/mpeg")


async def _close_clients(app: web.Application) -> None:
    await app[LOOKUP_CLIENT].close()
    await app[TTS_CLIENT].close()


def make_app(
    session_factory: async_sessionmaker[AsyncSession], settings: Settings, translator: Translator
) -> web.Application:
    # Every initData check would fail anyway, but a misconfigured process
    # should not start at all: fail loudly here, not with a 401 per request.
    if not settings.bot_token:
        raise ValueError("BOT_TOKEN is empty: the reader cannot check initData")
    app = web.Application(
        middlewares=[log_requests, check_init_data], client_max_size=CLIENT_MAX_SIZE
    )
    app.on_response_prepare.append(set_headers)
    app[SESSION_FACTORY] = session_factory
    app[SETTINGS] = settings
    app[TRANSLATOR] = translator
    app[READER_BREAKER] = CallBreaker(
        settings.reader_breaker_max_calls,
        timedelta(minutes=settings.reader_breaker_window_minutes),
        daily_max_calls=settings.reader_daily_max_calls,
    )
    app[LOOKUP_CLIENT] = make_client(
        settings.openai_api_key, settings.openai_timeout_seconds, LOOKUP_MAX_RETRIES
    )
    app[TTS_CLIENT] = make_tts_client(settings.openai_api_key, settings.openai_tts_timeout_seconds)
    app[LOOKUP_LOCKS] = KeyedLocks()
    app[TTS_LOCKS] = KeyedLocks()
    app.on_cleanup.append(_close_clients)
    install_log_redaction()
    index_route = app.router.add_get("/", index)
    favicon_route = app.router.add_get("/favicon.ico", favicon)
    app.router.add_get("/api/books", list_books)
    app.router.add_get("/api/chapters/{chapter_id}", get_chapter)
    app.router.add_post("/api/lookup", lookup_word)
    app.router.add_post("/api/tts", speak_word)
    static = app.router.add_static("/static/", STATIC_DIR)
    app[PUBLIC_RESOURCES] = frozenset({index_route.resource, favicon_route.resource, static})
    return app
