"""The reader's web side (plan 2026-11-20-kirjakaveri-reader, "П2"): an
aiohttp app inside the bot's process that serves the Mini App's static files
and its /api. Caddy terminates HTTPS in front of it; it listens on localhost.

Every /api request is checked against initData and the whitelist
(reader/auth.py). The static files are public - they are code, the book
only ever comes through /api.

Logs carry ids, statuses and chapter numbers - never book text or initData.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from datetime import timedelta
from pathlib import Path

from aiohttp import web
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kielikaveri.config import Settings
from kielikaveri.db.books import chapter_by_number, mark_opened
from kielikaveri.db.models import Book, BookChapter, ChapterStatus
from kielikaveri.llm.breaker import CallBreaker
from kielikaveri.logging_context import new_trace_id, trace_id_var
from kielikaveri.reader.auth import AuthError, user_id_from_header
from kielikaveri.translate.fst_check import split_words
from kielikaveri.translate.worker import Translator

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

SESSION_FACTORY = web.AppKey("session_factory", async_sessionmaker[AsyncSession])
SETTINGS = web.AppKey("settings", Settings)
TRANSLATOR = web.AppKey("translator", Translator)
# The reader's own breaker (plan 2026-11-20-kirjakaveri-reader, И6): word
# lookup and TTS at reading pace must not trip the bot's shared one.
READER_BREAKER = web.AppKey("reader_breaker", CallBreaker)
USER_ID = web.RequestKey("user_id", int)

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
    if not request.path.startswith("/api/"):
        response = await handler(request)
        # Telegram's WebView caches hard: without revalidation a deploy's
        # new app.js may never reach the phone.
        response.headers["Cache-Control"] = "no-cache"
        return response
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


async def index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "index.html")


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


async def get_chapter(request: web.Request) -> web.Response:
    """One translated chapter, each paragraph split as split_words does.
    Opening it is the reader's signal to the translator: chapter N+1 is
    queued and the translator woken (db.books.mark_opened)."""
    chapter_id = request.match_info["chapter_id"]
    async with request.app[SESSION_FACTORY]() as session:
        chapter = await session.get(BookChapter, chapter_id)
        book = await session.get(Book, chapter.book_id) if chapter else None
        # Someone else's chapter looks exactly like a missing one.
        if book is None or book.user_id != request[USER_ID]:
            raise web.HTTPNotFound()
        if chapter.status is not ChapterStatus.done:
            raise web.HTTPConflict()
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


def make_app(
    session_factory: async_sessionmaker[AsyncSession], settings: Settings, translator: Translator
) -> web.Application:
    app = web.Application(middlewares=[log_requests, check_init_data])
    app[SESSION_FACTORY] = session_factory
    app[SETTINGS] = settings
    app[TRANSLATOR] = translator
    app[READER_BREAKER] = CallBreaker(
        settings.reader_breaker_max_calls,
        timedelta(minutes=settings.reader_breaker_window_minutes),
    )
    app.router.add_get("/", index)
    app.router.add_get("/api/books", list_books)
    app.router.add_get("/api/chapters/{chapter_id}", get_chapter)
    app.router.add_static("/static/", STATIC_DIR)
    return app
