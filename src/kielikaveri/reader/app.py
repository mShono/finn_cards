"""The reader's web side (plan 2026-11-20-kirjakaveri-reader, "П2"): an
aiohttp app inside the bot's process that serves the Mini App's static files
and its /api. Caddy terminates HTTPS in front of it; it listens on localhost.

Every request is checked against initData and the whitelist
(reader/auth.py), except the few public routes: the page itself, its
static files and /favicon.ico - they are code, the book only ever comes through /api.

Logs carry ids, statuses and chapter numbers - never book text or initData.

П3 adds the word lookup and 🔊 (reader/lookup.py, reader/speech.py): POST
/api/lookup and /api/tts, both with {"chapter_id", "paragraph", "token"}.

П4 adds, with the same body: POST /api/lemma (the FST's lemmas at once,
before the lookup's answer comes; no OpenAI) and POST /api/cards (➕,
reader/cards.py); /api/lookup says whether the word is already in the
chapter's deck (in_deck). Opening a chapter and /api/lemma warm the OpenAI
connection up in the background (reader/warmup.py).

П5 adds 📚 learning under /api/learn/ (reader/learn.py): decks, start,
debt, review, undo and tts - the session's queue on the server.

П6 adds 🗂 decks (reader/decks.py): /api/decks - the list, a new deck,
renaming and deleting one, its words page by page - and /api/notes - the
search, a word's screen, its edits, moving it, deleting it and its 🔊.
"""

from __future__ import annotations

import asyncio
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

from finn_cards.morphology import lemmatize
from kielikaveri.cards_save import SaveStatus
from kielikaveri.config import Settings
from kielikaveri.db.books import chapter_by_number, mark_opened
from kielikaveri.db.models import Book, BookChapter, ChapterStatus
from kielikaveri.learn_core import LearnSide
from kielikaveri.llm.breaker import BudgetExceededError, CallBreaker, CircuitOpenError
from kielikaveri.llm.client import make_client
from kielikaveri.logging_context import new_trace_id, trace_id_var
from kielikaveri.reader import decks, learn
from kielikaveri.reader.auth import AuthError, user_id_from_header
from kielikaveri.reader.cards import add_card, in_chapter_deck
from kielikaveri.reader.lookup import (
    KeyedLocks,
    LookupFailedError,
    NoSuchWordError,
    Target,
    _book_text_in_flight,
    install_log_redaction,
    lookup,
    resolve_target,
)
from kielikaveri.reader.speech import EmptySpeechError, speak, speak_text
from kielikaveri.reader.warmup import Warmup
from kielikaveri.srs.scheduler import Rating
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
CARD_LOCKS = web.AppKey("card_locks", KeyedLocks)
# 📚 learning (reader/learn.py): one session per user, and the user's lock
# every learn call takes.
LEARN_SESSIONS = web.AppKey("learn_sessions", dict)
LEARN_LOCKS = web.AppKey("learn_locks", KeyedLocks)
# 🗂 decks (reader/decks.py): one lock per note and per deck, keyed apart.
DECK_LOCKS = web.AppKey("deck_locks", KeyedLocks)
# The time of the reader's OpenAI calls and the connection's warm-up.
WARMUP = web.AppKey("warmup", Warmup)
USER_ID = web.RequestKey("user_id", int)

# Only what the page loads: its own files, Telegram's script and its own
# /api, and the 🔊 WAV: fetched with the Authorization header (an <audio
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
# The lookup, TTS and learn bodies are a few short fields; the rest of /api is GET.
CLIENT_MAX_SIZE = 16 * 1024
# The OpenAI clients' key when OPENAI_API_KEY is empty (see make_app).
NO_API_KEY = "unset"
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
    _warm_up(request.app)
    return web.json_response(body)


def _no_openai(app: web.Application) -> web.Response | None:
    """503 {"reason": "no_openai"} when OPENAI_API_KEY is empty: the
    clients hold a stand-in key (make_app), a call would only fail as a 401
    and spend the reader's breaker."""
    if app[SETTINGS].openai_api_key:
        return None
    return web.json_response({"reason": "no_openai"}, status=503)


def _warm_up(app: web.Application) -> None:
    if not app[SETTINGS].openai_api_key:
        return
    app[WARMUP].maybe_start(app[LOOKUP_CLIENT], app[SETTINGS].openai_text_model)


def _is_index(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


async def _word_target(request: web.Request) -> Target:
    """The body's word, checked: 400 for a bad body, 404/409 for the
    chapter (user_chapter), 422 for a paragraph or word that isn't there."""
    target, _chapter, _book = await _word_in_chapter(request)
    return target


async def _word_in_chapter(request: web.Request) -> tuple[Target, BookChapter, Book]:
    """_word_target with the chapter and its book."""
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
        chapter, book = await user_chapter(session, body["chapter_id"], request[USER_ID])
        try:
            target = resolve_target(chapter, body["paragraph"], body["token"])
        except NoSuchWordError:
            raise web.HTTPUnprocessableEntity() from None
    return target, chapter, book


def _target_ids(target: Target) -> str:
    return f"chapter_id={target.chapter_id} paragraph={target.paragraph} token={target.token}"


def _call_failed(event: str, target: Target | str, exc: Exception) -> web.Response:
    """503 with the breaker's reason, 502 for OpenAI. Only the class is
    logged: an API error's message can quote the request. `target` - the
    word, or the log's ids ("key=value ...") of whatever else was asked."""
    ids = target if isinstance(target, str) else _target_ids(target)
    if isinstance(exc, CircuitOpenError):
        reason = "daily" if isinstance(exc, BudgetExceededError) else "rate"
        logger.warning("event=%s %s reason=%s", event, ids, reason)
        return web.json_response({"reason": reason}, status=503)
    logger.warning("event=%s %s error=%s", event, ids, type(exc).__name__)
    return web.json_response({"reason": "openai"}, status=502)


async def lookup_word(request: web.Request) -> web.Response:
    target = await _word_target(request)
    app = request.app
    if (refused := _no_openai(app)) is not None:
        return refused
    try:
        body = await lookup(
            app[SESSION_FACTORY],
            app[LOOKUP_CLIENT],
            app[READER_BREAKER],
            app[SETTINGS].openai_text_model,
            app[LOOKUP_LOCKS],
            target,
            app[WARMUP],
        )
    except (CircuitOpenError, LookupFailedError, OpenAIError) as exc:
        return _call_failed("reader.lookup_failed", target, exc)
    # Asked every time, cached answer or not: ➕ changes it.
    body["in_deck"] = await in_chapter_deck(
        app[SESSION_FACTORY], target.chapter_id, body["lemma"], body["pos"]
    )
    return web.json_response(body)


async def lemma_word(request: web.Request) -> web.Response:
    """The FST's lemmas of the tapped word, in milliseconds: the popup
    shows them while the lookup is on its way. No OpenAI, no breaker."""
    target = await _word_target(request)
    started = time.monotonic()
    # Nothing on this path logs the word today; should morphology ever
    # start to, its lines are cut like the lookup's (the thread gets a copy
    # of this context).
    flag = _book_text_in_flight.set(True)
    try:
        lemmas = await asyncio.to_thread(lemmatize, target.surface)
    finally:
        _book_text_in_flight.reset(flag)
    logger.info(
        "event=reader.lemma chapter_id=%s paragraph=%d token=%d count=%d duration_ms=%d",
        target.chapter_id,
        target.paragraph,
        target.token,
        len(lemmas),
        (time.monotonic() - started) * 1000,
    )
    _warm_up(request.app)
    return web.json_response(
        {"surface": target.surface, "lemmas": lemmas, "fst_flagged": target.fst_flagged}
    )


async def save_word(request: web.Request) -> web.Response:
    """➕: {"status": "saved" | "duplicate", "deck": name}; 422
    {"reason": "save_failed"} when the card can't be built."""
    target, chapter, book = await _word_in_chapter(request)
    app = request.app
    if (refused := _no_openai(app)) is not None:
        return refused
    try:
        result = await add_card(
            app[SESSION_FACTORY],
            app[LOOKUP_CLIENT],
            app[READER_BREAKER],
            app[SETTINGS].openai_text_model,
            app[LOOKUP_LOCKS],
            app[CARD_LOCKS],
            app[DECK_LOCKS],
            app[WARMUP],
            request[USER_ID],
            chapter,
            book,
            target,
        )
    except (CircuitOpenError, LookupFailedError, OpenAIError) as exc:
        return _call_failed("reader.card_failed", target, exc)
    except Exception as exc:  # noqa: BLE001 - logged by class, then 500
        # Not aiohttp's own handler: its traceback would carry the book's
        # text (an IntegrityError quotes its parameters). The class only.
        logger.error(
            "event=reader.card_failed chapter_id=%s paragraph=%d token=%d error=%s",
            target.chapter_id,
            target.paragraph,
            target.token,
            type(exc).__name__,
        )
        raise web.HTTPInternalServerError() from None
    if result.status is SaveStatus.failed:
        if isinstance(result.error, CircuitOpenError | OpenAIError):
            return _call_failed("reader.card_failed", target, result.error)
        return web.json_response({"reason": "save_failed"}, status=422)
    return web.json_response({"status": result.status.value, "deck": result.deck})


async def speak_word(request: web.Request) -> web.Response:
    target = await _word_target(request)
    app = request.app
    if (refused := _no_openai(app)) is not None:
        return refused
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
    return web.Response(body=audio, content_type="audio/wav")


async def _json_body(request: web.Request) -> dict:
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise web.HTTPBadRequest() from None
    if not isinstance(body, dict):
        raise web.HTTPBadRequest()
    return body


def _string(body: dict, name: str) -> str:
    value = body.get(name)
    if not isinstance(value, str) or not value:
        raise web.HTTPBadRequest()
    return value


def _conflict(reason: str, payload: dict | None = None) -> web.Response:
    return web.json_response({"reason": reason, **(payload or {})}, status=409)


async def _learn_call(call: Awaitable[dict]) -> web.Response:
    """A learn call's answer, its errors as the plan says: 404 for what isn't
    the user's, 409 with a reason for the session's own refusals."""
    try:
        return web.json_response(await call)
    except learn.NotFoundError:
        raise web.HTTPNotFound() from None
    except learn.SessionGoneError:
        return _conflict("session_gone")
    except learn.NothingToUndoError:
        return _conflict("nothing_to_undo")
    except learn.StaleError as exc:
        return _conflict("stale", exc.payload)


async def learn_decks(request: web.Request) -> web.Response:
    return web.json_response(await learn.decks(request.app[SESSION_FACTORY], request[USER_ID]))


async def learn_start(request: web.Request) -> web.Response:
    body = await _json_body(request)
    deck_id = body.get("deck_id")
    side = body.get("side")
    if deck_id is not None and not isinstance(deck_id, str):
        raise web.HTTPBadRequest()
    if not isinstance(side, str) or side not in LearnSide:
        raise web.HTTPBadRequest()
    app = request.app
    return await _learn_call(
        learn.start(
            app[SESSION_FACTORY],
            app[SETTINGS],
            app[LEARN_SESSIONS],
            app[LEARN_LOCKS],
            request[USER_ID],
            deck_id,
            LearnSide(side),
        )
    )


async def learn_debt(request: web.Request) -> web.Response:
    body = await _json_body(request)
    session_id = _string(body, "session")
    action = body.get("action")
    if action not in ("batch", "defer"):
        raise web.HTTPBadRequest()
    app = request.app
    return await _learn_call(
        learn.debt(
            app[SESSION_FACTORY],
            app[SETTINGS],
            app[LEARN_SESSIONS],
            app[LEARN_LOCKS],
            request[USER_ID],
            session_id,
            defer=action == "defer",
        )
    )


async def learn_review(request: web.Request) -> web.Response:
    body = await _json_body(request)
    session_id = _string(body, "session")
    card_id = _string(body, "card_id")
    rating = body.get("rating")
    if not isinstance(rating, int) or isinstance(rating, bool) or not 1 <= rating <= 4:
        raise web.HTTPBadRequest()
    app = request.app
    return await _learn_call(
        learn.review(
            app[SESSION_FACTORY],
            app[SETTINGS],
            app[LEARN_SESSIONS],
            app[LEARN_LOCKS],
            request[USER_ID],
            session_id,
            card_id,
            Rating(rating),
        )
    )


async def learn_undo(request: web.Request) -> web.Response:
    body = await _json_body(request)
    session_id = _string(body, "session")
    app = request.app
    return await _learn_call(
        learn.undo(
            app[SESSION_FACTORY],
            app[LEARN_SESSIONS],
            app[LEARN_LOCKS],
            request[USER_ID],
            session_id,
        )
    )


async def learn_tts(request: web.Request) -> web.Response:
    """🔊 of the card's example_fi: WAV, the reader's cache and breaker,
    the errors of /api/tts; 503 {"reason": "no_openai"} without a key."""
    body = await _json_body(request)
    session_id = _string(body, "session")
    card_id = _string(body, "card_id")
    app = request.app
    settings = app[SETTINGS]
    try:
        text = await learn.example_to_speak(
            app[SESSION_FACTORY],
            app[LEARN_SESSIONS],
            app[LEARN_LOCKS],
            request[USER_ID],
            session_id,
            card_id,
        )
    except learn.NotFoundError:
        raise web.HTTPNotFound() from None
    except learn.SessionGoneError:
        return _conflict("session_gone")
    if (refused := _no_openai(app)) is not None:
        return refused
    ids = f"card_id={card_id} source=learn"
    try:
        audio = await speak_text(
            app[TTS_CLIENT],
            app[READER_BREAKER],
            settings.reader_tts_model,
            settings.reader_tts_speed,
            Path(settings.reader_tts_cache_dir),
            app[TTS_LOCKS],
            text,
            ids,
        )
    except (CircuitOpenError, OpenAIError, EmptySpeechError) as exc:
        return _call_failed("reader.tts_failed", ids, exc)
    return web.Response(body=audio, content_type="audio/wav")


# --- 🗂 decks (reader/decks.py) -------------------------------------------------------


async def _decks_call(event: str, ids: str, call: Awaitable[dict | None]) -> web.Response:
    """A decks call's answer (None - 204), its errors as the plan says: 404
    for what isn't the user's, 409 and 422 with a reason, the lookup's
    503/502 for OpenAI. Anything else is logged by class only - a
    traceback of an IntegrityError would quote the word - and is a 500."""
    try:
        body = await call
    except web.HTTPException:
        raise
    except decks.NotFoundError:
        raise web.HTTPNotFound() from None
    except decks.ConflictError as exc:
        return _conflict(exc.reason, exc.payload)
    except decks.BadNameError:
        return web.json_response({"reason": "bad_name"}, status=422)
    except decks.NoOpenAIError:
        return web.json_response({"reason": "no_openai"}, status=503)
    except (CircuitOpenError, OpenAIError) as exc:
        return _call_failed(event, ids, exc)
    except Exception as exc:  # noqa: BLE001 - logged by class, then 500
        logger.error("event=%s %s error=%s", event, ids, type(exc).__name__)
        raise web.HTTPInternalServerError() from None
    if body is None:
        return web.Response(status=204)
    return web.json_response(body)


def _name(body: dict) -> str:
    """A deck's name as sent: not a string - 400; its length is checked by
    reader/decks.py (422 bad_name)."""
    name = body.get("name")
    if not isinstance(name, str):
        raise web.HTTPBadRequest()
    return name


def _query_int(request: web.Request, name: str, default: int) -> int:
    value = request.query.get(name)
    if value is None:
        return default
    if not value.isascii() or not value.isdigit():
        raise web.HTTPBadRequest()
    return int(value)


async def decks_list(request: web.Request) -> web.Response:
    return await _decks_call(
        "decks.failed",
        "op=list source=reader",
        decks.list_decks(request.app[SESSION_FACTORY], request[USER_ID]),
    )


async def decks_create(request: web.Request) -> web.Response:
    name = _name(await _json_body(request))
    return await _decks_call(
        "decks.failed",
        "op=create source=reader",
        decks.create_deck(request.app[SESSION_FACTORY], request[USER_ID], name),
    )


async def decks_rename(request: web.Request) -> web.Response:
    deck_id = request.match_info["deck_id"]
    name = _name(await _json_body(request))
    app = request.app
    return await _decks_call(
        "decks.failed",
        f"op=rename deck_id={deck_id} source=reader",
        decks.rename_deck(app[SESSION_FACTORY], app[DECK_LOCKS], request[USER_ID], deck_id, name),
    )


async def decks_delete(request: web.Request) -> web.Response:
    deck_id = request.match_info["deck_id"]
    app = request.app
    return await _decks_call(
        "decks.failed",
        f"op=delete deck_id={deck_id} source=reader",
        decks.delete_deck(app[SESSION_FACTORY], app[DECK_LOCKS], request[USER_ID], deck_id),
    )


async def decks_notes(request: web.Request) -> web.Response:
    """A page of the deck's words, newest first: ?offset= (0) &limit= (50,
    at most 100; 0 or not a number - 400)."""
    deck_id = request.match_info["deck_id"]
    offset = _query_int(request, "offset", 0)
    limit = _query_int(request, "limit", decks.DEFAULT_PAGE)
    if limit < 1:
        raise web.HTTPBadRequest()
    limit = min(limit, decks.MAX_PAGE)
    return await _decks_call(
        "decks.failed",
        f"op=notes deck_id={deck_id} source=reader",
        decks.deck_notes(request.app[SESSION_FACTORY], request[USER_ID], deck_id, offset, limit),
    )


async def notes_search(request: web.Request) -> web.Response:
    query = request.query.get("q", "")
    return await _decks_call(
        "decks.failed",
        "op=search source=reader",
        decks.search(request.app[SESSION_FACTORY], request[USER_ID], query),
    )


async def notes_get(request: web.Request) -> web.Response:
    note_id = request.match_info["note_id"]
    return await _decks_call(
        "decks.failed",
        f"op=note note_id={note_id} source=reader",
        decks.note(request.app[SESSION_FACTORY], request[USER_ID], note_id),
    )


async def notes_edit(request: web.Request) -> web.Response:
    """{"translation_ru"} or {"lemma"} - one of the two, a non-empty string
    once stripped (400 otherwise)."""
    body = await _json_body(request)
    fields = [name for name in ("lemma", "translation_ru") if name in body]
    if len(fields) != 1:
        raise web.HTTPBadRequest()
    field = fields[0]
    value = body[field]
    if not isinstance(value, str) or not value.strip():
        raise web.HTTPBadRequest()
    value = value.strip()
    note_id = request.match_info["note_id"]
    app = request.app
    ids = f"op=edit note_id={note_id} field={field} source=reader"
    if field == "translation_ru":
        call = decks.set_translation(
            app[SESSION_FACTORY], app[DECK_LOCKS], request[USER_ID], note_id, value
        )
        return await _decks_call("decks.failed", ids, call)
    call = decks.set_lemma(
        app[SESSION_FACTORY],
        app[LOOKUP_CLIENT],
        app[READER_BREAKER],
        app[SETTINGS].openai_text_model,
        bool(app[SETTINGS].openai_api_key),
        app[DECK_LOCKS],
        request[USER_ID],
        note_id,
        value,
    )
    return await _decks_call("decks.lemma_failed", ids, call)


async def notes_delete(request: web.Request) -> web.Response:
    """204; a second time (a double tap) - 404."""
    note_id = request.match_info["note_id"]
    app = request.app
    return await _decks_call(
        "decks.failed",
        f"op=delete_note note_id={note_id} source=reader",
        decks.delete_note(app[SESSION_FACTORY], app[DECK_LOCKS], request[USER_ID], note_id),
    )


async def notes_move(request: web.Request) -> web.Response:
    deck_id = _string(await _json_body(request), "deck_id")
    note_id = request.match_info["note_id"]
    app = request.app
    return await _decks_call(
        "decks.failed",
        f"op=move note_id={note_id} deck_id={deck_id} source=reader",
        decks.move_note(app[SESSION_FACTORY], app[DECK_LOCKS], request[USER_ID], note_id, deck_id),
    )


async def notes_tts(request: web.Request) -> web.Response:
    """🔊 of the word's lemma or example_fi ({"what": "lemma" | "example"}):
    WAV, the reader's cache and breaker, the errors of /api/learn/tts."""
    body = await _json_body(request)
    what = body.get("what")
    if what not in ("lemma", "example"):
        raise web.HTTPBadRequest()
    note_id = request.match_info["note_id"]
    app = request.app
    settings = app[SETTINGS]
    try:
        text = await decks.text_to_speak(app[SESSION_FACTORY], request[USER_ID], note_id, what)
    except decks.NotFoundError:
        raise web.HTTPNotFound() from None
    if (refused := _no_openai(app)) is not None:
        return refused
    ids = f"note_id={note_id} what={what} source=decks"
    try:
        audio = await speak_text(
            app[TTS_CLIENT],
            app[READER_BREAKER],
            settings.reader_tts_model,
            settings.reader_tts_speed,
            Path(settings.reader_tts_cache_dir),
            app[TTS_LOCKS],
            text,
            ids,
        )
    except (CircuitOpenError, OpenAIError, EmptySpeechError) as exc:
        return _call_failed("reader.tts_failed", ids, exc)
    return web.Response(body=audio, content_type="audio/wav")


async def _stop_warmup(app: web.Application) -> None:
    await app[WARMUP].stop()


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
    # The SDK refuses to build a client without a key, and learning must
    # still work then (П5): the clients get a stand-in nobody calls with -
    # lookup, ➕ and both TTS answer 503 no_openai, the warm-up is skipped.
    api_key = settings.openai_api_key or NO_API_KEY
    app[LOOKUP_CLIENT] = make_client(api_key, settings.openai_timeout_seconds, LOOKUP_MAX_RETRIES)
    app[TTS_CLIENT] = make_tts_client(api_key, settings.openai_tts_timeout_seconds)
    app[LOOKUP_LOCKS] = KeyedLocks()
    app[TTS_LOCKS] = KeyedLocks()
    app[CARD_LOCKS] = KeyedLocks()
    app[LEARN_SESSIONS] = {}
    app[LEARN_LOCKS] = KeyedLocks()
    app[DECK_LOCKS] = KeyedLocks()
    app[WARMUP] = Warmup()
    app.on_shutdown.append(_stop_warmup)
    app.on_cleanup.append(_close_clients)
    install_log_redaction()
    index_route = app.router.add_get("/", index)
    favicon_route = app.router.add_get("/favicon.ico", favicon)
    app.router.add_get("/api/books", list_books)
    app.router.add_get("/api/chapters/{chapter_id}", get_chapter)
    app.router.add_post("/api/lookup", lookup_word)
    app.router.add_post("/api/tts", speak_word)
    app.router.add_post("/api/lemma", lemma_word)
    app.router.add_post("/api/cards", save_word)
    app.router.add_get("/api/learn/decks", learn_decks)
    app.router.add_post("/api/learn/start", learn_start)
    app.router.add_post("/api/learn/debt", learn_debt)
    app.router.add_post("/api/learn/review", learn_review)
    app.router.add_post("/api/learn/undo", learn_undo)
    app.router.add_post("/api/learn/tts", learn_tts)
    app.router.add_get("/api/decks", decks_list)
    app.router.add_post("/api/decks", decks_create)
    app.router.add_patch("/api/decks/{deck_id}", decks_rename)
    app.router.add_delete("/api/decks/{deck_id}", decks_delete)
    app.router.add_get("/api/decks/{deck_id}/notes", decks_notes)
    app.router.add_get("/api/notes/search", notes_search)
    app.router.add_get("/api/notes/{note_id}", notes_get)
    app.router.add_patch("/api/notes/{note_id}", notes_edit)
    app.router.add_delete("/api/notes/{note_id}", notes_delete)
    app.router.add_post("/api/notes/{note_id}/move", notes_move)
    app.router.add_post("/api/notes/{note_id}/tts", notes_tts)
    static = app.router.add_static("/static/", STATIC_DIR)
    app[PUBLIC_RESOURCES] = frozenset({index_route.resource, favicon_route.resource, static})
    return app
