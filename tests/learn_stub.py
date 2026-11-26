"""A stand-in server for checking the learning screens (plan
2026-11-20-kirjakaveri-reader, "П5", stage У2) by hand, without the bot, the
database or Telegram: the real static files, under the real CSP, plus fake
/api/learn/* answers in the contract's shapes.

    uv run python tests/learn_stub.py [--port 8099] [--decks 2] [--debt]
                                      [--empty] [--delay 0.6]

then open http://localhost:8099/?screen=learn (a phone-sized window in the
browser's dev tools gives touch events for swipes). The page loads a fake
telegram-web-app.js: initData is "stub", the BackButton is a floating
"← Back" button, haptics go to the console. Every API call is printed, so a
fast double swipe can be seen to send one review. GET /stub/drop forgets the
session: the next rating answers 409 session_gone. /api/learn/review waits
--delay seconds, so the in-flight guard has time to matter.

Also used by tests/test_reader_static_learn.py, which keeps it in step with
the contract.
"""

import argparse
import asyncio
import io
import json
import math
import struct
import uuid
import wave
from dataclasses import dataclass, field

from aiohttp import web

from kielikaveri.reader.app import CONTENT_SECURITY_POLICY, STATIC_DIR

TELEGRAM_SCRIPT = "https://telegram.org/js/telegram-web-app.js"
STUB_SCRIPT = "stub/telegram-web-app.js"

# A fake Telegram.WebApp: just what app.js calls.
FAKE_TELEGRAM_JS = """\
"use strict";
(function () {
  let handlers = [];
  let button = null;
  function backButton() {
    if (!button) {
      button = document.createElement("button");
      button.type = "button";
      button.textContent = "← Back";
      button.className = "stub-back";
      button.style.position = "fixed";
      button.style.top = "4px";
      button.style.left = "4px";
      button.style.width = "auto";
      button.style.zIndex = "100";
      button.style.padding = "4px 10px";
      button.addEventListener("click", () => handlers.slice().forEach((fn) => fn()));
      document.body.append(button);
    }
    return button;
  }
  window.Telegram = {
    WebApp: {
      initData: "stub",
      ready() {},
      expand() {},
      BackButton: {
        onClick(fn) { handlers.push(fn); },
        offClick(fn) { handlers = handlers.filter((h) => h !== fn); },
        show() { backButton().hidden = false; },
        hide() { if (button) button.hidden = true; },
      },
      HapticFeedback: {
        impactOccurred(style) { console.log("haptic", style); },
      },
    },
  };
})();
"""

CARDS = [
    {
        "id": "c-talo",
        "type": "recognition",
        "lemma": "talo",
        "translation_ru": "дом",
        "example_fi": "Hän lähti talosta aikaisin aamulla.",
        "example_ru": "Он ушёл из дома рано утром.",
    },
    {
        "id": "c-nahda",
        "type": "production",
        "lemma": "nähdä",
        "translation_ru": "видеть",
        "example_fi": "Kukaan ei nähnyt, minne hän meni.",
        "example_ru": "Никто не видел, куда он пошёл.",
    },
    {
        "id": "c-talo-gen",
        "type": "inflection",
        "lemma": "talo",
        "translation_ru": "дом",
        "example_fi": "Talon ovi oli auki.",
        "example_ru": None,
        "cue": "kenen? minkä?",
        "form_answer": "talon",
        "form_label": "genetiivi - kenen? minkä? (-n)",
    },
    {
        "id": "c-aamu",
        "type": "recognition",
        "lemma": "aamu",
        "translation_ru": "утро",
        "example_fi": "",
        "example_ru": "",
    },
]
DECKS = [{"id": "d-book", "name": "Покаянные дни - гл.1"}, {"id": "d-add", "name": "Мои слова"}]


@dataclass
class StubState:
    decks: int = 2
    debt: bool = False
    empty: bool = False
    delay: float = 0.6
    session: str | None = None
    queue: list[int] = field(default_factory=list)
    done: int = 0
    last: int | None = None
    calls: list[tuple[str, dict]] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


STATE = web.AppKey("stub_state", StubState)


def _conflict(reason: str, **extra) -> web.Response:
    return web.json_response({"reason": reason, **extra}, status=409)


def _card_state(state: StubState, can_undo: bool) -> dict:
    return {
        "state": "card",
        "session": state.session,
        "card": CARDS[state.queue[0]],
        "progress": {"done": state.done, "total": state.done + len(state.queue)},
        "can_undo": can_undo,
    }


def _current(state: StubState) -> dict:
    """What a successful review would answer now."""
    if state.queue:
        return _card_state(state, can_undo=state.last is not None)
    return {
        "state": "end",
        "reviewed": state.done,
        "remaining": 0,
        "text": f"Сессия окончена: {state.done} пройдено.",
        "can_undo": state.last is not None,
    }


async def _body(request: web.Request, state: StubState) -> dict:
    body = await request.json() if request.can_read_body else {}
    state.calls.append((request.path, body))
    print(f"{request.method} {request.path} {json.dumps(body, ensure_ascii=False)}", flush=True)
    return body


@web.middleware
async def stub_headers(request: web.Request, handler) -> web.StreamResponse:
    if request.path.startswith("/api/") and not request.headers.get("Authorization", "").startswith(
        "tma "
    ):
        raise web.HTTPUnauthorized()
    response = await handler(request)
    response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
    response.headers["Cache-Control"] = "no-store"
    return response


async def index(request: web.Request) -> web.Response:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert TELEGRAM_SCRIPT in html
    return web.Response(text=html.replace(TELEGRAM_SCRIPT, STUB_SCRIPT), content_type="text/html")


async def telegram_js(request: web.Request) -> web.Response:
    return web.Response(text=FAKE_TELEGRAM_JS, content_type="application/javascript")


async def drop_session(request: web.Request) -> web.Response:
    request.app[STATE].session = None
    return web.Response(text="session dropped: the next learn call answers session_gone\n")


async def books(request: web.Request) -> web.Response:
    return web.json_response({"books": []})


async def decks(request: web.Request) -> web.Response:
    state = request.app[STATE]
    await _body(request, state)
    return web.json_response({"decks": DECKS[: state.decks]})


async def start(request: web.Request) -> web.Response:
    state = request.app[STATE]
    body = await _body(request, state)
    if body.get("side") not in ("fi", "ru", "mix"):
        return web.json_response({"reason": "bad_request"}, status=400)
    if state.empty:
        return web.json_response({"state": "empty", "text": "Сейчас нечего повторять."})
    state.session = uuid.uuid4().hex
    state.queue = list(range(len(CARDS)))
    state.done = 0
    state.last = None
    if state.debt:
        return web.json_response(
            {
                "state": "debt",
                "session": state.session,
                "overdue": 57,
                "max_cards": 20,
                "postpone_days": 7,
            }
        )
    return web.json_response(_card_state(state, can_undo=False))


async def debt(request: web.Request) -> web.Response:
    state = request.app[STATE]
    body = await _body(request, state)
    if body.get("session") != state.session or state.session is None:
        return _conflict("session_gone")
    if body.get("action") == "defer":
        state.queue = state.queue[:2]
    return web.json_response(_card_state(state, can_undo=False))


async def review(request: web.Request) -> web.Response:
    state = request.app[STATE]
    body = await _body(request, state)
    async with state.lock:
        await asyncio.sleep(state.delay)
        if body.get("session") != state.session or state.session is None:
            return _conflict("session_gone")
        rating = body.get("rating")
        if not isinstance(rating, int) or isinstance(rating, bool) or not 1 <= rating <= 4:
            return web.json_response({"reason": "bad_request"}, status=400)
        if not state.queue or CARDS[state.queue[0]]["id"] != body.get("card_id"):
            return _conflict("stale", **_current(state))
        state.last = state.queue.pop(0)
        state.done += 1
        answer = _current(state)
        answer.pop("session", None)
        return web.json_response(answer)


async def undo(request: web.Request) -> web.Response:
    state = request.app[STATE]
    body = await _body(request, state)
    async with state.lock:
        if body.get("session") != state.session or state.session is None:
            return _conflict("session_gone")
        if state.last is None:
            return _conflict("nothing_to_undo")
        state.queue.insert(0, state.last)
        state.done -= 1
        state.last = None
        answer = _card_state(state, can_undo=False)
        answer.pop("session")
        return web.json_response(answer)


def _beep_wav() -> bytes:
    rate = 16000
    frames = b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * 660 * i / rate)))
        for i in range(rate * 2 // 5)
    )
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(frames)
    return out.getvalue()


async def tts(request: web.Request) -> web.Response:
    state = request.app[STATE]
    body = await _body(request, state)
    if body.get("session") != state.session or state.session is None:
        return _conflict("session_gone")
    return web.Response(body=_beep_wav(), content_type="audio/wav")


def make_stub_app(**options) -> web.Application:
    app = web.Application(middlewares=[stub_headers])
    app[STATE] = StubState(**options)
    app.router.add_get("/", index)
    app.router.add_get("/stub/telegram-web-app.js", telegram_js)
    app.router.add_get("/stub/drop", drop_session)
    app.router.add_get("/api/books", books)
    app.router.add_get("/api/learn/decks", decks)
    app.router.add_post("/api/learn/start", start)
    app.router.add_post("/api/learn/debt", debt)
    app.router.add_post("/api/learn/review", review)
    app.router.add_post("/api/learn/undo", undo)
    app.router.add_post("/api/learn/tts", tts)
    app.router.add_static("/static/", STATIC_DIR)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--decks", type=int, default=2, choices=(0, 1, 2))
    parser.add_argument("--debt", action="store_true", help="start answers the debt prompt")
    parser.add_argument("--empty", action="store_true", help="start answers empty")
    parser.add_argument("--delay", type=float, default=0.6, help="seconds a review takes")
    args = parser.parse_args()
    app = make_stub_app(decks=args.decks, debt=args.debt, empty=args.empty, delay=args.delay)
    print(f"http://localhost:{args.port}/?screen=learn", flush=True)
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
