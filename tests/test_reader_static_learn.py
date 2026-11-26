"""Static checks on the reader's learning screens (plan
2026-11-20-kirjakaveri-reader, "П5", stage У2): the /api/learn/* calls, card
text through textContent only, ?screen=learn, the swipe's thresholds and
ratings, the in-flight guard, the 409 paths. And the test-side stub server
(tests/learn_stub.py) answers in the contract's shapes."""

import re

import pytest
from aiohttp.test_utils import TestClient, TestServer
from learn_stub import CARDS, STUB_SCRIPT, TELEGRAM_SCRIPT, make_stub_app
from test_reader_static import APP_JS, _calls, _routes, learn_merged

from kielikaveri.reader.app import CONTENT_SECURITY_POLICY, STATIC_DIR

STYLE_CSS = (STATIC_DIR / "style.css").read_text(encoding="utf-8")

LEARN_CALLS = {
    "/api/learn/decks": {"GET"},
    "/api/learn/start": {"POST"},
    "/api/learn/debt": {"POST"},
    "/api/learn/review": {"POST"},
    "/api/learn/undo": {"POST"},
    "/api/learn/tts": {"POST"},
}


def _function(name: str) -> str:
    """The body of a top-level `function name(...)` in app.js."""
    start = re.search(rf"^(async )?function {name}\(", APP_JS, re.MULTILINE)
    assert start, name
    return APP_JS[start.start() :].split("\n}\n", 1)[0]


def _const(name: str) -> str:
    match = re.search(rf"^const {name} = ([^;]+);", APP_JS, re.MULTILINE)
    assert match, name
    return match.group(1)


# --- calls and routes ---------------------------------------------------------


def test_learn_calls_and_methods():
    calls = _calls()
    for path, methods in LEARN_CALLS.items():
        assert calls.get(path) == methods, path


def test_learn_calls_are_routes_once_the_server_is_merged():
    routes = _routes()
    if not learn_merged(routes):
        pytest.skip("server routes (learn-api) not merged yet")
    for path, methods in LEARN_CALLS.items():
        assert path in routes, path
        assert methods <= routes[path], (path, routes[path])


def test_request_bodies_follow_the_contract():
    assert "{ deck_id: learn.deckId, side }" in _function("startSession")
    assert "{ session: learn.session, action }" in _function("showDebt")
    assert "{ session: learn.session, card_id: ui.card.id, rating }" in _function("rate")
    assert "{ session: learn.session }" in _function("undoLast")
    assert "{ session: learn.session, card_id: card.id }" in _function("showCard")
    assert '["fi", ' in APP_JS and '["ru", ' in APP_JS and '["mix", ' in APP_JS


# --- no stubs, no HTML sinks --------------------------------------------------


def test_no_api_stubs_in_app_js():
    # Fake answers live in tests/learn_stub.py, not in the page.
    for marker in ("stub", "mock", "fake", "c-talo"):
        assert marker not in APP_JS.lower(), marker
    assert APP_JS.count("fetch(") == 1


def test_card_text_goes_through_text_content_only():
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert sink not in APP_JS, sink
    # el() sets attributes with setAttribute and children with append(),
    # both text-safe; no style attribute (strict CSP for markup).
    assert 'setAttribute("style"' not in APP_JS
    assert "new Function" not in APP_JS
    card = _function("showCard") + _function("cardText")
    for field in (
        "lemma",
        "translation_ru",
        "example_fi",
        "example_ru",
        "cue",
        "form_answer",
        "form_label",
    ):
        assert f"card.{field}" in card, field


def test_empty_examples_are_hidden():
    body = _function("showCard")
    assert "if (card.example_fi) back.append(" in body
    assert "if (card.example_ru) back.append(" in body
    # 🔊 speaks example_fi: no example, no button.
    assert "if (card.example_fi) {" in body


def test_front_face_by_card_type():
    body = _function("cardText")
    assert '"production"' in body and "front: card.translation_ru" in body
    assert '"inflection"' in body and "`${card.lemma} → ${card.cue}`" in body
    assert "answer: card.form_answer" in body and "card.form_label" in body
    # recognition (the fallthrough): 🇫🇮 lemma, answer the translation.
    tail = body.rsplit("return", 1)[1]
    assert "front: card.lemma" in tail and "answer: card.translation_ru" in tail


# --- entry --------------------------------------------------------------------


def test_screen_learn_query_parameter_and_no_hash_writes():
    assert 'new URLSearchParams(window.location.search).get("screen") === "learn"' in APP_JS
    # Telegram keeps tgWebAppData in the hash: never write it.
    assert "location.hash" not in APP_JS
    assert "pushState" not in APP_JS and "replaceState" not in APP_JS


def test_switcher_on_both_root_screens():
    assert '"📖 Книги"' in APP_JS and '"📚 Учить"' in APP_JS
    assert 'switcher("books")' in _function("showBooks")
    assert 'switcher("learn")' in _function("showLearn")


def test_deck_picker_only_with_more_than_one_deck():
    body = _function("showLearn")
    assert "learn.decks.length > 1" in body
    assert "learn.decks.length === 1 ? learn.decks[0].id : null" in body


def test_back_button_steps():
    # Card, debt, end, empty and "session gone" go back to the side choice.
    for name in ("showCard", "showDebt", "showEnd", "showMessage", "showSessionGone"):
        assert "navigate(showSides)" in _function(name), name
    assert "navigate(learn.decks.length > 1 ? showDecks : null)" in _function("showSides")


# --- swipe --------------------------------------------------------------------


def test_swipe_constants_and_ratings():
    assert _const("SWIPE_THRESHOLD") == "0.3"
    assert re.search(r"again: 1, hard: 2, good: 3, easy: 4", _const("RATING"))
    body = _function("swipeRating")
    assert "width * SWIPE_THRESHOLD" in body
    assert "FLICK_PX_PER_MS" in body
    assert "return dx < 0 ? RATING.again : RATING.good;" in body
    assert "return null" in body


def test_buttons_under_the_card():
    body = _function("showCard")
    assert 'rateButton("✗ Забыл", RATING.again' in body
    assert 'rateButton("✓ Помню", RATING.good' in body
    assert 'rateButton("Трудно", RATING.hard' in body
    assert 'rateButton("Легко", RATING.easy' in body
    # Hidden until the flip.
    assert 'class: "rate", hidden: ""' in body
    assert "ui.rateArea.hidden = false" in _function("flip")


def test_swipe_only_after_flip_and_tap_flips():
    body = _function("attachSwipe")
    for event in ("pointerdown", "pointermove", "pointerup", "pointercancel"):
        assert f'"{event}"' in body, event
    assert "!ui.revealed" in body
    assert "flip(ui)" in body
    assert "snapBack(ui)" in body
    assert "rate(ui, rating" in body
    assert "!ui.revealed" in _function("rate")


def test_card_follows_the_finger_and_underlay_colours():
    body = _function("follow")
    assert "translateX(" in body and "rotate(" in body
    assert '"--swipe"' in body
    assert '"to-left"' in body and '"to-right"' in body
    assert "Забыл" in APP_JS and "Помню" in APP_JS
    assert "touch-action: pan-y" in STYLE_CSS
    assert ".to-left .underlay" in STYLE_CSS and ".to-right .underlay" in STYLE_CSS
    assert "var(--swipe)" in STYLE_CSS


def test_flip_animation_respects_reduced_motion():
    assert "rotateY(180deg)" in STYLE_CSS
    reduced = STYLE_CSS.split("prefers-reduced-motion: reduce", 1)[1]
    assert ".flashcard-inner" in reduced and "transition: none" in reduced


def test_keyboard_and_haptics():
    assert '"ArrowLeft"' in APP_JS and '"ArrowRight"' in APP_JS and 'event.key === " "' in APP_JS
    assert "rate(ui, RATING.again, -1)" in APP_JS and "rate(ui, RATING.good, 1)" in APP_JS
    assert "HapticFeedback" in _function("haptic")
    assert "haptic()" in _function("rate")


# --- in-flight guard and 409 --------------------------------------------------


def test_one_rating_request_in_flight():
    body = _function("rate")
    guard = body.index("if (ui.busy")
    busy = body.index("setBusy(ui, true)")
    call = body.index('api("api/learn/review"')
    # The guard and the busy flag come before the request leaves, with no
    # await in between: a second swipe in the same tick already sees busy.
    assert guard < busy < call
    assert "await" not in body
    assert "ui.busy" in _function("attachSwipe")
    assert "button.disabled = busy" in _function("setBusy")
    # Undo shares the card's guard.
    assert "ui.busy" in _function("undoLast")


def test_409_paths():
    assert "this.body = body" in APP_JS
    rate = _function("rate")
    assert 'isConflict(error, "stale") && error.body && error.body.state' in rate
    assert "showState(error.body)" in rate
    assert "isSessionLost(error)" in rate
    gone = _function("showSessionGone")
    assert "SESSION_GONE_TEXT" in gone and '"Начать заново"' in gone
    assert _const("SESSION_GONE_TEXT") == '"Сессия прервалась."'
    undo = _function("undoLast")
    assert 'isConflict(error, "nothing_to_undo")' in undo


def test_undo_on_next_card_and_end_screen():
    for name in ("showCard", "showEnd"):
        body = _function(name)
        assert "if (data.can_undo)" in body and '"↩ Отменить"' in body, name


def test_learn_tts_unlocks_audio_in_the_tap():
    body = _function("showCard")
    assert "playSound(speak, speakError" in body
    assert 'request("api/learn/tts"' in body


# --- the stub server ----------------------------------------------------------


async def _post(client, path, body):
    response = await client.post(path, json=body, headers={"Authorization": "tma stub"})
    return response.status, await response.json()


async def test_stub_serves_the_page_under_the_csp():
    async with TestClient(TestServer(make_stub_app(delay=0))) as client:
        page = await client.get("/")
        html = await page.text()
        assert STUB_SCRIPT in html and TELEGRAM_SCRIPT not in html
        assert page.headers["Content-Security-Policy"] == CONTENT_SECURITY_POLICY
        assert (await client.get("/static/app.js")).status == 200
        assert (await client.get("/" + STUB_SCRIPT)).status == 200
        assert (await client.get("/api/learn/decks")).status == 401


async def test_stub_walks_the_contract():
    async with TestClient(TestServer(make_stub_app(delay=0))) as client:
        auth = {"Authorization": "tma stub"}
        decks = await (await client.get("/api/learn/decks", headers=auth)).json()
        assert len(decks["decks"]) == 2

        status, data = await _post(client, "/api/learn/start", {"deck_id": None, "side": "mix"})
        assert status == 200 and data["state"] == "card" and data["can_undo"] is False
        session = data["session"]
        first = data["card"]["id"]
        assert data["progress"] == {"done": 0, "total": len(CARDS)}

        status, data = await _post(
            client, "/api/learn/review", {"session": session, "card_id": first, "rating": 1}
        )
        assert status == 200 and data["can_undo"] is True
        assert data["progress"]["done"] == 1

        # The same card again: stale, with the current state in the body.
        status, stale = await _post(
            client, "/api/learn/review", {"session": session, "card_id": first, "rating": 3}
        )
        assert status == 409 and stale["reason"] == "stale"
        assert stale["state"] == "card" and stale["card"]["id"] == data["card"]["id"]

        status, data = await _post(client, "/api/learn/undo", {"session": session})
        assert status == 200 and data["card"]["id"] == first and data["can_undo"] is False
        status, data = await _post(client, "/api/learn/undo", {"session": session})
        assert status == 409 and data["reason"] == "nothing_to_undo"

        for card in CARDS:
            status, data = await _post(
                client,
                "/api/learn/review",
                {"session": session, "card_id": card["id"], "rating": 3},
            )
        assert data["state"] == "end" and data["reviewed"] == len(CARDS) and data["can_undo"]

        tts = await client.post(
            "/api/learn/tts", json={"session": session, "card_id": first}, headers=auth
        )
        assert tts.status == 200 and tts.content_type == "audio/wav"

        await client.get("/stub/drop")
        status, data = await _post(client, "/api/learn/undo", {"session": session})
        assert status == 409 and data["reason"] == "session_gone"


async def test_stub_debt_and_empty():
    async with TestClient(TestServer(make_stub_app(delay=0, debt=True))) as client:
        status, data = await _post(client, "/api/learn/start", {"deck_id": None, "side": "fi"})
        assert data["state"] == "debt"
        assert {"session", "overdue", "max_cards", "postpone_days"} <= data.keys()
        status, data = await _post(
            client, "/api/learn/debt", {"session": data["session"], "action": "defer"}
        )
        assert status == 200 and data["state"] == "card"
    async with TestClient(TestServer(make_stub_app(delay=0, empty=True))) as client:
        status, data = await _post(client, "/api/learn/start", {"deck_id": None, "side": "ru"})
        assert data == {"state": "empty", "text": data["text"]}


# --- review fixes -------------------------------------------------------------


def test_tts_without_openai_and_session_gone_texts():
    body = _function("errorText")
    no_openai = body.index('error.reason === "no_openai"')
    assert no_openai < body.index("error.status === 503) {")
    assert '"Озвучка сейчас недоступна."' in body
    assert 'isConflict(error, "session_gone")) return SESSION_GONE_TEXT' in body
    # 🔊 on a card: session_gone ends the session like a rating does.
    tts = _function("showCard").split('request("api/learn/tts"', 1)[1]
    assert 'isConflict(error, "session_gone")' in tts and "showSessionGone()" in tts


def test_404_ends_the_session_and_reloads_decks():
    lost = _function("isSessionLost")
    assert 'isConflict(error, "session_gone")' in lost and "error.status === 404" in lost
    for name in ("rate", "undoLast", "showDebt"):
        assert "isSessionLost(error)" in _function(name), name
    start = _function("startSession")
    assert "error.status === 404) showLearn()" in start


def test_long_faces_scroll_in_full_and_hidden_face_takes_no_taps():
    face = STYLE_CSS.split("\n.face {", 1)[1].split("}", 1)[0]
    face = re.sub(r"/\*.*?\*/", "", face, flags=re.DOTALL)
    assert "justify-content" not in face
    assert "overflow-y: auto" in face
    assert ".face > :first-child { margin-top: auto; }" in STYLE_CSS
    assert ".face > :last-child { margin-bottom: auto; }" in STYLE_CSS
    assert re.search(
        r"\.flashcard-inner:not\(\.flipped\) \.face\.back,\s*"
        r"\.flashcard-inner\.flipped \.face\.front \{ pointer-events: none; \}",
        STYLE_CSS,
    )


def test_vertical_swipes_off_only_on_the_card():
    swipes = _function("verticalSwipes")
    assert '"enableVerticalSwipes"' in swipes and '"disableVerticalSwipes"' in swipes
    assert 'typeof tg[method] === "function"' in swipes
    # Every screen change turns them back on; the card turns them off.
    assert "verticalSwipes(true)" in _function("navigate")
    card = _function("showCard")
    assert card.index("navigate(showSides)") < card.index("verticalSwipes(false)")
