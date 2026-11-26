"""Static checks on the reader's decks screens (plan
2026-11-20-kirjakaveri-reader, "П6", stage К2): the /api/decks* and
/api/notes/* calls and their methods, the third tab and ?screen=decks|search,
text through textContent only, confirmations through tg.showConfirm, the
in-flight guard, the paged word list, the BackButton steps, the error texts.
And the test-side stub server (tests/decks_stub.py) answers in the
contract's shapes."""

import re

import pytest
from aiohttp.test_utils import TestClient, TestServer
from decks_stub import CHAPTER_ID, make_decks_stub_app
from learn_stub import FAKE_TELEGRAM_JS, STUB_SCRIPT
from test_reader_static import APP_JS, _calls, _routes, decks_merged
from test_reader_static_learn import STYLE_CSS, _const, _function

from kielikaveri.reader.app import CONTENT_SECURITY_POLICY

DECKS_CALLS = {
    "/api/decks": {"GET", "POST"},
    "/api/decks/{}": {"PATCH", "DELETE"},
    "/api/decks/{}/notes": {"GET"},
    "/api/notes/search": {"GET"},
    "/api/notes/{}": {"GET", "PATCH", "DELETE"},
    "/api/notes/{}/move": {"POST"},
    "/api/notes/{}/tts": {"POST"},
}
AUTH = {"Authorization": "tma stub"}


# --- calls and routes ---------------------------------------------------------


def test_decks_calls_and_methods():
    calls = _calls()
    for path, methods in DECKS_CALLS.items():
        assert calls.get(path) == methods, path


def test_decks_calls_are_routes_once_the_server_is_merged():
    routes = _routes()
    if not decks_merged(routes):
        pytest.skip("server routes (decks-api) not merged yet")
    for path, methods in DECKS_CALLS.items():
        assert path in routes, path
        assert methods <= routes[path], (path, routes[path])


def test_request_takes_a_method():
    body = _function("request")
    assert 'method: method || (body !== undefined ? "POST" : "GET")' in body
    # Still one fetch, the one with Authorization.
    assert APP_JS.count("fetch(") == 1


def test_request_bodies_follow_the_contract():
    assert 'api("api/decks", { body: { name } })' in _function("newDeckBlock")
    assert '{ method: "PATCH", body: { name } }' in _function("renderDeck")
    assert '{ method: "PATCH", body: { [field]: text } }' in _function("renderNote")
    note = _function("renderNote")
    assert 'editField("lemma", note.lemma' in note
    assert 'editField("translation_ru", note.translation_ru' in note
    assert "{ body: { deck_id: deck.id } }" in _function("moveNote")
    assert '{ method: "DELETE" }' in _function("deleteNote")
    assert '{ method: "DELETE" }' in _function("deleteDeck")
    assert "{ body: { what }, signal }" in note
    assert 'speakButton("lemma"' in note and 'speakButton("example"' in note


def test_ids_and_query_are_encoded():
    for path in re.findall(r"`api/(?:decks|notes)/[^`]*`", APP_JS):
        for value in re.findall(r"\$\{([^}]*)\}", path):
            assert value.startswith("encodeURIComponent(") or value in (
                "NOTES_PAGE",
                "pager.loaded",
            ), (path, value)
    assert "api/notes/search?q=${encodeURIComponent(q)}" in _function("liveSearch")


# --- text only through textContent --------------------------------------------


def test_no_html_sinks_or_inline_code():
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert sink not in APP_JS, sink
    assert "new Function" not in APP_JS
    assert 'setAttribute("style"' not in APP_JS
    # No inline handlers: el() turns onclick into addEventListener.
    assert not re.search(r"""setAttribute\(\s*["']on""", APP_JS)
    assert 'if (key === "onclick") node.addEventListener("click", value);' in APP_JS
    assert "javascript:" not in APP_JS


def test_no_api_stubs_in_app_js():
    for marker in ("stub", "mock", "fake", "sana001", "d-general"):
        assert marker not in APP_JS.lower(), marker


def test_word_screen_shows_the_contract_fields():
    body = _function("renderNote")
    for field in (
        "note.lemma",
        "note.pos",
        "note.puhekieli",
        "note.translation_ru",
        "note.example_fi",
        "note.example_ru",
        "note.forms_verified && note.forms",
        "note.deck.chapter",
        "note.cards",
    ):
        assert field in body, field
    assert "if (note.example_fi)" in body and "if (note.example_ru)" in body
    # deck may be null: a note from before decks.
    assert "note.deck ?" in body


def test_card_lines():
    body = _function("cardLine")
    assert 'card.status === "not_introduced"' in body and "ещё не открыта" in body
    assert "следующее повторение" in body
    assert "повторений ${card.reps || 0}" in body and "забыто ${card.lapses || 0}" in body
    assert "card.form" in body
    side = _const("CARD_SIDE")
    assert 'recognition: "🇫🇮→🇷🇺"' in side and 'production: "🇷🇺→🇫🇮"' in side
    assert '"дек"' in _const("MONTHS")


# --- entry and tabs -----------------------------------------------------------


def test_three_tabs_and_screen_parameter():
    switch = _function("switcher")
    assert 'tab("books", "📖 Книги", showBooks)' in switch
    assert 'tab("learn", "📚 Учить", showLearn)' in switch
    assert 'tab("decks", "🗂 Колоды", showDeckList)' in switch
    assert 'switcher("decks")' in _function("showDeckList")
    assert 'else if (screen === "decks") showDeckList();' in APP_JS
    assert 'else if (screen === "search") showSearch("");' in APP_JS
    assert "location.hash" not in APP_JS
    assert "pushState" not in APP_JS and "replaceState" not in APP_JS


def test_search_screen_focuses_its_field():
    body = _function("showSearch")
    assert "if (focus) {" in body and "input.focus()" in body
    # The list's field moves to the search screen on the first letter.
    assert "showSearch(search.value)" in _function("showDeckList")


def test_back_button_steps():
    # word -> deck or search -> list
    assert "navigate(back)" in _function("showNote")
    assert "noteItem(note, () => showSearch(q, false), true)" in _function("showSearch")
    assert "const back = () => showDeck(deck.id);" in _function("renderDeck")
    assert "navigate(showDeckList)" in _function("showDeck")
    assert "navigate(showDeckList)" in _function("showSearch")
    assert "navigate(null)" in _function("showDeckList")
    # 📖 Открыть главу comes back to the deck.
    assert "showChapter(chapterId, back)" in _function("renderDeck")
    assert "async function showChapter(chapterId, back = showBooks)" in APP_JS


def test_learn_this_deck():
    body = _function("learnDeck")
    assert "learn.deckId = deck.id" in body and "learn.fromDeck = deck.id" in body
    assert "showSides()" in body
    assert "learn.fromDeck = null" in _function("showLearn")
    assert '"📚 Учить эту колоду"' in _function("renderDeck")


# --- deck screen --------------------------------------------------------------


def test_deck_header():
    body = _function("renderDeck")
    assert "deckCounters(deck)" in body
    assert "формы: изучается ${deck.introduced}, ещё не открыто ${deck.not_introduced}" in body
    assert '"⋯"' in body and '"✍️ Переименовать"' in body and '"🗑 Удалить колоду"' in body
    # A chapter deck whose book is gone has chapter null.
    assert "if (deck.chapter && deck.chapter.id)" in body and '"📖 Открыть главу"' in body
    assert "📖 ${chapter.book_title}, гл. ${chapter.number}" in _function("deckCaption")
    assert "слов ${deck.notes} · к повторению ${deck.due}" in _function("deckCounters")


def test_words_load_50_at_a_time_while_scrolling():
    assert _const("NOTES_PAGE") == "50"
    show = _function("showDeck")
    assert "/notes?offset=0&limit=${NOTES_PAGE}" in show
    body = _function("deckWords")
    assert "/notes?offset=${pager.loaded}&limit=${NOTES_PAGE}" in body
    assert "IntersectionObserver" in body
    assert "if (pager.loading || pager.loaded >= pager.total) return;" in body
    assert "observer.disconnect()" in body


def test_deck_search_filters_the_server_search_by_deck():
    body = _function("deckWords")
    assert "liveSearch(search, view" in body
    assert "note.deck && note.deck.id === deck.id" in body


# --- confirmations, guards, refresh -------------------------------------------


def test_deletes_ask_through_show_confirm():
    confirm = _function("askConfirm")
    assert "tg.showConfirm(text" in confirm and "confirmOpen" in confirm
    for name in ("deleteDeck", "deleteNote"):
        assert "askConfirm(" in _function(name), name


def test_one_request_in_flight():
    assert "button.disabled = busy" in _function("setGuard")
    edit = _function("submitEdit")
    assert edit.index("if (editor.guard.busy) return;") < edit.index("editorBusy(editor, true)")
    assert edit.index("editorBusy(editor, true)") < edit.index("send()")
    for name in ("deleteNote", "deleteDeck", "moveNote"):
        body = _function(name)
        assert "guard.busy" in body, name
        assert body.index("setGuard(guard, true)") < body.index(
            "request(" if "request(" in body else "api("
        ), name
    # ✍️ Слово shows "…" while the model works, the error goes under the field.
    busy = _function("editorBusy")
    assert 'busy ? "…" : editor.label' in busy
    assert "editor.error.textContent = errorFor(error)" in edit
    for name in ("submitEdit", "moveNote", "deleteNote", "deleteDeck"):
        assert "await" not in _function(name), name


def test_answers_for_a_gone_screen_are_dropped():
    for name in ("submitEdit", "moveNote", "deleteNote", "deleteDeck", "liveSearch", "deckWords"):
        assert "view === navigation" in _function(name) or "view !== navigation" in _function(
            name
        ), name
    assert "mine === asked" in _function("liveSearch")


def test_everything_is_asked_again_after_a_change():
    # Rename -> the deck again, create/delete deck -> the list again,
    # edit/move -> the word again, delete word -> back (deck or search).
    assert "back, deckNameErrorText" in _function("renderDeck")
    assert "() => showDeckList()" in _function("newDeckBlock")
    assert "showDeckList()" in _function("deleteDeck")
    assert "reopen()" in _function("moveNote")
    assert "const reopen = () => showNote(note.id, back);" in _function("renderNote")
    assert "back()" in _function("deleteNote")
    # No client-side cache of decks or notes.
    assert "localStorage" not in APP_JS and "sessionStorage" not in APP_JS


def test_error_texts():
    edit = _function("noteEditErrorText")
    assert 'isConflict(error, "duplicate")' in edit and "уже есть в этой колоде" in edit
    assert 'error.reason === "no_openai"' in edit
    assert "error.status === 502" in edit
    assert "isNotFound(error)) return NOTE_GONE_TEXT" in edit
    assert edit.rstrip().endswith("return errorText(error);")
    assert 'isConflict(error, "duplicate_in_target")' in _function("moveNote")
    assert "showNotice(NOTE_GONE_TEXT, back)" in _function("deleteNote")
    assert _const("NOTE_GONE_TEXT") == '"Слово уже удалено."'
    name = _function("deckNameErrorText")
    assert "error.status === 422) return DECK_NAME_TEXT" in name


def test_styles_exist():
    for selector in (
        "input.search",
        "input.field",
        ".word-actions",
        ".deck-head",
        ".menu",
        ".pair",
        ".error",
    ):
        assert selector in STYLE_CSS, selector


def test_fake_telegram_confirms():
    assert "showConfirm(message, callback)" in FAKE_TELEGRAM_JS


# --- the stub server ----------------------------------------------------------


async def _json(response):
    return response.status, (await response.json() if response.status != 204 else None)


def _stub():
    return TestClient(TestServer(make_decks_stub_app(llm_delay=0, delete_delay=0, delay=0)))


async def test_stub_serves_the_page_under_the_csp():
    async with _stub() as client:
        page = await client.get("/?screen=decks")
        assert STUB_SCRIPT in await page.text()
        assert page.headers["Content-Security-Policy"] == CONTENT_SECURITY_POLICY
        assert (await client.get("/api/decks")).status == 401
        for path in ("/api/books", "/api/learn/decks", f"/api/chapters/{CHAPTER_ID}"):
            assert (await client.get(path, headers=AUTH)).status == 200, path


async def test_stub_decks_create_rename_delete():
    async with _stub() as client:
        status, data = await _json(await client.get("/api/decks", headers=AUTH))
        assert status == 200
        keys = {"id", "name", "notes", "due", "introduced", "not_introduced", "chapter"}
        assert all(deck.keys() == keys for deck in data["decks"])
        chapter = [deck for deck in data["decks"] if deck["chapter"]]
        assert chapter and chapter[0]["chapter"].keys() == {"id", "book_title", "number"}

        for bad in ("", "   ", "x" * 65):
            status, data = await _json(
                await client.post("/api/decks", json={"name": bad}, headers=AUTH)
            )
            assert (status, data) == (422, {"reason": "bad_name"}), bad
        status, data = await _json(
            await client.post("/api/decks", json={"name": "  Новая  "}, headers=AUTH)
        )
        assert status == 200 and data["deck"]["name"] == "Новая"
        deck_id = data["deck"]["id"]
        status, data = await _json(
            await client.patch(f"/api/decks/{deck_id}", json={"name": "Иная"}, headers=AUTH)
        )
        assert status == 200 and data["deck"]["name"] == "Иная"
        status, data = await _json(await client.delete(f"/api/decks/{deck_id}", headers=AUTH))
        assert (status, data) == (200, {"deleted_notes": 0})
        status, _ = await _json(await client.delete(f"/api/decks/{deck_id}", headers=AUTH))
        assert status == 404

        # No decks left: the list makes "Общая".
        _, data = await _json(await client.get("/api/decks", headers=AUTH))
        for deck in data["decks"]:
            await client.delete(f"/api/decks/{deck['id']}", headers=AUTH)
        _, data = await _json(await client.get("/api/decks", headers=AUTH))
        assert [deck["name"] for deck in data["decks"]] == ["Общая"]


async def test_stub_pages_a_big_deck():
    async with _stub() as client:
        seen = []
        offset = 0
        while True:
            status, data = await _json(
                await client.get(f"/api/decks/d-big/notes?offset={offset}&limit=50", headers=AUTH)
            )
            assert status == 200 and data["total"] >= 120
            if not data["notes"]:
                break
            assert all(
                note.keys() == {"id", "lemma", "translation_ru", "pos"} for note in data["notes"]
            )
            seen += [note["id"] for note in data["notes"]]
            offset += len(data["notes"])
        assert len(seen) == len(set(seen)) == data["total"]


async def test_stub_search_folds_case():
    async with _stub() as client:

        async def search(q):
            return (
                await _json(await client.get("/api/notes/search", params={"q": q}, headers=AUTH))
            )[1]["notes"]

        assert await search("") == []
        upper, lower = await search("Ä"), await search("ä")
        assert upper == lower and {"äiti", "pää", "hyvä"} <= {note["lemma"] for note in upper}
        assert [note["lemma"] for note in await search("ДОМ")][:2] == ["talo", "talo"]
        assert await search("дом") == await search("ДОМ")
        assert [note["lemma"] for note in await search("mennää")] == ["mennä"]
        kissa = await search("кошка")
        assert kissa[0]["deck"] is None
        assert kissa[0].keys() == {"id", "lemma", "translation_ru", "deck"}


async def _note_id(client, lemma, deck_id):
    notes = (await _json(await client.get(f"/api/decks/{deck_id}/notes?limit=100", headers=AUTH)))[
        1
    ]
    return next(note["id"] for note in notes["notes"] if note["lemma"] == lemma)


async def test_stub_note_detail_move_edit_delete():
    async with _stub() as client:
        talo = await _note_id(client, "talo", "d-chapter")
        status, note = await _json(await client.get(f"/api/notes/{talo}", headers=AUTH))
        assert status == 200
        assert note.keys() == {
            "id",
            "lemma",
            "pos",
            "translation_ru",
            "example_fi",
            "example_ru",
            "puhekieli",
            "forms",
            "forms_verified",
            "deck",
            "cards",
        }
        assert note["deck"]["chapter"]["id"] == CHAPTER_ID
        card_keys = {"type", "form", "status", "state", "due", "reps", "lapses"}
        assert all(card.keys() == card_keys for card in note["cards"])
        assert {card["status"] for card in note["cards"]} == {"introduced", "not_introduced"}

        # "talo" is in "Общая" already.
        status, data = await _json(
            await client.post(
                f"/api/notes/{talo}/move", json={"deck_id": "d-general"}, headers=AUTH
            )
        )
        assert (status, data) == (409, {"reason": "duplicate_in_target"})
        status, data = await _json(
            await client.post(
                f"/api/notes/{talo}/move", json={"deck_id": "d-chapter"}, headers=AUTH
            )
        )
        assert status == 200 and data["note"]["deck"]["id"] == "d-chapter"
        status, data = await _json(
            await client.post(f"/api/notes/{talo}/move", json={"deck_id": "d-big"}, headers=AUTH)
        )
        assert status == 200 and data["note"]["deck"]["id"] == "d-big"

        aiti = await _note_id(client, "äiti", "d-general")
        status, data = await _json(
            await client.patch(f"/api/notes/{aiti}", json={"lemma": "talo"}, headers=AUTH)
        )
        assert (status, data) == (409, {"reason": "duplicate"})
        status, data = await _json(
            await client.patch(f"/api/notes/{aiti}", json={"translation_ru": "мать"}, headers=AUTH)
        )
        assert status == 200 and data["note"]["translation_ru"] == "мать"
        for mode, expected in (("no_openai", 503), ("rate", 503), ("openai", 502)):
            await client.get(f"/stub/llm/{mode}")
            status, data = await _json(
                await client.patch(f"/api/notes/{aiti}", json={"lemma": "äitee"}, headers=AUTH)
            )
            assert (status, data["reason"]) == (expected, mode)
        await client.get("/stub/llm/ok")
        status, data = await _json(
            await client.patch(f"/api/notes/{aiti}", json={"lemma": "äitee"}, headers=AUTH)
        )
        assert status == 200 and data["note"]["lemma"] == "äitee"

        tts = await client.post(f"/api/notes/{aiti}/tts", json={"what": "example"}, headers=AUTH)
        assert tts.status == 200 and tts.content_type == "audio/wav"
        bad = await client.post(f"/api/notes/{aiti}/tts", json={"what": "x"}, headers=AUTH)
        assert bad.status == 400

        status, _ = await _json(await client.delete(f"/api/notes/{aiti}", headers=AUTH))
        assert status == 204
        status, _ = await _json(await client.delete(f"/api/notes/{aiti}", headers=AUTH))
        assert status == 404
        assert (await client.get(f"/api/notes/{aiti}", headers=AUTH)).status == 404
