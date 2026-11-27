"""Static checks on the reader's front end for "П4" (plan
2026-11-20-kirjakaveri-reader): the early FST lemma (/api/lemma) and ➕
(/api/cards). Every "api/..." path in app.js must be a route of make_app."""

import re

import pytest
from test_reader_static import APP_JS, _calls, _canonical, _routes, unmerged

# Any string or template literal starting with api/, not just api()/request()
# calls: a path kept in a constant is checked too.
PATH_RE = re.compile(r"""(["`])(api/[^"`]*)\1""")


def test_every_api_path_in_app_js_is_a_route():
    routes = _routes()
    if "/api/cards" not in routes:
        pytest.skip("server routes not merged yet")
    paths = {_canonical("/" + path.split("?", 1)[0]) for _, path in PATH_RE.findall(APP_JS)}
    assert {"/api/lemma", "/api/lookup", "/api/cards", "/api/tts"} <= paths
    for path in paths:
        if unmerged(path, routes):
            continue
        assert path in routes, path
    for path, methods in _calls().items():
        if unmerged(path, routes):
            continue
        assert methods <= routes[path], (path, methods, routes[path])


def test_lemma_and_cards_are_posts():
    calls = _calls()
    assert calls.get("/api/lemma") == {"POST"}
    assert calls.get("/api/cards") == {"POST"}


def test_add_card_ui():
    assert "innerHTML" not in APP_JS
    assert "AbortController" in APP_JS
    for text in ("in_deck", "save_failed", "✓ уже в колоде", "✓ в колоде"):
        assert text in APP_JS, text


def test_lemma_and_lookup_share_the_popup_abort():
    # Both requests of a tap are aborted together when the popup closes or
    # another word is tapped; ➕ is not.
    body = APP_JS.split("function openWord", 1)[1].split("\n}\n", 1)[0]
    assert body.count("signal: mine.controller.signal") == 2
    assert '"api/lemma"' in body and '"api/lookup"' in body
    add = APP_JS.split("function addCard", 1)[1].split("\n}\n", 1)[0]
    assert "signal" not in add
