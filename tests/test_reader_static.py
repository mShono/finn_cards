"""Static checks on the reader's front end (plan 2026-11-20-kirjakaveri-reader,
"П3"): no HTML injection sinks, nothing inline that the CSP would block, and
the API paths app.js calls match the routes make_app serves."""

import re
from unittest.mock import MagicMock

from bot_helpers import make_settings
from test_reader import OWNER, TOKEN

from kielikaveri.reader.app import STATIC_DIR, make_app

APP_JS = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
INDEX_HTML = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

# api("api/books"), api(`api/chapters/${id}`), request("api/tts", { body: ... })
CALL_RE = re.compile(r"""\b(api|request)\(\s*(["`])(api/[^"`]*)\2\s*(,\s*\{\s*body\b)?""")

# П3's endpoints come from the server branch (reader-lookup), merged
# separately: their routes are checked against make_app after the merge.
LOOKUP_PATHS = {"/api/lookup", "/api/tts"}
# П5's /api/learn/* come from the server branch (learn-api): until it is
# merged they are not routes of make_app (test_reader_static_learn.py).
LEARN_PREFIX = "/api/learn/"


def learn_merged(routes: dict[str, set[str]]) -> bool:
    return "/api/learn/start" in routes


def _canonical(path: str) -> str:
    # `${encodeURIComponent(id)}` in JS and `{chapter_id}` in aiohttp both
    # become `{}`.
    path = re.sub(r"\$\{[^}]*\}", "{}", path)
    return re.sub(r"\{[^}]*\}", "{}", path)


def _calls() -> dict[str, set[str]]:
    """Each API path app.js calls (made absolute) -> the HTTP methods used."""
    calls: dict[str, set[str]] = {}
    for _, _, path, body in CALL_RE.findall(APP_JS):
        calls.setdefault(_canonical("/" + path), set()).add("POST" if body else "GET")
    return calls


def _routes() -> dict[str, set[str]]:
    settings = make_settings(bot_token=TOKEN, whitelist_user_ids=str(OWNER))
    app = make_app(MagicMock(), settings, MagicMock())
    routes: dict[str, set[str]] = {}
    for route in app.router.routes():
        info = route.resource.get_info() if route.resource else {}
        path = info.get("path") or info.get("formatter")
        if path:
            routes.setdefault(_canonical(path), set()).add(route.method)
    return routes


def test_app_js_has_no_html_sinks():
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
        assert sink not in APP_JS, sink


def test_index_html_has_no_inline_scripts_or_handlers():
    for tag in re.findall(r"<script\b[^>]*>", INDEX_HTML):
        assert "src=" in tag, tag
    assert not re.search(r"<script\b[^>]*>\s*(?!</script>)\S", INDEX_HTML), "inline script body"
    assert not re.search(r"\son[a-z]+\s*=", INDEX_HTML, re.IGNORECASE), "inline handler"
    assert "javascript:" not in INDEX_HTML


def test_fetch_goes_through_one_helper():
    # One fetch, the one that sets Authorization: tma <initData>.
    assert APP_JS.count("fetch(") == 1
    assert 'Authorization: "tma "' in APP_JS


def test_existing_paths_match_make_app_routes():
    calls = _calls()
    routes = _routes()
    for path, methods in calls.items():
        if path in LOOKUP_PATHS:
            continue
        if path.startswith(LEARN_PREFIX) and not learn_merged(routes):
            continue
        assert path in routes, path
        assert methods <= routes[path], (path, methods, routes[path])
    assert "/api/books" in calls
    assert "/api/chapters/{}" in calls


def test_lookup_and_tts_are_posts():
    calls = _calls()
    for path in LOOKUP_PATHS:
        assert calls.get(path) == {"POST"}, path


def test_audio_is_unlocked_inside_the_tap():
    # An Audio element played after an awaited fetch stayed silent in
    # Telegram: the AudioContext must be resumed in the click handler itself,
    # i.e. in a synchronous playSound (the popup's and the flashcard's 🔊),
    # before the first await.
    assert "AudioContext" in APP_JS
    assert "decodeAudioData" in APP_JS
    assert "AbortController" in APP_JS
    assert "new Audio(" not in APP_JS
    assert "async function playSound" not in APP_JS
    body = APP_JS.split("function playSound", 1)[1].split("\n}\n", 1)[0]
    assert "unlockAudio()" in body
    assert "await" not in body
