"""A stand-in server for checking the decks screens (plan
2026-11-20-kirjakaveri-reader, "П6", stage К2) by hand, without the bot, the
database or Telegram: the real static files, under the real CSP, plus fake
/api/decks* and /api/notes* answers in the contract's shapes, kept in memory.
The learning tab is tests/learn_stub.py's (its fake Telegram too), the books
tab is empty, and chapter "ch-1" opens for "📖 Открыть главу".

    uv run python tests/decks_stub.py [--port 8099] [--llm-delay 1.5]
                                      [--llm-mode ok] [--delete-delay 0.4]

then open http://localhost:8099/?screen=decks (or ?screen=search; a
phone-sized window in the browser's dev tools shows the phone layout).
The fake Telegram's BackButton is a floating "← Back" button and
tg.showConfirm is the browser's confirm. Every API call is printed.

What to try:
- "Общая": search "ä"/"Ä" (äiti, pää, hyvä), "дом", "ДОМ", "mennää" (puhekieli);
  the search on the list screen finds "kissa" too, a word in no deck.
- "Большая колода": 130 words, loaded 50 at a time while scrolling.
- "Покаянные дни - гл.1": a chapter deck, "📖 Открыть главу"; its "talo"
  moved to "Общая" answers 409 duplicate_in_target.
- "Старая книга - гл.3": a chapter deck whose book is gone (chapter null).
- ✍️ Слово waits --llm-delay seconds ("…" on the button); renaming a word to
  one already in the deck (talo -> äiti in "Общая") answers 409 duplicate.
  GET /stub/llm/<mode> switches ✍️ Слово to answer no_openai (503), rate or
  daily (503), openai (502), or ok again.
- 🗑 Удалить waits --delete-delay seconds; a second delete of the same word
  answers 404 (GET /stub/undelete brings the seed back).
- Delete every deck: the next list makes "Общая" again, like the bot.

Also used by tests/test_reader_static_decks.py, which keeps it in step with
the contract.
"""

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from itertools import count

from aiohttp import web
from learn_stub import _beep_wav, make_stub_app

DECK_NAME_MAX = 64
SEARCH_MAX = 50
PAGE_DEFAULT = 50
PAGE_MAX = 100
LLM_MODES = {
    "ok": None,
    "no_openai": (503, "no_openai"),
    "rate": (503, "rate"),
    "daily": (503, "daily"),
    "openai": (502, "openai"),
}
GENERAL = "Общая"
CHAPTER_DECK = "d-chapter"
CHAPTER_ID = "ch-1"

NOUN_FORMS = {
    "genetiivi": "{}n",
    "partitiivi": "{}a",
    "inessiivi": "{}ssa",
    "elatiivi": "{}sta",
}


def _iso(delta: timedelta) -> str:
    return (datetime.now(UTC) + delta).isoformat()


def _cards(pos: str | None, forms: dict | None, *, reviewed: bool) -> list[dict]:
    """Two word cards (one reviewed a few times when `reviewed`), plus an
    inflection card per form: the first introduced, the rest not yet."""
    cards = [
        {
            "type": "recognition",
            "form": None,
            "status": "introduced",
            "state": "review" if reviewed else "learning",
            "due": _iso(timedelta(days=7) if reviewed else timedelta(hours=-1)),
            "reps": 5 if reviewed else 0,
            "lapses": 1 if reviewed else 0,
        },
        {
            "type": "production",
            "form": None,
            "status": "introduced",
            "state": "learning",
            "due": _iso(timedelta(hours=-2)),
            "reps": 2 if reviewed else 0,
            "lapses": 0,
        },
    ]
    for i, key in enumerate(forms or {}):
        cards.append(
            {
                "type": "inflection",
                "form": key,
                "status": "introduced" if i == 0 else "not_introduced",
                "state": "learning",
                "due": _iso(timedelta(days=400 if i == 0 else 0)),
                "reps": 1 if i == 0 else 0,
                "lapses": 0,
            }
        )
    return cards


def _note(
    lemma: str,
    translation: str,
    deck_id: str | None,
    *,
    pos: str | None = "noun",
    example_fi: str = "",
    example_ru: str = "",
    puhekieli: str | None = None,
    forms: bool = False,
    reviewed: bool = False,
) -> dict:
    principal = (
        {key: pattern.format(lemma) for key, pattern in NOUN_FORMS.items()} if forms else None
    )
    return {
        "lemma": lemma,
        "pos": pos,
        "translation_ru": translation,
        "example_fi": example_fi,
        "example_ru": example_ru,
        "puhekieli": puhekieli,
        "forms": principal,
        "forms_verified": forms,
        "deck_id": deck_id,
        "cards": _cards(pos, principal, reviewed=reviewed),
    }


def _seed() -> tuple[dict[str, dict], list[dict]]:
    decks = {
        "d-general": {"id": "d-general", "name": GENERAL, "chapter": None},
        CHAPTER_DECK: {
            "id": CHAPTER_DECK,
            "name": "Покаянные дни - гл.1",
            "chapter": {"id": CHAPTER_ID, "book_title": "Покаянные дни", "number": 1},
        },
        # The book was deleted: decks.chapter_id is NULL, chapter is null.
        "d-orphan": {"id": "d-orphan", "name": "Старая книга - гл.3", "chapter": None},
        "d-big": {"id": "d-big", "name": "Большая колода", "chapter": None},
    }
    notes = [
        _note(
            "talo",
            "дом",
            "d-general",
            example_fi="Hän lähti talosta aikaisin aamulla.",
            example_ru="Он ушёл из дома рано утром.",
            forms=True,
            reviewed=True,
        ),
        _note("äiti", "мама", "d-general", example_fi="Äiti tuli kotiin.", forms=True),
        _note("pää", "голова", "d-general"),
        _note("hyvä", "хороший", "d-general", pos="adj"),
        _note(
            "mennä",
            "идти",
            "d-general",
            pos="verb",
            puhekieli="mennää",
            example_fi="Me mennään kotiin.",
            example_ru="Мы идём домой.",
        ),
        _note("nähdä", "видеть", "d-general", pos="verb", example_fi="Kukaan ei nähnyt."),
        _note("olla kotona", "быть дома", "d-general", pos=None),
        _note("talo", "дом, здание", CHAPTER_DECK, example_fi="Talon ovi oli auki.", forms=True),
        _note("aamu", "утро", CHAPTER_DECK, reviewed=True),
        _note("lähteä", "уходить", CHAPTER_DECK, pos="verb"),
        _note("ääni", "звук, голос", "d-orphan"),
        _note("Äänekoski", "Ээнекоски (город)", "d-orphan", pos="propn"),
        # A note from before decks (deck_id NULL): only the search sees it.
        _note("kissa", "кошка", None),
    ]
    notes += [
        _note(f"sana{i:03d}", f"слово номер {i}", "d-big", reviewed=i % 3 == 0)
        for i in range(1, 131)
    ]
    return decks, notes


@dataclass
class DecksState:
    llm_delay: float = 1.5
    llm_mode: str = "ok"
    delete_delay: float = 0.4
    decks: dict[str, dict] = field(default_factory=dict)
    notes: dict[str, dict] = field(default_factory=dict)
    ids: count = field(default_factory=lambda: count(1))
    calls: list[tuple[str, str, dict]] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def reset(self) -> None:
        self.decks, seed = _seed()
        self.notes = {}
        # Seeded oldest first: the last one is the newest.
        for note in seed:
            self.add_note(note)

    def add_note(self, note: dict) -> dict:
        seq = next(self.ids)
        note = {**note, "id": note.get("id") or f"n{seq}", "created": seq}
        self.notes[note["id"]] = note
        return note


DECKS_STATE = web.AppKey("decks_stub_state", DecksState)


def _reason(status: int, reason: str) -> web.Response:
    return web.json_response({"reason": reason}, status=status)


async def _log(request: web.Request) -> dict:
    body = await request.json() if request.can_read_body else {}
    request.app[DECKS_STATE].calls.append((request.method, request.path_qs, body))
    print(f"{request.method} {request.path_qs} {json.dumps(body, ensure_ascii=False)}", flush=True)
    return body if isinstance(body, dict) else {}


def _now() -> datetime:
    return datetime.now(UTC)


def _deck_json(state: DecksState, deck: dict) -> dict:
    cards = [
        card
        for note in state.notes.values()
        if note["deck_id"] == deck["id"]
        for card in note["cards"]
    ]
    introduced = [card for card in cards if card["status"] == "introduced"]
    now = _now()
    return {
        "id": deck["id"],
        "name": deck["name"],
        "notes": sum(1 for note in state.notes.values() if note["deck_id"] == deck["id"]),
        "due": sum(1 for card in introduced if datetime.fromisoformat(card["due"]) <= now),
        "introduced": len(introduced),
        "not_introduced": len(cards) - len(introduced),
        "chapter": deck["chapter"],
    }


def _deck_ref(state: DecksState, note: dict, *, chapter: bool) -> dict | None:
    deck = state.decks.get(note["deck_id"]) if note["deck_id"] else None
    if deck is None:
        return None
    ref = {"id": deck["id"], "name": deck["name"]}
    if chapter:
        ref["chapter"] = deck["chapter"]
    return ref


def _note_detail(state: DecksState, note: dict) -> dict:
    return {
        "id": note["id"],
        "lemma": note["lemma"],
        "pos": note["pos"],
        "translation_ru": note["translation_ru"],
        "example_fi": note["example_fi"],
        "example_ru": note["example_ru"],
        "puhekieli": note["puhekieli"],
        "forms": note["forms"],
        "forms_verified": note["forms_verified"],
        "deck": _deck_ref(state, note, chapter=True),
        "cards": note["cards"],
    }


def _deck_or_404(request: web.Request) -> dict:
    deck = request.app[DECKS_STATE].decks.get(request.match_info["deck_id"])
    if deck is None:
        raise web.HTTPNotFound(text='{"reason": "not_found"}', content_type="application/json")
    return deck


def _note_or_404(request: web.Request) -> dict:
    note = request.app[DECKS_STATE].notes.get(request.match_info["note_id"])
    if note is None:
        raise web.HTTPNotFound(text='{"reason": "not_found"}', content_type="application/json")
    return note


def _deck_name(body: dict) -> str | None:
    name = body.get("name")
    if not isinstance(name, str):
        return None
    name = name.strip()
    return name if 1 <= len(name) <= DECK_NAME_MAX else None


# --- decks --------------------------------------------------------------------


async def list_decks(request: web.Request) -> web.Response:
    state = request.app[DECKS_STATE]
    await _log(request)
    if not state.decks:
        # Like the bot: no decks at all makes "Общая".
        deck_id = f"d{next(state.ids)}"
        state.decks[deck_id] = {"id": deck_id, "name": GENERAL, "chapter": None}
    return web.json_response({"decks": [_deck_json(state, deck) for deck in state.decks.values()]})


async def create_deck(request: web.Request) -> web.Response:
    state = request.app[DECKS_STATE]
    name = _deck_name(await _log(request))
    if name is None:
        return _reason(422, "bad_name")
    deck_id = f"d{next(state.ids)}"
    state.decks[deck_id] = {"id": deck_id, "name": name, "chapter": None}
    return web.json_response({"deck": _deck_json(state, state.decks[deck_id])})


async def rename_deck(request: web.Request) -> web.Response:
    state = request.app[DECKS_STATE]
    body = await _log(request)
    deck = _deck_or_404(request)
    name = _deck_name(body)
    if name is None:
        return _reason(422, "bad_name")
    deck["name"] = name
    return web.json_response({"deck": _deck_json(state, deck)})


async def delete_deck(request: web.Request) -> web.Response:
    state = request.app[DECKS_STATE]
    await _log(request)
    async with state.lock:
        deck = _deck_or_404(request)
        doomed = [nid for nid, note in state.notes.items() if note["deck_id"] == deck["id"]]
        for note_id in doomed:
            del state.notes[note_id]
        del state.decks[deck["id"]]
    return web.json_response({"deleted_notes": len(doomed)})


def _int_param(request: web.Request, name: str, default: int) -> int:
    raw = request.query.get(name, "")
    if raw == "":
        return default
    if not raw.isdigit():
        raise web.HTTPBadRequest(text='{"reason": "bad_request"}', content_type="application/json")
    return int(raw)


async def deck_notes(request: web.Request) -> web.Response:
    state = request.app[DECKS_STATE]
    await _log(request)
    deck = _deck_or_404(request)
    offset = _int_param(request, "offset", 0)
    limit = max(1, min(PAGE_MAX, _int_param(request, "limit", PAGE_DEFAULT)))
    notes = sorted(
        (note for note in state.notes.values() if note["deck_id"] == deck["id"]),
        key=lambda note: -note["created"],
    )
    page = [
        {
            "id": note["id"],
            "lemma": note["lemma"],
            "translation_ru": note["translation_ru"],
            "pos": note["pos"],
        }
        for note in notes[offset : offset + limit]
    ]
    return web.json_response({"notes": page, "total": len(notes)})


# --- notes --------------------------------------------------------------------


def _rank(note: dict, q: str) -> int | None:
    """0 exact, 1 a word starts with q, 2 q anywhere, None no match."""
    fields = [note["lemma"], note["translation_ru"], note["puhekieli"] or ""]
    folded = [text.casefold() for text in fields]
    if any(text == q for text in folded):
        return 0
    if any(word.startswith(q) for text in folded for word in text.replace(",", " ").split()):
        return 1
    if any(q in text for text in folded):
        return 2
    return None


async def search_notes(request: web.Request) -> web.Response:
    state = request.app[DECKS_STATE]
    await _log(request)
    q = request.query.get("q", "").casefold()
    if len(q) < 1:
        return web.json_response({"notes": []})
    ranked = []
    for note in state.notes.values():
        rank = _rank(note, q)
        if rank is not None:
            ranked.append((rank, -note["created"], note))
    ranked.sort(key=lambda item: item[:2])
    return web.json_response(
        {
            "notes": [
                {
                    "id": note["id"],
                    "lemma": note["lemma"],
                    "translation_ru": note["translation_ru"],
                    "deck": _deck_ref(state, note, chapter=False),
                }
                for _, _, note in ranked[:SEARCH_MAX]
            ]
        }
    )


async def get_note(request: web.Request) -> web.Response:
    state = request.app[DECKS_STATE]
    await _log(request)
    return web.json_response(_note_detail(state, _note_or_404(request)))


def _clash(state: DecksState, note: dict, lemma: str, deck_id: str | None) -> bool:
    return any(
        other["id"] != note["id"]
        and other["deck_id"] == deck_id
        and other["lemma"] == lemma
        and other["pos"] == note["pos"]
        for other in state.notes.values()
    )


async def edit_note(request: web.Request) -> web.Response:
    state = request.app[DECKS_STATE]
    body = await _log(request)
    note = _note_or_404(request)
    if "translation_ru" in body:
        value = body["translation_ru"]
        if not isinstance(value, str) or not value.strip():
            return _reason(422, "bad_value")
        note["translation_ru"] = value.strip()
        return web.json_response({"note": _note_detail(state, note)})
    value = body.get("lemma")
    if not isinstance(value, str) or not value.strip():
        return _reason(422, "bad_value")
    lemma = value.strip()
    # Checked before the model call, like the bot's ✍️ Слово.
    if _clash(state, note, lemma, note["deck_id"]):
        return _reason(409, "duplicate")
    failure = LLM_MODES[state.llm_mode]
    if failure == (503, "no_openai"):
        return _reason(*failure)
    await asyncio.sleep(state.llm_delay)
    if failure is not None:
        return _reason(*failure)
    if request.match_info["note_id"] not in state.notes:
        return _reason(404, "not_found")
    if _clash(state, note, lemma, note["deck_id"]):
        return _reason(409, "duplicate")
    note["lemma"] = lemma
    return web.json_response({"note": _note_detail(state, note)})


async def move_note(request: web.Request) -> web.Response:
    state = request.app[DECKS_STATE]
    body = await _log(request)
    note = _note_or_404(request)
    target = state.decks.get(body.get("deck_id"))
    if target is None:
        return _reason(404, "not_found")
    if note["deck_id"] == target["id"]:
        return web.json_response({"note": _note_detail(state, note)})
    if _clash(state, note, note["lemma"], target["id"]):
        return _reason(409, "duplicate_in_target")
    note["deck_id"] = target["id"]
    return web.json_response({"note": _note_detail(state, note)})


async def delete_note(request: web.Request) -> web.Response:
    state = request.app[DECKS_STATE]
    await _log(request)
    async with state.lock:
        await asyncio.sleep(state.delete_delay)
        note = _note_or_404(request)
        del state.notes[note["id"]]
    return web.Response(status=204)


async def note_tts(request: web.Request) -> web.Response:
    body = await _log(request)
    note = _note_or_404(request)
    what = body.get("what")
    if what not in ("lemma", "example"):
        return _reason(400, "bad_request")
    if what == "example" and not note["example_fi"]:
        return _reason(404, "no_example")
    return web.Response(body=_beep_wav(), content_type="audio/wav")


# --- the chapter behind "📖 Открыть главу", and the stub's own switches ------


async def chapter(request: web.Request) -> web.Response:
    await _log(request)
    if request.match_info["chapter_id"] != CHAPTER_ID:
        return _reason(404, "not_found")
    return web.json_response(
        {
            "id": CHAPTER_ID,
            "number": 1,
            "part": None,
            "title": "Aamu",
            "book": {"title": "Покаянные дни"},
            "paragraphs": [
                ["", "Hän", " ", "lähti", " ", "talosta", " ", "aikaisin", " ", "aamulla", "."],
                ["", "Talon", " ", "ovi", " ", "oli", " ", "auki", "."],
            ],
            "next": None,
        }
    )


async def set_llm(request: web.Request) -> web.Response:
    mode = request.match_info["mode"]
    if mode not in LLM_MODES:
        return web.Response(status=404, text=f"modes: {', '.join(LLM_MODES)}\n")
    request.app[DECKS_STATE].llm_mode = mode
    return web.Response(text=f"✍️ Слово now answers: {mode}\n")


async def undelete(request: web.Request) -> web.Response:
    request.app[DECKS_STATE].reset()
    return web.Response(text="decks and words are back to the seed\n")


def make_decks_stub_app(
    *, llm_delay: float = 1.5, llm_mode: str = "ok", delete_delay: float = 0.4, **learn
) -> web.Application:
    app = make_stub_app(**learn)
    state = DecksState(llm_delay=llm_delay, llm_mode=llm_mode, delete_delay=delete_delay)
    state.reset()
    app[DECKS_STATE] = state
    app.router.add_get("/stub/llm/{mode}", set_llm)
    app.router.add_get("/stub/undelete", undelete)
    app.router.add_get("/api/chapters/{chapter_id}", chapter)
    app.router.add_get("/api/decks", list_decks)
    app.router.add_post("/api/decks", create_deck)
    app.router.add_patch("/api/decks/{deck_id}", rename_deck)
    app.router.add_delete("/api/decks/{deck_id}", delete_deck)
    app.router.add_get("/api/decks/{deck_id}/notes", deck_notes)
    # Before /api/notes/{note_id}: routes match in the order they are added.
    app.router.add_get("/api/notes/search", search_notes)
    app.router.add_get("/api/notes/{note_id}", get_note)
    app.router.add_patch("/api/notes/{note_id}", edit_note)
    app.router.add_delete("/api/notes/{note_id}", delete_note)
    app.router.add_post("/api/notes/{note_id}/move", move_note)
    app.router.add_post("/api/notes/{note_id}/tts", note_tts)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--llm-delay", type=float, default=1.5, help="seconds ✍️ Слово takes")
    parser.add_argument("--llm-mode", choices=sorted(LLM_MODES), default="ok")
    parser.add_argument("--delete-delay", type=float, default=0.4, help="seconds a delete takes")
    args = parser.parse_args()
    app = make_decks_stub_app(
        llm_delay=args.llm_delay, llm_mode=args.llm_mode, delete_delay=args.delete_delay
    )
    print(f"http://localhost:{args.port}/?screen=decks", flush=True)
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
