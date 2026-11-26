"""/api/decks and /api/notes (plan 2026-11-20-kirjakaveri-reader, "П6", К1):
the deck screens of the Mini App on top of decks_core. OpenAI is a fake;
the FST is the real one."""

# The fixtures come from test_reader_learn and test_reader_lookup; tests
# naming them as parameters "redefine" the import for ruff.
# ruff: noqa: F811

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import openai
import pytest
from conftest import delete_book, log_fields
from sqlalchemy import func, select
from test_reader import OWNER, STRANGER, header
from test_reader_learn import (  # noqa: F401 - fixtures among them
    client,
    make_client,
    tts,
)
from test_reader_lookup import (  # noqa: F401 - fixtures among them
    make_book,
    session_factory,
)
from test_reader_tts import WAV

from finn_cards.morphology import generate_forms
from kielikaveri.db.models import (
    Card,
    CardState,
    CardStatus,
    CardType,
    Deck,
    Note,
    NoteKind,
    Review,
)
from kielikaveri.ingest import ResolvedForms
from kielikaveri.llm.breaker import CircuitOpenError
from kielikaveri.reader.app import LOOKUP_CLIENT, READER_BREAKER

NOW = datetime.now(UTC)
DUE = datetime(2026, 12, 3, 9, 30, tzinfo=UTC)
DECK_KEYS = {"id", "name", "notes", "due", "introduced", "not_introduced", "chapter"}
NOTE_KEYS = {
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


async def add_deck(session_factory, deck_id, user_id=OWNER, name=None, chapter_id=None) -> str:
    async with session_factory() as session:
        session.add(
            Deck(
                id=deck_id, user_id=user_id, name=name or f"Pakka {deck_id}", chapter_id=chapter_id
            )
        )
        await session.commit()
    return deck_id


async def add_note(
    session_factory,
    note_id,
    deck_id,
    user_id=OWNER,
    lemma=None,
    pos="substantiivi",
    created_at=None,
    **columns,
) -> str:
    defaults = {
        "translation_ru": f"перевод-{note_id}",
        "example_fi": f"Esimerkki {note_id}.",
        "example_ru": f"Пример {note_id}.",
        "kind": NoteKind.word,
        "meta": {},
    }
    async with session_factory() as session:
        session.add(
            Note(
                id=note_id,
                user_id=user_id,
                deck_id=deck_id,
                lemma=lemma or f"sana{note_id}",
                pos=pos,
                created_at=created_at or NOW,
                **{**defaults, **columns},
            )
        )
        await session.commit()
    return note_id


async def add_card(session_factory, card_id, note_id, user_id=OWNER, reviews=0, **columns) -> str:
    columns = {"type": CardType.recognition, "due": NOW - timedelta(days=1), **columns}
    async with session_factory() as session:
        session.add(Card(id=card_id, note_id=note_id, user_id=user_id, **columns))
        await session.flush()
        for _ in range(reviews):
            session.add(Review(card_id=card_id, user_id=user_id, rating=3, reviewed_at=NOW))
        await session.commit()
    return card_id


async def count(session_factory, model, *where) -> int:
    async with session_factory() as session:
        return await session.scalar(select(func.count()).select_from(model).where(*where))


async def get_note(session_factory, note_id) -> Note | None:
    async with session_factory() as session:
        return await session.get(Note, note_id)


def talo_meta() -> dict:
    result = generate_forms("talo", "substantiivi")
    return {
        "principal_forms": result.principal_forms,
        "forms_source": "fst",
        "forms_verified": True,
    }


class FakeResolve:
    """resolve_note_forms in reader/decks.py: records what it was called with."""

    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls: list[dict] = []

    async def __call__(self, client, breaker, model, lemma, pos, now, context=None):
        self.calls.append({"client": client, "breaker": breaker, "lemma": lemma, "pos": pos})
        if self.error is not None:
            raise self.error
        return ResolvedForms({"genetiivi": lemma + "n"}, "fst", True, pos), None


@pytest.fixture
def resolve(monkeypatch):
    fake = FakeResolve()
    monkeypatch.setattr("kielikaveri.reader.decks.resolve_note_forms", fake)
    return fake


# --- auth ----------------------------------------------------------------------------


async def test_decks_and_notes_need_init_data(client):
    for method, path in (
        ("GET", "/api/decks"),
        ("POST", "/api/decks"),
        ("PATCH", "/api/decks/x"),
        ("DELETE", "/api/decks/x"),
        ("GET", "/api/decks/x/notes"),
        ("GET", "/api/notes/search?q=a"),
        ("GET", "/api/notes/x"),
        ("PATCH", "/api/notes/x"),
        ("DELETE", "/api/notes/x"),
        ("POST", "/api/notes/x/move"),
        ("POST", "/api/notes/x/tts"),
    ):
        r = await client.request(method, path, json={})
        assert r.status == 401, (method, path)


# --- GET /api/decks ------------------------------------------------------------------


async def test_list_creates_общая_when_there_are_no_decks(client, session_factory):
    r = await client.get("/api/decks", headers=header())

    assert r.status == 200
    assert r.headers["Cache-Control"] == "no-store"
    [deck] = (await r.json())["decks"]
    assert deck["name"] == "Общая"
    assert deck == {
        "id": deck["id"],
        "name": "Общая",
        "notes": 0,
        "due": 0,
        "introduced": 0,
        "not_introduced": 0,
        "chapter": None,
    }
    # Once only.
    r = await client.get("/api/decks", headers=header())
    assert len((await r.json())["decks"]) == 1
    assert await count(session_factory, Deck, Deck.user_id == OWNER) == 1


async def test_list_counts_and_names_the_chapter(client, session_factory):
    _book_id, chapter_id, _ = await make_book(session_factory)
    await add_deck(session_factory, "a", name="Общая")
    await add_deck(session_factory, "ch", name="Kirja - гл.1", chapter_id=chapter_id)
    await add_deck(session_factory, "theirs", user_id=STRANGER)
    await add_note(session_factory, "n1", "a")
    await add_note(session_factory, "n2", "a")
    await add_card(session_factory, "c1", "n1")
    await add_card(session_factory, "c2", "n2", due=NOW + timedelta(days=3))
    await add_card(
        session_factory, "c3", "n2", type=CardType.inflection, status=CardStatus.not_introduced
    )

    r = await client.get("/api/decks", headers=header())

    decks = (await r.json())["decks"]
    assert [d["id"] for d in decks] == ["a", "ch"]
    assert all(set(d) == DECK_KEYS for d in decks)
    assert decks[0] == {
        "id": "a",
        "name": "Общая",
        "notes": 2,
        "due": 1,
        "introduced": 2,
        "not_introduced": 1,
        "chapter": None,
    }
    assert decks[1]["chapter"] == {"id": chapter_id, "book_title": "Kirja", "number": 1}
    assert decks[1]["notes"] == 0


# --- POST / PATCH / DELETE /api/decks ------------------------------------------------


async def test_create_deck_strips_the_name(client, session_factory):
    r = await client.post("/api/decks", json={"name": "  Matka  "}, headers=header())

    assert r.status == 200
    deck = (await r.json())["deck"]
    assert set(deck) == DECK_KEYS
    assert deck["name"] == "Matka"
    assert deck["notes"] == 0 and deck["chapter"] is None
    async with session_factory() as session:
        stored = await session.get(Deck, deck["id"])
    assert (stored.user_id, stored.name) == (OWNER, "Matka")


@pytest.mark.parametrize("name", ["", "   ", "x" * 65, " " + "x" * 65 + " "])
async def test_create_deck_refuses_a_bad_name(client, session_factory, name):
    r = await client.post("/api/decks", json={"name": name}, headers=header())

    assert (r.status, await r.json()) == (422, {"reason": "bad_name"})
    assert await count(session_factory, Deck) == 0


async def test_a_name_of_64_after_strip_is_fine(client):
    r = await client.post("/api/decks", json={"name": "  " + "ä" * 64 + "  "}, headers=header())
    assert r.status == 200
    assert (await r.json())["deck"]["name"] == "ä" * 64


@pytest.mark.parametrize("payload", [{}, {"name": 5}, {"name": None}, ["x"]])
async def test_create_deck_without_a_string_name_is_400(client, payload):
    r = await client.post("/api/decks", json=payload, headers=header())
    assert r.status == 400


async def test_create_deck_body_is_limited(client):
    r = await client.post("/api/decks", json={"name": "x" * 20_000}, headers=header())
    assert r.status == 413


async def test_rename_deck(client, session_factory):
    await add_deck(session_factory, "a", name="Vanha")
    await add_note(session_factory, "n", "a")

    r = await client.patch("/api/decks/a", json={"name": " Uusi "}, headers=header())

    assert r.status == 200
    deck = (await r.json())["deck"]
    assert set(deck) == DECK_KEYS
    assert (deck["id"], deck["name"], deck["notes"]) == ("a", "Uusi", 1)

    r = await client.patch("/api/decks/a", json={"name": "x" * 65}, headers=header())
    assert (r.status, await r.json()) == (422, {"reason": "bad_name"})
    async with session_factory() as session:
        assert (await session.get(Deck, "a")).name == "Uusi"


async def test_someone_elses_deck_is_404(client, session_factory):
    await add_deck(session_factory, "theirs", user_id=STRANGER, name="Heidän")
    await add_note(session_factory, "t", "theirs", user_id=STRANGER)

    assert (
        await client.patch("/api/decks/theirs", json={"name": "Minun"}, headers=header())
    ).status == 404
    assert (await client.delete("/api/decks/theirs", headers=header())).status == 404
    assert (await client.get("/api/decks/theirs/notes", headers=header())).status == 404
    assert (await client.get("/api/decks/missing/notes", headers=header())).status == 404
    async with session_factory() as session:
        assert (await session.get(Deck, "theirs")).name == "Heidän"
    assert await count(session_factory, Note) == 1


async def test_delete_deck_takes_its_words_and_history_and_nothing_else(client, session_factory):
    await add_deck(session_factory, "gone")
    await add_deck(session_factory, "kept")
    await add_deck(session_factory, "theirs", user_id=STRANGER)
    await add_note(session_factory, "g1", "gone")
    await add_note(session_factory, "g2", "gone")
    await add_card(session_factory, "gc1", "g1", reviews=2)
    await add_card(session_factory, "gc2", "g2", reviews=1)
    await add_note(session_factory, "k", "kept")
    await add_card(session_factory, "kc", "k", reviews=1)
    await add_note(session_factory, "t", "theirs", user_id=STRANGER)
    await add_card(session_factory, "tc", "t", user_id=STRANGER, reviews=1)

    r = await client.delete("/api/decks/gone", headers=header())

    assert (r.status, await r.json()) == (200, {"deleted_notes": 2})
    assert await count(session_factory, Deck, Deck.id == "gone") == 0
    assert await count(session_factory, Note, Note.id.in_(["g1", "g2"])) == 0
    assert await count(session_factory, Card, Card.id.in_(["gc1", "gc2"])) == 0
    assert await count(session_factory, Review, Review.card_id.in_(["gc1", "gc2"])) == 0
    for note_id, card_id in (("k", "kc"), ("t", "tc")):
        assert await count(session_factory, Note, Note.id == note_id) == 1
        assert await count(session_factory, Card, Card.id == card_id) == 1
        assert await count(session_factory, Review, Review.card_id == card_id) == 1
    assert await count(session_factory, Deck) == 2

    # A second time it is gone.
    assert (await client.delete("/api/decks/gone", headers=header())).status == 404


async def test_deleting_the_last_deck_brings_общая_back(client, session_factory):
    await add_deck(session_factory, "only", name="Ainoa")

    assert (await client.delete("/api/decks/only", headers=header())).status == 200

    r = await client.get("/api/decks", headers=header())
    assert [d["name"] for d in (await r.json())["decks"]] == ["Общая"]


async def test_a_double_delete_of_a_deck_deletes_once(client, session_factory):
    await add_deck(session_factory, "a")
    await add_note(session_factory, "n", "a")

    first, second = await asyncio.gather(
        client.delete("/api/decks/a", headers=header()),
        client.delete("/api/decks/a", headers=header()),
    )
    assert sorted([first.status, second.status]) == [200, 404]


# --- GET /api/decks/{id}/notes -------------------------------------------------------


async def test_deck_notes_newest_first_in_pages(client, session_factory):
    await add_deck(session_factory, "a")
    await add_deck(session_factory, "b")
    for i in range(5):
        await add_note(session_factory, f"n{i}", "a", created_at=NOW - timedelta(days=10 - i))
    await add_note(session_factory, "other", "b")

    r = await client.get("/api/decks/a/notes?offset=0&limit=2", headers=header())

    assert r.status == 200
    body = await r.json()
    assert body["total"] == 5
    assert body["notes"] == [
        {"id": "n4", "lemma": "sanan4", "translation_ru": "перевод-n4", "pos": "substantiivi"},
        {"id": "n3", "lemma": "sanan3", "translation_ru": "перевод-n3", "pos": "substantiivi"},
    ]
    r = await client.get("/api/decks/a/notes?offset=4&limit=2", headers=header())
    assert [n["id"] for n in (await r.json())["notes"]] == ["n0"]
    # Defaults: from the start, 50 at a time.
    r = await client.get("/api/decks/a/notes", headers=header())
    assert [n["id"] for n in (await r.json())["notes"]] == ["n4", "n3", "n2", "n1", "n0"]


async def test_deck_notes_pages_do_not_overlap_within_one_second(client, session_factory):
    await add_deck(session_factory, "a")
    for i in range(6):
        await add_note(session_factory, f"n{i}", "a", created_at=NOW)

    seen = []
    for offset in (0, 2, 4):
        r = await client.get(f"/api/decks/a/notes?offset={offset}&limit=2", headers=header())
        seen += [n["id"] for n in (await r.json())["notes"]]
    assert seen == ["n5", "n4", "n3", "n2", "n1", "n0"]


@pytest.mark.parametrize("query", ["offset=-1", "offset=x", "limit=0", "limit=1.5", "limit=-2"])
async def test_deck_notes_bad_paging_is_400(client, session_factory, query):
    await add_deck(session_factory, "a")
    r = await client.get(f"/api/decks/a/notes?{query}", headers=header())
    assert r.status == 400


# --- GET /api/notes/search -----------------------------------------------------------


async def search(client, query) -> list[dict]:
    r = await client.get("/api/notes/search", params={"q": query}, headers=header())
    assert r.status == 200
    return (await r.json())["notes"]


async def test_search_contract_and_case(client, session_factory):
    await add_deck(session_factory, "a", name="Общая")
    await add_deck(session_factory, "theirs", user_id=STRANGER)
    await add_note(session_factory, "fi", "a", lemma="Äiti", translation_ru="мама")
    await add_note(session_factory, "ru", "a", lemma="talo", translation_ru="Дом")
    await add_note(session_factory, "loose", None, lemma="äidinkieli", translation_ru="родной язык")
    await add_note(session_factory, "t", "theirs", user_id=STRANGER, lemma="äiti")

    assert await search(client, "ÄITI") == [
        {
            "id": "fi",
            "lemma": "Äiti",
            "translation_ru": "мама",
            "deck": {"id": "a", "name": "Общая"},
        }
    ]
    assert [n["id"] for n in await search(client, "äi")] == ["loose", "fi"]
    assert (await search(client, "äidin"))[0]["deck"] is None
    assert [n["id"] for n in await search(client, "ДОМ")] == ["ru"]
    assert [n["id"] for n in await search(client, "дом")] == ["ru"]
    assert await search(client, "") == []
    r = await client.get("/api/notes/search", headers=header())
    assert (r.status, await r.json()) == (200, {"notes": []})


async def test_search_order_exact_then_word_start_then_rest(client, session_factory):
    await add_deck(session_factory, "a")
    await add_note(session_factory, "inside", "a", lemma="sanakirja")
    await add_note(session_factory, "start", "a", lemma="kirjasto")
    await add_note(session_factory, "exact", "a", lemma="kirja")
    await add_note(session_factory, "puhe", "a", lemma="minä", meta={"puhekieli": "Kirja-mä"})

    assert [n["id"] for n in await search(client, "KIRJA")] == ["exact", "start", "puhe", "inside"]


async def test_search_answers_50_at_most(client, session_factory):
    await add_deck(session_factory, "a")
    for i in range(55):
        await add_note(session_factory, f"n{i}", "a", lemma=f"sana{i:02}")
    assert len(await search(client, "sana")) == 50


# --- GET /api/notes/{id} -------------------------------------------------------------


async def test_note_contract(client, session_factory):
    _book_id, chapter_id, _ = await make_book(session_factory)
    await add_deck(session_factory, "ch", name="Kirja - гл.1", chapter_id=chapter_id)
    meta = {
        "principal_forms": {"genetiivi": "talon", "partitiivi": "taloa"},
        "forms_source": "fst",
        "forms_verified": True,
        "puhekieli": "taloo",
    }
    await add_note(session_factory, "n", "ch", lemma="talo", meta=meta)
    await add_card(
        session_factory,
        "r",
        "n",
        due=DUE,
        reps=5,
        lapses=1,
        state=CardState.review,
    )
    await add_card(
        session_factory,
        "i",
        "n",
        type=CardType.inflection,
        form="partitiivi",
        status=CardStatus.not_introduced,
        due=DUE,
    )

    r = await client.get("/api/notes/n", headers=header())

    assert r.status == 200
    assert await r.json() == {
        "id": "n",
        "lemma": "talo",
        "pos": "substantiivi",
        "translation_ru": "перевод-n",
        "example_fi": "Esimerkki n.",
        "example_ru": "Пример n.",
        "puhekieli": "taloo",
        "forms": {"genetiivi": "talon", "partitiivi": "taloa"},
        "forms_verified": True,
        "deck": {
            "id": "ch",
            "name": "Kirja - гл.1",
            "chapter": {"id": chapter_id, "book_title": "Kirja", "number": 1},
        },
        "cards": [
            {
                "type": "recognition",
                "form": None,
                "status": "introduced",
                "state": "review",
                "due": "2026-12-03T09:30:00+00:00",
                "reps": 5,
                "lapses": 1,
            },
            {
                "type": "inflection",
                "form": "partitiivi",
                "status": "not_introduced",
                "state": "learning",
                "due": "2026-12-03T09:30:00+00:00",
                "reps": 0,
                "lapses": 0,
            },
        ],
    }


async def test_note_without_forms_or_deck(client, session_factory):
    await add_note(session_factory, "loose", None)

    body = await (await client.get("/api/notes/loose", headers=header())).json()

    assert set(body) == NOTE_KEYS
    assert (body["forms"], body["forms_verified"], body["deck"], body["puhekieli"]) == (
        None,
        False,
        None,
        None,
    )
    assert body["cards"] == []


async def test_a_chapter_deck_whose_book_is_gone_has_no_chapter(client, session_factory):
    book_id, chapter_id, _ = await make_book(session_factory)
    await add_deck(session_factory, "ch", chapter_id=chapter_id)
    await add_note(session_factory, "n", "ch")
    async with session_factory() as session:
        await delete_book(session, book_id)
        await session.commit()

    body = await (await client.get("/api/notes/n", headers=header())).json()
    assert body["deck"]["chapter"] is None


async def test_someone_elses_note_is_404_everywhere(client, session_factory, tts):
    await add_deck(session_factory, "mine")
    await add_deck(session_factory, "theirs", user_id=STRANGER)
    await add_note(session_factory, "t", "theirs", user_id=STRANGER, lemma="heidän")
    await add_card(session_factory, "tc", "t", user_id=STRANGER, reviews=1)

    for method, path, payload in (
        ("GET", "/api/notes/t", None),
        ("PATCH", "/api/notes/t", {"translation_ru": "моё"}),
        ("PATCH", "/api/notes/t", {"lemma": "minun"}),
        ("DELETE", "/api/notes/t", None),
        ("POST", "/api/notes/t/move", {"deck_id": "mine"}),
        ("POST", "/api/notes/t/tts", {"what": "lemma"}),
        ("GET", "/api/notes/missing", None),
    ):
        r = await client.request(method, path, json=payload, headers=header())
        assert r.status == 404, (method, path, payload)

    note = await get_note(session_factory, "t")
    assert (note.lemma, note.translation_ru, note.deck_id) == ("heidän", "перевод-t", "theirs")
    assert await count(session_factory, Review) == 1
    assert tts.calls == []


# --- PATCH /api/notes/{id} - translation ---------------------------------------------


async def test_edit_translation(client, session_factory, resolve):
    await add_deck(session_factory, "a")
    await add_note(session_factory, "n", "a")

    r = await client.patch("/api/notes/n", json={"translation_ru": " новый "}, headers=header())

    assert r.status == 200
    note = (await r.json())["note"]
    assert set(note) == NOTE_KEYS
    assert note["translation_ru"] == "новый"
    assert (await get_note(session_factory, "n")).translation_ru == "новый"
    assert resolve.calls == []


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"translation_ru": ""},
        {"translation_ru": "   "},
        {"translation_ru": 5},
        {"lemma": None},
        {"lemma": "talo", "translation_ru": "дом"},
    ],
)
async def test_edit_needs_one_non_empty_field(client, session_factory, payload):
    await add_deck(session_factory, "a")
    await add_note(session_factory, "n", "a")
    r = await client.patch("/api/notes/n", json=payload, headers=header())
    assert r.status == 400


# --- PATCH /api/notes/{id} - lemma ---------------------------------------------------


async def test_edit_lemma_goes_through_the_readers_breaker_and_client(
    make_client, session_factory, resolve
):
    client = await make_client()
    await add_deck(session_factory, "a")
    await add_note(session_factory, "n", "a", lemma="talo", meta=talo_meta())

    r = await client.patch("/api/notes/n", json={"lemma": "kissan"}, headers=header())

    assert r.status == 200
    note = (await r.json())["note"]
    # canonical_key: the FST's lemma of what was typed.
    assert note["lemma"] == "kissa"
    assert note["forms"] == {"genetiivi": "kissan"}
    [call] = resolve.calls
    app = client.server.app
    assert call["breaker"] is app[READER_BREAKER]
    assert call["client"] is app[LOOKUP_CLIENT]
    assert (call["lemma"], call["pos"]) == ("kissa", "substantiivi")


async def test_edit_lemma_with_the_real_fst_resyncs_the_form_cards(client, session_factory):
    await add_deck(session_factory, "a")
    await add_note(session_factory, "n", "a", lemma="talo", meta=talo_meta())
    await add_card(session_factory, "r", "n", reviews=1)

    r = await client.patch("/api/notes/n", json={"lemma": "kissa"}, headers=header())

    assert r.status == 200
    note = (await r.json())["note"]
    assert note["lemma"] == "kissa"
    assert note["forms"]["genetiivi"] == "kissan"
    assert note["forms_verified"] is True
    # The recognition card and its history stay.
    assert await count(session_factory, Review, Review.card_id == "r") == 1


async def test_edit_lemma_duplicate_is_409(client, session_factory, resolve):
    await add_deck(session_factory, "a")
    await add_deck(session_factory, "b")
    await add_note(session_factory, "n", "a", lemma="talo")
    await add_note(session_factory, "other", "a", lemma="kissa")
    # The same word in another deck is no clash.
    await add_note(session_factory, "elsewhere", "b", lemma="koira")

    r = await client.patch("/api/notes/n", json={"lemma": "kissan"}, headers=header())
    # The lemma as settled (canonical_key), for the bot's clash text.
    assert (r.status, await r.json()) == (409, {"reason": "duplicate", "lemma": "kissa"})
    assert (await get_note(session_factory, "n")).lemma == "talo"
    assert resolve.calls == []

    r = await client.patch("/api/notes/n", json={"lemma": "koira"}, headers=header())
    assert r.status == 200


async def test_edit_lemma_lost_race_is_409(client, session_factory, monkeypatch):
    await add_deck(session_factory, "a")
    await add_note(session_factory, "n", "a", lemma="talo")

    async def steal(client, breaker, model, lemma, pos, now, context=None):
        await add_note(session_factory, "thief", "a", lemma=lemma)
        return ResolvedForms({}, "llm", False, pos), None

    monkeypatch.setattr("kielikaveri.reader.decks.resolve_note_forms", steal)

    r = await client.patch("/api/notes/n", json={"lemma": "kissa"}, headers=header())
    assert (r.status, await r.json()) == (409, {"reason": "duplicate", "lemma": "kissa"})
    assert (await get_note(session_factory, "n")).lemma == "talo"


async def test_edit_lemma_without_a_key_is_503_and_changes_nothing(
    make_client, session_factory, resolve
):
    client = await make_client(openai_api_key="")
    await add_deck(session_factory, "a")
    await add_note(session_factory, "n", "a", lemma="talo", meta=talo_meta())

    r = await client.patch("/api/notes/n", json={"lemma": "kissa"}, headers=header())

    assert (r.status, await r.json()) == (503, {"reason": "no_openai"})
    assert resolve.calls == []
    note = await get_note(session_factory, "n")
    assert note.lemma == "talo" and note.meta == talo_meta()
    # A translation needs no key.
    r = await client.patch("/api/notes/n", json={"translation_ru": "дом"}, headers=header())
    assert r.status == 200


async def test_a_pattern_needs_no_llm_nor_a_key(make_client, session_factory, resolve):
    client = await make_client(openai_api_key="")
    await add_deck(session_factory, "a")
    await add_note(
        session_factory, "p", "a", lemma="olla jonkun mielestä", pos=None, kind=NoteKind.pattern
    )

    r = await client.patch("/api/notes/p", json={"lemma": "olla mieltä"}, headers=header())

    assert r.status == 200
    assert (await r.json())["note"]["lemma"] == "olla mieltä"
    assert resolve.calls == []


@pytest.mark.parametrize(
    ("error", "status", "reason"),
    [
        (CircuitOpenError("rate"), 503, "rate"),
        (openai.APIConnectionError(request=MagicMock()), 502, "openai"),
    ],
)
async def test_edit_lemma_llm_failure_changes_nothing(
    client, session_factory, monkeypatch, error, status, reason
):
    fake = FakeResolve(error)
    monkeypatch.setattr("kielikaveri.reader.decks.resolve_note_forms", fake)
    await add_deck(session_factory, "a")
    await add_note(session_factory, "n", "a", lemma="talo", meta=talo_meta())

    r = await client.patch("/api/notes/n", json={"lemma": "kissa"}, headers=header())

    assert (r.status, await r.json()) == (status, {"reason": reason})
    note = await get_note(session_factory, "n")
    assert note.lemma == "talo" and note.meta == talo_meta()


async def test_a_tripped_readers_breaker_refuses_the_lemma(
    make_client, session_factory, monkeypatch
):
    # resolve_note_forms calls the LLM only for some words, and checks the
    # breaker it is handed before each call: the reader's, spent here.
    client = await make_client(reader_breaker_max_calls=1)
    breaker = client.server.app[READER_BREAKER]
    breaker.check(datetime.now(UTC))
    seen = []

    async def resolve(client_, breaker_, model, lemma, pos, now, context=None):
        seen.append(breaker_)
        breaker_.check(now)
        return ResolvedForms({}, "llm", False, pos), None

    monkeypatch.setattr("kielikaveri.reader.decks.resolve_note_forms", resolve)
    await add_deck(session_factory, "a")
    await add_note(session_factory, "n", "a", lemma="talo")

    r = await client.patch("/api/notes/n", json={"lemma": "kissa"}, headers=header())

    assert (r.status, await r.json()) == (503, {"reason": "rate"})
    assert seen == [breaker]
    assert (await get_note(session_factory, "n")).lemma == "talo"


# --- POST /api/notes/{id}/move -------------------------------------------------------


async def test_move_keeps_cards_and_reviews(client, session_factory):
    await add_deck(session_factory, "a")
    await add_deck(session_factory, "b", name="Toinen")
    await add_note(session_factory, "n", "a")
    await add_card(session_factory, "c", "n", reviews=2)

    r = await client.post("/api/notes/n/move", json={"deck_id": "b"}, headers=header())

    assert r.status == 200
    note = (await r.json())["note"]
    assert set(note) == NOTE_KEYS
    assert note["deck"] == {"id": "b", "name": "Toinen", "chapter": None}
    assert (await get_note(session_factory, "n")).deck_id == "b"
    assert await count(session_factory, Card, Card.note_id == "n") == 1
    assert await count(session_factory, Review, Review.card_id == "c") == 2


async def test_move_conflict_and_same_deck(client, session_factory):
    await add_deck(session_factory, "a")
    await add_deck(session_factory, "b")
    await add_note(session_factory, "n", "a", lemma="talo")
    await add_note(session_factory, "there", "b", lemma="talo")

    r = await client.post("/api/notes/n/move", json={"deck_id": "b"}, headers=header())
    assert (r.status, await r.json()) == (409, {"reason": "duplicate_in_target"})
    assert (await get_note(session_factory, "n")).deck_id == "a"

    r = await client.post("/api/notes/n/move", json={"deck_id": "a"}, headers=header())
    assert r.status == 200
    assert (await r.json())["note"]["deck"]["id"] == "a"


async def test_move_into_someone_elses_deck_is_404(client, session_factory):
    await add_deck(session_factory, "a")
    await add_deck(session_factory, "theirs", user_id=STRANGER)
    await add_note(session_factory, "n", "a")

    for deck_id in ("theirs", "missing"):
        r = await client.post("/api/notes/n/move", json={"deck_id": deck_id}, headers=header())
        assert r.status == 404
    r = await client.post("/api/notes/n/move", json={"deck_id": 5}, headers=header())
    assert r.status == 400
    assert (await get_note(session_factory, "n")).deck_id == "a"


async def test_a_note_in_no_deck_can_be_moved_into_one(client, session_factory):
    await add_deck(session_factory, "a")
    await add_note(session_factory, "loose", None)
    r = await client.post("/api/notes/loose/move", json={"deck_id": "a"}, headers=header())
    assert r.status == 200
    assert (await get_note(session_factory, "loose")).deck_id == "a"


# --- DELETE /api/notes/{id} ----------------------------------------------------------


async def test_delete_note(client, session_factory):
    await add_deck(session_factory, "a")
    await add_note(session_factory, "n", "a")
    await add_note(session_factory, "kept", "a")
    await add_card(session_factory, "c", "n", reviews=2)
    await add_card(session_factory, "kc", "kept", reviews=1)

    r = await client.delete("/api/notes/n", headers=header())

    assert r.status == 204
    assert await count(session_factory, Note) == 1
    assert await count(session_factory, Card) == 1
    assert await count(session_factory, Review) == 1
    assert (await client.delete("/api/notes/n", headers=header())).status == 404


async def test_a_double_delete_gives_one_204_and_one_404(client, session_factory):
    await add_deck(session_factory, "a")
    await add_note(session_factory, "n", "a")
    await add_card(session_factory, "c", "n", reviews=1)

    first, second = await asyncio.gather(
        client.delete("/api/notes/n", headers=header()),
        client.delete("/api/notes/n", headers=header()),
    )

    assert sorted([first.status, second.status]) == [204, 404]
    assert await count(session_factory, Note) == 0


# --- POST /api/notes/{id}/tts --------------------------------------------------------


async def test_tts_of_lemma_and_example(client, session_factory, tts):
    await add_deck(session_factory, "a")
    await add_note(session_factory, "n", "a", lemma="talo", example_fi="Talo on iso.")

    for what in ("lemma", "example"):
        r = await client.post("/api/notes/n/tts", json={"what": what}, headers=header())
        assert r.status == 200
        assert r.content_type == "audio/wav"
        assert (await r.read())[:4] == WAV[:4]
    assert [call["input"] for call in tts.calls] == ["talo", "Talo on iso."]

    # Cached: the second time costs nothing.
    await client.post("/api/notes/n/tts", json={"what": "lemma"}, headers=header())
    assert len(tts.calls) == 2


@pytest.mark.parametrize("payload", [{}, {"what": "translation"}, {"what": None}])
async def test_tts_needs_lemma_or_example(client, session_factory, payload):
    await add_deck(session_factory, "a")
    await add_note(session_factory, "n", "a")
    r = await client.post("/api/notes/n/tts", json=payload, headers=header())
    assert r.status == 400


async def test_tts_errors_are_those_of_the_reader(make_client, session_factory, tts):
    await add_deck(session_factory, "a")
    await add_note(session_factory, "n", "a")
    await add_note(session_factory, "m", "a")

    no_key = await make_client(openai_api_key="")
    r = await no_key.post("/api/notes/n/tts", json={"what": "lemma"}, headers=header())
    assert (r.status, await r.json()) == (503, {"reason": "no_openai"})
    assert tts.calls == []

    client = await make_client(reader_breaker_max_calls=1)
    r = await client.post("/api/notes/n/tts", json={"what": "lemma"}, headers=header())
    assert r.status == 200
    r = await client.post("/api/notes/m/tts", json={"what": "lemma"}, headers=header())
    assert (r.status, await r.json()) == (503, {"reason": "rate"})

    tts.error = openai.APIConnectionError(request=MagicMock())
    fresh = await make_client()
    r = await fresh.post("/api/notes/n/tts", json={"what": "example"}, headers=header())
    assert (r.status, await r.json()) == (502, {"reason": "openai"})


# --- logs ----------------------------------------------------------------------------


async def test_logs_carry_ids_and_never_text(make_client, session_factory, caplog):
    caplog.set_level(logging.DEBUG)
    # aiosqlite at DEBUG prints SQL parameters; production keeps it at INFO.
    logging.getLogger("aiosqlite").setLevel(logging.INFO)
    client = await make_client()
    await add_deck(session_factory, "a", name="Salainen pakka")
    await add_deck(session_factory, "b", name="Toinen salainen")
    await add_note(
        session_factory,
        "n",
        "a",
        lemma="talo",
        translation_ru="секретный дом",
        meta=talo_meta(),
    )
    await add_note(session_factory, "o", "a", lemma="koira", translation_ru="секретный пёс")
    await add_note(session_factory, "m", "b", lemma="lintu", translation_ru="секретная птица")
    # talo_meta() above ran the FST in the test itself, not in the reader.
    caplog.clear()

    created = await client.post("/api/decks", json={"name": "Uusi salaisuus"}, headers=header())
    new_id = (await created.json())["deck"]["id"]
    await client.get("/api/decks", headers=header())
    await client.patch(f"/api/decks/{new_id}", json={"name": "Nimi salaisuus"}, headers=header())
    await client.get("/api/decks/a/notes", headers=header())
    await client.get("/api/notes/search", params={"q": "секрет"}, headers=header())
    await client.get("/api/notes/n", headers=header())
    await client.patch("/api/notes/n", json={"translation_ru": "тайный дом"}, headers=header())
    clash = await client.patch("/api/notes/n", json={"lemma": "koiran"}, headers=header())
    assert clash.status == 409
    edited = await client.patch("/api/notes/n", json={"lemma": "kissan"}, headers=header())
    assert (await edited.json())["note"]["lemma"] == "kissa"
    await client.post("/api/notes/n/move", json={"deck_id": "b"}, headers=header())
    await client.post("/api/notes/n/tts", json={"what": "lemma"}, headers=header())
    await client.delete("/api/notes/m", headers=header())
    await client.delete(f"/api/decks/{new_id}", headers=header())

    texts = [
        "talo",
        "koira",
        "kissa",
        "секрет",
        "тайный",
        "птица",
        "пёс",
        "lintu",
        "Salainen",
        "salainen",
        "salaisuus",
        "Esimerkki",
        "Пример",
    ]
    messages = [r.getMessage() for r in caplog.records]
    for message in messages:
        for piece in texts:
            assert piece not in message, message
    events = {
        f["event"]
        for f in map(log_fields, messages)
        if f.get("source") == "reader" and "event" in f
    }
    assert {
        "decks.list",
        "decks.create",
        "decks.rename",
        "decks.delete",
        "decks.notes",
        "decks.search",
        "decks.note",
        "decks.note_edit",
        "decks.lemma_clash",
        "decks.move",
        "decks.note_delete",
    } <= events
    # The lemma went through ingest and the FST: their lines are cut.
    assert any("(details hidden)" in m for m in messages)
