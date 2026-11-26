"""/api/learn/* (plan 2026-11-20-kirjakaveri-reader, "П5", У1): review
sessions in the Mini App, the queue kept on the server. OpenAI is a fake."""

# The database fixture comes from test_reader_lookup; tests naming it as a
# parameter "redefine" the import for ruff.
# ruff: noqa: F811

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer
from bot_helpers import make_settings
from conftest import log_fields
from sqlalchemy import delete, func, select, text
from test_reader import OWNER, STRANGER, TOKEN, header
from test_reader_lookup import (  # noqa: F401 - fixtures among them
    FakeOpenAI,
    book,
    session_factory,
)
from test_reader_lookup import body as word_body
from test_reader_tts import WAV, FakeTTS

from kielikaveri.db.models import Card, CardState, CardType, Deck, Note, NoteKind, Review
from kielikaveri.grammar import FORM_TASKS
from kielikaveri.learn_core import EMPTY_ALL_TEXT, NO_PRODUCTION_TEXT, record_review
from kielikaveri.reader.app import LEARN_SESSIONS, LOOKUP_CLIENT, TTS_CLIENT, WARMUP, make_app
from kielikaveri.srs.scheduler import Rating

NOW = datetime.now(UTC)
PAST = NOW - timedelta(days=1)
CARD_KEYS = {"id", "type", "lemma", "translation_ru", "example_fi", "example_ru"}
# A quizzable form, and its task.
FORM = next(iter(FORM_TASKS))
TASK = FORM_TASKS[FORM]


def build_app(session_factory, tts, tmp_path, **overrides):
    settings = make_settings(
        bot_token=TOKEN,
        whitelist_user_ids=f"{OWNER},{STRANGER}",
        reader_tts_cache_dir=str(tmp_path / "tts_cache"),
        **overrides,
    )
    app = make_app(session_factory, settings, MagicMock())
    app[TTS_CLIENT] = tts
    app[LOOKUP_CLIENT] = FakeOpenAI()
    return app


@pytest.fixture
def tts():
    return FakeTTS()


@pytest.fixture
async def make_client(session_factory, tts, tmp_path):
    clients = []

    async def make(**overrides):
        client = TestClient(TestServer(build_app(session_factory, tts, tmp_path, **overrides)))
        await client.start_server()
        clients.append(client)
        return client

    yield make
    for client in clients:
        await client.close()


@pytest.fixture
async def client(make_client):
    return await make_client()


async def add_deck(session_factory, user_id=OWNER, name="Общая") -> str:
    async with session_factory() as session:
        deck = Deck(user_id=user_id, name=name)
        session.add(deck)
        await session.commit()
        return deck.id


async def add_note(
    session_factory, deck_id, note_id, user_id=OWNER, lemma=None, meta=None, created_at=None
) -> str:
    lemma = lemma or f"sana{note_id}"
    async with session_factory() as session:
        session.add(
            Note(
                id=note_id,
                user_id=user_id,
                deck_id=deck_id,
                lemma=lemma,
                translation_ru=f"слово-{note_id}",
                example_fi=f"Tämä on {lemma} lause.",
                example_ru=f"Это предложение {note_id}.",
                kind=NoteKind.word,
                meta=meta or {},
                created_at=created_at or NOW - timedelta(days=30),
            )
        )
        await session.commit()
    return note_id


async def add_card(session_factory, card_id, note_id, user_id=OWNER, **columns) -> str:
    columns = {"type": CardType.recognition, "due": PAST, **columns}
    async with session_factory() as session:
        session.add(Card(id=card_id, note_id=note_id, user_id=user_id, **columns))
        await session.commit()
    return card_id


async def add_words(session_factory, count, deck_id=None, user_id=OWNER, **columns) -> list[str]:
    """`count` notes with one recognition card each, in queue order."""
    deck_id = deck_id or await add_deck(session_factory, user_id)
    ids = []
    for i in range(count):
        note = await add_note(
            session_factory,
            deck_id,
            f"{user_id}-n{i}",
            user_id=user_id,
            created_at=NOW - timedelta(days=30 - i),
        )
        ids.append(await add_card(session_factory, f"{user_id}-c{i}", note, user_id, **columns))
    return ids


async def post(client, path, payload, **header_kwargs):
    return await client.post(f"/api/learn/{path}", json=payload, headers=header(**header_kwargs))


async def start(client, side="mix", deck_id=None, **header_kwargs) -> dict:
    r = await post(client, "start", {"deck_id": deck_id, "side": side}, **header_kwargs)
    assert r.status == 200
    return await r.json()


async def rate(client, session, card_id, rating=3):
    return await post(client, "review", {"session": session, "card_id": card_id, "rating": rating})


async def review_rows(session_factory) -> list[Review]:
    async with session_factory() as session:
        return list((await session.scalars(select(Review))).all())


async def raw_srs(session_factory, card_id) -> tuple:
    """The card's SRS columns exactly as SQLite stores them."""
    async with session_factory() as session:
        row = await session.execute(
            text(
                "SELECT state, due, stability, difficulty, reps, lapses, step "
                "FROM cards WHERE id = :id"
            ),
            {"id": card_id},
        )
        return tuple(row.one())


# --- decks -------------------------------------------------------------------------


async def test_decks_lists_the_users_decks_only(client, session_factory):
    first = await add_deck(session_factory, name="Общая")
    second = await add_deck(session_factory, name="Kirja - гл.1")
    await add_deck(session_factory, user_id=STRANGER, name="Чужая")

    r = await client.get("/api/learn/decks", headers=header())

    assert r.status == 200
    assert r.headers["Cache-Control"] == "no-store"
    assert await r.json() == {
        "decks": [{"id": first, "name": "Общая"}, {"id": second, "name": "Kirja - гл.1"}]
    }


async def test_learn_needs_init_data(client):
    assert (await client.get("/api/learn/decks")).status == 401
    r = await client.post("/api/learn/start", json={"deck_id": None, "side": "mix"})
    assert r.status == 401


# --- start ---------------------------------------------------------------------------


async def test_start_gives_the_first_card(client, session_factory):
    ids = await add_words(session_factory, 2)

    body = await start(client)

    assert body["state"] == "card"
    assert isinstance(body["session"], str) and body["session"]
    assert body["progress"] == {"done": 0, "total": 2}
    assert body["can_undo"] is False
    assert body["card"] == {
        "id": ids[0],
        "type": "recognition",
        "lemma": "sana1-n0",
        "translation_ru": "слово-1-n0",
        "example_fi": "Tämä on sana1-n0 lause.",
        "example_ru": "Это предложение 1-n0.",
    }


async def test_an_inflection_card_carries_its_form(client, session_factory):
    deck = await add_deck(session_factory)
    note = await add_note(session_factory, deck, "n", meta={"principal_forms": {FORM: "sanaksi"}})
    card = await add_card(session_factory, "c", note, type=CardType.inflection, form=FORM)

    body = await start(client, side="fi")

    assert body["card"]["id"] == card
    assert body["card"]["type"] == "inflection"
    assert body["card"]["cue"] == TASK.cue
    assert body["card"]["form_answer"] == "sanaksi"
    assert body["card"]["form_label"] == TASK.label


async def test_a_form_without_a_task_shows_the_lemma_on_both_sides(client, session_factory):
    deck = await add_deck(session_factory)
    note = await add_note(session_factory, deck, "n", lemma="hakea", meta={})
    await add_card(session_factory, "c", note, type=CardType.inflection, form="nut_partisiippi")

    body = await start(client, side="fi")

    assert set(body["card"]) == CARD_KEYS | {"cue", "form_answer", "form_label"}
    assert body["card"]["cue"] is None
    assert body["card"]["form_answer"] == "hakea"
    assert body["card"]["form_label"] is None


async def test_start_with_nothing_due_is_empty(client, session_factory):
    assert await start(client) == {"state": "empty", "text": EMPTY_ALL_TEXT}


async def test_russian_side_says_it_is_still_closed(client, session_factory):
    await add_words(session_factory, 1)

    body = await start(client, side="ru")

    assert body["state"] == "empty"
    assert NO_PRODUCTION_TEXT in body["text"]


async def test_start_narrows_to_the_deck(client, session_factory):
    await add_words(session_factory, 1)
    other = await add_deck(session_factory, name="Toinen")
    note = await add_note(session_factory, other, "x")
    card = await add_card(session_factory, "x-card", note)

    body = await start(client, deck_id=other)

    assert body["card"]["id"] == card
    assert body["progress"]["total"] == 1


async def test_someone_elses_deck_is_404(client, session_factory):
    stranger_deck = await add_deck(session_factory, user_id=STRANGER)
    await add_words(session_factory, 1, deck_id=stranger_deck, user_id=STRANGER)

    r = await post(client, "start", {"deck_id": stranger_deck, "side": "mix"})
    assert r.status == 404
    r = await post(client, "start", {"deck_id": "no-such-deck", "side": "mix"})
    assert r.status == 404


@pytest.mark.parametrize(
    "payload",
    [
        {"deck_id": None, "side": "en"},
        {"deck_id": None},
        {"deck_id": 5, "side": "mix"},
        [],
    ],
)
async def test_bad_start_body_is_400(client, payload):
    assert (await post(client, "start", payload)).status == 400


async def test_unparsable_body_is_400(client):
    r = await client.post("/api/learn/start", data=b"{not json", headers=header())
    assert r.status == 400


async def test_a_huge_body_is_refused(client):
    r = await client.post(
        "/api/learn/start", data=b'{"side": "' + b"m" * 64 * 1024 + b'"}', headers=header()
    )
    assert r.status == 413


# --- review --------------------------------------------------------------------------


async def test_review_records_and_moves_on(client, session_factory):
    ids = await add_words(session_factory, 2)
    session = (await start(client))["session"]

    r = await rate(client, session, ids[0], 3)

    assert r.status == 200
    body = await r.json()
    assert body["state"] == "card"
    assert body["card"]["id"] == ids[1]
    assert body["progress"] == {"done": 1, "total": 2}
    assert body["can_undo"] is True
    assert "session" not in body
    [row] = await review_rows(session_factory)
    assert (row.card_id, row.user_id, row.rating) == (ids[0], OWNER, 3)
    async with session_factory() as db:
        assert (await db.get(Card, ids[0])).reps == 1


async def test_the_last_review_ends_the_session(client, session_factory):
    [card] = await add_words(session_factory, 1)
    session = (await start(client))["session"]

    body = await (await rate(client, session, card, 4)).json()

    assert body == {
        "state": "end",
        "reviewed": 1,
        "remaining": 0,
        "text": "Сессия окончена: 1 карточек пройдено - всё на сегодня!",
        "can_undo": True,
    }


@pytest.mark.parametrize("rating", [0, 5, "3", 3.0, True, None])
async def test_rating_outside_1_to_4_is_400(client, session_factory, rating):
    [card] = await add_words(session_factory, 1)
    session = (await start(client))["session"]

    assert (await rate(client, session, card, rating)).status == 400
    assert await review_rows(session_factory) == []


@pytest.mark.parametrize("missing", ["session", "card_id"])
async def test_review_without_its_ids_is_400(client, session_factory, missing):
    [card] = await add_words(session_factory, 1)
    payload = {"session": (await start(client))["session"], "card_id": card, "rating": 3}
    del payload[missing]
    assert (await post(client, "review", payload)).status == 400


async def test_two_parallel_reviews_record_one(client, session_factory):
    ids = await add_words(session_factory, 2)
    session = (await start(client))["session"]

    first, second = await asyncio.gather(
        rate(client, session, ids[0], 3), rate(client, session, ids[0], 3)
    )

    assert sorted([first.status, second.status]) == [200, 409]
    stale = await (first if first.status == 409 else second).json()
    assert stale["reason"] == "stale"
    assert stale["card"]["id"] == ids[1]
    assert len(await review_rows(session_factory)) == 1


async def test_a_card_not_at_the_head_is_stale_and_writes_nothing(client, session_factory):
    ids = await add_words(session_factory, 3)
    session = (await start(client))["session"]
    before = await raw_srs(session_factory, ids[1])

    r = await rate(client, session, ids[1], 3)

    assert r.status == 409
    assert await r.json() == {
        "reason": "stale",
        "state": "card",
        "card": (await start_card_json(session_factory, ids[0])),
        "progress": {"done": 0, "total": 3},
        "can_undo": False,
    }
    assert await review_rows(session_factory) == []
    assert await raw_srs(session_factory, ids[1]) == before


async def start_card_json(session_factory, card_id) -> dict:
    async with session_factory() as db:
        card = await db.get(Card, card_id)
        note = await db.get(Note, card.note_id)
    return {
        "id": card.id,
        "type": "recognition",
        "lemma": note.lemma,
        "translation_ru": note.translation_ru,
        "example_fi": note.example_fi,
        "example_ru": note.example_ru,
    }


async def test_after_the_end_a_review_is_stale_with_the_end(client, session_factory):
    [card] = await add_words(session_factory, 1)
    session = (await start(client))["session"]
    await rate(client, session, card)

    r = await rate(client, session, card)

    assert r.status == 409
    body = await r.json()
    assert (body["reason"], body["state"], body["can_undo"]) == ("stale", "end", True)
    assert len(await review_rows(session_factory)) == 1


async def test_a_replaced_session_is_gone(client, session_factory):
    ids = await add_words(session_factory, 2)
    old = (await start(client))["session"]
    new = (await start(client))["session"]
    assert old != new

    for path, payload in [
        ("review", {"session": old, "card_id": ids[0], "rating": 3}),
        ("undo", {"session": old}),
        ("debt", {"session": old, "action": "batch"}),
        ("tts", {"session": old, "card_id": ids[0]}),
    ]:
        r = await post(client, path, payload)
        assert r.status == 409, path
        assert await r.json() == {"reason": "session_gone"}
    assert await review_rows(session_factory) == []
    assert (await rate(client, new, ids[0])).status == 200


async def test_without_a_session_it_is_gone(client, session_factory):
    [card] = await add_words(session_factory, 1)
    r = await rate(client, "nope", card)
    assert r.status == 409
    assert await r.json() == {"reason": "session_gone"}


async def test_an_empty_start_ends_the_old_session(client, session_factory):
    [card] = await add_words(session_factory, 1)
    old = (await start(client))["session"]
    assert (await start(client, side="ru"))["state"] == "empty"
    assert await (await rate(client, old, card)).json() == {"reason": "session_gone"}


async def test_someone_elses_session_is_not_yours(client, session_factory):
    await add_words(session_factory, 1)
    stranger_cards = await add_words(session_factory, 1, user_id=STRANGER)
    mine = (await start(client))["session"]

    r = await post(
        client, "review", {"session": mine, "card_id": stranger_cards[0], "rating": 3}, user_id=2
    )
    assert await r.json() == {"reason": "session_gone"}


async def test_someone_elses_card_is_404(client, session_factory):
    await add_words(session_factory, 1)
    [stranger_card] = await add_words(session_factory, 1, user_id=STRANGER)
    session = (await start(client))["session"]

    assert (await rate(client, session, stranger_card)).status == 404
    assert (await rate(client, session, "no-such-card")).status == 404
    r = await post(client, "tts", {"session": session, "card_id": stranger_card})
    assert r.status == 404
    assert await review_rows(session_factory) == []


async def test_the_minutes_limit_ends_the_session_after_a_review(client, session_factory):
    ids = await add_words(session_factory, 3)
    session = (await start(client))["session"]
    live = client.server.app[LEARN_SESSIONS][OWNER]
    live.started_at -= timedelta(minutes=11)

    body = await (await rate(client, session, ids[0])).json()

    assert body["state"] == "end"
    assert (body["reviewed"], body["remaining"]) == (1, 2)
    assert body["text"].startswith("Сессия окончена: 1 карточек пройдено, осталось 2")
    assert (await rate(client, session, ids[1])).status == 409
    assert len(await review_rows(session_factory)) == 1


async def test_a_card_deleted_mid_session_is_skipped(client, session_factory):
    ids = await add_words(session_factory, 3)
    session = (await start(client))["session"]
    async with session_factory() as db:
        await db.execute(delete(Card).where(Card.id.in_([ids[0], ids[1]])))
        await db.commit()

    r = await rate(client, session, ids[0])

    assert r.status == 200
    body = await r.json()
    assert body["card"]["id"] == ids[2]
    assert body["progress"] == {"done": 0, "total": 1}
    assert await review_rows(session_factory) == []


async def test_a_skipped_card_is_logged_as_the_readers(client, session_factory, caplog):
    caplog.set_level(logging.INFO)
    ids = await add_words(session_factory, 2)
    session = (await start(client))["session"]
    await rate(client, session, ids[0])
    async with session_factory() as db:
        await db.execute(delete(Card).where(Card.id == ids[1]))
        await db.commit()

    await rate(client, session, ids[0])

    skipped = [
        log_fields(r.getMessage()) for r in caplog.records if "learn.skip_deleted" in r.getMessage()
    ]
    assert skipped and all(f["source"] == "reader" for f in skipped)
    assert {f["card_id"] for f in skipped} == {ids[1]}


async def test_a_deleted_last_card_ends_the_session(client, session_factory):
    ids = await add_words(session_factory, 2)
    session = (await start(client))["session"]
    await rate(client, session, ids[0])
    async with session_factory() as db:
        await db.execute(delete(Card).where(Card.id == ids[1]))
        await db.commit()

    # Re-read as stale (e.g. a double tap of the previous card).
    r = await rate(client, session, ids[0])
    body = await r.json()
    assert (r.status, body["state"], body["reviewed"], body["remaining"]) == (409, "end", 1, 0)


# --- undo ----------------------------------------------------------------------------


async def test_undo_restores_the_card_and_deletes_the_review(client, session_factory):
    ids = await add_words(session_factory, 2)
    # A card with a history, so every column has a value to restore.
    async with session_factory() as db:
        card = await db.get(Card, ids[0])
        card.state = CardState.review
        card.stability = 4.123456789
        card.difficulty = 6.5
        card.reps = 5
        card.lapses = 1
        card.step = None
        db.add(Review(card_id=card.id, user_id=OWNER, rating=3, reviewed_at=PAST - timedelta(5)))
        await db.commit()
    # Finnish: its stability has opened a production card, left out here.
    session = (await start(client, side="fi"))["session"]
    before = await raw_srs(session_factory, ids[0])

    await rate(client, session, ids[0], 1)
    assert await raw_srs(session_factory, ids[0]) != before
    r = await post(client, "undo", {"session": session})

    assert r.status == 200
    assert await r.json() == {
        "state": "card",
        "card": await start_card_json(session_factory, ids[0]),
        "progress": {"done": 0, "total": 2},
        "can_undo": False,
    }
    assert await raw_srs(session_factory, ids[0]) == before
    [old] = await review_rows(session_factory)
    assert old.reviewed_at < NOW
    # The card is the head again: it can be rated anew.
    body = await (await rate(client, session, ids[0], 3)).json()
    assert body["card"]["id"] == ids[1]


async def test_only_the_last_rating_can_be_undone(client, session_factory):
    ids = await add_words(session_factory, 3)
    session = (await start(client))["session"]
    r = await post(client, "undo", {"session": session})
    assert (r.status, await r.json()) == (409, {"reason": "nothing_to_undo"})

    await rate(client, session, ids[0])
    await rate(client, session, ids[1])
    assert (await post(client, "undo", {"session": session})).status == 200
    r = await post(client, "undo", {"session": session})
    assert (r.status, await r.json()) == (409, {"reason": "nothing_to_undo"})
    assert [row.card_id for row in await review_rows(session_factory)] == [ids[0]]


async def test_undo_from_the_end_screen(client, session_factory):
    [card] = await add_words(session_factory, 1)
    session = (await start(client))["session"]
    assert (await (await rate(client, session, card)).json())["state"] == "end"

    body = await (await post(client, "undo", {"session": session})).json()

    assert (body["state"], body["card"]["id"], body["can_undo"]) == ("card", card, False)
    assert body["progress"] == {"done": 0, "total": 1}
    assert await review_rows(session_factory) == []
    assert (await rate(client, session, card)).status == 200


async def test_undo_refuses_a_card_rated_again_elsewhere(client, session_factory):
    ids = await add_words(session_factory, 2)
    session = (await start(client))["session"]
    await rate(client, session, ids[0], 1)
    # Meanwhile the bot rates the same card.
    async with session_factory() as db:
        card = await db.get(Card, ids[0])
        await record_review(db, card, Rating.Good, datetime.now(UTC))
    after_bot = await raw_srs(session_factory, ids[0])

    r = await post(client, "undo", {"session": session})

    assert (r.status, await r.json()) == (409, {"reason": "nothing_to_undo"})
    assert await raw_srs(session_factory, ids[0]) == after_bot
    assert len(await review_rows(session_factory)) == 2
    # The session goes on where it was.
    body = await (await rate(client, session, ids[1])).json()
    assert body["state"] == "end"


async def test_undo_refuses_when_a_newer_review_exists(client, session_factory):
    ids = await add_words(session_factory, 2)
    session = (await start(client))["session"]
    await rate(client, session, ids[0])
    async with session_factory() as db:
        db.add(Review(card_id=ids[0], user_id=OWNER, rating=3, reviewed_at=datetime.now(UTC)))
        await db.commit()

    r = await post(client, "undo", {"session": session})

    assert (r.status, await r.json()) == (409, {"reason": "nothing_to_undo"})
    assert len(await review_rows(session_factory)) == 2


async def test_undo_deletes_the_production_card_that_rating_opened(client, session_factory):
    deck = await add_deck(session_factory)
    note = await add_note(session_factory, deck, "n")
    # Known well enough to be close to the production threshold (3 days).
    card = await add_card(
        session_factory,
        "rec",
        note,
        state=CardState.review,
        stability=2.5,
        difficulty=5.0,
        reps=3,
        due=PAST,
    )
    async with session_factory() as db:
        db.add(Review(card_id=card, user_id=OWNER, rating=3, reviewed_at=NOW - timedelta(days=4)))
        await db.commit()
    session = (await start(client, side="fi"))["session"]

    assert (await rate(client, session, card, 4)).status == 200
    async with session_factory() as db:
        production = await db.scalar(select(Card).where(Card.type == CardType.production))
    assert production is not None

    assert (await post(client, "undo", {"session": session})).status == 200
    async with session_factory() as db:
        count = await db.scalar(
            select(func.count()).select_from(Card).where(Card.type == CardType.production)
        )
    assert count == 0


# --- debt ----------------------------------------------------------------------------


async def overdue_words(session_factory, count) -> list[str]:
    ids = await add_words(
        session_factory, count, state=CardState.review, stability=5.0, difficulty=5.0, reps=2
    )
    return ids


async def test_a_big_debt_asks_first(make_client, session_factory):
    await overdue_words(session_factory, 3)
    client = await make_client(debt_threshold=2, session_max_cards=1, debt_postpone_days=7)

    body = await start(client)

    assert body == {
        "state": "debt",
        "session": body["session"],
        "overdue": 3,
        "max_cards": 1,
        "postpone_days": 7,
    }


async def test_debt_batch_studies_as_usual(make_client, session_factory):
    ids = await overdue_words(session_factory, 3)
    client = await make_client(debt_threshold=2, session_max_cards=2)
    session = (await start(client))["session"]

    r = await post(client, "debt", {"session": session, "action": "batch"})

    body = await r.json()
    assert r.status == 200
    assert body["state"] == "card"
    assert body["session"] == session
    assert body["card"]["id"] == ids[0]
    assert body["progress"] == {"done": 0, "total": 2}
    assert body["can_undo"] is False
    async with session_factory() as db:
        dues = [(await db.get(Card, i)).due for i in ids]
    assert all(due < NOW for due in dues)


async def test_debt_defer_postpones_the_tail(make_client, session_factory):
    ids = await overdue_words(session_factory, 3)
    client = await make_client(debt_threshold=2, session_max_cards=1, debt_postpone_days=7)
    session = (await start(client))["session"]

    body = await (await post(client, "debt", {"session": session, "action": "defer"})).json()

    assert (body["state"], body["session"], body["card"]["id"]) == ("card", session, ids[0])
    assert body["progress"] == {"done": 0, "total": 1}
    async with session_factory() as db:
        dues = [(await db.get(Card, i)).due for i in ids]
    assert dues[0] < NOW
    assert all(due > NOW + timedelta(days=6) for due in dues[1:])


async def test_a_second_debt_answer_is_stale(make_client, session_factory):
    ids = await overdue_words(session_factory, 3)
    client = await make_client(debt_threshold=2, session_max_cards=1)
    session = (await start(client))["session"]
    await post(client, "debt", {"session": session, "action": "batch"})

    r = await post(client, "debt", {"session": session, "action": "defer"})

    body = await r.json()
    assert (r.status, body["reason"], body["card"]["id"]) == (409, "stale", ids[0])
    async with session_factory() as db:
        dues = [(await db.get(Card, i)).due for i in ids]
    assert all(due < NOW for due in dues)


async def test_a_review_while_the_debt_is_open_is_stale(make_client, session_factory):
    ids = await overdue_words(session_factory, 3)
    client = await make_client(debt_threshold=2)
    session = (await start(client))["session"]

    r = await rate(client, session, ids[0])

    body = await r.json()
    assert (r.status, body["reason"], body["state"]) == (409, "stale", "debt")
    assert await review_rows(session_factory) == []


async def test_bad_debt_action_is_400(make_client, session_factory):
    await overdue_words(session_factory, 3)
    client = await make_client(debt_threshold=2)
    session = (await start(client))["session"]
    assert (await post(client, "debt", {"session": session, "action": "skip"})).status == 400


# --- tts -----------------------------------------------------------------------------


async def test_tts_speaks_the_example(client, session_factory, tts):
    [card] = await add_words(session_factory, 1)
    session = (await start(client))["session"]

    r = await post(client, "tts", {"session": session, "card_id": card})

    assert r.status == 200
    assert r.content_type == "audio/wav"
    assert await r.read() == WAV
    assert [call["input"] for call in tts.calls] == ["Tämä on sana1-n0 lause."]
    # Cached: the second time costs nothing.
    await post(client, "tts", {"session": session, "card_id": card})
    assert len(tts.calls) == 1


async def test_tts_errors_are_those_of_the_reader(make_client, session_factory, tts):
    import openai

    tts.error = openai.APIConnectionError(request=MagicMock())
    [card] = await add_words(session_factory, 1)
    client = await make_client()
    session = (await start(client))["session"]

    r = await post(client, "tts", {"session": session, "card_id": card})
    assert (r.status, await r.json()) == (502, {"reason": "openai"})


async def test_tts_breaker_is_503(make_client, session_factory):
    ids = await add_words(session_factory, 2)
    client = await make_client(reader_breaker_max_calls=1)
    session = (await start(client))["session"]

    assert (await post(client, "tts", {"session": session, "card_id": ids[0]})).status == 200
    r = await post(client, "tts", {"session": session, "card_id": ids[1]})
    assert (r.status, await r.json()) == (503, {"reason": "rate"})


async def test_learning_works_without_openai(make_client, session_factory, tts):
    ids = await add_words(session_factory, 2)
    client = await make_client(openai_api_key="")

    assert (await client.get("/api/learn/decks", headers=header())).status == 200
    session = (await start(client))["session"]
    assert (await rate(client, session, ids[0])).status == 200
    assert (await post(client, "undo", {"session": session})).status == 200
    r = await post(client, "tts", {"session": session, "card_id": ids[0]})
    assert (r.status, await r.json()) == (503, {"reason": "no_openai"})
    assert tts.calls == []


async def test_without_openai_the_reader_calls_nothing(make_client, book, tts):
    _, chapter_id, _ = book
    client = await make_client(openai_api_key="")
    lookup_client = client.server.app[LOOKUP_CLIENT]
    warmup = client.server.app[WARMUP]
    warmup.maybe_start = MagicMock()

    for path in ("/api/lookup", "/api/tts", "/api/cards"):
        r = await client.post(path, json=word_body(chapter_id), headers=header())
        assert (r.status, await r.json()) == (503, {"reason": "no_openai"}), path
    # The body is still checked first.
    r = await client.post("/api/lookup", json={"chapter_id": 1}, headers=header())
    assert r.status == 400
    assert (await client.get(f"/api/chapters/{chapter_id}", headers=header())).status == 200
    assert (
        await client.post("/api/lemma", json=word_body(chapter_id), headers=header())
    ).status == 200

    assert lookup_client.requests == []
    assert tts.calls == []
    warmup.maybe_start.assert_not_called()


# --- logs ----------------------------------------------------------------------------


async def test_logs_carry_ids_and_never_card_text(make_client, session_factory, caplog):
    caplog.set_level(logging.DEBUG)
    # aiosqlite at DEBUG prints SQL parameters; production keeps it at INFO.
    logging.getLogger("aiosqlite").setLevel(logging.INFO)
    ids = await overdue_words(session_factory, 3)
    client = await make_client(debt_threshold=2, session_max_cards=2)
    session = (await start(client))["session"]
    await post(client, "debt", {"session": session, "action": "batch"})
    await rate(client, session, ids[0])
    await rate(client, session, ids[0])  # stale
    await post(client, "undo", {"session": session})
    await post(client, "tts", {"session": session, "card_id": ids[0]})
    await rate(client, session, ids[0])
    await rate(client, session, ids[1])

    async with session_factory() as db:
        notes = (await db.scalars(select(Note))).all()
    texts = [t for n in notes for t in (n.lemma, n.translation_ru, n.example_fi, n.example_ru)]
    messages = [r.getMessage() for r in caplog.records]
    for message in messages:
        for piece in texts:
            assert piece not in message
    events = [log_fields(m) for m in messages]
    learn_events = {f["event"] for f in events if f.get("source") == "reader"}
    assert {
        "learn.debt_prompt",
        "learn.debt_choice",
        "learn.session_start",
        "learn.review",
        "learn.stale",
        "learn.undo",
        "learn.session_end",
    } <= learn_events
    undo = next(f for f in events if f.get("event") == "learn.undo")
    assert undo["card_id"] == ids[0]
