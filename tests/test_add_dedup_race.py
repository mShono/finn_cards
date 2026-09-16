"""The dedup invariant under real concurrency: one note per (user, deck, lemma, pos).

bot/add.py checks for an existing note (ingest.existing_note_keys) in one
session and inserts in another, with an LLM call in between - a window wide
enough for two /add turns for the same word to both pass the lookup and both
insert. tests/test_db_models.py pins the index itself, tests/test_edit.py
pins the same window on /edit; what was missing is the /add path driven
end-to-end by two genuinely concurrent turns.

Nothing here reaches for Note() directly: every insert goes through the
handler a deck-button tap actually lands in (add_deck_choice ->
_save_candidates_and_report), with the real dedup lookup, the real schema
validation and the real INSERT. The only patched call is the OpenAI one,
which is also where the test-only barrier sits - see _barrier_in_the_race_window.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from sqlalchemy import select, text

from kielikaveri.bot.add import AddStates, add_deck_choice
from kielikaveri.config import Settings
from kielikaveri.db.decks import create_deck
from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import (
    NOTE_UNIQUE_INDEX,
    Card,
    CardType,
    Note,
    User,
)
from kielikaveri.ingest import ResolvedForms
from kielikaveri.llm.breaker import CallBreaker
from kielikaveri.srs.graduation import sync_user_card_types

USER_ID = 1
NOW = datetime(2026, 9, 15, 10, 0, tzinfo=UTC)

# A compound the FST does not know, so canonical_key() leaves it untouched and
# the key under test is exactly the one the unique index is built on.
CANDIDATE = {
    "lemma": "seurojentalo",
    "pos": "substantiivi",
    "translation_ru": "дом собраний",
    "example_fi": "Seurojentalo on kylän keskellä.",
    "example_ru": "Дом собраний в центре деревни.",
    "kind": "word",
    "meta": {"topics": ["kylä"]},
}


@pytest.fixture
async def make_db(tmp_path):
    """Real file-backed SQLite, optionally with the unique index dropped.

    A dropped index, not an older alembic revision: the point is to change
    exactly one thing (the DB-level constraint) while every other column,
    default and code path stays what production runs.
    """
    engines = []

    async def _make(name: str, *, unique_index: bool = True):
        engine = make_engine(f"sqlite+aiosqlite:///{tmp_path}/{name}.db")
        await create_all(engine)
        if not unique_index:
            async with engine.begin() as conn:
                await conn.execute(text(f"DROP INDEX {NOTE_UNIQUE_INDEX}"))
        engines.append(engine)
        return make_session_factory(engine)

    yield _make
    for engine in engines:
        await engine.dispose()


async def seed(session_factory, *deck_names: str) -> list[str]:
    async with session_factory() as session:
        session.add(User(id=USER_ID))
        deck_ids = [(await create_deck(session, USER_ID, name)).id for name in deck_names]
        await session.commit()
    return deck_ids


def make_settings() -> Settings:
    return Settings(
        openai_api_key="sk-test",
        openai_text_model="gpt-5.6-terra",
        openai_timeout_seconds=1.0,
        breaker_max_calls=60,
        breaker_window_minutes=10,
    )


def make_breaker() -> CallBreaker:
    return CallBreaker(max_calls=60, window=timedelta(minutes=10))


def _barrier_in_the_race_window(monkeypatch, parties: int) -> None:
    """Hold every turn between its dedup lookup and its INSERT.

    _save_candidates_and_report reads the existing keys, then awaits
    resolve_note_forms (the OpenAI call), then inserts - so a barrier inside
    that call is exactly the window the race lives in, and it guarantees the
    interleaving instead of hoping the event loop produces it. Test-only:
    production still calls the real resolve_note_forms, which is patched out
    in every other /add test too (it would otherwise hit the network).
    """
    barrier = asyncio.Barrier(parties)

    async def resolve_at_the_barrier(*args, **kwargs):
        await barrier.wait()
        return ResolvedForms({"partitiivi": "seurojentaloa"}, "fst", True), None

    monkeypatch.setattr("kielikaveri.bot.add.resolve_note_forms", resolve_at_the_barrier)


async def tap_deck_button(session_factory, deck_id: str, chat_id: int):
    """One /add turn, resumed at the point the learner picks a deck.

    Each turn gets its own FSM key (aiogram keys state by chat), which is what
    two updates racing each other look like - a message resent from a second
    device, or a first one retried. Everything after the tap is production
    code: add_deck_choice validates the batch, clears the state and hands over
    to _save_candidates_and_report.
    """
    batch_id = f"batch{chat_id:06d}"
    state = FSMContext(
        storage=MemoryStorage(), key=StorageKey(bot_id=0, chat_id=chat_id, user_id=USER_ID)
    )
    await state.set_state(AddStates.choosing_deck)
    await state.update_data(batch_id=batch_id, candidates=[CANDIDATE])
    callback = SimpleNamespace(
        data=f"adddeck:{batch_id}:{deck_id}",
        from_user=SimpleNamespace(id=USER_ID),
        answer=AsyncMock(),
        message=SimpleNamespace(answer=AsyncMock()),
    )
    await add_deck_choice(callback, state, session_factory, make_settings(), make_breaker())
    return callback


def reports_of(callback) -> str:
    return "\n".join(call.args[0] for call in callback.message.answer.call_args_list)


async def notes_in(session_factory) -> list[Note]:
    async with session_factory() as session:
        return list(await session.scalars(select(Note)))


async def test_two_concurrent_adds_of_the_same_word_leave_exactly_one_note(make_db, monkeypatch):
    session_factory = await make_db("same_deck")
    (deck_id,) = await seed(session_factory, "Общая")
    _barrier_in_the_race_window(monkeypatch, parties=2)

    first, second = await asyncio.gather(
        tap_deck_button(session_factory, deck_id, chat_id=1),
        tap_deck_button(session_factory, deck_id, chat_id=2),
    )

    notes = await notes_in(session_factory)
    assert len(notes) == 1
    assert (notes[0].user_id, notes[0].deck_id, notes[0].lemma, notes[0].pos) == (
        USER_ID,
        deck_id,
        "seurojentalo",
        "substantiivi",
    )

    # Whichever turn lost is the one that must have reported a duplicate - the
    # API promises nothing about which, only that neither turn crashes and
    # neither silently drops the word (add_deck_choice's IntegrityError branch).
    texts = [reports_of(first), reports_of(second)]
    assert sum("🇫🇮 seurojentalo" in report for report in texts) == 1
    assert sum("не дублирую" in report and "seurojentalo" in report for report in texts) == 1


async def test_the_surviving_note_opens_exactly_one_set_of_cards(make_db, monkeypatch):
    # /add itself writes no cards - /learn opens them lazily per note
    # (srs/graduation). So "no two sets of cards" is only really answered by
    # running that same opening path over whatever the race left behind.
    session_factory = await make_db("cards")
    (deck_id,) = await seed(session_factory, "Общая")
    _barrier_in_the_race_window(monkeypatch, parties=2)

    await asyncio.gather(
        tap_deck_button(session_factory, deck_id, chat_id=1),
        tap_deck_button(session_factory, deck_id, chat_id=2),
    )

    async with session_factory() as session:
        await sync_user_card_types(session, USER_ID, NOW)
        await session.commit()

    notes = await notes_in(session_factory)
    async with session_factory() as session:
        cards = list(await session.scalars(select(Card)))

    # One set: every card hangs off the single surviving note, and no
    # (type, form) is opened twice. A second note would have doubled this
    # exactly - ensure_card_types runs per note and dedups only within one.
    assert len(notes) == 1
    assert {card.note_id for card in cards} == {notes[0].id}
    assert len({(card.type, card.form) for card in cards}) == len(cards)
    assert [card.type for card in cards].count(CardType.recognition) == 1


async def test_two_concurrent_adds_of_the_same_word_into_different_decks_both_land(
    make_db, monkeypatch
):
    # The guard above must stay deck-scoped: a word already learned in one
    # deck is still addable to another (per-deck dedup, requested 03.09.2026).
    session_factory = await make_db("two_decks")
    (deck_a, deck_b) = await seed(session_factory, "Общая", "talkoot")
    _barrier_in_the_race_window(monkeypatch, parties=2)

    first, second = await asyncio.gather(
        tap_deck_button(session_factory, deck_a, chat_id=1),
        tap_deck_button(session_factory, deck_b, chat_id=2),
    )

    notes = await notes_in(session_factory)
    assert len(notes) == 2
    assert {note.deck_id for note in notes} == {deck_a, deck_b}
    assert {note.lemma for note in notes} == {"seurojentalo"}
    for callback in (first, second):
        assert "🇫🇮 seurojentalo" in reports_of(callback)


async def test_without_the_unique_index_the_same_race_writes_two_notes(make_db, monkeypatch):
    """Proof the test above tests something.

    Same handler, same barrier, same two turns - only the DB-level index is
    gone. Both turns pass the Python dedup lookup (neither can see the other's
    uncommitted insert) and both commit, so the pre-insert check on its own
    leaves two notes for one word. Nothing in src/ changes to get here.
    """
    session_factory = await make_db("no_index", unique_index=False)
    (deck_id,) = await seed(session_factory, "Общая")
    _barrier_in_the_race_window(monkeypatch, parties=2)

    first, second = await asyncio.gather(
        tap_deck_button(session_factory, deck_id, chat_id=1),
        tap_deck_button(session_factory, deck_id, chat_id=2),
    )

    notes = await notes_in(session_factory)
    assert len(notes) == 2
    assert {(note.lemma, note.pos, note.deck_id) for note in notes} == {
        ("seurojentalo", "substantiivi", deck_id)
    }
    for callback in (first, second):
        assert "🇫🇮 seurojentalo" in reports_of(callback)
