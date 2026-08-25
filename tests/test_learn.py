import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from sqlalchemy import select

from kielikaveri.bot.learn import (
    LearnStates,
    Rating,
    _show_next_card,
    learn_debt_choice,
    learn_rate,
    learn_reveal,
    learn_start,
    render_card,
)
from kielikaveri.config import Settings
from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import Card, CardType, Note, NoteKind, Review, User

NOW = datetime(2026, 8, 24, 10, 0, tzinfo=UTC)


@pytest.fixture
async def session_factory(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    await create_all(engine)
    yield make_session_factory(engine)
    await engine.dispose()


def make_state() -> FSMContext:
    return FSMContext(storage=MemoryStorage(), key=StorageKey(bot_id=0, chat_id=1, user_id=1))


def make_settings(**overrides) -> Settings:
    defaults = {
        "session_max_cards": 20,
        "session_max_minutes": 10,
        "daily_new_limit": 10,
        "day_boundary_hour": 4,
        "debt_threshold": 100,
        "debt_postpone_days": 7,
    }
    return Settings(**{**defaults, **overrides})


def make_note(note_id: str = "note-1", user_id: int = 1) -> Note:
    return Note(
        id=note_id,
        user_id=user_id,
        lemma="hakea",
        translation_ru="искать",
        example_fi="Haen töitä.",
        example_ru="Я ищу работу.",
        kind=NoteKind.word,
        meta={},
    )


def make_card(card_id: str, note_id: str, user_id: int, due: datetime, reps: int = 0) -> Card:
    return Card(
        id=card_id, note_id=note_id, user_id=user_id, type=CardType.recognition, due=due, reps=reps
    )


def make_callback(data: str) -> SimpleNamespace:
    return SimpleNamespace(
        data=data,
        from_user=SimpleNamespace(id=1),
        answer=AsyncMock(),
        message=SimpleNamespace(answer=AsyncMock()),
    )


def make_message() -> SimpleNamespace:
    return SimpleNamespace(from_user=SimpleNamespace(id=1), answer=AsyncMock())


async def _seed_reviewed_card(session_factory, card_id: str) -> None:
    """A card already rated once, mirroring the state right after learn_rate
    processed it and moved the queue past it."""
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(make_note())
        await session.flush()
        session.add(make_card(card_id, "note-1", 1, due=NOW, reps=1))
        session.add(Review(card_id=card_id, user_id=1, rating=Rating.Good.value, reviewed_at=NOW))
        await session.commit()


# --- render_card: one branch per CardType --------------------------------
# Only the recognition branch was exercised before (indirectly, via
# learn_reveal/_show_next_card tests) - production and inflection had zero
# coverage, including inflection's random.choice over principal_forms.


def test_render_card_recognition_shows_finnish_front_and_translation_back():
    note = make_note()
    card = make_card("card-A", "note-1", 1, due=NOW)
    card.type = CardType.recognition

    front, back = render_card(card, note)

    assert front == "🇫🇮 hakea"
    assert "искать" in back
    assert "Haen töitä." in back  # example_fi


def test_render_card_production_shows_translation_front_and_finnish_back():
    note = make_note()
    card = make_card("card-A", "note-1", 1, due=NOW)
    card.type = CardType.production

    front, back = render_card(card, note)

    assert front == "🇷🇺 искать"
    assert back == "hakea\n\nHaen töitä."


def test_render_card_inflection_quizzes_one_of_the_principal_forms():
    note = make_note()
    card = make_card("card-A", "note-1", 1, due=NOW)
    card.type = CardType.inflection
    # A single-entry dict makes random.choice's pick deterministic, so the
    # test doesn't depend on which form gets quizzed.
    note.meta = {"principal_forms": {"preesens_1s": "haen"}}

    front, back = render_card(card, note)

    assert front == "hakea → preesens_1s?"
    assert back == "haen"


def test_render_card_inflection_without_principal_forms_falls_back_to_the_lemma():
    # Defensive path: ensure_card_types only creates an inflection card once
    # principal_forms is populated, but render_card doesn't re-check that -
    # if meta were ever edited afterward to drop the forms, this is what
    # /learn would show instead of crashing on an empty random.choice().
    note = make_note()
    note.meta = {}
    card = make_card("card-A", "note-1", 1, due=NOW)
    card.type = CardType.inflection

    front, back = render_card(card, note)

    assert front == back == "hakea"


# --- learn_rate: stale-button guard (regression for the double-tap bug) ----


async def test_stale_rating_on_a_card_no_longer_at_queue_head_is_ignored(session_factory):
    # Reproduces a duplicate tap on an old "Хорошо"/"Забыл" button: Telegram
    # never disables a used button, so a second tap on card-A's message can
    # arrive after the queue has already moved on to card-B. Applying it
    # again would silently record a review the user never made this time
    # and skew card-A's FSRS history - see learn.py's guard.
    await _seed_reviewed_card(session_factory, "card-A")
    state = make_state()
    await state.update_data(queue=["card-B"], reviewed_count=1)
    callback = make_callback("learn:rate:card-A:3")

    await learn_rate(callback, state, session_factory)

    async with session_factory() as session:
        reviews = (await session.scalars(select(Review).where(Review.card_id == "card-A"))).all()
        card = await session.get(Card, "card-A")
    assert len(reviews) == 1  # no phantom second review recorded
    assert card.reps == 1

    data = await state.get_data()
    assert data["queue"] == ["card-B"]  # untouched
    assert data["reviewed_count"] == 1  # untouched
    callback.answer.assert_awaited_once()


async def test_concurrent_double_tap_on_the_current_head_card_records_only_one_review(
    session_factory,
):
    # A real double-tap: two callback_query updates for the *same still-head*
    # card, both reaching learn_rate before either has finished. The
    # stale-button guard above only rejects a tap on a card that's already
    # been popped from the queue - it does nothing here, because both calls
    # read the queue before either writes it back. Reproduced with real
    # asyncio concurrency (aiosqlite genuinely yields to the loop on I/O),
    # not a mock: without a fix, this records two reviews and advances the
    # card's FSRS stability twice off a single rating.
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(make_note())
        await session.flush()
        session.add(make_card("card-A", "note-1", 1, due=NOW))
        await session.commit()

    state = make_state()
    await state.update_data(
        queue=["card-A"],
        reviewed_count=0,
        session_started_at=datetime.now(UTC).isoformat(),
        session_max_cards=20,
        session_max_minutes=10,
    )
    callback_1 = make_callback("learn:rate:card-A:3")
    callback_2 = make_callback("learn:rate:card-A:3")

    await asyncio.gather(
        learn_rate(callback_1, state, session_factory),
        learn_rate(callback_2, state, session_factory),
    )

    async with session_factory() as session:
        reviews = (await session.scalars(select(Review).where(Review.card_id == "card-A"))).all()
        card = await session.get(Card, "card-A")
    assert len(reviews) == 1  # not 2 - the second concurrent tap must be rejected
    assert card.reps == 1  # not 2


async def test_rating_the_card_at_queue_head_records_exactly_one_review(session_factory):
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(make_note())
        await session.flush()
        session.add(make_card("card-A", "note-1", 1, due=NOW))
        await session.commit()

    state = make_state()
    await state.update_data(
        queue=["card-A"],
        reviewed_count=0,
        # Real time, not the fixed NOW - the empty-queue check short-circuits
        # before elapsed time is ever evaluated *today*, but a stale
        # timestamp here is a trap for whoever next touches this test (see
        # the debt_choice tests below, which hit exactly this).
        session_started_at=datetime.now(UTC).isoformat(),
        session_max_cards=20,
        session_max_minutes=10,
    )
    callback = make_callback("learn:rate:card-A:3")

    await learn_rate(callback, state, session_factory)

    async with session_factory() as session:
        reviews = (await session.scalars(select(Review).where(Review.card_id == "card-A"))).all()
        card = await session.get(Card, "card-A")
    assert len(reviews) == 1
    assert card.reps == 1

    # The queue is now empty, so _show_next_card ends the session and clears
    # the FSM data - nothing left over for a subsequent stray callback to act on.
    assert await state.get_data() == {}
    callback.message.answer.assert_awaited_once()


# --- _show_next_card: rendering and the two session-end conditions --------


async def test_show_next_card_displays_the_head_of_a_nonempty_queue(session_factory):
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(make_note())
        await session.flush()
        session.add(make_card("card-A", "note-1", 1, due=NOW))
        session.add(make_card("card-B", "note-1", 1, due=NOW))
        await session.commit()

    state = make_state()
    await state.update_data(
        queue=["card-A", "card-B"],
        reviewed_count=0,
        session_started_at=datetime.now(UTC).isoformat(),
        session_max_cards=20,
        session_max_minutes=10,
    )
    answer_to = SimpleNamespace(answer=AsyncMock())

    await _show_next_card(answer_to, state, session_factory)

    answer_to.answer.assert_awaited_once()
    _text, kwargs = answer_to.answer.call_args.args, answer_to.answer.call_args.kwargs
    assert "hakea" in _text[0]  # card-A's front, not card-B's
    reveal_button = kwargs["reply_markup"].inline_keyboard[0][0]
    assert reveal_button.callback_data == "learn:reveal:card-A"
    # Session still open - state untouched, nothing cleared.
    assert (await state.get_data())["queue"] == ["card-A", "card-B"]


async def test_session_ends_once_the_card_limit_is_reached_even_with_cards_left(session_factory):
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(make_note())
        await session.flush()
        session.add(make_card("card-A", "note-1", 1, due=NOW))
        session.add(make_card("card-B", "note-1", 1, due=NOW))
        await session.commit()

    state = make_state()
    await state.update_data(
        queue=["card-A", "card-B"],
        reviewed_count=2,  # already at the cap
        session_started_at=datetime.now(UTC).isoformat(),
        session_max_cards=2,
        session_max_minutes=10,
    )
    answer_to = SimpleNamespace(answer=AsyncMock())

    await _show_next_card(answer_to, state, session_factory)

    text = answer_to.answer.call_args.args[0]
    assert "Сессия окончена" in text
    assert "осталось 2" in text  # both cards are still due, session just capped
    assert await state.get_data() == {}


async def test_session_ends_once_the_time_limit_is_reached(session_factory):
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(make_note())
        await session.flush()
        session.add(make_card("card-A", "note-1", 1, due=NOW))
        await session.commit()

    state = make_state()
    started_11_minutes_ago = (datetime.now(UTC) - timedelta(minutes=11)).isoformat()
    await state.update_data(
        queue=["card-A"],
        reviewed_count=0,
        session_started_at=started_11_minutes_ago,
        session_max_cards=20,
        session_max_minutes=10,
    )
    answer_to = SimpleNamespace(answer=AsyncMock())

    await _show_next_card(answer_to, state, session_factory)

    text = answer_to.answer.call_args.args[0]
    assert "Сессия окончена" in text
    assert await state.get_data() == {}


# --- learn_reveal ------------------------------------------------------------


async def test_learn_reveal_shows_the_back_and_a_rating_keyboard_for_that_card(session_factory):
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(make_note())
        await session.flush()
        session.add(make_card("card-A", "note-1", 1, due=NOW))
        await session.commit()

    callback = make_callback("learn:reveal:card-A")

    await learn_reveal(callback, session_factory)

    callback.message.answer.assert_awaited_once()
    text, kwargs = (
        callback.message.answer.call_args.args[0],
        callback.message.answer.call_args.kwargs,
    )
    assert "искать" in text  # note.translation_ru, part of the back
    rate_buttons = kwargs["reply_markup"].inline_keyboard[0]
    assert all(button.callback_data.startswith("learn:rate:card-A:") for button in rate_buttons)
    callback.answer.assert_awaited_once()


# --- learn_start: debt-threshold branching ----------------------------------


async def test_learn_start_offers_debt_choice_when_overdue_exceeds_threshold(session_factory):
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(make_note())
        await session.flush()
        for i in range(3):
            session.add(make_card(f"card-{i}", "note-1", 1, due=NOW - timedelta(days=1)))
        await session.commit()

    state = make_state()
    settings = make_settings(debt_threshold=2)
    message = make_message()

    await learn_start(message, state, session_factory, settings)

    assert await state.get_state() == LearnStates.debt_choice
    message.answer.assert_awaited_once()
    assert "Просрочено 3" in message.answer.call_args.args[0]


async def test_learn_start_goes_straight_to_reviewing_when_under_the_debt_threshold(
    session_factory,
):
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(make_note())
        await session.flush()
        session.add(make_card("card-A", "note-1", 1, due=NOW - timedelta(days=1)))
        await session.commit()

    state = make_state()
    settings = make_settings(debt_threshold=100)
    message = make_message()

    await learn_start(message, state, session_factory, settings)

    assert await state.get_state() == LearnStates.reviewing
    message.answer.assert_awaited_once()
    assert "hakea" in message.answer.call_args.args[0]  # the card's front, not a debt prompt


# --- learn_debt_choice -------------------------------------------------------


async def _seed_five_overdue_cards(session_factory, now: datetime) -> None:
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(make_note())
        await session.flush()
        for i in range(5):
            session.add(make_card(f"card-{i}", "note-1", 1, due=now - timedelta(days=5 - i)))
        await session.commit()


async def test_learn_debt_choice_defer_postpones_the_tail_then_starts_a_session(session_factory):
    # debt_now must be close to real time: _show_next_card (called at the end
    # of _start_session) measures elapsed session time against the real
    # clock, so a stale/fixed "now" here would make the session look like it
    # had already run past session_max_minutes and end before it starts.
    now = datetime.now(UTC)
    await _seed_five_overdue_cards(session_factory, now)
    state = make_state()
    await state.set_state(LearnStates.debt_choice)
    await state.update_data(debt_now=now.isoformat())
    settings = make_settings(session_max_cards=2, debt_postpone_days=7)
    callback = make_callback("learn:debt:defer")

    await learn_debt_choice(callback, state, session_factory, settings)

    async with session_factory() as session:
        cards = (await session.scalars(select(Card))).all()
    postponed = [c for c in cards if c.due > now]
    assert len(postponed) == 3  # 5 overdue, keep_n=session_max_cards=2, rest deferred

    messages = [call.args[0] for call in callback.message.answer.call_args_list]
    assert any("Отложено 3" in m for m in messages)
    assert await state.get_state() == LearnStates.reviewing  # session started right after


async def test_learn_debt_choice_batch_starts_a_session_without_deferring_anything(
    session_factory,
):
    now = datetime.now(UTC)
    await _seed_five_overdue_cards(session_factory, now)
    state = make_state()
    await state.set_state(LearnStates.debt_choice)
    await state.update_data(debt_now=now.isoformat())
    settings = make_settings(session_max_cards=2)
    callback = make_callback("learn:debt:batch")

    await learn_debt_choice(callback, state, session_factory, settings)

    async with session_factory() as session:
        cards = (await session.scalars(select(Card))).all()
    assert all(c.due <= now for c in cards)  # nothing pushed forward

    messages = [call.args[0] for call in callback.message.answer.call_args_list]
    assert not any("Отложено" in m for m in messages)
    assert await state.get_state() == LearnStates.reviewing
