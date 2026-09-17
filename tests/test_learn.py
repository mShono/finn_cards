import asyncio
import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import fsrs
import pytest
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from conftest import log_fields
from sqlalchemy import func, select

from kielikaveri.bot.learn import (
    DECK_ALL_TOKEN,
    LearnStates,
    Rating,
    _show_next_card,
    learn_debt_choice,
    learn_deck_choice,
    learn_listen,
    learn_rate,
    learn_reveal,
    learn_start,
    render_card,
)
from kielikaveri.config import Settings
from kielikaveri.db.decks import create_deck
from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import Card, CardState, CardType, Note, NoteKind, Review, User
from kielikaveri.srs import scheduler as srs_scheduler

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


def make_note(note_id: str = "note-1", user_id: int = 1, deck_id: str | None = None) -> Note:
    return Note(
        id=note_id,
        user_id=user_id,
        lemma="hakea",
        translation_ru="искать",
        example_fi="Haen töitä.",
        example_ru="Я ищу работу.",
        kind=NoteKind.word,
        deck_id=deck_id,
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
        message=SimpleNamespace(answer=AsyncMock(), answer_audio=AsyncMock()),
    )


def make_message() -> SimpleNamespace:
    return SimpleNamespace(from_user=SimpleNamespace(id=1), answer=AsyncMock())


async def _drain_session(
    state, session_factory, settings, message, max_taps: int = 50
) -> list[str]:
    """Run one /learn session through the real handlers and return the card
    ids it actually showed.

    Deliberately no hand-built FSM data: the queue comes from the real
    build_session_queue and every card is advanced by the real learn_rate,
    so the card cap under test is the one production applies.
    """
    await learn_start(message, state, session_factory, settings)
    shown: list[str] = []
    for _ in range(max_taps):
        markup = message.answer.call_args.kwargs.get("reply_markup")
        if markup is None:
            break  # the session-end message carries no keyboard
        card_id = markup.inline_keyboard[0][0].callback_data.split(":", 2)[2]
        shown.append(card_id)
        reveal = make_callback(f"learn:reveal:{card_id}")
        reveal.message = message
        await learn_reveal(reveal, session_factory)
        rate = make_callback(f"learn:rate:{card_id}:3")
        rate.message = message
        await learn_rate(rate, state, session_factory)
    return shown


async def _seed_due_cards(session_factory, count: int, now: datetime) -> None:
    """`count` already-reviewed cards, each overdue by a different amount so
    the queue order is deterministic."""
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(make_note())
        await session.flush()
        for i in range(count):
            session.add(
                make_card(f"card-{i}", "note-1", 1, due=now - timedelta(days=count - i), reps=1)
            )
        await session.commit()


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


def test_render_card_inflection_asks_by_context_not_by_the_forms_name():
    note = make_note()
    card = make_card("card-A", "note-1", 1, due=NOW)
    card.type = CardType.inflection
    card.form = "preesens_1s"
    note.meta = {"principal_forms": {"preesens_1s": "haen"}}

    front, back = render_card(card, note)

    # The front must not name the category - that is what the card is
    # testing, and it is only revealed on the back.
    assert front == "hakea → minä, nyt → ?"
    assert "preesens" not in front
    assert back.startswith("haen")
    assert "preesens, 1. persoona yksikkö (minä)" in back


def test_render_card_inflection_asks_the_form_the_card_is_scheduled_for():
    # The question is fixed per card, not drawn at random: that is what
    # makes the card's own FSRS interval mean anything.
    note = make_note()
    note.meta = {"principal_forms": {"genetiivi": "kaupan", "illatiivi": "kauppaan"}}
    card = make_card("card-A", "note-1", 1, due=NOW)
    card.type = CardType.inflection
    card.form = "illatiivi"

    assert [render_card(card, note) for _ in range(5)] == [
        ("hakea → mihin?", "kauppaan\n\n✅ illatiivi - mihin? sisään (-Vn, -seen, -hVn)")
    ] * 5


def test_render_card_inflection_falls_back_when_the_card_has_no_usable_form():
    # Defensive: a card left form-less (pre-migration row not yet adopted)
    # or pointing at a form the note no longer has must not crash /learn.
    note = make_note()
    note.meta = {"principal_forms": {"genetiivi": "kaupan"}}
    card = make_card("card-A", "note-1", 1, due=NOW)
    card.type = CardType.inflection

    assert render_card(card, note) == ("hakea", "hakea")

    card.form = "translatiivi"
    assert render_card(card, note) == ("hakea", "hakea")


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


# --- _show_next_card and the two session-end conditions ------------------


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


async def test_a_session_never_serves_more_than_session_max_cards(session_factory):
    # Five cards are due but the cap is two. The limit lives in
    # build_session_queue alone, so the proof has to be the real flow: start a
    # session, rate whatever it offers, and count what actually got reviewed.
    now = datetime.now(UTC)
    await _seed_due_cards(session_factory, 5, now)
    state = make_state()
    message = make_message()

    shown = await _drain_session(
        state, session_factory, make_settings(session_max_cards=2, daily_new_limit=10), message
    )

    assert len(shown) == 2  # not 5 - the queue was capped before the first card
    async with session_factory() as session:
        reviews = (await session.scalars(select(Review))).all()
    assert len(reviews) == 2

    text = message.answer.call_args.args[0]
    assert "Сессия окончена: 2 карточек пройдено" in text
    assert await state.get_data() == {}  # session closed, nothing left in the FSM


async def test_the_next_session_picks_up_the_cards_the_cap_left_behind(session_factory):
    # The cap defers cards, it doesn't drop them: the three the first session
    # couldn't reach are still due and lead the next one.
    now = datetime.now(UTC)
    await _seed_due_cards(session_factory, 5, now)
    settings = make_settings(session_max_cards=2, daily_new_limit=10)

    first = await _drain_session(make_state(), session_factory, settings, make_message())
    second = await _drain_session(make_state(), session_factory, settings, make_message())

    assert len(second) == 2
    assert not set(first) & set(second)  # the cap moved the window, didn't repeat it


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
    listen_button = kwargs["reply_markup"].inline_keyboard[1][0]
    assert listen_button.callback_data == "learn:listen:card-A"
    callback.answer.assert_awaited_once()


# --- learn_listen -------------------------------------------------------------


async def test_learn_listen_sends_synthesized_audio_of_the_example_sentence(
    session_factory, monkeypatch
):
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(make_note())
        await session.flush()
        session.add(make_card("card-A", "note-1", 1, due=NOW))
        await session.commit()

    monkeypatch.setattr(
        "kielikaveri.bot.learn.synthesize_speech",
        lambda client, model, text, speed: b"fake-mp3-bytes",
    )
    callback = make_callback("learn:listen:card-A")
    settings = make_settings(openai_api_key="sk-test", openai_tts_model="tts-1")

    await learn_listen(callback, session_factory, settings)

    callback.message.answer_audio.assert_awaited_once()
    audio = callback.message.answer_audio.call_args.args[0]
    assert audio.data == b"fake-mp3-bytes"
    callback.answer.assert_awaited_once()


async def test_learn_listen_without_an_openai_key_answers_gracefully(session_factory):
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(make_note())
        await session.flush()
        session.add(make_card("card-A", "note-1", 1, due=NOW))
        await session.commit()

    callback = make_callback("learn:listen:card-A")
    settings = make_settings(openai_api_key="")

    await learn_listen(callback, session_factory, settings)

    callback.answer.assert_awaited_once_with(
        "Озвучка недоступна - не настроен OpenAI.", show_alert=True
    )
    callback.message.answer_audio.assert_not_awaited()


# --- learn_start: debt-threshold branching ----------------------------------


async def test_learn_start_offers_debt_choice_when_overdue_exceeds_threshold(session_factory):
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(make_note())
        await session.flush()
        for i in range(3):
            # reps=1 - only missed reviews count as debt, see
            # _seed_five_overdue_cards.
            session.add(make_card(f"card-{i}", "note-1", 1, due=NOW - timedelta(days=1), reps=1))
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


# --- learn_start / learn_deck_choice: deck picking ---------------------------


async def test_learn_start_skips_the_picker_with_a_single_deck(session_factory):
    async with session_factory() as session:
        session.add(User(id=1))
        deck = await create_deck(session, 1, "Общая")
        await session.flush()
        session.add(make_note(deck_id=deck.id))
        await session.flush()
        session.add(make_card("card-A", "note-1", 1, due=NOW - timedelta(days=1)))
        await session.commit()

    state = make_state()
    message = make_message()

    await learn_start(message, state, session_factory, make_settings())

    # Straight into a session - no deck question asked.
    assert await state.get_state() == LearnStates.reviewing
    assert "hakea" in message.answer.call_args.args[0]


async def test_learn_start_asks_which_deck_when_more_than_one_exists(session_factory):
    async with session_factory() as session:
        session.add(User(id=1))
        await create_deck(session, 1, "Общая")
        await create_deck(session, 1, "Из книги")
        await session.commit()

    state = make_state()
    message = make_message()

    await learn_start(message, state, session_factory, make_settings())

    assert await state.get_state() == LearnStates.deck_choice
    buttons = [
        b for row in message.answer.call_args.kwargs["reply_markup"].inline_keyboard for b in row
    ]
    labels = [b.text for b in buttons]
    assert "Общая" in labels
    assert "Из книги" in labels
    assert "Все" in labels


async def test_learn_deck_choice_only_queues_cards_from_the_chosen_deck(session_factory):
    async with session_factory() as session:
        session.add(User(id=1))
        deck_a = await create_deck(session, 1, "Общая")
        deck_b = await create_deck(session, 1, "Из книги")
        await session.flush()
        session.add(make_note("note-a", 1, deck_id=deck_a.id))
        session.add(make_note("note-b", 1, deck_id=deck_b.id))
        await session.flush()
        session.add(make_card("card-a", "note-a", 1, due=NOW - timedelta(days=1)))
        session.add(make_card("card-b", "note-b", 1, due=NOW - timedelta(days=1)))
        await session.commit()

    state = make_state()
    await state.set_state(LearnStates.deck_choice)
    callback = make_callback(f"learn:deck:{deck_b.id}")

    await learn_deck_choice(callback, state, session_factory, make_settings())

    data = await state.get_data()
    assert data["queue"] == ["card-b"]


async def test_learn_deck_choice_all_queues_cards_from_every_deck(session_factory):
    async with session_factory() as session:
        session.add(User(id=1))
        deck_a = await create_deck(session, 1, "Общая")
        deck_b = await create_deck(session, 1, "Из книги")
        await session.flush()
        session.add(make_note("note-a", 1, deck_id=deck_a.id))
        session.add(make_note("note-b", 1, deck_id=deck_b.id))
        await session.flush()
        session.add(make_card("card-a", "note-a", 1, due=NOW - timedelta(days=1)))
        session.add(make_card("card-b", "note-b", 1, due=NOW - timedelta(days=1)))
        await session.commit()

    state = make_state()
    await state.set_state(LearnStates.deck_choice)
    callback = make_callback(f"learn:deck:{DECK_ALL_TOKEN}")

    await learn_deck_choice(callback, state, session_factory, make_settings())

    data = await state.get_data()
    assert set(data["queue"]) == {"card-a", "card-b"}


# --- learn_debt_choice -------------------------------------------------------


async def _seed_five_overdue_cards(session_factory, now: datetime) -> None:
    # reps=1: debt means reviews that came due and were missed. Cards that
    # have never been reviewed are new material waiting under the daily
    # new-card limit, and neither the prompt nor the defer touches them
    # (see tests/test_queue.py).
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(make_note())
        await session.flush()
        for i in range(5):
            session.add(
                make_card(f"card-{i}", "note-1", 1, due=now - timedelta(days=5 - i), reps=1)
            )
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


# --- application-level logging ------------------------------------------------


async def test_learn_start_single_deck_logs_session_start(session_factory, caplog):
    async with session_factory() as session:
        session.add(User(id=1))
        deck = await create_deck(session, 1, "Общая")
        await session.flush()
        session.add(make_note(deck_id=deck.id))
        await session.flush()
        session.add(make_card("card-A", "note-1", 1, due=NOW - timedelta(days=1)))
        await session.commit()

    state = make_state()
    message = make_message()

    with caplog.at_level(logging.INFO, logger="kielikaveri.bot.learn"):
        await learn_start(message, state, session_factory, make_settings())

    events = [log_fields(r.message) for r in caplog.records]
    start = next(f for f in events if f.get("event") == "learn.session_start")
    assert start["queue"] == "1"


async def test_learn_start_with_no_due_cards_logs_session_empty(session_factory, caplog):
    async with session_factory() as session:
        session.add(User(id=1))
        await session.commit()

    state = make_state()
    message = make_message()

    with caplog.at_level(logging.INFO, logger="kielikaveri.bot.learn"):
        await learn_start(message, state, session_factory, make_settings())

    events = [log_fields(r.message) for r in caplog.records]
    assert any(f.get("event") == "learn.session_empty" for f in events)


async def test_show_next_card_logs_session_end_with_reason_and_counts(session_factory, caplog):
    # A session that runs out of time mid-queue - the only way session_end
    # reports a non-zero `remaining`, now that the card cap is the queue's job
    # and a capped queue simply ends empty.
    now = datetime.now(UTC)
    await _seed_due_cards(session_factory, 2, now)
    state = make_state()
    message = make_message()
    settings = make_settings(session_max_cards=2, session_max_minutes=10, daily_new_limit=10)

    await learn_start(message, state, session_factory, settings)
    first_card = message.answer.call_args.kwargs["reply_markup"].inline_keyboard[0][0]
    card_id = first_card.callback_data.split(":", 2)[2]
    rate = make_callback(f"learn:rate:{card_id}:3")
    rate.message = message

    # The learner walked away for 11 minutes between the first card and the
    # second - only the clock is moved, the queue and the count stay as the
    # real flow left them.
    await state.update_data(
        session_started_at=(datetime.now(UTC) - timedelta(minutes=11)).isoformat()
    )

    with caplog.at_level(logging.INFO, logger="kielikaveri.bot.learn"):
        await learn_rate(rate, state, session_factory)

    events = [log_fields(r.message) for r in caplog.records]
    end = next(f for f in events if f.get("event") == "learn.session_end")
    assert end["reason"] == "max_minutes"
    assert end["reviewed"] == "1"
    assert end["remaining"] == "1"  # the second card was never shown


async def test_learn_reveal_logs_reveal_event(session_factory, caplog):
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(make_note())
        await session.flush()
        session.add(make_card("card-A", "note-1", 1, due=NOW))
        await session.commit()

    callback = make_callback("learn:reveal:card-A")

    with caplog.at_level(logging.DEBUG, logger="kielikaveri.bot.learn"):
        await learn_reveal(callback, session_factory)

    events = [log_fields(r.message) for r in caplog.records]
    reveal = next(f for f in events if f.get("event") == "learn.reveal")
    assert reveal["card_id"] == "card-A"


async def test_learn_rate_logs_db_save_with_rating(session_factory, caplog):
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
        session_max_minutes=10,
    )
    callback = make_callback("learn:rate:card-A:3")

    with caplog.at_level(logging.INFO, logger="kielikaveri.bot.learn"):
        await learn_rate(callback, state, session_factory)

    events = [log_fields(r.message) for r in caplog.records]
    saved = next(f for f in events if f.get("event") == "db.save")
    assert saved["entity"] == "review"
    assert saved["card_id"] == "card-A"
    assert saved["rating"] == "3"


async def test_learn_rate_stale_button_logs_debug_not_a_save(session_factory, caplog):
    await _seed_reviewed_card(session_factory, "card-A")
    state = make_state()
    await state.update_data(queue=["card-B"], reviewed_count=1)
    callback = make_callback("learn:rate:card-A:3")

    with caplog.at_level(logging.DEBUG, logger="kielikaveri.bot.learn"):
        await learn_rate(callback, state, session_factory)

    events = [log_fields(r.message) for r in caplog.records]
    assert any(f.get("event") == "learn.rate_stale" for f in events)
    assert not any(f.get("event") == "db.save" for f in events)


# --- learn_rate feeds FSRS the card's real review history ---------------------


async def test_learn_rate_schedules_off_the_previous_review_not_the_current_one(
    session_factory, monkeypatch
):
    """The end-to-end half of the `last_review` contract (the wrapper's own
    half lives in test_scheduler.py): the timestamp FSRS schedules from has
    to come out of the `reviews` rows already in the database, and the row
    this very answer writes must not be one of them.

    Both halves fail loudly here. Passing no history at all reads as "fully
    forgotten, yet recalled" and multiplies the interval by orders of
    magnitude; letting autoflush slip the new row into the max() reads as
    "answered again a moment later" and shortens it instead - the `bugged`
    card below is exactly that second outcome, which is why it is asserted
    against rather than just compared to the correct one.

    Real py-fsrs on both sides, never a mock - a faked scheduler would
    happily agree with whatever the wrapper did.
    """
    # Fuzzing randomizes Review-state intervals, so neither side gets it.
    monkeypatch.setattr(srs_scheduler._scheduler, "enable_fuzzing", False)
    reference_scheduler = fsrs.Scheduler(enable_fuzzing=False)

    # A history the learner really could have: first seen 20 days ago, then
    # answered again at each due date - the last of them long enough ago
    # that the gap since actually matters to FSRS.
    started = datetime.now(UTC).replace(microsecond=0) - timedelta(days=20)
    reference = fsrs.Card(due=started)
    history: list[datetime] = []
    when = started
    for _ in range(3):
        reference, _ = reference_scheduler.review_card(reference, Rating.Good, when)
        history.append(when)
        when = reference.due

    async with session_factory() as session:
        session.add(User(id=1))
        session.add(make_note())
        await session.flush()
        session.add(
            Card(
                id="card-A",
                note_id="note-1",
                user_id=1,
                type=CardType.recognition,
                state=CardState(reference.state.name.lower()),
                step=reference.step,
                stability=reference.stability,
                difficulty=reference.difficulty,
                due=reference.due,
                reps=len(history),
            )
        )
        for reviewed_at in history:
            session.add(
                Review(
                    card_id="card-A", user_id=1, rating=Rating.Good.value, reviewed_at=reviewed_at
                )
            )
        await session.commit()

    async with session_factory() as session:
        stored = await session.scalar(
            select(func.max(Review.reviewed_at)).where(Review.card_id == "card-A")
        )
    assert stored == history[-1]  # the previous answer really is in the DB

    state = make_state()
    await state.update_data(
        queue=["card-A"],
        reviewed_count=0,
        session_started_at=datetime.now(UTC).isoformat(),
        session_max_minutes=10,
    )
    await learn_rate(make_callback("learn:rate:card-A:3"), state, session_factory)

    async with session_factory() as session:
        card = await session.get(Card, "card-A")
        reviews = (
            await session.scalars(
                select(Review).where(Review.card_id == "card-A").order_by(Review.reviewed_at)
            )
        ).all()

    assert len(reviews) == len(history) + 1
    # learn_rate stamps its own `now`; the row it wrote is what it used.
    now = reviews[-1].reviewed_at
    expected, _ = reference_scheduler.review_card(reference, Rating.Good, now)
    answered_a_moment_ago = fsrs.Card(
        state=reference.state,
        step=reference.step,
        stability=reference.stability,
        difficulty=reference.difficulty,
        due=reference.due,
        last_review=now,
    )
    bugged, _ = reference_scheduler.review_card(answered_a_moment_ago, Rating.Good, now)

    assert card.state.value == expected.state.name.lower()
    assert card.step == expected.step
    assert card.stability == pytest.approx(expected.stability)
    assert card.difficulty == pytest.approx(expected.difficulty)
    assert abs(card.due - expected.due) <= timedelta(seconds=1)
    # Not the shorter interval an autoflushed `max()` would have produced.
    assert card.stability != pytest.approx(bugged.stability)
