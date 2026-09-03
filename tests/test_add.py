import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import openai
import pytest
from aiogram.filters import CommandObject
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from sqlalchemy import select

from kielikaveri.bot.add import (
    AddStates,
    add_command,
    add_deck_choice,
    add_deck_choice_stray,
    chat_message,
    delete_cancel,
    delete_command,
    delete_confirm,
)
from kielikaveri.config import Settings
from kielikaveri.db.decks import active_deck, create_deck, set_active_deck
from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import Card, IngestCache, Note, Review, User
from kielikaveri.ingest import ResolvedForms, TokenUsage
from kielikaveri.llm.breaker import CallBreaker, CircuitOpenError

NOW = datetime(2026, 8, 26, 10, 0, tzinfo=UTC)

WORD_CANDIDATE = {
    "lemma": "hakea",
    "pos": "verbi",
    "translation_ru": "искать",
    "example_fi": "Haen töitä kaupungista.",
    "example_ru": "Я ищу работу в городе.",
    "kind": "word",
    "meta": {"cefr": "B1"},
}

PATTERN_CANDIDATE = {
    "lemma": "hakea + partitiivi",
    "pos": None,
    "translation_ru": "искать + партитив",
    "example_fi": "Haen töitä.",
    "example_ru": "Я ищу работу.",
    "kind": "pattern",
    "meta": {},
}


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
        "openai_api_key": "sk-test",
        "openai_text_model": "gpt-5.6-terra",
        "openai_timeout_seconds": 1.0,
        "breaker_max_calls": 60,
        "breaker_window_minutes": 10,
    }
    return Settings(**{**defaults, **overrides})


def make_breaker(**overrides) -> CallBreaker:
    defaults = {"max_calls": 60, "window": timedelta(minutes=10)}
    return CallBreaker(**{**defaults, **overrides})


def make_message(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text, from_user=SimpleNamespace(id=1), answer=AsyncMock())


def make_command(args: str | None) -> CommandObject:
    return CommandObject(prefix="/", command="add", args=args)


def make_callback(data: str) -> SimpleNamespace:
    return SimpleNamespace(
        data=data,
        from_user=SimpleNamespace(id=1),
        answer=AsyncMock(),
        message=SimpleNamespace(answer=AsyncMock()),
    )


def patch_check_and_suggest(
    monkeypatch, reply_ru: str, candidates: list[dict], needs_clarification: bool = False
) -> AsyncMock:
    mock = AsyncMock(
        return_value=(reply_ru, needs_clarification, candidates, TokenUsage(10, 5, 15))
    )
    monkeypatch.setattr("kielikaveri.bot.add.check_and_suggest", mock)
    return mock


def patch_resolve_note_forms(monkeypatch, forms: dict | None = None) -> AsyncMock:
    mock = AsyncMock(return_value=(ResolvedForms(forms or {}, "fst", True), None))
    monkeypatch.setattr("kielikaveri.bot.add.resolve_note_forms", mock)
    return mock


def texts_of(mock_answer: AsyncMock) -> list[str]:
    return [call.args[0] for call in mock_answer.call_args_list]


# --- add_command (no args / with args) ---------------------------------------------


async def test_add_without_text_prompts_for_it(session_factory):
    message = make_message("/add")

    await add_command(
        message, make_command(None), make_state(), session_factory, make_settings(), make_breaker()
    )

    message.answer.assert_awaited_once()
    assert "текст" in message.answer.call_args.args[0]


async def test_add_with_args_runs_the_same_analyzer_as_plain_chat(session_factory, monkeypatch):
    mock = patch_check_and_suggest(monkeypatch, "Нашла кое-что.", [])
    message = make_message("/add Haen töitä.")

    await add_command(
        message,
        make_command("Haen töitä."),
        make_state(),
        session_factory,
        make_settings(),
        make_breaker(),
    )

    mock.assert_awaited_once()
    assert mock.await_args.args[3] == "Haen töitä."


# --- chat_message: errors / empty replies --------------------------------------------


async def test_chat_without_openai_key_answers_gracefully(session_factory):
    message = make_message("Haen töitä.")
    settings = make_settings(openai_api_key="")

    await chat_message(message, make_state(), session_factory, settings, make_breaker())

    message.answer.assert_awaited_once_with("Ответить не могу - не настроен OpenAI.")


async def test_chat_reports_a_tripped_breaker_honestly(session_factory, monkeypatch):
    monkeypatch.setattr(
        "kielikaveri.bot.add.check_and_suggest",
        AsyncMock(side_effect=CircuitOpenError("stopped")),
    )
    message = make_message("Haen töitä.")

    await chat_message(message, make_state(), session_factory, make_settings(), make_breaker())

    assert (
        "предохранитель" in message.answer.call_args.args[0].lower()
        or "баг" in message.answer.call_args.args[0]
    )


async def test_chat_reports_openai_unavailable_honestly(session_factory, monkeypatch):
    monkeypatch.setattr(
        "kielikaveri.bot.add.check_and_suggest",
        AsyncMock(side_effect=openai.APIConnectionError(request=SimpleNamespace())),
    )
    message = make_message("Haen töitä.")

    await chat_message(message, make_state(), session_factory, make_settings(), make_breaker())

    assert "недоступен" in message.answer.call_args.args[0]
    assert "/learn" in message.answer.call_args.args[0]


async def test_chat_falls_back_when_the_llm_returns_an_empty_reply_with_no_candidates(
    session_factory, monkeypatch
):
    patch_check_and_suggest(monkeypatch, "   ", [])
    message = make_message("...")

    await chat_message(message, make_state(), session_factory, make_settings(), make_breaker())

    message.answer.assert_awaited_once_with("Не нашла, что ответить - попробуй переформулировать.")


async def test_chat_with_no_candidates_only_sends_the_reply(session_factory, monkeypatch):
    patch_check_and_suggest(monkeypatch, "Просто ответ, без карточек.", [])
    message = make_message("Что значит kiitos?")

    await chat_message(message, make_state(), session_factory, make_settings(), make_breaker())

    message.answer.assert_awaited_once_with("Просто ответ, без карточек.")


async def test_chat_concurrent_identical_text_does_not_crash_on_cache_write(
    session_factory, monkeypatch
):
    patch_resolve_note_forms(monkeypatch)
    patch_check_and_suggest(monkeypatch, "Нашла кое-что.", [WORD_CANDIDATE])
    message_1 = make_message("Haen töitä.")
    message_2 = make_message("Haen töitä.")

    await asyncio.gather(
        chat_message(message_1, make_state(), session_factory, make_settings(), make_breaker()),
        chat_message(message_2, make_state(), session_factory, make_settings(), make_breaker()),
    )

    async with session_factory() as session:
        cached = (await session.scalars(select(IngestCache))).all()
    assert len(cached) == 1


async def test_chat_uses_the_cache_and_skips_the_llm_call(session_factory, monkeypatch):
    from kielikaveri.ingest import hash_text, store_cached_chat

    async with session_factory() as session:
        await store_cached_chat(
            session,
            hash_text("Haen töitä.", "gpt-5.6-terra", is_follow_up=False),
            "gpt-5.6-terra",
            "Из кэша.",
            False,
            [WORD_CANDIDATE],
        )
        await session.commit()

    mock = AsyncMock()
    monkeypatch.setattr("kielikaveri.bot.add.check_and_suggest", mock)
    patch_resolve_note_forms(monkeypatch)
    message = make_message("Haen töitä.")

    await chat_message(message, make_state(), session_factory, make_settings(), make_breaker())

    mock.assert_not_called()
    assert message.answer.call_args_list[0].args[0] == "Из кэша."


async def test_chat_drops_duplicates_of_existing_notes(session_factory, monkeypatch):
    async with session_factory() as session:
        session.add(
            Note(
                id="n1",
                user_id=1,
                lemma="hakea",
                pos="verbi",
                translation_ru="искать",
                example_fi="x",
                example_ru="y",
                kind="word",
                meta={},
            )
        )
        await session.commit()

    patch_check_and_suggest(monkeypatch, "Нашла кое-что.", [WORD_CANDIDATE])
    message = make_message("Haen töitä.")

    await chat_message(message, make_state(), session_factory, make_settings(), make_breaker())

    assert texts_of(message.answer) == ["Нашла кое-что.", "Все 1 кандидатов уже есть в базе."]


# --- needs_clarification: bare text gets a question, not an unsolicited answer ------


async def test_chat_with_bare_text_asks_instead_of_acting(session_factory, monkeypatch):
    patch_check_and_suggest(
        monkeypatch, "Пришлёшь перевод или назовёшь слова?", [], needs_clarification=True
    )
    message = make_message("Naapurit auttavat talkoissa.")
    state = make_state()

    await chat_message(message, state, session_factory, make_settings(), make_breaker())

    message.answer.assert_awaited_once_with("Пришлёшь перевод или назовёшь слова?")
    assert await state.get_state() == AddStates.awaiting_instruction.state
    data = await state.get_data()
    assert data["pending_text"] == "Naapurit auttavat talkoissa."


async def test_chat_follow_up_passes_the_pending_text_as_context(session_factory, monkeypatch):
    mock = patch_check_and_suggest(monkeypatch, "Вот перевод: сосед.", [])
    state = make_state()
    await state.set_state(AddStates.awaiting_instruction)
    await state.update_data(pending_text="Naapurit auttavat talkoissa.")
    message = make_message("переведи naapuri")

    await chat_message(message, state, session_factory, make_settings(), make_breaker())

    assert mock.await_args.kwargs["context_text"] == "Naapurit auttavat talkoissa."
    assert mock.await_args.args[3] == "переведи naapuri"
    assert await state.get_state() is None


async def test_chat_follow_up_clears_pending_state_even_on_error(session_factory, monkeypatch):
    monkeypatch.setattr(
        "kielikaveri.bot.add.check_and_suggest",
        AsyncMock(side_effect=CircuitOpenError("stopped")),
    )
    state = make_state()
    await state.set_state(AddStates.awaiting_instruction)
    await state.update_data(pending_text="Naapurit auttavat talkoissa.")
    message = make_message("переведи naapuri")

    await chat_message(message, state, session_factory, make_settings(), make_breaker())

    assert await state.get_state() is None


# --- saving: always asks which deck, even with just the default one ---------------


async def test_chat_asks_which_deck_even_when_only_one_exists(session_factory, monkeypatch):
    patch_check_and_suggest(monkeypatch, "Добавляю.", [WORD_CANDIDATE])
    message = make_message("добавь hakea")
    state = make_state()

    await chat_message(message, state, session_factory, make_settings(), make_breaker())

    async with session_factory() as session:
        assert (await session.scalars(select(Note))).all() == []
    assert await state.get_state() == AddStates.choosing_deck.state
    keyboard = message.answer.call_args.kwargs["reply_markup"]
    labels = {btn.text for row in keyboard.inline_keyboard for btn in row}
    assert labels == {"Общая"}


async def test_deck_choice_callback_data_fits_telegrams_64_byte_limit(session_factory, monkeypatch):
    # Regression, found live 27.08.2026: callback_data packed a full uuid4
    # batch_id (36 chars) plus a full uuid4 deck id (36 chars) - 81 bytes,
    # over Telegram's 64-byte cap. Telegram silently rejected the whole
    # message (BUTTON_DATA_INVALID) and, since nothing caught that yet at
    # the time, the user saw "Добавляю." and then nothing, ever - a mocked
    # message.answer() in every other test never validates this, because
    # only the real Telegram API enforces the limit.
    patch_check_and_suggest(monkeypatch, "Добавляю.", [WORD_CANDIDATE])
    message = make_message("добавь hakea")
    state = make_state()

    await chat_message(message, state, session_factory, make_settings(), make_breaker())

    keyboard = message.answer.call_args.kwargs["reply_markup"]
    for row in keyboard.inline_keyboard:
        for btn in row:
            assert len(btn.callback_data.encode("utf-8")) <= 64, btn.callback_data


async def test_chat_asks_which_deck_when_more_than_one_exists(session_factory, monkeypatch):
    async with session_factory() as session:
        await create_deck(session, 1, "Общая")
        await create_deck(session, 1, "Из книги")
        await session.commit()

    patch_check_and_suggest(monkeypatch, "Добавляю.", [WORD_CANDIDATE])
    message = make_message("добавь hakea")
    state = make_state()

    await chat_message(message, state, session_factory, make_settings(), make_breaker())

    assert await state.get_state() == AddStates.choosing_deck.state
    data = await state.get_data()
    assert data["candidates"] == [WORD_CANDIDATE]
    async with session_factory() as session:
        assert (await session.scalars(select(Note))).all() == []
    # The keyboard offers both decks.
    keyboard = message.answer.call_args.kwargs["reply_markup"]
    labels = {btn.text for row in keyboard.inline_keyboard for btn in row}
    assert labels == {"Общая", "Из книги"}


async def test_add_deck_choice_saves_into_the_picked_deck_and_reports_the_new_count(
    session_factory, monkeypatch
):
    patch_resolve_note_forms(monkeypatch)
    async with session_factory() as session:
        deck_a = await create_deck(session, 1, "Общая")
        deck_b = await create_deck(session, 1, "Из книги")
        await set_active_deck(session, 1, deck_a.id)
        await session.commit()
        source = await _make_source(session)

    state = make_state()
    await state.set_state(AddStates.choosing_deck)
    await state.update_data(batch_id="batch-1", candidates=[WORD_CANDIDATE], source_id=source)
    callback = make_callback(f"adddeck:batch-1:{deck_b.id}")

    await add_deck_choice(callback, state, session_factory, make_settings(), make_breaker())

    async with session_factory() as session:
        note = (await session.scalars(select(Note))).one()
        user = await session.get(User, 1)
    assert note.deck_id == deck_b.id
    assert user.last_deck_id == deck_b.id
    report = callback.message.answer.call_args.args[0]
    assert "Из книги" in report
    assert "1 слов" in report
    assert await state.get_state() is None


async def test_add_deck_choice_rejects_a_stale_batch(session_factory):
    state = make_state()
    await state.set_state(AddStates.choosing_deck)
    await state.update_data(batch_id="batch-1", candidates=[WORD_CANDIDATE], source_id="src")
    callback = make_callback("adddeck:old-batch:deck-x")

    await add_deck_choice(callback, state, session_factory, make_settings(), make_breaker())

    callback.answer.assert_awaited_once_with(
        "Эта подборка уже неактуальна - пришли текст ещё раз.", show_alert=True
    )
    async with session_factory() as session:
        assert (await session.scalars(select(Note))).all() == []


async def test_add_deck_choice_stray_callback_is_rejected_outside_the_state():
    callback = make_callback("adddeck:batch-1:deck-x")

    await add_deck_choice_stray(callback)

    callback.answer.assert_awaited_once_with(
        "Эта подборка уже неактуальна - пришли текст ещё раз.", show_alert=True
    )


async def test_chat_reports_a_failed_candidate_without_blocking_the_others(
    session_factory, monkeypatch
):
    async def fake_resolve(client, breaker, model, lemma, pos, now):
        if lemma == "hakea":
            raise CircuitOpenError("stopped")
        return ResolvedForms({}, "fst", True), None

    monkeypatch.setattr(
        "kielikaveri.bot.add.resolve_note_forms", AsyncMock(side_effect=fake_resolve)
    )
    patch_check_and_suggest(monkeypatch, "Добавляю.", [WORD_CANDIDATE, PATTERN_CANDIDATE])
    message = make_message("добавь hakea, hakea + partitiivi")
    state = make_state()

    await chat_message(message, state, session_factory, make_settings(), make_breaker())

    data = await state.get_data()
    async with session_factory() as session:
        deck_id = (await active_deck(session, 1)).id
    callback = make_callback(f"adddeck:{data['batch_id']}:{deck_id}")
    await add_deck_choice(callback, state, session_factory, make_settings(), make_breaker())

    async with session_factory() as session:
        notes = (await session.scalars(select(Note))).all()
    assert [n.lemma for n in notes] == ["hakea + partitiivi"]
    report = callback.message.answer.call_args.args[0]
    assert "Не удалось сохранить «hakea»" in report
    assert "hakea + partitiivi" in report


async def test_chat_reports_an_honest_error_instead_of_dying_silently(session_factory, monkeypatch):
    # Regression, found live 27.08.2026: an exception anywhere in the
    # dedup-and-save path used to die silently after "Добавляю." - the user
    # saw a confirmation-sounding reply and then nothing, with no way to
    # tell whether anything was saved.
    patch_check_and_suggest(monkeypatch, "Добавляю.", [WORD_CANDIDATE])
    monkeypatch.setattr(
        "kielikaveri.bot.add.canonical_key", MagicMock(side_effect=RuntimeError("fst exploded"))
    )
    message = make_message("добавь hakea")

    await chat_message(message, make_state(), session_factory, make_settings(), make_breaker())

    async with session_factory() as session:
        assert (await session.scalars(select(Note))).all() == []
    report = texts_of(message.answer)[-1]
    assert "не получилось" in report.lower()


# --- /delete ---------------------------------------------------------------------


async def _make_source(session) -> str:
    from kielikaveri.db.models import Source

    source = Source(type="other", ref="test", context_fi="x")
    session.add(source)
    await session.commit()
    return source.id


async def _add_note(session_factory, *, lemma="naapuri", pos="substantiivi", deck_name="Общая"):
    async with session_factory() as session:
        deck = await create_deck(session, 1, deck_name)
        session.add(
            Note(
                id=f"note-{lemma}-{deck_name}",
                user_id=1,
                lemma=lemma,
                pos=pos,
                translation_ru="сосед",
                example_fi="x",
                example_ru="y",
                kind="word",
                deck_id=deck.id,
                meta={},
            )
        )
        await session.commit()
        return deck.id


async def test_delete_without_a_word_prompts_for_it(session_factory):
    message = make_message("/delete")

    await delete_command(message, make_command(None), session_factory)

    assert "слово" in message.answer.call_args.args[0]


async def test_delete_reports_no_match(session_factory):
    message = make_message("/delete naapuri")

    await delete_command(message, make_command("naapuri"), session_factory)

    message.answer.assert_awaited_once_with("Не нашла такое слово.")


async def test_delete_shows_a_confirm_button_for_a_single_match(session_factory):
    await _add_note(session_factory)
    message = make_message("/delete naapuri")

    await delete_command(message, make_command("naapuri"), session_factory)

    keyboard = message.answer.call_args.kwargs["reply_markup"]
    buttons = [btn for row in keyboard.inline_keyboard for btn in row]
    assert any("naapuri" in btn.text for btn in buttons)
    assert any(btn.callback_data == "delnote:cancel" for btn in buttons)


async def test_delete_lists_every_match_when_ambiguous(session_factory):
    await _add_note(session_factory, deck_name="Общая")
    await _add_note(session_factory, deck_name="Из книги")
    message = make_message("/delete naapuri")

    await delete_command(message, make_command("naapuri"), session_factory)

    keyboard = message.answer.call_args.kwargs["reply_markup"]
    delete_buttons = [
        btn
        for row in keyboard.inline_keyboard
        for btn in row
        if btn.callback_data.startswith("delnote:") and btn.callback_data != "delnote:cancel"
    ]
    assert len(delete_buttons) == 2


async def test_delete_matches_a_finnish_capital_letter_case_insensitively(session_factory):
    # SQLite's built-in lower() only folds ASCII - func.lower('Äiti') stays
    # 'Äiti', so a SQL-side comparison against the user's typed "äiti" would
    # silently find nothing. Matching must happen in Python (str.lower()).
    await _add_note(session_factory, lemma="Äiti")
    message = make_message("/delete äiti")

    await delete_command(message, make_command("äiti"), session_factory)

    keyboard = message.answer.call_args.kwargs["reply_markup"]
    buttons = [btn for row in keyboard.inline_keyboard for btn in row]
    assert any("Äiti" in btn.text for btn in buttons)


async def test_delete_confirm_removes_the_note_and_its_cards(session_factory):
    await _add_note(session_factory)
    async with session_factory() as session:
        note = (await session.scalars(select(Note))).one()
        session.add(Card(id="card-1", note_id=note.id, user_id=1, type="recognition"))
        await session.commit()
        session.add(Review(card_id="card-1", user_id=1, rating=3))
        await session.commit()

    callback = make_callback(f"delnote:{note.id}")
    await delete_confirm(callback, session_factory)

    async with session_factory() as session:
        assert (await session.scalars(select(Note))).all() == []
        assert (await session.scalars(select(Card))).all() == []
        assert (await session.scalars(select(Review))).all() == []
    report = callback.message.answer.call_args.args[0]
    assert "naapuri" in report
    assert "Осталось 0 слов" in report


async def test_delete_confirm_on_an_already_deleted_note_is_honest(session_factory):
    callback = make_callback("delnote:does-not-exist")

    await delete_confirm(callback, session_factory)

    callback.answer.assert_awaited_once_with("Уже удалено.", show_alert=True)


async def test_delete_confirm_concurrent_double_tap_deletes_only_once(session_factory):
    # Same real-concurrency race class as learn_rate/ingest_cache: two
    # callback_query updates for one genuine double-tap on the same confirm
    # button both fetch the note before either commits. Proven with
    # asyncio.gather() on real aiosqlite, not mocked.
    await _add_note(session_factory)
    async with session_factory() as session:
        note = (await session.scalars(select(Note))).one()
    callback_1 = make_callback(f"delnote:{note.id}")
    callback_2 = make_callback(f"delnote:{note.id}")

    await asyncio.gather(
        delete_confirm(callback_1, session_factory),
        delete_confirm(callback_2, session_factory),
    )

    async with session_factory() as session:
        assert (await session.scalars(select(Note))).all() == []
    successes = [c for c in (callback_1, callback_2) if c.message.answer.call_args_list]
    assert len(successes) == 1
    losers = [c for c in (callback_1, callback_2) if c not in successes]
    losers[0].answer.assert_awaited_once_with("Уже удалено.", show_alert=True)


async def test_delete_cancel_deletes_nothing():
    callback = make_callback("delnote:cancel")

    await delete_cancel(callback)

    callback.answer.assert_awaited_once_with("Отменено.")
