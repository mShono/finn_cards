from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage

from kielikaveri.bot.edit import (
    EditStates,
    note_edit_apply,
    note_edit_cancel,
    note_edit_field_choice,
    note_edit_menu,
)
from kielikaveri.config import Settings
from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import Note, NoteKind
from kielikaveri.ingest import ResolvedForms
from kielikaveri.llm.breaker import CallBreaker, CircuitOpenError

NOW = datetime(2026, 8, 27, 10, 0, tzinfo=UTC)


@pytest.fixture
async def session_factory(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    await create_all(engine)
    yield make_session_factory(engine)
    await engine.dispose()


def make_state() -> FSMContext:
    return FSMContext(storage=MemoryStorage(), key=StorageKey(bot_id=0, chat_id=1, user_id=1))


def make_message(text: str = "") -> SimpleNamespace:
    return SimpleNamespace(text=text, from_user=SimpleNamespace(id=1), answer=AsyncMock())


def make_callback(data: str, user_id: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        data=data,
        from_user=SimpleNamespace(id=user_id),
        answer=AsyncMock(),
        message=SimpleNamespace(answer=AsyncMock()),
    )


def make_settings(**overrides) -> Settings:
    defaults = {
        "openai_api_key": "sk-test",
        "openai_text_model": "gpt-5.6-terra",
        "openai_timeout_seconds": 1.0,
        "breaker_max_calls": 60,
        "breaker_window_minutes": 10,
    }
    return Settings(**{**defaults, **overrides})


def make_breaker() -> CallBreaker:
    return CallBreaker(max_calls=60, window=timedelta(minutes=10))


async def _add_note(session_factory, **overrides) -> Note:
    defaults = {
        "id": "n1",
        "user_id": 1,
        "lemma": "hakea",
        "pos": "verbi",
        "translation_ru": "искать",
        "example_fi": "Haen töitä.",
        "example_ru": "Я ищу работу.",
        "kind": NoteKind.word,
        "meta": {"cefr": "B1"},
    }
    defaults.update(overrides)
    async with session_factory() as session:
        note = Note(**defaults)
        session.add(note)
        await session.commit()
    return note


# --- note_edit_menu ------------------------------------------------------------------


async def test_note_edit_menu_shows_field_buttons(session_factory):
    await _add_note(session_factory)
    callback = make_callback("noteedit:n1")

    await note_edit_menu(callback, session_factory)

    text = callback.message.answer.call_args.args[0]
    assert "hakea" in text and "искать" in text
    keyboard = callback.message.answer.call_args.kwargs["reply_markup"]
    codes = {b.callback_data for row in keyboard.inline_keyboard for b in row}
    assert codes == {"noteeditfield:n1:lm", "noteeditfield:n1:tr", "noteeditcancel"}


async def test_note_edit_menu_rejects_another_users_note(session_factory):
    await _add_note(session_factory)
    callback = make_callback("noteedit:n1", user_id=2)

    await note_edit_menu(callback, session_factory)

    callback.answer.assert_awaited_once_with("Не нашла эту карточку.", show_alert=True)
    callback.message.answer.assert_not_awaited()


# --- note_edit_field_choice / cancel ---------------------------------------------------


async def test_note_edit_field_choice_prompts_and_sets_state(session_factory):
    await _add_note(session_factory)
    state = make_state()
    callback = make_callback("noteeditfield:n1:tr")

    await note_edit_field_choice(callback, state, session_factory)

    assert await state.get_state() == EditStates.awaiting_value.state
    assert await state.get_data() == {"note_id": "n1", "field": "translation_ru"}
    assert "искать" in callback.message.answer.call_args.args[0]


async def test_note_edit_cancel_clears_state(session_factory):
    state = make_state()
    await state.set_state(EditStates.awaiting_value)
    await state.update_data(note_id="n1", field="lemma")
    callback = make_callback("noteeditcancel")

    await note_edit_cancel(callback, state)

    assert await state.get_state() is None
    callback.answer.assert_awaited_once_with("Отменено.")


# --- note_edit_apply: translation ------------------------------------------------------


async def test_note_edit_apply_updates_translation(session_factory):
    await _add_note(session_factory)
    state = make_state()
    await state.set_state(EditStates.awaiting_value)
    await state.update_data(note_id="n1", field="translation_ru")
    message = make_message("подавать заявление")

    await note_edit_apply(message, state, session_factory, make_settings(), make_breaker())

    async with session_factory() as session:
        note = await session.get(Note, "n1")
    assert note.translation_ru == "подавать заявление"
    assert await state.get_state() is None
    report = message.answer.call_args.args[0]
    assert "искать" in report and "подавать заявление" in report


async def test_note_edit_apply_rejects_empty_value(session_factory):
    await _add_note(session_factory)
    state = make_state()
    await state.set_state(EditStates.awaiting_value)
    await state.update_data(note_id="n1", field="translation_ru")
    message = make_message("   ")

    await note_edit_apply(message, state, session_factory, make_settings(), make_breaker())

    async with session_factory() as session:
        note = await session.get(Note, "n1")
    assert note.translation_ru == "искать"
    assert await state.get_state() == EditStates.awaiting_value.state


async def test_note_edit_apply_cancel_word_clears_state_without_saving(session_factory):
    await _add_note(session_factory)
    state = make_state()
    await state.set_state(EditStates.awaiting_value)
    await state.update_data(note_id="n1", field="translation_ru")
    message = make_message("отмена")

    await note_edit_apply(message, state, session_factory, make_settings(), make_breaker())

    async with session_factory() as session:
        note = await session.get(Note, "n1")
    assert note.translation_ru == "искать"
    assert await state.get_state() is None


# --- note_edit_apply: lemma, forms recompute -------------------------------------------


async def test_note_edit_apply_updates_lemma_and_recomputes_forms(session_factory, monkeypatch):
    await _add_note(session_factory)
    mock = AsyncMock(return_value=(ResolvedForms({"preesens_1s": "menen"}, "fst", True), None))
    monkeypatch.setattr("kielikaveri.bot.edit.resolve_note_forms", mock)
    state = make_state()
    await state.set_state(EditStates.awaiting_value)
    await state.update_data(note_id="n1", field="lemma")
    message = make_message("mennä")

    await note_edit_apply(message, state, session_factory, make_settings(), make_breaker())

    async with session_factory() as session:
        note = await session.get(Note, "n1")
    assert note.lemma == "mennä"
    assert note.meta["principal_forms"] == {"preesens_1s": "menen"}
    assert note.meta["cefr"] == "B1"  # untouched fields survive the meta rewrite
    mock.assert_awaited_once()
    report = message.answer.call_args.args[0]
    assert "hakea" in report and "mennä" in report


async def test_note_edit_apply_lemma_rejects_a_clash_with_an_existing_note(session_factory):
    await _add_note(session_factory, id="n1", lemma="hakea", pos="verbi")
    await _add_note(session_factory, id="n2", lemma="mennä", pos="verbi", translation_ru="идти")
    state = make_state()
    await state.set_state(EditStates.awaiting_value)
    await state.update_data(note_id="n1", field="lemma")
    message = make_message("mennä")

    await note_edit_apply(message, state, session_factory, make_settings(), make_breaker())

    async with session_factory() as session:
        note = await session.get(Note, "n1")
    assert note.lemma == "hakea"
    assert "уже есть" in message.answer.call_args.args[0]


async def test_note_edit_apply_lemma_saves_even_when_the_breaker_has_tripped(
    session_factory, monkeypatch
):
    await _add_note(session_factory)
    mock = AsyncMock(side_effect=CircuitOpenError("stopped"))
    monkeypatch.setattr("kielikaveri.bot.edit.resolve_note_forms", mock)
    state = make_state()
    await state.set_state(EditStates.awaiting_value)
    await state.update_data(note_id="n1", field="lemma")
    message = make_message("mennä")

    await note_edit_apply(message, state, session_factory, make_settings(), make_breaker())

    async with session_factory() as session:
        note = await session.get(Note, "n1")
    assert note.lemma == "mennä"
    reports = [call.args[0] for call in message.answer.call_args_list]
    assert any("предохранитель" in r for r in reports)
    assert any("mennä" in r for r in reports)


async def test_note_edit_apply_lemma_on_a_pattern_note_skips_forms_recompute(
    session_factory, monkeypatch
):
    await _add_note(
        session_factory,
        lemma="hakea + partitiivi",
        pos=None,
        kind=NoteKind.pattern,
        translation_ru="искать + партитив",
    )
    mock = AsyncMock()
    monkeypatch.setattr("kielikaveri.bot.edit.resolve_note_forms", mock)
    state = make_state()
    await state.set_state(EditStates.awaiting_value)
    await state.update_data(note_id="n1", field="lemma")
    message = make_message("hakea + elatiivi")

    await note_edit_apply(message, state, session_factory, make_settings(), make_breaker())

    async with session_factory() as session:
        note = await session.get(Note, "n1")
    assert note.lemma == "hakea + elatiivi"
    mock.assert_not_awaited()
