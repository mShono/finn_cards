import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import openai
import pytest
from aiogram.filters import CommandObject
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from sqlalchemy import select

from kielikaveri.bot.add import add_command, chat_add_keep, chat_message
from kielikaveri.config import Settings
from kielikaveri.db.decks import create_deck, set_active_deck
from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import IngestCache, Note, Source
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


def patch_check_and_suggest(monkeypatch, reply_ru: str, candidates: list[dict]) -> AsyncMock:
    mock = AsyncMock(return_value=(reply_ru, candidates, TokenUsage(10, 5, 15)))
    monkeypatch.setattr("kielikaveri.bot.add.check_and_suggest", mock)
    return mock


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


# --- chat_message: no candidates / errors -------------------------------------------


async def test_chat_without_openai_key_answers_gracefully(session_factory):
    message = make_message("Haen töitä.")
    settings = make_settings(openai_api_key="")

    await chat_message(message, make_state(), session_factory, settings, make_breaker())

    message.answer.assert_awaited_once_with("Ответить не могу - не настроен OpenAI.")


async def test_chat_shows_the_reply_and_all_candidates_at_once(session_factory, monkeypatch):
    patch_check_and_suggest(
        monkeypatch, "Хорошее предложение.", [WORD_CANDIDATE, PATTERN_CANDIDATE]
    )
    message = make_message("Haen töitä kaupungista.")
    state = make_state()

    await chat_message(message, state, session_factory, make_settings(), make_breaker())

    # reply + a deck header + one message per candidate - not one at a time.
    texts = [call.args[0] for call in message.answer.call_args_list]
    assert texts[0] == "Хорошее предложение."
    assert any("hakea" in t for t in texts)
    assert any("partitiivi" in t for t in texts)
    data = await state.get_data()
    assert len(data["candidates"]) == 2
    assert data["added"] == []


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

    texts = [call.args[0] for call in message.answer.call_args_list]
    assert texts == ["Нашла кое-что.", "Все 1 кандидатов уже есть в базе."]


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
            session, hash_text("Haen töitä."), "gpt-5.6-terra", "Из кэша.", [WORD_CANDIDATE]
        )
        await session.commit()

    mock = AsyncMock()
    monkeypatch.setattr("kielikaveri.bot.add.check_and_suggest", mock)
    message = make_message("Haen töitä.")

    await chat_message(message, make_state(), session_factory, make_settings(), make_breaker())

    mock.assert_not_called()
    assert message.answer.call_args_list[0].args[0] == "Из кэша."


# --- chat_add_keep -------------------------------------------------------------------


async def _start_batch(
    session_factory, candidates: list[dict], deck_id: str | None = None
) -> tuple[FSMContext, str]:
    async with session_factory() as session:
        if deck_id is None:
            deck = await create_deck(session, 1, "Общая")
            deck_id = deck.id
        source = Source(type="other", ref="test", context_fi="Haen töitä.")
        session.add(source)
        await session.commit()
        source_id = source.id

    state = make_state()
    batch_id = "batch-1"
    await state.update_data(
        batch_id=batch_id,
        candidates=candidates,
        added=[],
        source_id=source_id,
        deck_id=deck_id,
    )
    return state, batch_id


async def test_chat_add_keep_saves_a_word_note_with_resolved_forms_and_the_active_deck(
    session_factory, monkeypatch
):
    monkeypatch.setattr(
        "kielikaveri.bot.add.resolve_note_forms",
        AsyncMock(
            return_value=(
                ResolvedForms({"preesens_1s": "haen"}, "fst", True),
                None,
            )
        ),
    )
    state, batch_id = await _start_batch(session_factory, [WORD_CANDIDATE])
    callback = make_callback(f"chat:add:{batch_id}:0")

    await chat_add_keep(callback, state, session_factory, make_settings(), make_breaker())

    async with session_factory() as session:
        notes = (await session.scalars(select(Note))).all()
    assert len(notes) == 1
    assert notes[0].lemma == "hakea"
    assert notes[0].meta["principal_forms"] == {"preesens_1s": "haen"}
    assert notes[0].meta["origin"] == "text"
    assert notes[0].deck_id is not None
    assert notes[0].source_id is not None

    async with session_factory() as session:
        sources = (await session.scalars(select(Source))).all()
    assert len(sources) == 1
    assert sources[0].context_fi == "Haen töitä."
    callback.answer.assert_awaited_once_with("Добавлено.")


async def test_chat_add_keep_rejects_a_stale_batch_id(session_factory):
    state, _batch_id = await _start_batch(session_factory, [WORD_CANDIDATE])
    callback = make_callback("chat:add:old-batch:0")

    await chat_add_keep(callback, state, session_factory, make_settings(), make_breaker())

    callback.answer.assert_awaited_once_with(
        "Эта подборка уже неактуальна - пришли текст ещё раз.", show_alert=True
    )
    async with session_factory() as session:
        assert (await session.scalars(select(Note))).all() == []


async def test_chat_add_keep_rejects_an_already_added_index(session_factory, monkeypatch):
    monkeypatch.setattr(
        "kielikaveri.bot.add.resolve_note_forms",
        AsyncMock(return_value=(ResolvedForms({}, "fst", True), None)),
    )
    state, batch_id = await _start_batch(session_factory, [WORD_CANDIDATE])
    callback_1 = make_callback(f"chat:add:{batch_id}:0")
    callback_2 = make_callback(f"chat:add:{batch_id}:0")

    await chat_add_keep(callback_1, state, session_factory, make_settings(), make_breaker())
    await chat_add_keep(callback_2, state, session_factory, make_settings(), make_breaker())

    callback_2.answer.assert_awaited_once_with("Уже добавлено.", show_alert=True)
    async with session_factory() as session:
        assert len((await session.scalars(select(Note))).all()) == 1


async def test_chat_add_keep_can_save_candidates_out_of_order(session_factory, monkeypatch):
    # Not sequential any more - the second candidate can be added before the
    # first, unlike the old cursor-based swipe.
    monkeypatch.setattr(
        "kielikaveri.bot.add.resolve_note_forms",
        AsyncMock(return_value=(ResolvedForms({}, "fst", True), None)),
    )
    state, batch_id = await _start_batch(session_factory, [WORD_CANDIDATE, PATTERN_CANDIDATE])
    callback = make_callback(f"chat:add:{batch_id}:1")

    await chat_add_keep(callback, state, session_factory, make_settings(), make_breaker())

    async with session_factory() as session:
        notes = (await session.scalars(select(Note))).all()
    assert len(notes) == 1
    assert notes[0].lemma == "hakea + partitiivi"


async def test_chat_add_keep_reuses_one_source_across_a_batch(session_factory, monkeypatch):
    monkeypatch.setattr(
        "kielikaveri.bot.add.resolve_note_forms",
        AsyncMock(return_value=(ResolvedForms({}, "fst", True), None)),
    )
    state, batch_id = await _start_batch(session_factory, [WORD_CANDIDATE, PATTERN_CANDIDATE])

    await chat_add_keep(
        make_callback(f"chat:add:{batch_id}:0"),
        state,
        session_factory,
        make_settings(),
        make_breaker(),
    )
    await chat_add_keep(
        make_callback(f"chat:add:{batch_id}:1"),
        state,
        session_factory,
        make_settings(),
        make_breaker(),
    )

    async with session_factory() as session:
        sources = (await session.scalars(select(Source))).all()
        notes = (await session.scalars(select(Note))).all()
    assert len(sources) == 1
    assert {n.source_id for n in notes} == {sources[0].id}


async def test_chat_add_keep_reports_a_tripped_breaker_and_does_not_save(
    session_factory, monkeypatch
):
    monkeypatch.setattr(
        "kielikaveri.bot.add.resolve_note_forms",
        AsyncMock(side_effect=CircuitOpenError("stopped")),
    )
    state, batch_id = await _start_batch(session_factory, [WORD_CANDIDATE])
    callback = make_callback(f"chat:add:{batch_id}:0")

    await chat_add_keep(callback, state, session_factory, make_settings(), make_breaker())

    callback.answer.assert_awaited_once_with(
        "Предохранитель сработал - карточка не сохранена.", show_alert=True
    )
    async with session_factory() as session:
        assert (await session.scalars(select(Note))).all() == []


async def test_chat_add_keep_reports_openai_unavailable_and_does_not_save(
    session_factory, monkeypatch
):
    monkeypatch.setattr(
        "kielikaveri.bot.add.resolve_note_forms",
        AsyncMock(side_effect=openai.APIConnectionError(request=SimpleNamespace())),
    )
    state, batch_id = await _start_batch(session_factory, [WORD_CANDIDATE])
    callback = make_callback(f"chat:add:{batch_id}:0")

    await chat_add_keep(callback, state, session_factory, make_settings(), make_breaker())

    callback.answer.assert_awaited_once_with(
        "OpenAI недоступен - карточка не сохранена, попробуй позже.", show_alert=True
    )
    async with session_factory() as session:
        assert (await session.scalars(select(Note))).all() == []


async def test_chat_add_keep_rejects_a_schema_invalid_note(session_factory, monkeypatch):
    monkeypatch.setattr(
        "kielikaveri.bot.add.resolve_note_forms",
        AsyncMock(return_value=(ResolvedForms({}, "fst", True), None)),
    )
    monkeypatch.setattr(
        "kielikaveri.bot.add.build_full_note",
        lambda candidate, resolved: {"lemma": "hakea"},
    )
    state, batch_id = await _start_batch(session_factory, [WORD_CANDIDATE])
    callback = make_callback(f"chat:add:{batch_id}:0")

    await chat_add_keep(callback, state, session_factory, make_settings(), make_breaker())

    callback.answer.assert_awaited_once_with(
        "Не удалось сохранить - невалидные данные от модели.", show_alert=True
    )
    async with session_factory() as session:
        assert (await session.scalars(select(Note))).all() == []


async def test_chat_add_keep_saves_into_the_users_active_deck(session_factory, monkeypatch):
    monkeypatch.setattr(
        "kielikaveri.bot.add.resolve_note_forms",
        AsyncMock(return_value=(ResolvedForms({}, "fst", True), None)),
    )
    async with session_factory() as session:
        deck_a = await create_deck(session, 1, "Общая")
        deck_b = await create_deck(session, 1, "Из книги")
        await set_active_deck(session, 1, deck_b.id)
        await session.commit()

    state, batch_id = await _start_batch(session_factory, [WORD_CANDIDATE], deck_id=deck_b.id)
    callback = make_callback(f"chat:add:{batch_id}:0")

    await chat_add_keep(callback, state, session_factory, make_settings(), make_breaker())

    async with session_factory() as session:
        note = (await session.scalars(select(Note))).one()
    assert note.deck_id == deck_b.id
    assert note.deck_id != deck_a.id
