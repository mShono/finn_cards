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

from kielikaveri.bot.add import add_keep, add_skip, add_start, add_stray_callback
from kielikaveri.config import Settings
from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import Note, Source
from kielikaveri.ingest import ResolvedForms, TokenUsage
from kielikaveri.llm.breaker import CallBreaker, CircuitOpenError

NOW = datetime(2026, 8, 25, 10, 0, tzinfo=UTC)

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


# --- add_start ---------------------------------------------------------------------


async def test_add_without_text_prompts_for_it(session_factory):
    message = make_message("/add")
    await add_start(
        message, make_command(None), make_state(), session_factory, make_settings(), make_breaker()
    )

    message.answer.assert_awaited_once()
    assert "текст" in message.answer.call_args.args[0]


async def test_add_without_openai_key_answers_gracefully(session_factory):
    message = make_message("/add Haen töitä.")
    settings = make_settings(openai_api_key="")

    await add_start(
        message,
        make_command("Haen töitä."),
        make_state(),
        session_factory,
        settings,
        make_breaker(),
    )

    message.answer.assert_awaited_once_with("Добавление недоступно - не настроен OpenAI.")


async def test_add_generates_and_shows_the_first_candidate(session_factory, monkeypatch):
    monkeypatch.setattr(
        "kielikaveri.bot.add.generate_candidates",
        AsyncMock(return_value=([WORD_CANDIDATE], TokenUsage(10, 5, 15))),
    )
    message = make_message("/add Haen töitä.")
    state = make_state()

    await add_start(
        message,
        make_command("Haen töitä."),
        state,
        session_factory,
        make_settings(),
        make_breaker(),
    )

    message.answer.assert_awaited_once()
    assert "hakea" in message.answer.call_args.args[0]
    data = await state.get_data()
    assert data["candidates"] == [WORD_CANDIDATE]
    assert data["cursor"] == 0


async def test_add_drops_duplicates_of_existing_notes(session_factory, monkeypatch):
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

    monkeypatch.setattr(
        "kielikaveri.bot.add.generate_candidates",
        AsyncMock(return_value=([WORD_CANDIDATE], TokenUsage(10, 5, 15))),
    )
    message = make_message("/add Haen töitä.")

    await add_start(
        message,
        make_command("Haen töitä."),
        make_state(),
        session_factory,
        make_settings(),
        make_breaker(),
    )

    message.answer.assert_awaited_once_with("Все 1 кандидатов уже есть в базе - добавлять нечего.")


async def test_add_reports_a_tripped_breaker_honestly(session_factory, monkeypatch):
    monkeypatch.setattr(
        "kielikaveri.bot.add.generate_candidates",
        AsyncMock(side_effect=CircuitOpenError("stopped")),
    )
    message = make_message("/add Haen töitä.")

    await add_start(
        message,
        make_command("Haen töitä."),
        make_state(),
        session_factory,
        make_settings(),
        make_breaker(),
    )

    assert (
        "предохранитель" in message.answer.call_args.args[0].lower()
        or "баг" in message.answer.call_args.args[0]
    )


async def test_add_reports_openai_unavailable_honestly(session_factory, monkeypatch):
    monkeypatch.setattr(
        "kielikaveri.bot.add.generate_candidates",
        AsyncMock(side_effect=openai.APIConnectionError(request=SimpleNamespace())),
    )
    message = make_message("/add Haen töitä.")

    await add_start(
        message,
        make_command("Haen töitä."),
        make_state(),
        session_factory,
        make_settings(),
        make_breaker(),
    )

    assert "недоступен" in message.answer.call_args.args[0]
    assert "/learn" in message.answer.call_args.args[0]


async def test_add_uses_the_cache_and_skips_the_llm_call(session_factory, monkeypatch):
    from kielikaveri.ingest import hash_text, store_cached_candidates

    async with session_factory() as session:
        await store_cached_candidates(
            session, hash_text("Haen töitä."), "gpt-5.6-terra", [WORD_CANDIDATE]
        )
        await session.commit()

    generate = AsyncMock()
    monkeypatch.setattr("kielikaveri.bot.add.generate_candidates", generate)
    message = make_message("/add Haen töitä.")

    await add_start(
        message,
        make_command("Haen töitä."),
        make_state(),
        session_factory,
        make_settings(),
        make_breaker(),
    )

    generate.assert_not_called()
    message.answer.assert_awaited_once()


# --- add_skip / add_keep ------------------------------------------------------------


async def _start_review(session_factory, candidates: list[dict]) -> FSMContext:
    state = make_state()
    await state.set_state("AddStates:reviewing")
    await state.update_data(
        candidates=candidates,
        source_text="Haen töitä.",
        source_id=None,
        cursor=0,
        kept_count=0,
        skipped_count=0,
        duplicates_count=0,
    )
    return state


async def test_add_skip_advances_without_saving(session_factory):
    state = await _start_review(session_factory, [WORD_CANDIDATE])
    callback = make_callback("add:skip:0")

    await add_skip(callback, state)

    data = await state.get_data()
    assert data == {}  # cleared - it was the last (only) candidate
    callback.message.answer.assert_awaited_once()
    assert "пропущено 1" in callback.message.answer.call_args.args[0]

    async with session_factory() as session:
        assert (await session.scalars(select(Note))).all() == []


async def test_add_skip_rejects_a_stale_cursor(session_factory):
    state = await _start_review(session_factory, [WORD_CANDIDATE, PATTERN_CANDIDATE])
    callback = make_callback("add:skip:1")  # cursor is actually 0

    await add_skip(callback, state)

    callback.answer.assert_awaited_once_with("Эта карточка уже обработана.", show_alert=True)
    data = await state.get_data()
    assert data["cursor"] == 0


async def test_add_keep_saves_a_word_note_with_resolved_forms(session_factory, monkeypatch):
    monkeypatch.setattr(
        "kielikaveri.bot.add.resolve_note_forms",
        AsyncMock(
            return_value=(
                ResolvedForms({"preesens_1s": "haen"}, "fst", True),
                None,
            )
        ),
    )
    state = await _start_review(session_factory, [WORD_CANDIDATE])
    callback = make_callback("add:keep:0")

    await add_keep(callback, state, session_factory, make_settings(), make_breaker())

    async with session_factory() as session:
        notes = (await session.scalars(select(Note))).all()
    assert len(notes) == 1
    assert notes[0].lemma == "hakea"
    assert notes[0].meta["principal_forms"] == {"preesens_1s": "haen"}
    assert notes[0].meta["forms_source"] == "fst"
    assert notes[0].meta["origin"] == "text"
    assert notes[0].source_id is not None

    async with session_factory() as session:
        sources = (await session.scalars(select(Source))).all()
    assert len(sources) == 1
    assert sources[0].context_fi == "Haen töitä."


async def test_add_keep_pattern_kind_skips_the_llm_form_call(session_factory, monkeypatch):
    resolve = AsyncMock()
    monkeypatch.setattr("kielikaveri.bot.add.resolve_note_forms", resolve)
    state = await _start_review(session_factory, [PATTERN_CANDIDATE])
    callback = make_callback("add:keep:0")

    await add_keep(callback, state, session_factory, make_settings(), make_breaker())

    resolve.assert_not_called()
    async with session_factory() as session:
        notes = (await session.scalars(select(Note))).all()
    assert notes[0].meta["forms_source"] == "llm"
    assert notes[0].meta["forms_verified"] is False


async def test_add_keep_reuses_one_source_across_a_batch(session_factory, monkeypatch):
    monkeypatch.setattr(
        "kielikaveri.bot.add.resolve_note_forms",
        AsyncMock(return_value=(ResolvedForms({}, "fst", True), None)),
    )
    state = await _start_review(session_factory, [WORD_CANDIDATE, PATTERN_CANDIDATE])

    await add_keep(
        make_callback("add:keep:0"), state, session_factory, make_settings(), make_breaker()
    )
    await add_keep(
        make_callback("add:keep:1"), state, session_factory, make_settings(), make_breaker()
    )

    async with session_factory() as session:
        sources = (await session.scalars(select(Source))).all()
        notes = (await session.scalars(select(Note))).all()
    assert len(sources) == 1
    assert {n.source_id for n in notes} == {sources[0].id}


async def test_add_keep_reports_a_tripped_breaker_and_does_not_save(session_factory, monkeypatch):
    monkeypatch.setattr(
        "kielikaveri.bot.add.resolve_note_forms",
        AsyncMock(side_effect=CircuitOpenError("stopped")),
    )
    state = await _start_review(session_factory, [WORD_CANDIDATE])
    callback = make_callback("add:keep:0")

    await add_keep(callback, state, session_factory, make_settings(), make_breaker())

    callback.answer.assert_awaited_once_with(
        "Предохранитель сработал - карточка не сохранена.", show_alert=True
    )
    async with session_factory() as session:
        assert (await session.scalars(select(Note))).all() == []


async def test_add_keep_rejects_a_stale_cursor(session_factory):
    state = await _start_review(session_factory, [WORD_CANDIDATE, PATTERN_CANDIDATE])
    callback = make_callback("add:keep:1")  # cursor is actually 0

    await add_keep(callback, state, session_factory, make_settings(), make_breaker())

    callback.answer.assert_awaited_once_with("Эта карточка уже обработана.", show_alert=True)
    async with session_factory() as session:
        assert (await session.scalars(select(Note))).all() == []


# --- stray callback -------------------------------------------------------------------


async def test_add_stray_callback_after_the_session_ended():
    callback = make_callback("add:keep:0")

    await add_stray_callback(callback)

    callback.answer.assert_awaited_once_with(
        "Эта сессия добавления уже неактуальна - начните заново через /add", show_alert=True
    )
