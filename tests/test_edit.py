import json
import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import openai
import pytest
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from conftest import log_fields

from finn_cards.morphology import NOMINAL_FORMS, VERB_FORMS, generate_forms
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


def _fst_meta(lemma: str, pos: str) -> dict:
    # What /add stores for a verified note - built by the real FST, so a
    # stale set in a test is exactly the one production would have kept.
    result = generate_forms(lemma, pos)
    assert not result.ambiguous
    return {
        "principal_forms": result.principal_forms,
        "forms_source": "fst",
        "forms_verified": True,
    }


def patch_openai_client(monkeypatch, create: AsyncMock | None = None) -> SimpleNamespace:
    # The only mocked boundary on the real-FST path: the OpenAI client itself.
    client = SimpleNamespace(responses=SimpleNamespace(create=create or AsyncMock()))
    monkeypatch.setattr("kielikaveri.bot.edit.make_client", lambda *args, **kwargs: client)
    return client


def fake_llm_response(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(
        output_text=json.dumps(payload, ensure_ascii=False),
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
    )


# --- note_edit_menu ------------------------------------------------------------------


async def test_note_edit_menu_shows_field_buttons(session_factory):
    await _add_note(session_factory)
    callback = make_callback("noteedit:n1")

    await note_edit_menu(callback, session_factory)

    text = callback.message.answer.call_args.args[0]
    assert "hakea" in text and "искать" in text
    keyboard = callback.message.answer.call_args.kwargs["reply_markup"]
    codes = {b.callback_data for row in keyboard.inline_keyboard for b in row}
    assert codes == {"noteeditfield:n1:lm", "noteeditfield:n1:tr", "delnote:n1", "noteeditcancel"}


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
    # Real FST, real resolve_note_forms: the forms must be the NEW lemma's.
    # Same part of speech on both sides, so only the lemma is under test here.
    await _add_note(
        session_factory,
        lemma="puhua",
        translation_ru="говорить",
        example_fi="Puhun suomea.",
        meta={"cefr": "B1", **_fst_meta("puhua", "verbi")},
    )
    client = patch_openai_client(monkeypatch)
    state = make_state()
    await state.set_state(EditStates.awaiting_value)
    await state.update_data(note_id="n1", field="lemma")
    message = make_message("hakea")

    await note_edit_apply(message, state, session_factory, make_settings(), make_breaker())

    async with session_factory() as session:
        note = await session.get(Note, "n1")
    assert note.lemma == "hakea"
    assert note.pos == "verbi"
    assert note.meta["principal_forms"] == {
        "preesens_1s": "haen",
        "preesens_3s": "hakee",
        "imperfekti_3s": "haki",
        "konditionaali_1s": "hakisin",
        "imperatiivi_2s": "hae",
        "nut_partisiippi": "hakenut",
        "passiivi": "haetaan",
    }
    assert note.meta["forms_source"] == "fst"
    assert note.meta["forms_verified"] is True
    assert note.meta["cefr"] == "B1"  # untouched fields survive the meta rewrite
    client.responses.create.assert_not_called()  # FST alone was enough
    report = message.answer.call_args.args[0]
    assert "puhua" in report and "hakea" in report


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


async def test_note_edit_apply_lemma_handles_a_clash_that_appears_during_the_llm_call(
    session_factory, monkeypatch
):
    # The race uq_notes_user_deck_lemma_pos exists for, reproduced in the real
    # window: the clash lookup above runs before resolve_note_forms, so anything
    # that takes "mennä" while that call is in flight only surfaces at commit.
    await _add_note(session_factory)

    async def steal_the_lemma(*args, **kwargs):
        await _add_note(session_factory, id="n2", lemma="mennä", translation_ru="идти")
        return ResolvedForms({"preesens_1s": "menen"}, "fst", True), None

    monkeypatch.setattr("kielikaveri.bot.edit.resolve_note_forms", steal_the_lemma)
    state = make_state()
    await state.set_state(EditStates.awaiting_value)
    await state.update_data(note_id="n1", field="lemma")
    message = make_message("mennä")

    await note_edit_apply(message, state, session_factory, make_settings(), make_breaker())

    async with session_factory() as session:
        note = await session.get(Note, "n1")
    assert note.lemma == "hakea"  # kept its old lemma, nothing half-written
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


# --- application-level logging ---------------------------------------------


async def test_note_edit_apply_translation_logs_save_event(session_factory, caplog):
    await _add_note(session_factory)
    state = make_state()
    await state.set_state(EditStates.awaiting_value)
    await state.update_data(note_id="n1", field="translation_ru")
    message = make_message("подавать заявление")

    with caplog.at_level(logging.INFO, logger="kielikaveri.bot.edit"):
        await note_edit_apply(message, state, session_factory, make_settings(), make_breaker())

    saves = [
        log_fields(r.message)
        for r in caplog.records
        if log_fields(r.message).get("event") == "edit.save"
    ]
    assert len(saves) == 1
    assert saves[0]["field"] == "translation_ru"
    assert saves[0]["note_id"] == "n1"


async def test_note_edit_apply_lemma_breaker_trip_logs_warning(
    session_factory, monkeypatch, caplog
):
    await _add_note(session_factory)
    monkeypatch.setattr(
        "kielikaveri.bot.edit.resolve_note_forms",
        AsyncMock(side_effect=CircuitOpenError("stopped")),
    )
    state = make_state()
    await state.set_state(EditStates.awaiting_value)
    await state.update_data(note_id="n1", field="lemma")
    message = make_message("mennä")

    with caplog.at_level(logging.WARNING, logger="kielikaveri.bot.edit"):
        await note_edit_apply(message, state, session_factory, make_settings(), make_breaker())

    warnings = [
        log_fields(r.message)
        for r in caplog.records
        if r.levelno == logging.WARNING
        and log_fields(r.message).get("event") == "edit.lemma_forms_skipped"
    ]
    assert len(warnings) == 1
    assert warnings[0]["reason"] == "breaker_open"


# --- note_edit_apply: lemma edit on the real FST/resolve path --------------------------
#
# Nothing below mocks resolve_note_forms, the FST or the forms/POS checks -
# only the OpenAI client, and only where the lemma is really ambiguous.


async def test_note_edit_apply_lemma_saves_the_pos_the_new_lemma_resolved_to(
    session_factory, monkeypatch
):
    # hakea (verbi) -> talo: the FST only knows talo as a noun, so the forms
    # are nominal ones and the note must say substantiivi, not keep verbi.
    await _add_note(session_factory, meta={"cefr": "B1", **_fst_meta("hakea", "verbi")})
    client = patch_openai_client(monkeypatch)
    state = make_state()
    await state.set_state(EditStates.awaiting_value)
    await state.update_data(note_id="n1", field="lemma")

    await note_edit_apply(
        make_message("talo"), state, session_factory, make_settings(), make_breaker()
    )

    async with session_factory() as session:
        note = await session.get(Note, "n1")
    assert note.lemma == "talo"
    assert note.pos == "substantiivi"
    forms = note.meta["principal_forms"]
    assert set(forms) == set(NOMINAL_FORMS)
    assert not set(forms) & set(VERB_FORMS)  # nothing left of hakea's verb table
    assert forms["nominatiivi"] == "talo"
    assert forms["genetiivi"] == "talon"
    assert forms["partitiivi"] == "taloa"
    assert forms["inessiivi"] == "talossa"
    assert forms["monikon_partitiivi"] == "taloja"
    assert "haen" not in forms.values()
    assert note.meta["forms_source"] == "fst"
    assert note.meta["forms_verified"] is True
    assert note.meta["cefr"] == "B1"
    client.responses.create.assert_not_called()


async def test_note_edit_apply_lemma_to_an_ambiguous_lemma_asks_the_llm_about_the_new_one(
    session_factory, monkeypatch
):
    # talo (substantiivi) -> mennä: POS flips to verbi, and two of mennä's
    # forms are FST-ambiguous, so the tie-break call goes out - for mennä,
    # with mennä's own FST candidates, and its answer lands on the note.
    await _add_note(
        session_factory,
        lemma="talo",
        pos="substantiivi",
        translation_ru="дом",
        example_fi="Talo on iso.",
        meta=_fst_meta("talo", "substantiivi"),
    )
    create = AsyncMock(
        return_value=fake_llm_response({"preesens_1s": "menen", "imperatiivi_2s": "mene"})
    )
    client = patch_openai_client(monkeypatch, create)
    state = make_state()
    await state.set_state(EditStates.awaiting_value)
    await state.update_data(note_id="n1", field="lemma")

    await note_edit_apply(
        make_message("mennä"), state, session_factory, make_settings(), make_breaker()
    )

    async with session_factory() as session:
        note = await session.get(Note, "n1")
    assert note.lemma == "mennä"
    assert note.pos == "verbi"
    assert note.meta["principal_forms"] == {
        "preesens_1s": "menen",
        "preesens_3s": "menee",
        "imperfekti_3s": "meni",
        "konditionaali_1s": "menisin",
        "imperatiivi_2s": "mene",
        "nut_partisiippi": "mennyt",
        "passiivi": "mennään",
    }
    assert note.meta["forms_source"] == "fst+llm"
    assert note.meta["forms_verified"] is True
    client.responses.create.assert_awaited_once()
    request = client.responses.create.call_args.kwargs
    assert "'mennä' (verbi)" in request["instructions"]
    assert json.loads(request["input"]) == {
        "preesens_1s": ["meen", "menen"],
        "imperatiivi_2s": ["mee", "mene"],
    }


@pytest.mark.parametrize("failure", ["breaker", "openai"])
async def test_note_edit_apply_lemma_drops_the_old_lemmas_forms_when_recompute_fails(
    session_factory, monkeypatch, failure
):
    # hakea's verified forms must not survive under the lemma mennä: the
    # inflection card would ask "mennä → 1s" and reveal "haen".
    await _add_note(session_factory, meta={"cefr": "B1", **_fst_meta("hakea", "verbi")})
    if failure == "breaker":
        client = patch_openai_client(monkeypatch)
        breaker = CallBreaker(max_calls=0, window=timedelta(minutes=10))
    else:
        client = patch_openai_client(
            monkeypatch,
            AsyncMock(side_effect=openai.APIConnectionError(request=SimpleNamespace())),
        )
        breaker = make_breaker()
    state = make_state()
    await state.set_state(EditStates.awaiting_value)
    await state.update_data(note_id="n1", field="lemma")
    message = make_message("mennä")

    await note_edit_apply(message, state, session_factory, make_settings(), breaker)

    async with session_factory() as session:
        note = await session.get(Note, "n1")
    assert note.lemma == "mennä"
    assert note.pos == "verbi"
    assert note.meta == {"cefr": "B1"}
    if failure == "breaker":
        client.responses.create.assert_not_called()
    else:
        client.responses.create.assert_awaited_once()
    reports = [call.args[0] for call in message.answer.call_args_list]
    assert any("не пересчитала" in r for r in reports)


async def test_note_edit_apply_same_lemma_keeps_its_forms_when_recompute_fails(
    session_factory, monkeypatch
):
    # The flip side of the test above: resubmitting the lemma the note
    # already has leaves nothing stale, so its forms stay.
    meta = {
        "principal_forms": {
            "preesens_1s": "menen",
            "preesens_3s": "menee",
            "imperfekti_3s": "meni",
            "konditionaali_1s": "menisin",
            "imperatiivi_2s": "mene",
            "nut_partisiippi": "mennyt",
            "passiivi": "mennään",
        },
        "forms_source": "fst+llm",
        "forms_verified": True,
    }
    await _add_note(session_factory, lemma="mennä", translation_ru="идти", meta=dict(meta))
    patch_openai_client(monkeypatch)
    state = make_state()
    await state.set_state(EditStates.awaiting_value)
    await state.update_data(note_id="n1", field="lemma")

    await note_edit_apply(
        make_message("mennä"),
        state,
        session_factory,
        make_settings(),
        CallBreaker(max_calls=0, window=timedelta(minutes=10)),
    )

    async with session_factory() as session:
        note = await session.get(Note, "n1")
    assert note.lemma == "mennä"
    assert note.meta == meta


async def test_note_edit_apply_lemma_clash_under_the_resolved_pos_keeps_the_old_note(
    session_factory, monkeypatch
):
    # The pre-resolve clash lookup filters by the OLD pos (verbi) and so
    # misses talo/substantiivi; the corrected pos must still hit the unique
    # index at commit, leaving hakea exactly as it was.
    hakea_meta = _fst_meta("hakea", "verbi")
    await _add_note(session_factory, meta=dict(hakea_meta))
    await _add_note(
        session_factory,
        id="n2",
        lemma="talo",
        pos="substantiivi",
        translation_ru="дом",
        meta=_fst_meta("talo", "substantiivi"),
    )
    patch_openai_client(monkeypatch)
    state = make_state()
    await state.set_state(EditStates.awaiting_value)
    await state.update_data(note_id="n1", field="lemma")
    message = make_message("talo")

    await note_edit_apply(message, state, session_factory, make_settings(), make_breaker())

    async with session_factory() as session:
        note = await session.get(Note, "n1")
    assert note.lemma == "hakea"
    assert note.pos == "verbi"
    assert note.meta == hakea_meta
    assert "уже есть" in message.answer.call_args.args[0]
