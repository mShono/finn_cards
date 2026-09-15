import json
import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from conftest import log_fields

from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import Note, NoteKind
from kielikaveri.ingest import (
    TokenUsage,
    _chat_schema,
    _load_note_schema,
    _log_llm_request,
    _log_llm_response,
    build_full_note,
    canonical_key,
    check_and_suggest,
    existing_note_keys,
    get_cached_chat,
    hash_text,
    resolve_ambiguous_forms,
    resolve_note_forms,
    resolve_note_pos,
    store_cached_chat,
)
from kielikaveri.llm.breaker import CallBreaker, CircuitOpenError

NOW = datetime(2026, 8, 25, 10, 0, tzinfo=UTC)


@pytest.fixture
async def session_factory(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    await create_all(engine)
    yield make_session_factory(engine)
    await engine.dispose()


def make_breaker() -> CallBreaker:
    return CallBreaker(max_calls=60, window=timedelta(minutes=10))


def fake_response(payload: dict, input_tokens=10, output_tokens=5) -> SimpleNamespace:
    return SimpleNamespace(
        output_text=json.dumps(payload, ensure_ascii=False),
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )


# --- hash_text / canonical_key -------------------------------------------------


def test_hash_text_is_stable_and_ignores_surrounding_whitespace():
    assert hash_text("Haen töitä.", "gpt-5.6-terra") == hash_text(
        "  Haen töitä.  ", "gpt-5.6-terra"
    )


def test_hash_text_differs_for_different_text():
    assert hash_text("Haen töitä.", "gpt-5.6-terra") != hash_text("Haen taloa.", "gpt-5.6-terra")


def test_hash_text_differs_for_different_model():
    # A model swap must miss the cache, not replay an old model's answer -
    # previously `model` was recorded in IngestCache but never checked back.
    assert hash_text("Haen töitä.", "gpt-5.6-terra") != hash_text("Haen töitä.", "gpt-6")


def test_hash_text_differs_for_follow_up_vs_fresh_turn():
    assert hash_text("Haen töitä.", "gpt-5.6-terra", is_follow_up=True) != hash_text(
        "Haen töitä.", "gpt-5.6-terra", is_follow_up=False
    )


def test_hash_text_differs_when_the_prompt_text_changes(monkeypatch):
    # Regression, found live 27.08.2026: editing _chat_instructions() (e.g. the
    # "don't claim a save you haven't made" fix) must invalidate old cache rows
    # for text sent before the edit - otherwise a resend of the same text keeps
    # replaying the pre-fix answer forever, with no LLM call and no visible sign.
    monkeypatch.setattr("kielikaveri.ingest._chat_instructions", lambda *, is_follow_up: "v1")
    before = hash_text("Haen töitä.", "gpt-5.6-terra")
    monkeypatch.setattr("kielikaveri.ingest._chat_instructions", lambda *, is_follow_up: "v2")
    after = hash_text("Haen töitä.", "gpt-5.6-terra")
    assert before != after


def test_canonical_key_lemmatizes_an_inflected_llm_lemma():
    # cards/instructions.md: the LLM sometimes returns a word form as "lemma".
    assert canonical_key("töitä", "substantiivi") == ("työ", "substantiivi")


def test_canonical_key_leaves_a_real_lemma_alone():
    assert canonical_key("hakea", "verbi") == ("hakea", "verbi")


def test_canonical_key_pattern_kind_uses_the_raw_construction():
    assert canonical_key("hakea + partitiivi", None) == ("hakea + partitiivi", None)


def test_canonical_key_resolves_a_capitalised_word_to_its_dictionary_form():
    # Case is handled here and nowhere else - whatever this returns is what
    # reaches the (user, deck, lemma, pos) unique index. Not by casefold():
    # the FST's own lemma is the canonical form, see the next test.
    assert canonical_key("Hakea", "verbi") == ("hakea", "verbi")
    assert canonical_key("Töitä", "substantiivi") == ("työ", "substantiivi")


def test_canonical_key_keeps_a_proper_noun_capitalised():
    # Why casefold() is not the normalisation to add on top: it would turn
    # Helsinki into helsinki.
    assert canonical_key("Helsingissä", "substantiivi") == ("Helsinki", "substantiivi")


def test_canonical_key_is_not_case_insensitive_for_an_ambiguous_lemma():
    # Pins a known pre-existing gap, not a contract worth keeping: when the FST
    # offers several lemmas, `if lemma in lemmas` can only match a lowercase
    # input, so a capitalised one falls through to lemmas[0] - a *different*
    # word. This is the mechanism behind the duplicate `seurojentalo` pair
    # found in production 11.09.2026. The unique index deliberately does not
    # paper over it: it compares what canonical_key() produced, so fixing the
    # case asymmetry means fixing it here, in the one canonicalisation.
    assert canonical_key("seurojentalo", "substantiivi") == ("seurojentalo", "substantiivi")
    assert canonical_key("Seurojentalo", "substantiivi") == ("seuratalo", "substantiivi")


# --- strict schema wrapper -------------------------------------------------------


def test_chat_schema_wraps_the_note_schema_and_excludes_fst_fields():
    schema = _chat_schema(_load_note_schema())
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["reply_ru", "needs_clarification", "candidates"]
    note_item = schema["properties"]["candidates"]["items"]
    assert "principal_forms" not in note_item["properties"]["meta"]["properties"]
    assert "origin" not in note_item["properties"]["meta"]["properties"]


# --- check_and_suggest ---------------------------------------------------------


async def test_check_and_suggest_parses_the_response_and_returns_usage():
    client = MagicMock()
    client.responses.create = AsyncMock(
        return_value=fake_response(
            {
                "reply_ru": "Нашла кое-что.",
                "needs_clarification": False,
                "candidates": [{"lemma": "hakea"}],
            }
        )
    )
    breaker = make_breaker()

    reply_ru, needs_clarification, candidates, usage = await check_and_suggest(
        client, breaker, "gpt-5.6-terra", "text", NOW
    )

    assert reply_ru == "Нашла кое-что."
    assert needs_clarification is False
    assert candidates == [{"lemma": "hakea"}]
    assert usage.input_tokens == 10
    assert usage.output_tokens == 5
    assert usage.total_tokens == 15
    client.responses.create.assert_called_once()
    assert client.responses.create.call_args.kwargs["model"] == "gpt-5.6-terra"
    assert client.responses.create.call_args.kwargs["input"] == "text"


async def test_check_and_suggest_combines_context_and_reply_for_a_follow_up_turn():
    client = MagicMock()
    client.responses.create = AsyncMock(
        return_value=fake_response(
            {"reply_ru": "Готово.", "needs_clarification": False, "candidates": []}
        )
    )
    breaker = make_breaker()

    await check_and_suggest(
        client,
        breaker,
        "gpt-5.6-terra",
        "переведи naapuri",
        NOW,
        context_text="Naapurit auttavat talkoissa.",
    )

    sent_input = client.responses.create.call_args.kwargs["input"]
    assert "Naapurit auttavat talkoissa." in sent_input
    assert "переведи naapuri" in sent_input
    sent_instructions = client.responses.create.call_args.kwargs["instructions"]
    assert "продолжение диалога" in sent_instructions


async def test_check_and_suggest_respects_a_tripped_breaker():
    client = MagicMock()
    client.responses.create = AsyncMock()
    breaker = CallBreaker(max_calls=0, window=timedelta(minutes=10))

    with pytest.raises(CircuitOpenError):
        await check_and_suggest(client, breaker, "gpt-5.6-terra", "text", NOW)

    client.responses.create.assert_not_called()


# --- resolve_note_forms -----------------------------------------------------------


async def test_resolve_note_forms_skips_the_llm_when_fst_is_unambiguous():
    client = MagicMock()
    client.responses.create = AsyncMock()
    breaker = make_breaker()

    resolved, usage = await resolve_note_forms(
        client, breaker, "gpt-5.6-terra", "hakea", "verbi", NOW
    )

    assert resolved.forms_source == "fst"
    assert resolved.forms_verified is True
    assert resolved.principal_forms["preesens_1s"] == "haen"
    assert usage is None
    client.responses.create.assert_not_called()


async def test_resolve_note_forms_asks_the_llm_only_for_the_ambiguous_form():
    client = MagicMock()
    client.responses.create = AsyncMock(
        return_value=fake_response({"monikon_genetiivi": "hampaiden"})
    )
    breaker = make_breaker()

    resolved, usage = await resolve_note_forms(
        client, breaker, "gpt-5.6-terra", "hammas", "substantiivi", NOW
    )

    assert resolved.forms_source == "fst+llm"
    assert resolved.forms_verified is True
    assert resolved.principal_forms["monikon_genetiivi"] == "hampaiden"
    assert (
        resolved.principal_forms["genetiivi"] == "hampaan"
    )  # untouched FST form survives the merge
    assert usage is not None
    client.responses.create.assert_called_once()


async def test_resolve_note_forms_handles_a_pos_with_no_forms_table():
    # adverbi is a valid note.pos (cards/schema.json), but forms_for_pos()
    # only covers verbi/substantiivi/adjektiivi - must degrade, not crash.
    client = MagicMock()
    client.responses.create = AsyncMock()
    breaker = make_breaker()

    resolved, usage = await resolve_note_forms(
        client, breaker, "gpt-5.6-terra", "kuitenkin", "adverbi", NOW
    )

    assert resolved.forms_source == "llm"
    assert resolved.forms_verified is False
    assert resolved.principal_forms == {}
    assert usage is None
    client.responses.create.assert_not_called()


# --- resolve_note_pos -------------------------------------------------------------
#
# Integration-level on the morphology side on purpose: the FST is the real
# one, only the OpenAI client is faked. The whole bug class these cover is
# the LLM naming a part of speech that belongs to a *different* lemma which
# merely spells the same, so a mocked pos_set_for_lemma() would test nothing.


async def test_resolve_note_pos_keeps_a_pos_the_fst_confirms():
    client = MagicMock()
    client.responses.create = AsyncMock()

    pos, usage = await resolve_note_pos(
        client, make_breaker(), "gpt-5.6-terra", "kissa", "substantiivi", "Kissa nukkuu.", NOW
    )

    assert pos == "substantiivi"
    assert usage is None
    client.responses.create.assert_not_called()


async def test_resolve_note_pos_corrects_when_the_lemma_allows_exactly_one():
    # The live bug: "hän tuli kotiin" makes the LLM answer verbi, which is
    # right about the sentence and wrong about the lemma - "tuli" the lemma
    # is only ever the noun "fire". One allowed pos means there is nothing
    # to choose between, so no second LLM call.
    client = MagicMock()
    client.responses.create = AsyncMock()

    pos, usage = await resolve_note_pos(
        client, make_breaker(), "gpt-5.6-terra", "tuli", "verbi", "Hän tuli kotiin.", NOW
    )

    assert pos == "substantiivi"
    assert usage is None
    client.responses.create.assert_not_called()


async def test_resolve_note_pos_keeps_the_llm_pos_for_a_lemma_the_fst_does_not_know():
    # Unknown words must behave exactly as before: the FST can neither
    # confirm nor refute, so nothing is corrected and nothing is asked.
    client = MagicMock()
    client.responses.create = AsyncMock()

    pos, usage = await resolve_note_pos(
        client, make_breaker(), "gpt-5.6-terra", "xyzquu", "substantiivi", "Xyzquu on täällä.", NOW
    )

    assert pos == "substantiivi"
    assert usage is None
    client.responses.create.assert_not_called()


async def test_resolve_note_pos_never_picks_the_first_of_several_allowed():
    # "hakea" really is both a verb and a noun under one lemma. The FST has
    # no ranking to offer (both readings weigh 0.0), so the code must ask
    # rather than take whichever came out of the analyzer first.
    client = MagicMock()
    client.responses.create = AsyncMock(return_value=fake_response({"pos": "verbi"}))

    pos, usage = await resolve_note_pos(
        client, make_breaker(), "gpt-5.6-terra", "hakea", "adjektiivi", "Haen töitä.", NOW
    )

    client.responses.create.assert_called_once()
    assert pos == "verbi"
    assert usage is not None


async def test_resolve_note_pos_asks_the_llm_with_an_enum_built_from_the_fst():
    client = MagicMock()
    client.responses.create = AsyncMock(return_value=fake_response({"pos": "substantiivi"}))

    pos, _usage = await resolve_note_pos(
        client, make_breaker(), "gpt-5.6-terra", "hakea", "adjektiivi", "Hakea oli pitkä.", NOW
    )

    kwargs = client.responses.create.call_args.kwargs
    schema = kwargs["text"]["format"]["schema"]
    # sorted() of the FST set, so the enum can't smuggle in an order that
    # means something - and "adjektiivi", the LLM's own first answer, is not
    # among the options precisely because the FST ruled it out.
    assert schema["properties"]["pos"]["enum"] == ["substantiivi", "verbi"]
    assert kwargs["text"]["format"]["strict"] is True
    assert "Hakea oli pitkä." in kwargs["input"]
    assert pos == "substantiivi"


async def test_resolve_note_pos_rejects_a_choice_outside_the_fst_set():
    # The strict enum should make this impossible; if it ever happens we keep
    # the LLM's original answer rather than inventing a pick of our own.
    client = MagicMock()
    client.responses.create = AsyncMock(return_value=fake_response({"pos": "adverbi"}))

    pos, _usage = await resolve_note_pos(
        client, make_breaker(), "gpt-5.6-terra", "hakea", "adjektiivi", "Haen töitä.", NOW
    )

    assert pos == "adjektiivi"


async def test_resolve_note_pos_leaves_a_missing_pos_alone():
    # kind="pattern" and the strict-schema nullable pos (see
    # resolve_note_forms' docstring) both arrive as None - nothing to check.
    client = MagicMock()
    client.responses.create = AsyncMock()

    pos, usage = await resolve_note_pos(
        client, make_breaker(), "gpt-5.6-terra", "hakea", None, None, NOW
    )

    assert pos is None
    assert usage is None
    client.responses.create.assert_not_called()


async def test_resolve_note_pos_does_not_borrow_voida_readings_for_voi():
    # "voi" as a lemma is the noun/particle/interjection; the five verb
    # readings all belong to "voida". So verbi is not an option here, and
    # because three options remain the choice goes back to the LLM.
    client = MagicMock()
    client.responses.create = AsyncMock(return_value=fake_response({"pos": "substantiivi"}))

    pos, _usage = await resolve_note_pos(
        client, make_breaker(), "gpt-5.6-terra", "voi", "verbi", "Ostin voita.", NOW
    )

    enum = client.responses.create.call_args.kwargs["text"]["format"]["schema"]["properties"][
        "pos"
    ]["enum"]
    assert "verbi" not in enum
    assert enum == ["interjektio", "partikkeli", "substantiivi"]
    assert pos == "substantiivi"


async def test_resolve_note_pos_does_not_use_another_lemmas_reading_for_kuusi():
    # "kuu+N+Sg+Nom+PxSg2" ("your moon") spells "kuusi" too. It must not
    # widen the lemma "kuusi"'s own set, which is noun + numeral.
    client = MagicMock()
    client.responses.create = AsyncMock(return_value=fake_response({"pos": "numeraali"}))

    pos, _usage = await resolve_note_pos(
        client, make_breaker(), "gpt-5.6-terra", "kuusi", "verbi", "Minulla on kuusi kirjaa.", NOW
    )

    enum = client.responses.create.call_args.kwargs["text"]["format"]["schema"]["properties"][
        "pos"
    ]["enum"]
    assert enum == ["numeraali", "substantiivi"]
    assert pos == "numeraali"


async def test_resolve_note_pos_does_not_fall_back_to_a_pick_when_the_breaker_is_open():
    # The important half of "never take the first pos": when the tie-break
    # call cannot happen at all, the answer is an honest failure, not a
    # quietly chosen member of the set. /add reports the word as failed.
    client = MagicMock()
    client.responses.create = AsyncMock()
    breaker = CallBreaker(max_calls=0, window=timedelta(minutes=10))

    with pytest.raises(CircuitOpenError):
        await resolve_note_pos(
            client, breaker, "gpt-5.6-terra", "hakea", "adjektiivi", "Haen töitä.", NOW
        )

    client.responses.create.assert_not_called()


# --- resolve_note_forms: pos correction ------------------------------------------


async def test_resolve_note_forms_recovers_the_grammar_forms_for_tuli():
    # Regression for the production bug. Before: the LLM answered verbi for
    # "hän tuli kotiin", generate_forms("tuli", "verbi") returned nothing at
    # all, and the note was saved with empty principal_forms and
    # forms_verified=False - so srs.graduation never issued a grammar card
    # for it. After: the pos is corrected to the lemma's only real one and
    # the full nominal paradigm comes back, FST-verified.
    client = MagicMock()
    client.responses.create = AsyncMock()

    resolved, usage = await resolve_note_forms(
        client,
        make_breaker(),
        "gpt-5.6-terra",
        "tuli",
        "verbi",
        NOW,
        context="Hän tuli kotiin.",
    )

    assert resolved.pos == "substantiivi"
    assert resolved.forms_source == "fst"
    assert resolved.forms_verified is True
    assert resolved.principal_forms["partitiivi"] == "tulta"
    assert resolved.principal_forms["genetiivi"] == "tulen"
    assert usage is None
    client.responses.create.assert_not_called()


async def test_resolve_note_forms_reports_the_pos_it_actually_used():
    client = MagicMock()
    client.responses.create = AsyncMock()

    resolved, _usage = await resolve_note_forms(
        client, make_breaker(), "gpt-5.6-terra", "hakea", "verbi", NOW, context="Haen töitä."
    )

    assert resolved.pos == "verbi"
    assert resolved.principal_forms["preesens_1s"] == "haen"


async def test_resolve_note_forms_asks_before_switching_a_genuinely_ambiguous_lemma():
    # "hakea" is the multi-pos case: the code must not silently take the
    # verb just because it is the bigger forms table or the first reading.
    client = MagicMock()
    client.responses.create = AsyncMock(return_value=fake_response({"pos": "verbi"}))

    resolved, usage = await resolve_note_forms(
        client, make_breaker(), "gpt-5.6-terra", "hakea", "adverbi", NOW, context="Haen töitä."
    )

    assert client.responses.create.await_count == 1
    assert (
        client.responses.create.call_args.kwargs["text"]["format"]["name"]
        == "kielikaveri_pos_choice"
    )
    assert resolved.pos == "verbi"
    assert resolved.forms_verified is True
    assert usage is not None  # the pos tie-break is billed like any other call


async def test_resolve_note_forms_unknown_lemma_keeps_its_old_behaviour():
    # An unknown word still degrades the same way it did before the pos
    # check existed: no correction, no crash, forms_verified=False.
    client = MagicMock()
    client.responses.create = AsyncMock()

    resolved, usage = await resolve_note_forms(
        client, make_breaker(), "gpt-5.6-terra", "xyzquu", "substantiivi", NOW
    )

    assert resolved.pos == "substantiivi"
    assert resolved.forms_source == "llm"
    assert resolved.forms_verified is False
    assert resolved.principal_forms == {}
    assert usage is None
    client.responses.create.assert_not_called()


# --- application-level logging -----------------------------------------------------


async def test_check_and_suggest_logs_request_and_response_with_duration_and_usage(caplog):
    client = MagicMock()
    client.responses.create = AsyncMock(
        return_value=fake_response(
            {"reply_ru": "Нашла кое-что.", "needs_clarification": False, "candidates": []}
        )
    )
    breaker = make_breaker()

    with caplog.at_level(logging.DEBUG, logger="kielikaveri.ingest"):
        await check_and_suggest(client, breaker, "gpt-5.6-terra", "text", NOW)

    events = [log_fields(r.message) for r in caplog.records]
    request = next(f for f in events if f.get("event") == "llm.request")
    assert request["op"] == "check_and_suggest"
    assert request["model"] == "gpt-5.6-terra"

    response = next(f for f in events if f.get("event") == "llm.response")
    assert response["op"] == "check_and_suggest"
    assert response["total_tokens"] == "15"
    assert int(response["duration_ms"]) >= 0


# The fixed field set every op=llm.request / op=llm.response must carry,
# regardless of which LLM call produced it - this is what _log_llm_request/
# _log_llm_response exist to guarantee structurally, not just by convention.
_LLM_REQUEST_CORE_FIELDS = {"event", "op", "model"}
_LLM_RESPONSE_CORE_FIELDS = {
    "event",
    "op",
    "model",
    "duration_ms",
    "input_tokens",
    "output_tokens",
    "total_tokens",
}


def test_log_llm_request_always_carries_the_core_fields(caplog):
    with caplog.at_level(logging.DEBUG, logger="kielikaveri.ingest"):
        _log_llm_request("some_future_op", "gpt-5.6-terra", extra_field="whatever")

    fields = log_fields(caplog.records[0].message)
    assert _LLM_REQUEST_CORE_FIELDS <= fields.keys()
    assert fields["op"] == "some_future_op"
    assert fields["model"] == "gpt-5.6-terra"


def test_log_llm_response_always_carries_the_core_fields(caplog):
    usage = TokenUsage(input_tokens=1, output_tokens=2, total_tokens=3)

    with caplog.at_level(logging.INFO, logger="kielikaveri.ingest"):
        _log_llm_response("some_future_op", "gpt-5.6-terra", 42, usage, extra_field="whatever")

    fields = log_fields(caplog.records[0].message)
    assert _LLM_RESPONSE_CORE_FIELDS <= fields.keys()
    assert fields["duration_ms"] == "42"
    assert fields["total_tokens"] == "3"


async def test_llm_response_core_fields_match_across_different_ops(caplog):
    """The actual regression this guards against: check_and_suggest and
    resolve_ambiguous_forms are two independent call sites - nothing but
    _log_llm_response stops one of them from quietly dropping a field
    (e.g. model, or total_tokens) that the other still has."""
    client = MagicMock()
    client.responses.create = AsyncMock(
        return_value=fake_response(
            {"reply_ru": "Ok.", "needs_clarification": False, "candidates": []}
        )
    )
    breaker = make_breaker()

    with caplog.at_level(logging.INFO, logger="kielikaveri.ingest"):
        await check_and_suggest(client, breaker, "gpt-5.6-terra", "text", NOW)
    check_and_suggest_fields = log_fields(
        next(r.message for r in caplog.records if "event=llm.response" in r.message)
    )
    caplog.clear()

    client2 = MagicMock()
    client2.responses.create = AsyncMock(
        return_value=fake_response({"monikon_genetiivi": "hampaiden"})
    )
    with caplog.at_level(logging.INFO, logger="kielikaveri.ingest"):
        await resolve_ambiguous_forms(
            client2,
            breaker,
            "gpt-5.6-terra",
            "hammas",
            "substantiivi",
            {"monikon_genetiivi": ["hampaiden"]},
            NOW,
        )
    resolve_ambiguous_forms_fields = log_fields(
        next(r.message for r in caplog.records if "event=llm.response" in r.message)
    )

    # Same core field set present on both - the exact values differ (different
    # calls), but neither op is missing a field the other one has.
    assert _LLM_RESPONSE_CORE_FIELDS <= check_and_suggest_fields.keys()
    assert _LLM_RESPONSE_CORE_FIELDS <= resolve_ambiguous_forms_fields.keys()


async def test_resolve_note_forms_fst_only_logs_fst_resolve_from_morphology(caplog):
    client = MagicMock()
    client.responses.create = AsyncMock()
    breaker = make_breaker()

    with caplog.at_level(logging.DEBUG, logger="finn_cards.morphology"):
        await resolve_note_forms(client, breaker, "gpt-5.6-terra", "hakea", "verbi", NOW)

    events = [log_fields(r.message) for r in caplog.records]
    resolve_events = [f for f in events if f.get("event") == "fst.resolve"]
    assert len(resolve_events) == 1
    assert resolve_events[0]["lemma"] == "hakea"
    assert resolve_events[0]["forms_source"] == "fst"


async def test_resolve_note_forms_ambiguous_logs_debug_not_warning(caplog):
    client = MagicMock()
    client.responses.create = AsyncMock(
        return_value=fake_response({"monikon_genetiivi": "hampaiden"})
    )
    breaker = make_breaker()

    with caplog.at_level(logging.DEBUG, logger="kielikaveri.ingest"):
        await resolve_note_forms(client, breaker, "gpt-5.6-terra", "hammas", "substantiivi", NOW)

    ambiguous = [
        r
        for r in caplog.records
        if log_fields(r.message).get("event") == "resolve_note_forms.ambiguous"
    ]
    assert len(ambiguous) == 1
    assert ambiguous[0].levelno == logging.DEBUG


async def test_resolve_note_forms_missing_table_logs_warning_fallback(caplog):
    client = MagicMock()
    client.responses.create = AsyncMock()
    breaker = make_breaker()

    with caplog.at_level(logging.WARNING, logger="kielikaveri.ingest"):
        await resolve_note_forms(client, breaker, "gpt-5.6-terra", "kuitenkin", "adverbi", NOW)

    fallbacks = [
        log_fields(r.message)
        for r in caplog.records
        if log_fields(r.message).get("event") == "resolve_note_forms.fallback"
    ]
    assert len(fallbacks) == 1
    assert fallbacks[0]["reason"] == "no_fst_table"


async def test_resolve_note_forms_handles_a_missing_pos():
    # Strict mode drops schema.json's "kind=word requires pos" allOf, so the
    # LLM can hand back pos=None even for kind="word" - must degrade too.
    client = MagicMock()
    client.responses.create = AsyncMock()
    breaker = make_breaker()

    resolved, usage = await resolve_note_forms(client, breaker, "gpt-5.6-terra", "hakea", None, NOW)

    assert resolved.forms_source == "llm"
    assert resolved.forms_verified is False
    assert usage is None
    client.responses.create.assert_not_called()


async def test_resolve_ambiguous_forms_schema_only_allows_the_fst_candidates():
    client = MagicMock()
    client.responses.create = AsyncMock(
        return_value=fake_response({"monikon_genetiivi": "hampaiden"})
    )
    breaker = make_breaker()

    chosen, _usage = await resolve_ambiguous_forms(
        client,
        breaker,
        "gpt-5.6-terra",
        "hammas",
        "substantiivi",
        {"monikon_genetiivi": ["hampaitten", "hampaiden"]},
        NOW,
    )

    assert chosen == {"monikon_genetiivi": "hampaiden"}
    schema = client.responses.create.call_args.kwargs["text"]["format"]["schema"]
    assert schema["properties"]["monikon_genetiivi"]["enum"] == ["hampaitten", "hampaiden"]


# --- build_full_note ---------------------------------------------------------------


def test_build_full_note_fills_in_the_fst_only_fields():
    from kielikaveri.ingest import ResolvedForms

    candidate = {
        "lemma": "hakea",
        "pos": "verbi",
        "translation_ru": "искать",
        "example_fi": "Haen töitä.",
        "example_ru": "Я ищу работу.",
        "kind": "word",
        "meta": {"cefr": "B1"},
    }
    resolved = ResolvedForms({"preesens_1s": "haen"}, "fst", True)

    note = build_full_note(candidate, resolved)

    assert note["meta"]["principal_forms"] == {"preesens_1s": "haen"}
    assert note["meta"]["forms_source"] == "fst"
    assert note["meta"]["forms_verified"] is True
    assert note["meta"]["origin"] == "text"
    assert note["meta"]["cefr"] == "B1"


def test_build_full_note_pattern_kind_gets_placeholder_forms():
    candidate = {
        "lemma": "hakea + partitiivi",
        "pos": None,
        "translation_ru": "искать + партитив",
        "example_fi": "Haen töitä.",
        "example_ru": "Я ищу работу.",
        "kind": "pattern",
        "meta": {},
    }

    note = build_full_note(candidate, None)

    assert note["meta"]["principal_forms"] == {}
    assert note["meta"]["forms_source"] == "llm"
    assert note["meta"]["forms_verified"] is False


# --- cache -------------------------------------------------------------------------


async def test_cache_round_trips(session_factory):
    async with session_factory() as session:
        assert await get_cached_chat(session, "abc") is None

        await store_cached_chat(
            session, "abc", "gpt-5.6-terra", "Нашла кое-что.", False, [{"lemma": "hakea"}]
        )
        await session.commit()

    async with session_factory() as session:
        assert await get_cached_chat(session, "abc") == (
            "Нашла кое-что.",
            False,
            [{"lemma": "hakea"}],
        )


async def test_cache_round_trips_needs_clarification(session_factory):
    async with session_factory() as session:
        await store_cached_chat(session, "xyz", "gpt-5.6-terra", "Что с этим сделать?", True, [])
        await session.commit()

    async with session_factory() as session:
        assert await get_cached_chat(session, "xyz") == ("Что с этим сделать?", True, [])


async def test_cache_treats_a_pre_reply_ru_row_as_a_miss(session_factory):
    # Rows written by the old generate_candidates()-based cache have no
    # reply_ru at all - get_cached_chat must not replay them with an empty
    # reply, it should look like nothing was cached.
    from kielikaveri.db.models import IngestCache

    async with session_factory() as session:
        session.add(
            IngestCache(text_hash="legacy", model="gpt-5.6-terra", candidates=[{"lemma": "hakea"}])
        )
        await session.commit()

    async with session_factory() as session:
        assert await get_cached_chat(session, "legacy") is None


# --- existing_note_keys -------------------------------------------------------------


async def test_existing_note_keys_scopes_to_the_user(session_factory):
    async with session_factory() as session:
        session.add(
            Note(
                id="n1",
                user_id=1,
                lemma="hakea",
                pos="verbi",
                translation_ru="искать",
                example_fi="Haen töitä.",
                example_ru="Я ищу работу.",
                kind=NoteKind.word,
                meta={},
            )
        )
        session.add(
            Note(
                id="n2",
                user_id=2,
                lemma="pitää",
                pos="verbi",
                translation_ru="держать",
                example_fi="Pidän tästä.",
                example_ru="Мне это нравится.",
                kind=NoteKind.word,
                meta={},
            )
        )
        await session.commit()

    async with session_factory() as session:
        keys = await existing_note_keys(session, user_id=1)

    assert keys == {("hakea", "verbi")}
