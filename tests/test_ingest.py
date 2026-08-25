import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import Note, NoteKind
from kielikaveri.ingest import (
    _candidates_schema,
    _load_note_schema,
    build_full_note,
    canonical_key,
    existing_note_keys,
    generate_candidates,
    get_cached_candidates,
    hash_text,
    resolve_ambiguous_forms,
    resolve_note_forms,
    store_cached_candidates,
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
    assert hash_text("Haen töitä.") == hash_text("  Haen töitä.  ")


def test_hash_text_differs_for_different_text():
    assert hash_text("Haen töitä.") != hash_text("Haen taloa.")


def test_canonical_key_lemmatizes_an_inflected_llm_lemma():
    # cards/instructions.md: the LLM sometimes returns a word form as "lemma".
    assert canonical_key("töitä", "substantiivi") == ("työ", "substantiivi")


def test_canonical_key_leaves_a_real_lemma_alone():
    assert canonical_key("hakea", "verbi") == ("hakea", "verbi")


def test_canonical_key_pattern_kind_uses_the_raw_construction():
    assert canonical_key("hakea + partitiivi", None) == ("hakea + partitiivi", None)


# --- strict schema wrapper -------------------------------------------------------


def test_candidates_schema_wraps_the_note_schema_and_excludes_fst_fields():
    schema = _candidates_schema(_load_note_schema())
    assert schema["additionalProperties"] is False
    note_item = schema["properties"]["candidates"]["items"]
    assert "principal_forms" not in note_item["properties"]["meta"]["properties"]
    assert "origin" not in note_item["properties"]["meta"]["properties"]


# --- generate_candidates ---------------------------------------------------------


async def test_generate_candidates_parses_the_response_and_returns_usage():
    client = MagicMock()
    client.responses.create = AsyncMock(
        return_value=fake_response({"candidates": [{"lemma": "hakea"}]})
    )
    breaker = make_breaker()

    candidates, usage = await generate_candidates(client, breaker, "gpt-5.6-terra", "text", NOW)

    assert candidates == [{"lemma": "hakea"}]
    assert usage.input_tokens == 10
    assert usage.output_tokens == 5
    assert usage.total_tokens == 15
    client.responses.create.assert_called_once()
    assert client.responses.create.call_args.kwargs["model"] == "gpt-5.6-terra"


async def test_generate_candidates_respects_a_tripped_breaker():
    client = MagicMock()
    client.responses.create = AsyncMock()
    breaker = CallBreaker(max_calls=0, window=timedelta(minutes=10))

    with pytest.raises(CircuitOpenError):
        await generate_candidates(client, breaker, "gpt-5.6-terra", "text", NOW)

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
        assert await get_cached_candidates(session, "abc") is None

        await store_cached_candidates(session, "abc", "gpt-5.6-terra", [{"lemma": "hakea"}])
        await session.commit()

    async with session_factory() as session:
        assert await get_cached_candidates(session, "abc") == [{"lemma": "hakea"}]


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
