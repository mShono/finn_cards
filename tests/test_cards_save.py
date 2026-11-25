"""cards_save: one candidate saved as a note without UI (plan
2026-11-20-kirjakaveri-reader, "П4") - the steps /add has always run, shared
with the reader's ➕. The FST is real; OpenAI is a fake that must not be
called for these words."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import jsonschema
import openai
import pytest
from conftest import add_users
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from kielikaveri.cards_save import (
    REASON_BREAKER,
    REASON_INVALID,
    REASON_OPENAI,
    DedupState,
    SaveStatus,
    SettledCandidate,
    save_candidate,
    settle_lemma,
    store_note,
)
from kielikaveri.db.decks import create_deck
from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import Card, Note
from kielikaveri.ingest import SURFACE_FIELD, ResolvedForms
from kielikaveri.llm.breaker import CallBreaker, CircuitOpenError

USER = 1
NOW = datetime(2026, 11, 25, 10, 0, tzinfo=UTC)
MODEL = "gpt-test"

CANDIDATE = {
    "kind": "word",
    "lemma": "talo",
    "pos": "substantiivi",
    "translation_ru": "дом",
    "example_fi": "Talo on vanha.",
    "example_ru": "Дом старый.",
    SURFACE_FIELD: "talossa",
}


class NoOpenAI:
    """Any call is a test failure: the FST decides these words alone."""

    def __init__(self):
        self.responses = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs):
        raise AssertionError("OpenAI must not be called")


@pytest.fixture
async def session_factory(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    await create_all(engine)
    factory = make_session_factory(engine)
    await add_users(factory, USER)
    yield factory
    await engine.dispose()


@pytest.fixture
async def deck_id(session_factory):
    async with session_factory() as session:
        deck = await create_deck(session, USER, "Kirja - гл.1")
        await session.commit()
        return deck.id


def breaker() -> CallBreaker:
    return CallBreaker(60, timedelta(minutes=10))


async def save(session_factory, deck_id, candidate=CANDIDATE, dedup=None, **kwargs):
    return await save_candidate(
        session_factory,
        kwargs.pop("client", NoOpenAI()),
        kwargs.pop("breaker", breaker()),
        MODEL,
        USER,
        deck_id,
        candidate,
        dedup if dedup is not None else DedupState(set()),
        NOW,
        **kwargs,
    )


async def notes(session_factory) -> list[Note]:
    async with session_factory() as session:
        return list((await session.scalars(select(Note))).all())


async def test_a_new_word_is_saved_with_fst_forms(session_factory, deck_id):
    outcome = await save(session_factory, deck_id)

    assert outcome.status is SaveStatus.saved
    assert outcome.lemma == "talo"
    assert outcome.translation_ru == "дом"
    assert outcome.in_dictionary is True
    [note] = await notes(session_factory)
    assert (note.lemma, note.pos, note.deck_id) == ("talo", "substantiivi", deck_id)
    assert note.example_fi == "Talo on vanha."
    assert note.meta["origin"] == "text"
    assert note.meta["forms_source"] == "fst"
    assert note.meta["principal_forms"]


async def test_no_cards_are_created_learn_does_that(session_factory, deck_id):
    await save(session_factory, deck_id)

    async with session_factory() as session:
        assert (await session.scalars(select(Card))).all() == []


async def test_an_inflected_lemma_is_saved_under_the_fsts(session_factory, deck_id):
    outcome = await save(session_factory, deck_id, {**CANDIDATE, "lemma": "talossa"})

    assert outcome.status is SaveStatus.saved
    assert outcome.lemma == "talo"


async def test_a_word_already_in_the_deck_is_a_duplicate(session_factory, deck_id):
    dedup = DedupState({("talo", "substantiivi")})

    outcome = await save(session_factory, deck_id, dedup=dedup)

    assert outcome.status is SaveStatus.duplicate
    assert outcome.lemma == "talo"
    assert await notes(session_factory) == []


async def test_the_batch_dedups_itself(session_factory, deck_id):
    dedup = DedupState(set())
    first = await save(session_factory, deck_id, dedup=dedup)
    second = await save(session_factory, deck_id, dedup=dedup)

    assert (first.status, second.status) == (SaveStatus.saved, SaveStatus.duplicate)


async def test_losing_the_race_to_the_index_is_a_duplicate(session_factory, deck_id):
    # Both saves passed the Python check (separate dedup states): the
    # unique index turns the second INSERT away.
    first = await save(session_factory, deck_id, dedup=DedupState(set()))
    second = await save(session_factory, deck_id, dedup=DedupState(set()))

    assert first.status is SaveStatus.saved
    assert second.status is SaveStatus.duplicate
    assert len(await notes(session_factory)) == 1


async def test_another_integrity_error_is_raised(session_factory, deck_id):
    # A broken foreign key is a bug, not a duplicate.
    with pytest.raises(IntegrityError, match="FOREIGN KEY"):
        await save(session_factory, "no-such-deck")


async def test_the_breaker_is_a_failure_with_its_reason(session_factory, deck_id):
    async def tripped(*args, **kwargs):
        raise CircuitOpenError("stop")

    settled = SettledCandidate(dict(CANDIDATE), True)
    outcome = await store_note(
        session_factory,
        NoOpenAI(),
        breaker(),
        MODEL,
        USER,
        deck_id,
        settled,
        DedupState(set()),
        NOW,
        resolve_forms=tripped,
    )

    assert outcome.status is SaveStatus.failed
    assert outcome.reason == REASON_BREAKER
    assert isinstance(outcome.error, CircuitOpenError)


async def test_openai_failure_is_a_failure_with_its_reason(session_factory, deck_id):
    async def broken(*args, **kwargs):
        raise openai.APIConnectionError(request=SimpleNamespace())

    outcome = await store_note(
        session_factory,
        NoOpenAI(),
        breaker(),
        MODEL,
        USER,
        deck_id,
        SettledCandidate(dict(CANDIDATE), True),
        DedupState(set()),
        NOW,
        resolve_forms=broken,
    )

    assert outcome.status is SaveStatus.failed
    assert outcome.reason == REASON_OPENAI


async def test_invalid_note_is_a_failure(session_factory, deck_id):
    outcome = await save(session_factory, deck_id, {**CANDIDATE, "example_fi": 5})

    assert outcome.status is SaveStatus.failed
    assert outcome.reason == REASON_INVALID
    assert isinstance(outcome.error, jsonschema.ValidationError)
    assert await notes(session_factory) == []


async def test_the_pos_fixed_by_the_fst_is_saved_and_dedup_rerun(session_factory, deck_id):
    async def as_adjective(*args, **kwargs):
        return ResolvedForms({"partitiivi": "taloa"}, "fst", True, "adjektiivi"), None

    dedup = DedupState({("talo", "adjektiivi")})
    outcome = await store_note(
        session_factory,
        NoOpenAI(),
        breaker(),
        MODEL,
        USER,
        deck_id,
        SettledCandidate(dict(CANDIDATE), True),
        dedup,
        NOW,
        resolve_forms=as_adjective,
    )

    assert outcome.status is SaveStatus.duplicate


async def test_context_overrides_the_example_for_the_lemma(session_factory, deck_id):
    seen = {}

    async def spy(client, breaker, model, lemma, pos, now, context=None):
        seen["context"] = context
        return ResolvedForms({}, "llm", False, pos), None

    dedup = DedupState(set())
    settled = await settle_lemma(
        NoOpenAI(), breaker(), MODEL, CANDIDATE, dedup, NOW, context="Hän asuu talossa."
    )
    assert isinstance(settled, SettledCandidate)
    await store_note(
        session_factory,
        NoOpenAI(),
        breaker(),
        MODEL,
        USER,
        deck_id,
        settled,
        dedup,
        NOW,
        context="Hän asuu talossa.",
        resolve_forms=spy,
    )

    assert seen["context"] == "Hän asuu talossa."
