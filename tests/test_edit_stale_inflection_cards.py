"""The lifecycle of a note's inflection cards across /edit.

/edit rewrites meta.principal_forms wholesale when the lemma changes - and
with it the part of speech, when the new lemma can't carry the stored one
(ingest.resolve_note_pos). The cards are keyed by form name, so a rewritten
form set leaves cards quizzing forms the note no longer has: render_card
falls back to (lemma, lemma) for them, and they still sit in the queue,
spend the daily form budget and hold up the curriculum's level gate.

graduation.resync_inflection_cards is what /edit now runs inside the same
transaction as the save: cards whose form is gone are deleted with their
reviews, the forms the note has get cards through the usual
_ensure_inflection_cards path, and a form set that did not change is left
completely alone (puhua -> hakea keeps every card and its FSRS history).

Real SQLite, real FST, real resolve_note_forms, real queue/curriculum/
handlers; OpenAI is replaced at the client boundary only.
"""

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import openai
import pytest
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from sqlalchemy import func, select

from finn_cards.morphology import NOMINAL_FORMS, VERB_FORMS
from kielikaveri.bot.edit import EditStates, note_edit_apply
from kielikaveri.bot.learn import learn_rate, learn_reveal, learn_start, render_card
from kielikaveri.config import Settings
from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import Card, CardStatus, CardType, Note, NoteKind, Review, User
from kielikaveri.grammar import FORM_TASKS
from kielikaveri.ingest import resolve_note_forms
from kielikaveri.llm.breaker import CallBreaker
from kielikaveri.srs.curriculum import introduce_due_forms
from kielikaveri.srs.graduation import sync_user_card_types
from kielikaveri.srs.queue import build_session_queue, card_counters, overdue_count
from kielikaveri.srs.scheduler import Rating, apply_review

VERB_CARD_FORMS = {name for name in VERB_FORMS if name in FORM_TASKS}
NOUN_CARD_FORMS = {name for name in NOMINAL_FORMS if name in FORM_TASKS}


@pytest.fixture
async def session_factory(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    await create_all(engine)
    yield make_session_factory(engine)
    await engine.dispose()


def make_breaker(max_calls: int = 60) -> CallBreaker:
    return CallBreaker(max_calls=max_calls, window=timedelta(minutes=10))


def make_settings() -> Settings:
    return Settings(
        openai_api_key="sk-test",
        openai_timeout_seconds=1.0,
        session_max_cards=50,
        session_max_minutes=10,
        daily_new_limit=50,
        daily_new_forms=4,
        day_boundary_hour=4,
        debt_threshold=100,
    )


def llm_response(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(
        output_text=json.dumps(payload, ensure_ascii=False),
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
    )


def patch_openai(monkeypatch, *responses) -> AsyncMock:
    # The only mocked boundary: the OpenAI client /edit builds.
    create = AsyncMock(side_effect=list(responses))
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    monkeypatch.setattr("kielikaveri.bot.edit.make_client", lambda *args, **kwargs: client)
    return create


def make_fsm() -> FSMContext:
    return FSMContext(storage=MemoryStorage(), key=StorageKey(bot_id=0, chat_id=1, user_id=1))


async def answer(session_factory, card_id: str, rating: Rating, at: datetime) -> None:
    """One real review, as learn_rate records it."""
    async with session_factory() as session:
        card = await session.get(Card, card_id)
        last = await session.scalar(
            select(func.max(Review.reviewed_at)).where(Review.card_id == card_id)
        )
        apply_review(card, rating, at, last)
        session.add(Review(card_id=card_id, user_id=1, rating=rating.value, reviewed_at=at))
        await session.commit()


async def add_note(session_factory, lemma: str, pos: str, meta: dict, created: datetime) -> None:
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(
            Note(
                id="n1",
                user_id=1,
                lemma=lemma,
                pos=pos,
                translation_ru="перевод",
                example_fi="Esimerkki.",
                example_ru="Пример.",
                kind=NoteKind.word,
                meta=meta,
                created_at=created,
            )
        )
        await session.flush()
        await sync_user_card_types(session, 1, created)
        await session.commit()


async def create_learned_note(session_factory, lemma: str, pos: str, created: datetime) -> None:
    """A note as /add leaves it, then studied: the word known, core forms opened,
    the first of them answered twice.

    Forms come from the real resolve_note_forms; the lemmas used here resolve
    on the FST alone, so the client is never called.
    """
    client = SimpleNamespace(responses=SimpleNamespace(create=AsyncMock()))
    resolved, _ = await resolve_note_forms(client, make_breaker(), "m", lemma, pos, created)
    assert resolved.forms_verified and resolved.pos == pos
    client.responses.create.assert_not_called()
    await add_note(
        session_factory,
        lemma,
        pos,
        {
            "principal_forms": resolved.principal_forms,
            "forms_source": resolved.forms_source,
            "forms_verified": True,
        },
        created,
    )
    async with session_factory() as session:
        recognition = await session.scalar(select(Card.id).where(Card.type == CardType.recognition))

    await answer(session_factory, recognition, Rating.Good, created)
    await answer(session_factory, recognition, Rating.Good, created + timedelta(hours=1))
    opened_at = created + timedelta(days=2)
    async with session_factory() as session:
        opened = await introduce_due_forms(
            session, 1, opened_at, daily_new_forms=4, boundary_hour=4
        )
        await session.commit()
    assert opened
    await answer(session_factory, opened[0].id, Rating.Good, opened_at)
    await answer(session_factory, opened[0].id, Rating.Good, opened_at + timedelta(hours=1))


async def edit_lemma(session_factory, value: str, breaker: CallBreaker | None = None) -> list:
    state = make_fsm()
    await state.set_state(EditStates.awaiting_value)
    await state.update_data(note_id="n1", field="lemma")
    message = SimpleNamespace(text=value, from_user=SimpleNamespace(id=1), answer=AsyncMock())
    await note_edit_apply(
        message, state, session_factory, make_settings(), breaker or make_breaker()
    )
    return [call.args[0] for call in message.answer.call_args_list]


async def snapshot(session_factory) -> dict[str, tuple]:
    """Every card keyed by id: the fields /edit could conceivably have touched."""
    async with session_factory() as session:
        cards = (await session.scalars(select(Card))).all()
        reviews = dict(
            (await session.execute(select(Review.card_id, func.count()).group_by(Review.card_id)))
            .tuples()
            .all()
        )
    return {
        c.id: (c.type, c.form, c.status, c.state, c.due, c.reps, c.stability, reviews.get(c.id, 0))
        for c in cards
    }


async def load(session_factory) -> tuple[Note, list[Card]]:
    async with session_factory() as session:
        note = await session.get(Note, "n1")
        cards = list((await session.scalars(select(Card).where(Card.note_id == "n1"))).all())
    return note, cards


def by_form(cards: list[Card]) -> dict[str, Card]:
    return {c.form: c for c in cards if c.type == CardType.inflection}


def word_card(cards: list[Card]) -> Card:
    return next(c for c in cards if c.type == CardType.recognition)


async def review_owners(session_factory) -> set[str]:
    """Card ids the review log still points at."""
    async with session_factory() as session:
        return set(await session.scalars(select(Review.card_id).distinct()))


async def orphan_reviews(session_factory) -> set[str]:
    """Reviews whose card is gone - nothing in this schema deletes them for us."""
    async with session_factory() as session:
        return set(
            await session.scalars(
                select(Review.card_id).where(Review.card_id.not_in(select(Card.id)))
            )
        )


async def run_learn(session_factory) -> list[tuple[str, str, str]]:
    """A full /learn session through the real handlers, every card rated Good.

    Returns (card id, front, back) for each card the learner was shown.
    """
    state = make_fsm()
    message = SimpleNamespace(from_user=SimpleNamespace(id=1), answer=AsyncMock())
    await learn_start(message, state, session_factory, make_settings())
    shown = []
    for _ in range(100):
        markup = message.answer.call_args.kwargs.get("reply_markup")
        if markup is None:
            break
        card_id = markup.inline_keyboard[0][0].callback_data.split(":", 2)[2]
        front = message.answer.call_args.args[0]
        reveal = SimpleNamespace(
            data=f"learn:reveal:{card_id}",
            from_user=SimpleNamespace(id=1),
            answer=AsyncMock(),
            message=message,
        )
        await learn_reveal(reveal, session_factory)
        back = message.answer.call_args.args[0]
        rate = SimpleNamespace(
            data=f"learn:rate:{card_id}:{Rating.Good.value}",
            from_user=SimpleNamespace(id=1),
            answer=AsyncMock(),
            message=message,
        )
        await learn_rate(rate, state, session_factory)
        shown.append((card_id, front, back))
    else:
        pytest.fail("session did not end")
    return shown


# --- A: verbi -> substantiivi -------------------------------------------------------


@pytest.mark.parametrize(
    ("new_lemma", "llm"),
    [
        # talo is only a noun to the FST - pos is corrected without asking.
        ("talo", []),
        # kuusi is a noun or a numeral - the POS tie-break goes to the LLM,
        # then kuusi's own ambiguous plural genitive does too.
        (
            "kuusi",
            [
                llm_response({"pos": "substantiivi"}),
                llm_response({"monikon_genetiivi": "kuusten"}),
            ],
        ),
    ],
)
async def test_verb_to_noun_replaces_the_verb_cards_with_the_noun_ones(
    session_factory, monkeypatch, new_lemma, llm
):
    now = datetime.now(UTC)
    await create_learned_note(session_factory, "puhua", "verbi", now - timedelta(days=10))
    _note, before_cards = await load(session_factory)
    before = await snapshot(session_factory)
    verb_ids = {c.id for c in before_cards if c.type == CardType.inflection}
    word = word_card(before_cards)
    assert {c.form for c in before_cards if c.type == CardType.inflection} == VERB_CARD_FORMS
    create = patch_openai(monkeypatch, *llm)

    replies = await edit_lemma(session_factory, new_lemma)

    assert replies == [f"Слово: «puhua» → «{new_lemma}»."]  # no error surfaced
    assert create.await_count == len(llm)
    note, cards = await load(session_factory)
    assert (note.lemma, note.pos) == (new_lemma, "substantiivi")

    # Every verb card is physically gone, not suspended and not kept around.
    after = await snapshot(session_factory)
    assert verb_ids & set(after) == set()
    assert not [c for c in cards if c.status == CardStatus.suspended]

    # The noun's forms took their place, as brand new cards.
    noun_cards = by_form(cards)
    assert set(noun_cards) == set(note.meta["principal_forms"]) & NOUN_CARD_FORMS
    assert {c.id for c in noun_cards.values()} & verb_ids == set()
    assert all(c.status == CardStatus.not_introduced for c in noun_cards.values())
    assert all(
        (c.reps, c.stability, c.introduced_at) == (0, None, None) for c in noun_cards.values()
    )
    for card in noun_cards.values():
        front, back = render_card(card, note)
        assert (front, back) != (note.lemma, note.lemma)

    # The word card is untouched, history and all.
    assert after[word.id] == before[word.id]

    # The deleted cards' reviews went with them - no row survives pointing at
    # a card that no longer exists, and none of them reaches a new card.
    assert await orphan_reviews(session_factory) == set()
    assert await review_owners(session_factory) == {word.id}


# --- B: substantiivi -> verbi -------------------------------------------------------


async def test_noun_to_verb_replaces_the_noun_cards_with_the_verb_ones(
    session_factory, monkeypatch
):
    now = datetime.now(UTC)
    await create_learned_note(session_factory, "talo", "substantiivi", now - timedelta(days=10))
    _note, before_cards = await load(session_factory)
    before = await snapshot(session_factory)
    noun_ids = {c.id for c in before_cards if c.type == CardType.inflection}
    word = word_card(before_cards)
    assert {c.form for c in before_cards if c.type == CardType.inflection} == NOUN_CARD_FORMS
    patch_openai(monkeypatch)

    await edit_lemma(session_factory, "puhua")

    note, cards = await load(session_factory)
    assert (note.lemma, note.pos) == ("puhua", "verbi")
    after = await snapshot(session_factory)
    assert noun_ids & set(after) == set()
    assert set(by_form(cards)) == VERB_CARD_FORMS
    assert {c.id for c in by_form(cards).values()} & noun_ids == set()
    assert all(c.status == CardStatus.not_introduced for c in by_form(cards).values())
    assert after[word.id] == before[word.id]
    assert await orphan_reviews(session_factory) == set()
    assert await review_owners(session_factory) == {word.id}


# --- C: the recompute failed --------------------------------------------------------


@pytest.mark.parametrize("failure", ["breaker", "openai"])
async def test_lemma_change_without_recompute_deletes_the_cards_and_creates_none(
    session_factory, monkeypatch, failure
):
    # puhua -> mennä: mennä has FST-ambiguous forms, so the recompute needs
    # the LLM; when it can't run, /edit drops the old forms - and the cards
    # built from them have nothing left to quiz.
    now = datetime.now(UTC)
    await create_learned_note(session_factory, "puhua", "verbi", now - timedelta(days=10))
    _note, before_cards = await load(session_factory)
    before = await snapshot(session_factory)
    word = word_card(before_cards)
    if failure == "breaker":
        patch_openai(monkeypatch)
        breaker = make_breaker(max_calls=0)
    else:
        patch_openai(monkeypatch, openai.APIConnectionError(request=SimpleNamespace()))
        breaker = make_breaker()

    replies = await edit_lemma(session_factory, "mennä", breaker)

    assert any("не пересчитала" in r for r in replies)
    note, cards = await load(session_factory)
    assert (note.lemma, note.pos) == ("mennä", "verbi")
    assert "principal_forms" not in note.meta
    assert by_form(cards) == {}
    assert [c.type for c in cards] == [CardType.recognition]
    assert (await snapshot(session_factory))[word.id] == before[word.id]
    assert await orphan_reviews(session_factory) == set()
    assert await review_owners(session_factory) == {word.id}

    # /learn sees the word card and nothing else: no unanswerable leftovers,
    # and no verified forms for the sync to build new cards from.
    shown = await run_learn(session_factory)
    assert [card_id for card_id, _front, _back in shown] == [word.id]
    _note, cards_after = await load(session_factory)
    assert by_form(cards_after) == {}


# --- D: the form set changes inside one part of speech ------------------------------


async def test_a_form_the_new_lemma_lacks_loses_its_card_while_the_others_stay(
    session_factory, monkeypatch
):
    # Same POS, smaller form set: kuusi's plural genitive is FST-ambiguous and
    # the LLM's pick is not one of the candidates, so resolve_note_forms
    # rejects it and the form is simply absent from the note.
    now = datetime.now(UTC)
    await create_learned_note(session_factory, "talo", "substantiivi", now - timedelta(days=10))
    _note, before_cards = await load(session_factory)
    before = await snapshot(session_factory)
    dropped = by_form(before_cards)["monikon_genetiivi"]
    patch_openai(
        monkeypatch,
        llm_response({"pos": "substantiivi"}),
        llm_response({"monikon_genetiivi": "kuusia"}),  # not an FST candidate
    )

    await edit_lemma(session_factory, "kuusi")

    note, cards = await load(session_factory)
    assert (note.lemma, note.pos) == ("kuusi", "substantiivi")
    assert note.meta["forms_verified"] is False
    assert "monikon_genetiivi" not in note.meta["principal_forms"]

    after = await snapshot(session_factory)
    assert dropped.id not in after
    # Every other card - word card included - is byte-for-byte what it was.
    assert after == {k: v for k, v in before.items() if k != dropped.id}
    assert set(by_form(cards)) == NOUN_CARD_FORMS - {"monikon_genetiivi"}
    assert await orphan_reviews(session_factory) == set()


async def test_forms_the_note_gained_get_cards_while_the_shared_ones_keep_their_history(
    session_factory, monkeypatch
):
    # A note carrying only part of the nominal table (an early import) edited
    # to a lemma the FST resolves in full: the three shared forms keep their
    # cards and their reviews, the ten new ones get cards of their own.
    now = datetime.now(UTC)
    # nominatiivi is in the FST table but not in FORM_TASKS (it is the lemma
    # itself), so it never has a card - the shared keys here are asked ones.
    kept_forms = {"genetiivi": "talon", "partitiivi": "taloa", "illatiivi": "taloon"}
    await add_note(
        session_factory,
        "talo",
        "substantiivi",
        {"principal_forms": kept_forms, "forms_source": "fst", "forms_verified": True},
        now - timedelta(days=10),
    )
    _note, before_cards = await load(session_factory)
    genetiivi = by_form(before_cards)["genetiivi"]
    await answer(session_factory, genetiivi.id, Rating.Good, now - timedelta(days=3))
    await answer(session_factory, genetiivi.id, Rating.Good, now - timedelta(days=2))
    before = await snapshot(session_factory)
    patch_openai(monkeypatch)

    await edit_lemma(session_factory, "kissa")

    note, cards = await load(session_factory)
    assert (note.lemma, note.pos) == ("kissa", "substantiivi")
    after = await snapshot(session_factory)
    cards_by_form = by_form(cards)
    assert set(cards_by_form) == NOUN_CARD_FORMS
    # Shared keys: same rows, same FSRS state, same reviews - now quizzing kissa.
    for name in kept_forms:
        assert after[cards_by_form[name].id] == before[cards_by_form[name].id]
    assert cards_by_form["genetiivi"].id == genetiivi.id
    assert render_card(cards_by_form["genetiivi"], note)[1].startswith("kissan")
    # New keys: new cards, waiting for the curriculum.
    for name in NOUN_CARD_FORMS - set(kept_forms):
        card = cards_by_form[name]
        assert card.id not in before
        assert (card.status, card.reps) == (CardStatus.not_introduced, 0)
    # The only reviews in this note's history are the genetiivi card's own.
    assert await review_owners(session_factory) == {genetiivi.id}
    assert await orphan_reviews(session_factory) == set()


# --- E: the form set does not change ------------------------------------------------


@pytest.mark.parametrize("new_lemma", ["hakea", "puhua"])
async def test_an_unchanged_form_set_keeps_every_card_and_its_history(
    session_factory, monkeypatch, new_lemma
):
    # puhua -> hakea is the everyday correction: same POS, same seven form
    # keys. Re-submitting the same lemma must be just as inert.
    now = datetime.now(UTC)
    await create_learned_note(session_factory, "puhua", "verbi", now - timedelta(days=10))
    before = await snapshot(session_factory)
    patch_openai(monkeypatch)

    await edit_lemma(session_factory, new_lemma)

    note, cards = await load(session_factory)
    assert (note.lemma, note.pos) == (new_lemma, "verbi")
    assert set(by_form(cards)) == VERB_CARD_FORMS
    # Same card ids, same FSRS state, same review counts - nothing recreated.
    assert await snapshot(session_factory) == before
    first = by_form(cards)["preesens_1s"]
    assert before[first.id][5] == 2  # reps survived the edit
    expected = "haen" if new_lemma == "hakea" else "puhun"
    assert render_card(first, note) == (
        f"{new_lemma} → minä, nyt → ?",
        f"{expected}\n\n✅ {FORM_TASKS['preesens_1s'].label}",
    )

    shown = await run_learn(session_factory)
    backs = {back.split("\n")[0] for _id, _front, back in shown}
    assert expected in backs


async def test_same_lemma_with_a_pos_the_fst_rules_out_builds_the_new_tables_cards(
    session_factory, monkeypatch
):
    # The only way /edit changes POS without changing the lemma: the stored
    # one is outside the FST's set. tuli/verbi is the LLM mistake
    # resolve_note_pos exists for; such a note never had verified forms, so
    # there are no cards to delete - only the noun's to create.
    now = datetime.now(UTC)
    await add_note(
        session_factory,
        "tuli",
        "verbi",
        {"principal_forms": {}, "forms_source": "llm", "forms_verified": False},
        now - timedelta(days=3),
    )
    _note, cards = await load(session_factory)
    assert [c.type for c in cards] == [CardType.recognition]
    patch_openai(monkeypatch)

    await edit_lemma(session_factory, "tuli")

    note, cards = await load(session_factory)
    assert (note.lemma, note.pos) == ("tuli", "substantiivi")
    assert note.meta["forms_verified"] is True
    assert set(by_form(cards)) == NOUN_CARD_FORMS


# --- F: what /learn, the queue and the curriculum see afterwards --------------------


async def test_learn_never_serves_a_card_of_the_previous_part_of_speech(
    session_factory, monkeypatch
):
    now = datetime.now(UTC)
    await create_learned_note(session_factory, "puhua", "verbi", now - timedelta(days=10))
    _note, before_cards = await load(session_factory)
    verb_ids = {c.id for c in before_cards if c.type == CardType.inflection}
    word = word_card(before_cards)
    patch_openai(monkeypatch)

    await edit_lemma(session_factory, "talo")

    async with session_factory() as session:
        queue = await build_session_queue(session, 1, now, 50, 50, 4)
        assert set(queue) & verb_ids == set()
        assert queue == [word.id]  # the noun forms are not introduced yet
        # Only the word card is overdue now; the three answered verb forms
        # are gone, and with them their share of the debt.
        assert await overdue_count(session, 1, now) == 1
        assert await overdue_count(session, 1, now, reviewed_only=True) == 1
        counters = await card_counters(session, 1, now)
    # 12 asked nominal forms (nominatiivi has no card) plus the word card.
    assert (counters.total, counters.introduced, counters.not_introduced) == (13, 1, 12)
    assert (counters.due, counters.overdue) == (1, 1)

    shown = await run_learn(session_factory)
    note, cards = await load(session_factory)
    assert {card_id for card_id, _f, _b in shown} & verb_ids == set()
    # Everything the learner met renders as a real question, not (lemma, lemma).
    for _card_id, front, back in shown:
        assert (front, back) != (note.lemma, note.lemma)
    # The curriculum opened the new table's core forms, out of today's budget.
    opened = {c.form for c in cards if c.status == CardStatus.introduced and c.form}
    assert opened == {"genetiivi", "partitiivi", "illatiivi", "monikon_genetiivi"}
    assert "talo → mihin?" in [front for _id, front, _back in shown]


async def test_the_level_gate_counts_only_the_new_parts_of_speech_forms(
    session_factory, monkeypatch
):
    # curriculum.eligible_forms gates a level behind the cards of the levels
    # below it. Leftover cards of the old POS used to sit in that count and
    # could never be answered; with them deleted, the verb's own core forms
    # are the whole gate.
    start = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)
    await create_learned_note(session_factory, "talo", "substantiivi", start)
    patch_openai(monkeypatch)
    await edit_lemma(session_factory, "puhua")

    day = start + timedelta(days=3)
    async with session_factory() as session:
        await sync_user_card_types(session, 1, day)
        opened = await introduce_due_forms(session, 1, day, daily_new_forms=10, boundary_hour=4)
        await session.commit()
    assert {c.form for c in opened} == {"preesens_1s", "preesens_3s", "imperfekti_3s"}

    _note, cards = await load(session_factory)
    for card in cards:
        if card.form in {"preesens_1s", "preesens_3s", "imperfekti_3s"}:
            await answer(session_factory, card.id, Rating.Good, day)
            await answer(session_factory, card.id, Rating.Good, day + timedelta(hours=1))

    later = day + timedelta(days=2)
    async with session_factory() as session:
        unblocked = await introduce_due_forms(session, 1, later, daily_new_forms=4, boundary_hour=4)
        await session.commit()
    # The extended verb forms open next - nothing of talo's is left to block them.
    assert {c.form for c in unblocked} <= VERB_CARD_FORMS
    assert unblocked
    note, _cards = await load(session_factory)
    assert all(c.form in note.meta["principal_forms"] for c in unblocked)
