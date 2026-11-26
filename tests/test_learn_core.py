"""learn_core: the database side of a review session - preparing it (card
sync, curriculum, debt), its queue by deck and side, one rating, the card
as data. No aiogram, no HTTP: the reader's /api/learn/* runs on top of
this (tests/test_reader_learn.py covers the session around it)."""

import logging
from datetime import UTC, datetime, timedelta

import fsrs
import pytest
from conftest import log_fields
from sqlalchemy import delete, func, select

from finn_cards.morphology import NOMINAL_FORMS
from kielikaveri.config import Settings
from kielikaveri.db.decks import create_deck
from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import (
    Card,
    CardState,
    CardStatus,
    CardType,
    Note,
    NoteKind,
    Review,
    User,
)
from kielikaveri.grammar import FORM_TASKS, CurriculumLevel
from kielikaveri.learn_core import (
    EMPTY_ALL_TEXT,
    EMPTY_SIDE_TEXT,
    NO_PRODUCTION_TEXT,
    LearnSide,
    build_queue,
    card_view,
    drop_deleted_head,
    prepare_session,
    record_review,
    session_end_text,
)
from kielikaveri.srs import scheduler as srs_scheduler
from kielikaveri.srs.scheduler import Rating

NOW = datetime(2026, 8, 24, 10, 0, tzinfo=UTC)


@pytest.fixture
async def session_factory(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    await create_all(engine)
    yield make_session_factory(engine)
    await engine.dispose()


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


def make_card(
    card_id: str,
    note_id: str,
    due: datetime,
    reps: int = 0,
    card_type: CardType = CardType.recognition,
) -> Card:
    return Card(id=card_id, note_id=note_id, user_id=1, type=card_type, due=due, reps=reps)


async def seed(session_factory, *cards: Card, notes: tuple[Note, ...] = ()) -> None:
    async with session_factory() as session:
        session.add(User(id=1))
        await session.flush()
        session.add_all(notes or (make_note(),))
        await session.flush()
        session.add_all(cards)
        await session.commit()


async def prepare(session_factory, side: LearnSide, deck_id=None, **settings) -> int:
    async with session_factory() as session:
        return await prepare_session(
            session, make_settings(**settings), 1, datetime.now(UTC), deck_id, side
        )


async def queue_of(session_factory, side: LearnSide = LearnSide.mix, deck_id=None, **settings):
    async with session_factory() as session:
        return await build_queue(
            session, make_settings(**settings), 1, datetime.now(UTC), deck_id, side
        )


async def rate_all(session_factory, card_ids: list[str]) -> None:
    for card_id in card_ids:
        async with session_factory() as session:
            card = await session.get(Card, card_id)
            await record_review(session, card, Rating.Good, datetime.now(UTC))


# --- card_view: the card's content by type ----------------------------------------


def test_card_view_of_a_word_card_carries_the_note_and_no_form():
    note = make_note()
    for card_type in (CardType.recognition, CardType.production):
        view = card_view(make_card("c", "note-1", NOW, card_type=card_type), note)

        assert (view.lemma, view.translation_ru, view.example_fi, view.example_ru) == (
            "hakea",
            "искать",
            "Haen töitä.",
            "Я ищу работу.",
        )
        assert (view.cue, view.form_answer, view.form_label) == (None, None, None)
        assert "cue" not in view.as_json()


def test_card_view_asks_the_form_the_card_is_scheduled_for():
    # The question is fixed per card, not drawn at random: that is what
    # makes the card's own FSRS interval mean anything.
    note = make_note()
    note.meta = {"principal_forms": {"genetiivi": "kaupan", "illatiivi": "kauppaan"}}
    card = make_card("c", "note-1", NOW, card_type=CardType.inflection)
    card.form = "illatiivi"

    views = {card_view(card, note) for _ in range(5)}

    assert len(views) == 1
    (view,) = views
    task = FORM_TASKS["illatiivi"]
    assert (view.cue, view.form_answer, view.form_label) == (task.cue, "kauppaan", task.label)
    # The cue asks by context - it must not name the form it tests.
    assert "illatiivi" not in view.cue


@pytest.mark.parametrize(
    "form, meta",
    [
        # A form-less card (pre-migration row not yet adopted).
        (None, {"principal_forms": {"genetiivi": "kaupan"}}),
        # A form the note no longer has.
        ("translatiivi", {"principal_forms": {"genetiivi": "kaupan"}}),
        # A valid form, but the note's forms were dropped afterwards.
        ("illatiivi", {}),
    ],
    ids=["no-form", "form-not-in-note", "no-principal-forms"],
)
def test_card_view_falls_back_to_the_lemma_without_a_usable_form(form, meta):
    note = make_note()
    note.meta = meta
    card = make_card("c", "note-1", NOW, card_type=CardType.inflection)
    card.form = form

    view = card_view(card, note)

    assert (view.cue, view.form_answer, view.form_label) == (None, "hakea", None)


# --- prepare_session: the debt ------------------------------------------------------


async def test_never_reviewed_cards_are_not_debt(session_factory):
    # Freshly introduced forms get due=now: without reviewed_only every
    # batch of new forms would read as missed reviews.
    past = datetime.now(UTC) - timedelta(days=1)
    await seed(
        session_factory,
        *[make_card(f"new-{i}", "note-1", past) for i in range(3)],
        *[make_card(f"old-{i}", "note-1", past, reps=1) for i in range(2)],
    )

    assert await prepare(session_factory, LearnSide.mix) == 2


async def test_the_debt_counts_only_the_chosen_side(session_factory):
    past = datetime.now(UTC) - timedelta(days=1)
    await seed(
        session_factory,
        *[make_card(f"rec-{i}", "note-1", past, reps=1) for i in range(3)],
        make_card("prod", "note-1", past, reps=1, card_type=CardType.production),
    )

    assert await prepare(session_factory, LearnSide.fi) == 3
    assert await prepare(session_factory, LearnSide.ru) == 1
    assert await prepare(session_factory, LearnSide.mix) == 4


async def test_the_debt_counts_only_the_chosen_deck(session_factory):
    past = datetime.now(UTC) - timedelta(days=1)
    async with session_factory() as session:
        session.add(User(id=1))
        await session.flush()
        deck_a = await create_deck(session, 1, "Общая")
        deck_b = await create_deck(session, 1, "Из книги")
        await session.flush()
        session.add_all(
            [make_note("note-a", deck_id=deck_a.id), make_note("note-b", deck_id=deck_b.id)]
        )
        await session.flush()
        session.add_all([make_card(f"a-{i}", "note-a", past, reps=1) for i in range(5)])
        session.add(make_card("b-0", "note-b", past, reps=1))
        await session.commit()

    assert await prepare(session_factory, LearnSide.mix, deck_a.id) == 5
    assert await prepare(session_factory, LearnSide.mix, deck_b.id) == 1
    assert await prepare(session_factory, LearnSide.mix) == 6


# --- prepare_session: the curriculum ------------------------------------------------


def noun_with_forms(note_id: str, lemma: str, deck_id: str, created_at: datetime) -> Note:
    core = [n for n in NOMINAL_FORMS if n in FORM_TASKS]
    core = [n for n in core if FORM_TASKS[n].level is CurriculumLevel.core]
    return Note(
        id=note_id,
        user_id=1,
        lemma=lemma,
        pos="substantiivi",
        translation_ru="x",
        example_fi="x",
        example_ru="x",
        kind=NoteKind.word,
        deck_id=deck_id,
        meta={"forms_verified": True, "principal_forms": {n: f"{lemma}-{n}" for n in core}},
        created_at=created_at,
    )


def known_word_card(session, note_id: str, now: datetime) -> None:
    """The note's recognition card, answered Good twice - the word is known,
    only the deck and the budget stand between it and its forms."""
    card_id = f"{note_id}-rec"
    session.add(
        Card(
            id=card_id,
            note_id=note_id,
            user_id=1,
            type=CardType.recognition,
            due=now - timedelta(days=1),
            reps=2,
        )
    )
    for days in (3, 2):
        session.add(
            Review(
                card_id=card_id,
                user_id=1,
                rating=Rating.Good.value,
                reviewed_at=now - timedelta(days=days),
            )
        )


async def inflection_cards(session_factory) -> list[Card]:
    async with session_factory() as session:
        return list(
            (await session.scalars(select(Card).where(Card.type == CardType.inflection))).all()
        )


@pytest.mark.parametrize("side, opens", [("ru", False), ("fi", True), ("mix", True)])
async def test_forms_open_only_for_a_side_that_shows_them(session_factory, side, opens):
    # A Russian session never shows a form - opening forms there would spend
    # the day's daily_new_forms budget on cards nobody sees.
    now = datetime.now(UTC)
    async with session_factory() as session:
        session.add(User(id=1))
        await session.flush()
        deck = await create_deck(session, 1, "Общая")
        await session.flush()
        session.add(noun_with_forms("note-a", "talo", deck.id, now - timedelta(days=20)))
        await session.flush()
        known_word_card(session, "note-a", now)
        await session.commit()

    await prepare(session_factory, LearnSide(side), daily_new_forms=2)

    forms = await inflection_cards(session_factory)
    assert forms  # the cards exist either way - only opening them is held back
    opened = [c for c in forms if c.status == CardStatus.introduced]
    assert (len(opened) == 2) is opens
    assert all(c.introduced_at is None for c in forms if c.status != CardStatus.introduced)


async def test_forms_open_only_in_the_chosen_deck_and_join_its_queue(session_factory):
    now = datetime.now(UTC)
    async with session_factory() as session:
        session.add(User(id=1))
        await session.flush()
        deck_a = await create_deck(session, 1, "Общая")
        deck_b = await create_deck(session, 1, "Из книги")
        await session.flush()
        # Deck A's note is older, so a curriculum walking every deck would
        # spend the whole form budget on it before reaching deck B.
        session.add(noun_with_forms("note-a", "talo", deck_a.id, now - timedelta(days=20)))
        session.add(noun_with_forms("note-b", "kirja", deck_b.id, now - timedelta(days=10)))
        await session.flush()
        known_word_card(session, "note-a", now)
        known_word_card(session, "note-b", now)
        await session.commit()

    await prepare(session_factory, LearnSide.mix, deck_b.id, daily_new_forms=2)
    built = await queue_of(session_factory, LearnSide.mix, deck_b.id)

    forms = await inflection_cards(session_factory)
    opened = {c.id: c.note_id for c in forms if c.status == CardStatus.introduced}
    # Every note's form cards are created (the sync is not deck-scoped)...
    assert {c.note_id for c in forms} == {"note-a", "note-b"}
    # ...but only the chosen deck's are opened, with all of today's budget.
    assert sorted(opened.values()) == ["note-b", "note-b"]
    assert built.queue[0] == "note-b-rec"
    assert sorted(built.queue[1:]) == sorted(opened)


# --- build_queue: side, deck, cap ---------------------------------------------------


@pytest.mark.parametrize(
    "side, expected",
    [
        ("fi", {"recognition", "inflection"}),
        ("ru", {"production"}),
        ("mix", {"recognition", "production", "inflection"}),
    ],
)
async def test_a_side_queues_only_its_card_types(session_factory, side, expected):
    past = datetime.now(UTC) - timedelta(days=1)
    await seed(
        session_factory, *[make_card(t.value, "note-1", past, card_type=t) for t in CardType]
    )

    built = await queue_of(session_factory, LearnSide(side))

    assert set(built.queue) == expected


async def test_a_deck_narrows_the_queue_and_none_takes_every_deck(session_factory):
    past = datetime.now(UTC) - timedelta(days=1)
    async with session_factory() as session:
        session.add(User(id=1))
        await session.flush()
        deck_a = await create_deck(session, 1, "Общая")
        deck_b = await create_deck(session, 1, "Из книги")
        await session.flush()
        session.add_all(
            [make_note("note-a", deck_id=deck_a.id), make_note("note-b", deck_id=deck_b.id)]
        )
        await session.flush()
        session.add_all([make_card("card-a", "note-a", past), make_card("card-b", "note-b", past)])
        await session.commit()

    assert (await queue_of(session_factory, deck_id=deck_b.id)).queue == ["card-b"]
    assert set((await queue_of(session_factory)).queue) == {"card-a", "card-b"}


async def test_the_next_session_picks_up_the_cards_the_cap_left_behind(session_factory):
    # The cap defers cards, it doesn't drop them.
    now = datetime.now(UTC)
    await seed(
        session_factory,
        *[make_card(f"card-{i}", "note-1", now - timedelta(days=5 - i), reps=1) for i in range(5)],
    )

    first = (await queue_of(session_factory, session_max_cards=2)).queue
    assert first == ["card-0", "card-1"]
    await rate_all(session_factory, first)
    second = (await queue_of(session_factory, session_max_cards=2)).queue

    assert second == ["card-2", "card-3"]


# --- empty sessions and the end of one ----------------------------------------------


async def test_an_empty_russian_side_explains_why_until_a_production_card_exists(
    session_factory,
):
    # Production cards open only after recognition settles - early on a
    # Russian session is empty, and that is not a bug.
    past = datetime.now(UTC) - timedelta(days=1)
    await seed(session_factory, make_card("rec", "note-1", past))

    built = await queue_of(session_factory, LearnSide.ru)

    assert (built.queue, built.no_production) == ([], True)
    assert built.empty_text == f"{EMPTY_SIDE_TEXT}\n{NO_PRODUCTION_TEXT}"


async def test_an_empty_russian_side_with_production_not_due_gives_no_hint(session_factory):
    now = datetime.now(UTC)
    await seed(
        session_factory,
        make_card("rec", "note-1", now - timedelta(days=1)),
        make_card("prod", "note-1", now + timedelta(days=3), reps=1, card_type=CardType.production),
    )

    built = await queue_of(session_factory, LearnSide.ru)

    assert (built.queue, built.no_production) == ([], False)
    assert built.empty_text == EMPTY_SIDE_TEXT


async def test_an_empty_side_does_not_claim_all_is_done(session_factory):
    # Only the Russian side has work today - "всё выучено" would send the
    # learner away from it.
    now = datetime.now(UTC)
    await seed(
        session_factory,
        make_card("rec", "note-1", now + timedelta(days=3), reps=1),
        make_card("prod", "note-1", now - timedelta(days=1), reps=1, card_type=CardType.production),
    )

    built = await queue_of(session_factory, LearnSide.fi)

    assert built.queue == []
    assert built.empty_text == EMPTY_SIDE_TEXT


async def test_an_empty_mix_session_says_everything_is_done(session_factory):
    await seed(session_factory, make_card("rec", "note-1", datetime.now(UTC) + timedelta(days=3)))

    built = await queue_of(session_factory, LearnSide.mix)

    assert (built.queue, built.no_production) == ([], False)
    assert built.empty_text == EMPTY_ALL_TEXT


@pytest.mark.parametrize(
    "side, remaining, expected",
    [
        ("mix", 0, "Сессия окончена: 3 карточек пройдено - всё на сегодня!"),
        ("fi", 0, "Сессия окончена: 3 карточек пройдено - на этой стороне всё на сегодня!"),
        ("ru", 0, "Сессия окончена: 3 карточек пройдено - на этой стороне всё на сегодня!"),
        ("mix", 2, "Сессия окончена: 3 карточек пройдено, осталось 2."),
    ],
)
def test_session_end_text(side, remaining, expected):
    assert session_end_text(3, remaining, LearnSide(side), ".") == expected


# --- record_review --------------------------------------------------------------------


async def test_a_rating_that_lifts_recognition_past_the_threshold_opens_production_at_once(
    session_factory,
):
    # The production card is written in the same commit as the review that
    # earned it, not left for the next session's sync - the deck screen
    # counts it straight away.
    now = datetime.now(UTC)
    card = make_card("card-A", "note-1", now, reps=3)
    card.state = CardState.review
    card.stability = 2.0  # below the threshold until this rating lifts it
    card.difficulty = 5.0
    await seed(session_factory, card)
    async with session_factory() as session:
        session.add(
            Review(card_id="card-A", user_id=1, rating=3, reviewed_at=now - timedelta(days=10))
        )
        await session.commit()

    async with session_factory() as session:
        record = await record_review(session, await session.get(Card, "card-A"), Rating.Good, now)

    async with session_factory() as session:
        cards = (await session.scalars(select(Card).where(Card.note_id == "note-1"))).all()
        reviews = (await session.scalars(select(Review))).all()
    production = [c.id for c in cards if c.type == CardType.production]
    assert len(production) == 1
    assert record.created_card_ids == production
    assert len(reviews) == 2
    assert record.review_id == max(r.id for r in reviews)


async def test_a_rating_schedules_off_the_previous_review_not_the_current_one(
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

    await seed(
        session_factory,
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
        ),
    )
    async with session_factory() as session:
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

    now = datetime.now(UTC)
    async with session_factory() as session:
        await record_review(session, await session.get(Card, "card-A"), Rating.Good, now)

    async with session_factory() as session:
        card = await session.get(Card, "card-A")
        reviews = (
            await session.scalars(
                select(Review).where(Review.card_id == "card-A").order_by(Review.reviewed_at)
            )
        ).all()

    assert len(reviews) == len(history) + 1
    now = reviews[-1].reviewed_at  # as SQLite gives it back
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


# --- drop_deleted_head: a word deleted mid-session ----------------------------------


async def test_deleted_cards_are_dropped_from_the_head_only(session_factory, caplog):
    # Deleting a word knows nothing about sessions: its card ids stay in
    # the queue. Only the head is checked - a deleted card further down is
    # dropped once it gets there.
    doomed = make_note("note-2")
    doomed.lemma = "naapuri"
    await seed(
        session_factory,
        *[make_card(c, n, NOW) for c, n in [("A", "note-2"), ("B", "note-1"), ("C", "note-2")]],
        notes=(make_note("note-1"), doomed),
    )
    async with session_factory() as session:
        await session.execute(delete(Card).where(Card.note_id == "note-2"))
        await session.commit()

    with caplog.at_level(logging.INFO, logger="kielikaveri.learn_core"):
        async with session_factory() as session:
            assert await drop_deleted_head(session, ["A", "B", "C"]) == ["B", "C"]
            assert await drop_deleted_head(session, ["C"], source="reader") == []

    skipped = [log_fields(r.getMessage()) for r in caplog.records]
    assert skipped == [
        {"event": "learn.skip_deleted", "card_id": "A"},
        {"event": "learn.skip_deleted", "card_id": "C", "source": "reader"},
    ]
