from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from finn_cards.morphology import NOMINAL_FORMS
from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import Card, CardState, CardStatus, CardType, Note, NoteKind, User
from kielikaveri.grammar import FORM_TASKS, CurriculumLevel
from kielikaveri.srs.curriculum import MASTERY_STABILITY_DAYS, eligible_forms, introduce_due_forms
from kielikaveri.srs.graduation import sync_user_card_types
from kielikaveri.srs.queue import build_session_queue, card_counters, overdue_count
from kielikaveri.srs.scheduler import Rating, SrsState
from kielikaveri.srs.scheduler import review as apply_review

NOW = datetime(2026, 9, 10, 9, 0, tzinfo=UTC)

CORE_NOUN_FORMS = [n for n, t in FORM_TASKS.items() if t.level is CurriculumLevel.core]
QUIZZABLE_NOUN_FORMS = [n for n in NOMINAL_FORMS if n in FORM_TASKS]


@pytest.fixture
async def session_factory(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    await create_all(engine)
    yield make_session_factory(engine)
    await engine.dispose()


def make_noun(note_id: str, lemma: str, user_id: int = 1, created_at: datetime = NOW) -> Note:
    forms = {"nominatiivi": lemma} | {name: f"{lemma}-{name}" for name in QUIZZABLE_NOUN_FORMS}
    return Note(
        id=note_id,
        user_id=user_id,
        lemma=lemma,
        pos="substantiivi",
        translation_ru="x",
        example_fi="x",
        example_ru="x",
        kind=NoteKind.word,
        meta={"forms_verified": True, "principal_forms": forms},
        created_at=created_at,
    )


async def seed_three_nouns(session_factory) -> None:
    async with session_factory() as session:
        session.add(User(id=1))
        for i, lemma in enumerate(["kauppa", "talo", "kirje"]):
            session.add(make_noun(f"n{i}", lemma, created_at=NOW + timedelta(seconds=i)))
        await session.flush()
        await sync_user_card_types(session, 1, NOW)
        await session.commit()


async def master_recognition(session_factory, note_id: str) -> None:
    """Bring a note's recognition card past the curriculum's unlock threshold."""
    async with session_factory() as session:
        card = (
            await session.scalars(
                select(Card).where(Card.note_id == note_id, Card.type == CardType.recognition)
            )
        ).one()
        card.stability = MASTERY_STABILITY_DAYS + 1
        card.state = CardState.review
        card.reps = 2
        await session.commit()


async def master_introduced_forms(session_factory) -> None:
    """Make every already-introduced inflection card count as retained."""
    async with session_factory() as session:
        for card in (
            await session.scalars(
                select(Card).where(
                    Card.type == CardType.inflection, Card.status == CardStatus.introduced
                )
            )
        ).all():
            card.stability = MASTERY_STABILITY_DAYS + 1
        await session.commit()


async def cards_of(session_factory, note_id: str, **filters) -> list[Card]:
    async with session_factory() as session:
        stmt = select(Card).where(Card.note_id == note_id)
        for column, value in filters.items():
            stmt = stmt.where(getattr(Card, column) == value)
        return list((await session.scalars(stmt.order_by(Card.form))).all())


# --- 1-2: the cards exist, the learner is not buried in them ------------------


async def test_three_nouns_create_all_their_form_cards_without_introducing_them(session_factory):
    await seed_three_nouns(session_factory)

    async with session_factory() as session:
        inflection = (
            await session.scalars(select(Card).where(Card.type == CardType.inflection))
        ).all()
        counters = await card_counters(session, 1, NOW)

    assert len(inflection) == 3 * len(QUIZZABLE_NOUN_FORMS) == 36
    # Every one of them exists and none of them is the learner's problem yet.
    assert all(c.status == CardStatus.not_introduced for c in inflection)
    assert counters.total == 39  # 36 inflection + 3 recognition
    assert counters.not_introduced == 36
    assert counters.introduced == 3
    assert counters.overdue == 0


async def test_not_introduced_forms_are_neither_due_nor_overdue(session_factory):
    await seed_three_nouns(session_factory)

    async with session_factory() as session:
        counters = await card_counters(session, 1, NOW)
        debt = await overdue_count(session, 1, NOW, reviewed_only=True)
        queue = await build_session_queue(
            session, 1, NOW, session_max_cards=20, daily_new_limit=10, boundary_hour=4
        )

    assert counters.due == 3  # only the three recognition cards
    assert debt == 0
    assert len(queue) == 3


# --- 3: the curriculum gates introduction -------------------------------------


async def test_no_form_opens_while_the_word_itself_is_still_new(session_factory):
    # Drilling the cases of a word you cannot yet recognise is pointless -
    # the note's recognition card has to mature first.
    await seed_three_nouns(session_factory)

    async with session_factory() as session:
        introduced = await introduce_due_forms(session, 1, NOW, daily_new_forms=4, boundary_hour=4)
        await session.commit()

    assert introduced == []


async def test_core_forms_open_first_and_only_up_to_the_daily_form_budget(session_factory):
    await seed_three_nouns(session_factory)
    await master_recognition(session_factory, "n0")

    async with session_factory() as session:
        introduced = await introduce_due_forms(session, 1, NOW, daily_new_forms=3, boundary_hour=4)
        await session.commit()

    assert len(introduced) == 3
    assert all(FORM_TASKS[c.form].level is CurriculumLevel.core for c in introduced)


async def test_the_form_budget_is_spent_once_per_study_day(session_factory):
    await seed_three_nouns(session_factory)
    await master_recognition(session_factory, "n0")

    async with session_factory() as session:
        first = await introduce_due_forms(session, 1, NOW, daily_new_forms=3, boundary_hour=4)
        await session.commit()
        # Same study day, a second /learn - the budget is already gone.
        again = await introduce_due_forms(session, 1, NOW, daily_new_forms=3, boundary_hour=4)
        await session.commit()
        # Next study day it refills.
        tomorrow = NOW + timedelta(days=1)
        third = await introduce_due_forms(session, 1, tomorrow, daily_new_forms=3, boundary_hour=4)
        await session.commit()

    assert len(first) == 3
    assert again == []
    # Two, not three: n0 has five core forms and three are already open, while
    # extended stays shut until the core ones are retained and the other two
    # notes are still unknown words. The budget is a ceiling, not a quota.
    assert len(third) == 2


async def test_extended_forms_wait_until_the_core_ones_are_mastered(session_factory):
    await seed_three_nouns(session_factory)
    await master_recognition(session_factory, "n0")

    # Open every core form, but leave them fragile.
    async with session_factory() as session:
        await introduce_due_forms(session, 1, NOW, daily_new_forms=99, boundary_hour=4)
        await session.commit()

    core_cards = await cards_of(session_factory, "n0", type=CardType.inflection)
    assert {c.form for c in core_cards if c.status == CardStatus.introduced} == set(
        CORE_NOUN_FORMS
    ) & set(QUIZZABLE_NOUN_FORMS)

    tomorrow = NOW + timedelta(days=1)
    async with session_factory() as session:
        blocked = await introduce_due_forms(
            session, 1, tomorrow, daily_new_forms=99, boundary_hour=4
        )
        await session.commit()
    assert blocked == []  # core is open but not yet retained

    async with session_factory() as session:
        for card in (
            await session.scalars(
                select(Card).where(
                    Card.note_id == "n0",
                    Card.type == CardType.inflection,
                    Card.status == CardStatus.introduced,
                )
            )
        ).all():
            card.stability = MASTERY_STABILITY_DAYS + 1
        await session.commit()

    async with session_factory() as session:
        opened = await introduce_due_forms(session, 1, tomorrow, daily_new_forms=2, boundary_hour=4)
        await session.commit()

    assert len(opened) == 2
    assert all(FORM_TASKS[c.form].level is CurriculumLevel.extended for c in opened)


async def test_a_grammatical_group_opens_together_rather_than_scattered(session_factory):
    # missä? / mistä? are one system - they should arrive back to back, not
    # months apart.
    await seed_three_nouns(session_factory)
    await master_recognition(session_factory, "n0")
    async with session_factory() as session:
        await introduce_due_forms(session, 1, NOW, daily_new_forms=99, boundary_hour=4)
        for card in (
            await session.scalars(
                select(Card).where(Card.note_id == "n0", Card.type == CardType.inflection)
            )
        ).all():
            if card.status == CardStatus.introduced:
                card.stability = MASTERY_STABILITY_DAYS + 1
        await session.commit()

    async with session_factory() as session:
        opened = await introduce_due_forms(
            session, 1, NOW + timedelta(days=1), daily_new_forms=2, boundary_hour=4
        )
        await session.commit()

    assert [c.form for c in opened] == ["inessiivi", "elatiivi"]


async def test_suspended_cards_are_never_shown(session_factory):
    await seed_three_nouns(session_factory)
    async with session_factory() as session:
        card = (
            await session.scalars(
                select(Card).where(Card.note_id == "n0", Card.type == CardType.recognition)
            )
        ).one()
        card.status = CardStatus.suspended
        await session.commit()

        queue = await build_session_queue(
            session, 1, NOW, session_max_cards=20, daily_new_limit=10, boundary_hour=4
        )
        counters = await card_counters(session, 1, NOW)

    assert card.id not in queue
    assert counters.due == 2
    assert counters.introduced == 2


# --- 4-7: FSRS still owns everything that has been introduced -----------------


async def test_reviewing_one_form_moves_only_that_forms_schedule(session_factory):
    await seed_three_nouns(session_factory)
    await master_recognition(session_factory, "n0")
    async with session_factory() as session:
        await introduce_due_forms(session, 1, NOW, daily_new_forms=99, boundary_hour=4)
        await session.commit()

    async with session_factory() as session:
        illative = (
            await session.scalars(
                select(Card).where(Card.note_id == "n0", Card.form == "illatiivi")
            )
        ).one()
        before = {
            c.form: (c.due, c.stability, c.reps)
            for c in (
                await session.scalars(
                    select(Card).where(Card.note_id == "n0", Card.form != "illatiivi")
                )
            ).all()
        }
        state = apply_review(
            SrsState(
                state=illative.state,
                due=illative.due,
                stability=illative.stability,
                difficulty=illative.difficulty,
                reps=illative.reps,
                lapses=illative.lapses,
                step=illative.step,
            ),
            Rating.Easy,
            NOW,
        )
        illative.due, illative.stability, illative.reps = state.due, state.stability, state.reps
        await session.commit()

    async with session_factory() as session:
        after = {
            c.form: (c.due, c.stability, c.reps)
            for c in (
                await session.scalars(
                    select(Card).where(Card.note_id == "n0", Card.form != "illatiivi")
                )
            ).all()
        }
        moved = await session.get(Card, illative.id)

    assert after == before  # nothing else budged
    assert moved.due > NOW and moved.reps == 1


async def test_an_unintroduced_form_has_no_schedule_to_be_moved(session_factory):
    await seed_three_nouns(session_factory)
    await master_recognition(session_factory, "n0")
    async with session_factory() as session:
        await introduce_due_forms(session, 1, NOW, daily_new_forms=99, boundary_hour=4)
        await session.commit()

    translative = (await cards_of(session_factory, "n0", form="translatiivi"))[0]

    assert translative.status == CardStatus.not_introduced
    assert translative.reps == 0
    assert translative.stability is None

    async with session_factory() as session:
        queue = await build_session_queue(
            session, 1, NOW, session_max_cards=99, daily_new_limit=99, boundary_hour=4
        )
    assert translative.id not in queue


async def test_a_form_joins_fsrs_once_introduced(session_factory):
    await seed_three_nouns(session_factory)
    await master_recognition(session_factory, "n0")

    async with session_factory() as session:
        before = await build_session_queue(
            session, 1, NOW, session_max_cards=99, daily_new_limit=99, boundary_hour=4
        )
        opened = await introduce_due_forms(session, 1, NOW, daily_new_forms=1, boundary_hour=4)
        await session.commit()
        after = await build_session_queue(
            session, 1, NOW, session_max_cards=99, daily_new_limit=99, boundary_hour=4
        )

    assert opened[0].id not in before
    assert opened[0].id in after


# --- 8-9: the session caps still hold -----------------------------------------


async def test_daily_new_limit_still_caps_what_a_session_admits(session_factory):
    await seed_three_nouns(session_factory)
    for note_id in ("n0", "n1", "n2"):
        await master_recognition(session_factory, note_id)
    async with session_factory() as session:
        # Open far more forms than a day's session should ever show.
        await introduce_due_forms(session, 1, NOW, daily_new_forms=99, boundary_hour=4)
        await session.commit()

        queue = await build_session_queue(
            session, 1, NOW, session_max_cards=99, daily_new_limit=10, boundary_hour=4
        )
        cards = [await session.get(Card, card_id) for card_id in queue]

    # The limit caps never-reviewed cards only; the three mastered
    # recognition cards are reviews and are never held back by it.
    assert len([c for c in cards if c.reps == 0]) == 10
    assert len(queue) == 13


async def test_session_max_cards_still_caps_the_queue(session_factory):
    await seed_three_nouns(session_factory)
    for note_id in ("n0", "n1", "n2"):
        await master_recognition(session_factory, note_id)
    async with session_factory() as session:
        await introduce_due_forms(session, 1, NOW, daily_new_forms=99, boundary_hour=4)
        await session.commit()
    await master_introduced_forms(session_factory)
    async with session_factory() as session:
        # Core is retained now, so the extended layer opens too - more than
        # enough due cards to run into the session cap.
        await introduce_due_forms(session, 1, NOW, daily_new_forms=99, boundary_hour=4)
        await session.commit()

        queue = await build_session_queue(
            session, 1, NOW, session_max_cards=20, daily_new_limit=99, boundary_hour=4
        )

    assert len(queue) == 20


# --- 10: idempotence ----------------------------------------------------------


async def test_rerunning_sync_creates_no_duplicates_and_keeps_introductions(session_factory):
    await seed_three_nouns(session_factory)
    await master_recognition(session_factory, "n0")
    async with session_factory() as session:
        opened = await introduce_due_forms(session, 1, NOW, daily_new_forms=2, boundary_hour=4)
        opened_ids = [c.id for c in opened]
        await session.commit()

    async with session_factory() as session:
        created_again = await sync_user_card_types(session, 1, NOW + timedelta(days=1))
        await session.commit()

    async with session_factory() as session:
        cards = (await session.scalars(select(Card))).all()
        still_introduced = [
            c.id for c in cards if c.id in opened_ids and c.status == CardStatus.introduced
        ]

    inflection = [c for c in cards if c.type == CardType.inflection]
    # Not a duplicate: n0's recognition card matured, which is exactly the
    # condition that opens its production card. No inflection card is
    # created twice, and nothing already introduced falls back.
    assert [c.type for c in created_again] == [CardType.production]
    assert len(inflection) == 36
    assert len({(c.note_id, c.form) for c in inflection}) == 36
    assert sorted(still_introduced) == sorted(opened_ids)


def test_eligible_forms_is_pure_and_needs_no_session():
    # The policy itself is a function of the note's cards - testable without
    # a database, which is what keeps it easy to re-tune.
    recognition = Card(
        id="r",
        note_id="n",
        user_id=1,
        type=CardType.recognition,
        due=NOW,
        stability=9.0,
        status=CardStatus.introduced,
    )
    forms = [
        Card(
            id=name,
            note_id="n",
            user_id=1,
            type=CardType.inflection,
            form=name,
            due=NOW,
            status=CardStatus.not_introduced,
        )
        for name in ("essiivi", "genetiivi", "inessiivi")
    ]

    eligible = eligible_forms([recognition, *forms])

    assert [c.form for c in eligible] == ["genetiivi"]  # core before the rest
