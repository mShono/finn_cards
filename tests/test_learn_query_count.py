"""/learn must not issue one SQL statement per note.

Both of /learn's per-note walks - graduation.sync_user_card_types and
curriculum.introduce_due_forms - used to run `SELECT ... WHERE cards.note_id
= ?` inside their loop. Measured on this schema before the fix: 50 notes ->
51 statements, 200 -> 201, 500 -> 501, i.e. linear in the size of the
learner's collection, on a path that runs at the start of every session.

The tests count statements at the driver level (`before_cursor_execute`),
not by patching or asserting on the functions themselves: a mock that made
the loop stop querying would also make the measurement meaningless. Whatever
the code really sends to SQLite is what is counted here.

Behaviour is pinned separately, by comparing against the per-note walk the
batched code replaced - see test_..._matches_the_per_note_walk. That
reference calls the very same production functions, one note at a time
(ensure_card_types still reads a note's cards itself when it isn't handed
them), so it is the old implementation, not a reimplementation of it.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import event, inspect, select

from finn_cards.morphology import NOMINAL_FORMS
from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import Card, Note, NoteKind, Review, User
from kielikaveri.grammar import FORM_TASKS
from kielikaveri.srs.curriculum import (
    eligible_forms,
    introduce_due_forms,
    successful_answer_counts,
)
from kielikaveri.srs.graduation import ensure_card_types, sync_user_card_types
from kielikaveri.srs.scheduler import Rating

NOW = datetime(2026, 9, 15, 10, 0, tzinfo=UTC)
QUIZZABLE_NOUN_FORMS = [n for n in NOMINAL_FORMS if n in FORM_TASKS]


class StatementCounter:
    """Counts every statement the driver actually executes on this engine."""

    def __init__(self, engine):
        self.statements: list[str] = []
        event.listen(engine.sync_engine, "before_cursor_execute", self._record)

    def _record(self, conn, cursor, statement, params, context, executemany) -> None:
        self.statements.append(statement)

    def reset(self) -> None:
        self.statements.clear()

    @property
    def card_selects(self) -> int:
        """Statements reading from `cards` - the ones that used to be per-note."""
        return sum(
            1
            for s in self.statements
            if s.lstrip().upper().startswith("SELECT") and "FROM cards" in s
        )


@pytest.fixture
async def counted(tmp_path):
    """A session factory plus a live count of the SQL it causes."""
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    await create_all(engine)
    yield make_session_factory(engine), StatementCounter(engine)
    await engine.dispose()


def make_noun(note_id: str, lemma: str, created_at: datetime, user_id: int = 1) -> Note:
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


async def seed_notes(session_factory, count: int) -> None:
    async with session_factory() as session:
        session.add(User(id=1))
        for i in range(count):
            # created_at spread out so introduce_due_forms' note order is total.
            session.add(make_noun(f"n{i}", f"sana{i}", NOW + timedelta(seconds=i)))
        await session.commit()


async def unlock_every_word(session_factory) -> None:
    """Rate every recognition card Good twice, so the curriculum has work to do.

    Written straight to the review log, which is where
    successful_answer_counts reads the unlock threshold from.
    """
    async with session_factory() as session:
        cards = (await session.scalars(select(Card))).all()
        for card in cards:
            if card.type.value == "recognition":
                for _ in range(2):
                    session.add(
                        Review(
                            card_id=card.id,
                            user_id=1,
                            rating=Rating.Good.value,
                            reviewed_at=NOW - timedelta(days=1),
                        )
                    )
        await session.commit()


# --- the N+1 regressions themselves -----------------------------------------


@pytest.mark.parametrize("note_count", [5, 40])
async def test_sync_user_card_types_reads_cards_once_whatever_the_note_count(counted, note_count):
    """Creating cards for N notes must not cost N SELECTs.

    Parametrized rather than asserting a magic number: a count that does not
    move when the collection grows eightfold is what "batched" means, and it
    stays true if the function ever legitimately gains a query.
    """
    session_factory, counter = counted
    await seed_notes(session_factory, note_count)

    async with session_factory() as session:
        counter.reset()
        created = await sync_user_card_types(session, 1, NOW)
        await session.commit()

    assert len(created) == note_count * (1 + len(QUIZZABLE_NOUN_FORMS))
    assert counter.card_selects == 1


@pytest.mark.parametrize("note_count", [5, 40])
async def test_sync_user_card_types_reads_cards_once_when_nothing_is_missing(counted, note_count):
    """The ordinary case: every card already exists and the run creates nothing.

    This is what a returning learner's /learn does, and it was the worst of
    the two - a full statement per note to conclude there was nothing to do.
    """
    session_factory, counter = counted
    await seed_notes(session_factory, note_count)
    async with session_factory() as session:
        await sync_user_card_types(session, 1, NOW)
        await session.commit()

    async with session_factory() as session:
        counter.reset()
        created = await sync_user_card_types(session, 1, NOW)

    assert created == []
    assert counter.card_selects == 1


@pytest.mark.parametrize("note_count", [5, 40])
async def test_introduce_due_forms_reads_cards_once_when_nothing_is_eligible(counted, note_count):
    """No word is known yet, so the loop walks every note and opens nothing.

    The budget never fills, which is precisely when the old code paid its
    full N statements - and it is the normal state of a session, not a
    corner case.
    """
    session_factory, counter = counted
    await seed_notes(session_factory, note_count)
    async with session_factory() as session:
        await sync_user_card_types(session, 1, NOW)
        await session.commit()

    async with session_factory() as session:
        counter.reset()
        opened = await introduce_due_forms(session, 1, NOW, daily_new_forms=10, boundary_hour=4)

    assert opened == []
    assert counter.card_selects == 2  # count_introduced_today + the batched fetch


@pytest.mark.parametrize("note_count", [5, 40])
async def test_introduce_due_forms_reads_cards_once_when_every_word_is_known(counted, note_count):
    session_factory, counter = counted
    await seed_notes(session_factory, note_count)
    async with session_factory() as session:
        await sync_user_card_types(session, 1, NOW)
        await session.commit()
    await unlock_every_word(session_factory)

    async with session_factory() as session:
        counter.reset()
        opened = await introduce_due_forms(session, 1, NOW, daily_new_forms=10, boundary_hour=4)
        await session.commit()

    assert len(opened) == 10
    assert counter.card_selects == 2


# --- behaviour is unchanged --------------------------------------------------


def snapshot(cards) -> list[tuple]:
    return sorted(
        (c.note_id, c.type.value, c.form, c.status.value, c.introduced_at, c.due) for c in cards
    )


async def per_note_walk(session_factory, note_count: int) -> list[tuple]:
    """The implementation that was replaced, run against its own database.

    sync_user_card_types' loop and introduce_due_forms' loop as they were,
    both calling the production functions one note at a time - so this is the
    old code path, not a second opinion about what it did.
    """
    async with session_factory() as session:
        notes = (await session.scalars(select(Note).where(Note.user_id == 1))).all()
        for note in notes:
            await ensure_card_types(session, note, NOW)
        await session.commit()

    await unlock_every_word(session_factory)

    async with session_factory() as session:
        successes = await successful_answer_counts(session, 1)
        notes = (
            await session.scalars(
                select(Note).where(Note.user_id == 1).order_by(Note.created_at, Note.id)
            )
        ).all()
        introduced: list[Card] = []
        for note in notes:
            if len(introduced) >= 10:
                break
            cards = list((await session.scalars(select(Card).where(Card.note_id == note.id))).all())
            for card in eligible_forms(cards, successes):
                if len(introduced) >= 10:
                    break
                card.status = card.status.__class__.introduced
                card.introduced_at = NOW
                card.due = NOW
                introduced.append(card)
        await session.commit()

    async with session_factory() as session:
        return snapshot((await session.scalars(select(Card))).all())


@pytest.fixture
async def plain_session_factory(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path}/reference.db")
    await create_all(engine)
    yield make_session_factory(engine)
    await engine.dispose()


async def test_batched_learn_prologue_matches_the_per_note_walk(counted, plain_session_factory):
    """Same cards, same statuses, same due dates - only fewer statements.

    Twelve notes: enough that the form budget (10) runs out mid-collection,
    so the early-exit from the note loop is exercised on both sides.
    """
    session_factory, _counter = counted
    await seed_notes(session_factory, 12)
    await seed_notes(plain_session_factory, 12)

    expected = await per_note_walk(plain_session_factory, 12)

    async with session_factory() as session:
        await sync_user_card_types(session, 1, NOW)
        await session.commit()
    await unlock_every_word(session_factory)
    async with session_factory() as session:
        await introduce_due_forms(session, 1, NOW, daily_new_forms=10, boundary_hour=4)
        await session.commit()

    async with session_factory() as session:
        actual = snapshot((await session.scalars(select(Card))).all())

    assert actual == expected


async def test_introduce_due_forms_batch_stays_inside_the_chosen_deck(counted):
    """The batched fetch is deck-filtered exactly as the note query is.

    A note in another deck must not have its forms opened, and must not eat
    the budget either.
    """
    session_factory, _counter = counted
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(make_noun("in", "sisalla", NOW))
        session.add(make_noun("out", "ulkona", NOW + timedelta(seconds=1)))
        await session.flush()
        (await session.get(Note, "in")).deck_id = "deck-a"
        (await session.get(Note, "out")).deck_id = "deck-b"
        await session.commit()
    async with session_factory() as session:
        await sync_user_card_types(session, 1, NOW)
        await session.commit()
    await unlock_every_word(session_factory)

    async with session_factory() as session:
        opened = await introduce_due_forms(
            session, 1, NOW, daily_new_forms=99, boundary_hour=4, deck_id="deck-a"
        )
        await session.commit()

    assert opened
    assert {c.note_id for c in opened} == {"in"}


async def test_cards_note_id_is_indexed(counted):
    """Without it SQLite scans all of `cards` for every note lookup - it
    creates no index for a foreign key on its own.
    """
    session_factory, _counter = counted
    async with session_factory() as session:
        indexes = await session.run_sync(
            lambda sync_session: inspect(sync_session.get_bind()).get_indexes("cards")
        )
    assert any(ix["column_names"] == ["note_id"] for ix in indexes)
