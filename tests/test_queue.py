from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import Card, CardType, Note, NoteKind, Review, User
from kielikaveri.srs.queue import (
    build_session_queue,
    count_new_cards_today,
    defer_overdue_tail,
    due_cards,
    overdue_count,
    study_day_bounds,
)

HELSINKI = ZoneInfo("Europe/Helsinki")


@pytest.fixture
async def session_factory(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    await create_all(engine)
    yield make_session_factory(engine)
    await engine.dispose()


def make_note(note_id: str, user_id: int) -> Note:
    return Note(
        id=note_id,
        user_id=user_id,
        lemma="hakea",
        translation_ru="искать",
        example_fi="Haen töitä.",
        example_ru="Я ищу работу.",
        kind=NoteKind.word,
        meta={},
    )


def make_card(card_id: str, note_id: str, user_id: int, due: datetime, reps: int = 0) -> Card:
    return Card(
        id=card_id, note_id=note_id, user_id=user_id, type=CardType.recognition, due=due, reps=reps
    )


# --- study_day_bounds -------------------------------------------------------


def test_before_boundary_hour_belongs_to_the_previous_study_day():
    now = datetime(2026, 8, 24, 2, 0, tzinfo=HELSINKI).astimezone(UTC)
    start, end = study_day_bounds(now, boundary_hour=4)
    assert start == datetime(2026, 8, 23, 4, 0, tzinfo=HELSINKI).astimezone(UTC)
    assert end == datetime(2026, 8, 24, 4, 0, tzinfo=HELSINKI).astimezone(UTC)


def test_after_boundary_hour_belongs_to_todays_study_day():
    now = datetime(2026, 8, 24, 10, 0, tzinfo=HELSINKI).astimezone(UTC)
    start, end = study_day_bounds(now, boundary_hour=4)
    assert start == datetime(2026, 8, 24, 4, 0, tzinfo=HELSINKI).astimezone(UTC)
    assert end == datetime(2026, 8, 25, 4, 0, tzinfo=HELSINKI).astimezone(UTC)


# --- due_cards / overdue_count ----------------------------------------------


async def test_due_cards_does_not_leak_another_users_cards(session_factory):
    now = datetime(2026, 8, 24, 10, 0, tzinfo=UTC)
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(User(id=2))
        session.add(make_note("note-1", 1))
        session.add(make_note("note-2", 2))
        await session.flush()
        session.add(make_card("card-1", "note-1", 1, due=now - timedelta(days=1)))
        session.add(make_card("card-2", "note-2", 2, due=now - timedelta(days=1)))
        await session.commit()

        cards = await due_cards(session, 1, now, limit=10)

    assert [c.id for c in cards] == ["card-1"]


async def test_overdue_count_ignores_cards_not_yet_due(session_factory):
    now = datetime(2026, 8, 24, 10, 0, tzinfo=UTC)
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(make_note("note-1", 1))
        await session.flush()
        session.add(make_card("card-1", "note-1", 1, due=now - timedelta(days=1)))
        session.add(make_card("card-2", "note-1", 1, due=now + timedelta(days=1)))
        await session.commit()

        count = await overdue_count(session, 1, now)

    assert count == 1


# --- count_new_cards_today ---------------------------------------------------


async def test_count_new_cards_today_counts_first_reviews_in_the_window(session_factory):
    now = datetime(2026, 8, 24, 10, 0, tzinfo=UTC)
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(make_note("note-1", 1))
        await session.flush()
        session.add(make_card("card-1", "note-1", 1, due=now, reps=1))
        session.add(make_card("card-2", "note-1", 1, due=now, reps=1))
        await session.flush()
        # card-1's first review is today, card-2's was yesterday (a review
        # today, but not its *first*, must not count as "new").
        session.add(Review(card_id="card-1", user_id=1, rating=3, reviewed_at=now))
        session.add(
            Review(card_id="card-2", user_id=1, rating=1, reviewed_at=now - timedelta(days=1))
        )
        session.add(Review(card_id="card-2", user_id=1, rating=3, reviewed_at=now))
        await session.commit()

        count = await count_new_cards_today(session, 1, now, boundary_hour=4)

    assert count == 1


# --- build_session_queue -----------------------------------------------------


async def test_build_session_queue_caps_total_size(session_factory):
    now = datetime(2026, 8, 24, 10, 0, tzinfo=UTC)
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(make_note("note-1", 1))
        await session.flush()
        for i in range(5):
            session.add(make_card(f"card-{i}", "note-1", 1, due=now - timedelta(minutes=i), reps=1))
        await session.commit()

        queue = await build_session_queue(
            session, 1, now, session_max_cards=3, daily_new_limit=10, boundary_hour=4
        )

    assert len(queue) == 3


async def test_build_session_queue_holds_back_new_cards_past_the_daily_limit(session_factory):
    now = datetime(2026, 8, 24, 10, 0, tzinfo=UTC)
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(make_note("note-1", 1))
        await session.flush()
        # 3 never-reviewed cards, all due, daily_new_limit=1.
        for i in range(3):
            session.add(make_card(f"new-{i}", "note-1", 1, due=now - timedelta(minutes=i), reps=0))
        # 1 already-reviewed (due) card - must not be held back by the new-limit.
        session.add(make_card("review-1", "note-1", 1, due=now, reps=2))
        await session.commit()

        queue = await build_session_queue(
            session, 1, now, session_max_cards=20, daily_new_limit=1, boundary_hour=4
        )

    new_in_queue = [card_id for card_id in queue if card_id.startswith("new-")]
    assert len(new_in_queue) == 1
    assert "review-1" in queue


# --- defer_overdue_tail -------------------------------------------------------


async def test_defer_overdue_tail_postpones_everything_past_keep_n(session_factory):
    now = datetime(2026, 8, 24, 10, 0, tzinfo=UTC)
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(make_note("note-1", 1))
        await session.flush()
        for i in range(5):
            session.add(
                make_card(f"card-{i}", "note-1", 1, due=now - timedelta(days=5 - i), reps=1)
            )
        await session.commit()

        postponed = await defer_overdue_tail(session, 1, now, keep_n=2, postpone_days=7)
        await session.commit()

        remaining_overdue = await overdue_count(session, 1, now)

    assert postponed == 3
    assert remaining_overdue == 2
