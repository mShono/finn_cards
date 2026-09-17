from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import event, select

from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import Card, CardStatus, CardType, Deck, Note, NoteKind, Review, User
from kielikaveri.srs.queue import (
    CardCounters,
    build_session_queue,
    card_counters,
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


def make_note(note_id: str, user_id: int, deck_id: str | None = None) -> Note:
    # Lemma derived from note_id: notes are unique per (user, deck, lemma, pos),
    # and every call here means a genuinely different word.
    return Note(
        id=note_id,
        user_id=user_id,
        lemma=f"hakea-{note_id}",
        translation_ru="искать",
        example_fi="Haen töitä.",
        example_ru="Я ищу работу.",
        kind=NoteKind.word,
        deck_id=deck_id,
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


async def test_a_new_card_backlog_larger_than_the_old_fetch_window_does_not_starve_a_review_card(
    session_factory,
):
    # Regression: candidates used to come from one window of
    # session_max_cards * 5 rows, oldest due first. More new cards than that
    # due *before* a review filled the whole window, all but the daily budget
    # were skipped, and the review - due, and already known to the learner -
    # was never fetched. Reviews now have a window of their own.
    now = datetime(2026, 8, 24, 10, 0, tzinfo=UTC)
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(make_note("note-1", 1))
        await session.flush()
        # session_max_cards=3 -> the old window was 15. 20 new cards, all due
        # *before* the review card, would fill it on their own.
        for i in range(20):
            session.add(
                make_card(f"new-{i}", "note-1", 1, due=now - timedelta(days=1, seconds=i), reps=0)
            )
        session.add(make_card("review-1", "note-1", 1, due=now - timedelta(minutes=1), reps=2))
        await session.commit()

        queue = await build_session_queue(
            session, 1, now, session_max_cards=3, daily_new_limit=1, boundary_hour=4
        )

    # The one new card the budget allows is due earlier, so it still leads.
    assert queue == ["new-19", "review-1"]


@pytest.mark.parametrize("spent_by", ["zero_limit", "reviewed_today"])
async def test_an_exhausted_new_budget_still_leaves_due_reviews_in_the_queue(
    session_factory, spent_by
):
    # With the budget gone, the old single window turned into an empty
    # session - "nothing to review" - while reviews were due.
    now = datetime(2026, 8, 24, 10, 0, tzinfo=UTC)
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(make_note("note-1", 1))
        await session.flush()
        for i in range(30):
            session.add(make_card(f"new-{i:02}", "note-1", 1, due=now - timedelta(days=1), reps=0))
        session.add(make_card("review-1", "note-1", 1, due=now - timedelta(hours=1), reps=2))
        session.add(make_card("review-2", "note-1", 1, due=now - timedelta(days=3), reps=5))
        daily_new_limit = 0
        if spent_by == "reviewed_today":
            # Two cards first reviewed today use up a limit of two.
            daily_new_limit = 2
            for i in range(2):
                session.add(
                    make_card(f"done-{i}", "note-1", 1, due=now + timedelta(days=1), reps=1)
                )
            await session.flush()
            for i in range(2):
                session.add(
                    Review(
                        card_id=f"done-{i}",
                        user_id=1,
                        rating=3,
                        reviewed_at=now - timedelta(minutes=5),
                    )
                )
        await session.commit()

        queue = await build_session_queue(
            session, 1, now, session_max_cards=3, daily_new_limit=daily_new_limit, boundary_hour=4
        )

    assert queue == ["review-2", "review-1"]


async def test_more_reviews_than_the_session_cap_returns_the_first_ones_in_due_order(
    session_factory,
):
    now = datetime(2026, 8, 24, 10, 0, tzinfo=UTC)
    tied = now - timedelta(hours=2)
    async with session_factory() as session:
        session.add(User(id=1))
        # Inserted younger-note-first, so insert order contradicts the order.
        younger = make_note("note-younger", 1)
        younger.created_at = now
        session.add(younger)
        older = make_note("note-older", 1)
        older.created_at = now - timedelta(days=1)
        session.add(older)
        await session.flush()
        session.add(make_card("r-latest", "note-older", 1, due=now - timedelta(minutes=1), reps=1))
        session.add(make_card("r-tied-younger", "note-younger", 1, due=tied, reps=1))
        session.add(make_card("r-tied-older", "note-older", 1, due=tied, reps=1))
        session.add(make_card("r-oldest", "note-younger", 1, due=now - timedelta(days=4), reps=3))
        session.add(make_card("r-middle", "note-older", 1, due=now - timedelta(hours=1), reps=1))
        await session.commit()

        queue = await build_session_queue(
            session, 1, now, session_max_cards=3, daily_new_limit=10, boundary_hour=4
        )
        in_due_order = await due_cards(session, 1, now, limit=None)

    assert queue == ["r-oldest", "r-tied-older", "r-tied-younger"]
    assert queue == [card.id for card in in_due_order][:3]


async def test_only_the_remaining_new_budget_is_admitted_oldest_due_first(session_factory):
    now = datetime(2026, 8, 24, 10, 0, tzinfo=UTC)
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(make_note("note-1", 1))
        await session.flush()
        for i in range(6):
            session.add(make_card(f"new-{i}", "note-1", 1, due=now - timedelta(hours=i), reps=0))
        session.add(make_card("seen-today", "note-1", 1, due=now + timedelta(days=1), reps=1))
        await session.flush()
        session.add(Review(card_id="seen-today", user_id=1, rating=3, reviewed_at=now))
        await session.commit()

        # A limit of 3, one already used today -> 2 new cards left.
        queue = await build_session_queue(
            session, 1, now, session_max_cards=20, daily_new_limit=3, boundary_hour=4
        )

    assert queue == ["new-5", "new-4"]


async def test_reviews_and_new_cards_interleave_by_due_order_not_by_group(session_factory):
    # No priority for reviews: whichever is due earlier comes first, and a
    # due-second tie falls to the syllabus - here a new word card before a
    # review of the same note's form.
    now = datetime(2026, 8, 24, 10, 0, tzinfo=UTC)
    tied = now - timedelta(hours=2)
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(make_note("note-1", 1))
        await session.flush()
        session.add(make_card("new-early", "note-1", 1, due=now - timedelta(days=1), reps=0))
        session.add(
            make_inflection_card("review-form", "note-1", 1, due=tied, form="genetiivi", reps=2)
        )
        session.add(make_card("new-word", "note-1", 1, due=tied, reps=0))
        session.add(make_card("review-late", "note-1", 1, due=now - timedelta(hours=1), reps=1))
        session.add(make_card("new-late", "note-1", 1, due=now - timedelta(minutes=1), reps=0))
        await session.commit()

        queue = await build_session_queue(
            session, 1, now, session_max_cards=20, daily_new_limit=10, boundary_hour=4
        )

    assert queue == ["new-early", "new-word", "review-form", "review-late", "new-late"]


async def test_a_zero_new_budget_sends_no_query_for_new_cards(tmp_path):
    # Counted at the driver: whatever SQL really reaches SQLite. The positive
    # run proves the check can see the new-card query when it does happen.
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    await create_all(engine)
    statements: list[str] = []
    event.listen(
        engine.sync_engine,
        "before_cursor_execute",
        lambda conn, cursor, statement, *args: statements.append(statement),
    )
    now = datetime(2026, 8, 24, 10, 0, tzinfo=UTC)
    try:
        async with make_session_factory(engine)() as session:
            session.add(User(id=1))
            session.add(make_note("note-1", 1))
            await session.flush()
            session.add(make_card("new-1", "note-1", 1, due=now - timedelta(days=1), reps=0))
            session.add(make_card("review-1", "note-1", 1, due=now - timedelta(hours=1), reps=2))
            await session.commit()

            def new_card_queries() -> int:
                return sum(1 for s in statements if "cards.reps = " in s)

            statements.clear()
            no_budget = await build_session_queue(
                session, 1, now, session_max_cards=5, daily_new_limit=0, boundary_hour=4
            )
            queries_without_budget = new_card_queries()

            statements.clear()
            with_budget = await build_session_queue(
                session, 1, now, session_max_cards=5, daily_new_limit=1, boundary_hour=4
            )
            queries_with_budget = new_card_queries()
    finally:
        await engine.dispose()

    assert no_budget == ["review-1"]
    assert queries_without_budget == 0
    assert with_budget == ["new-1", "review-1"]
    assert queries_with_budget == 1


async def _old_single_window_queue(session, user_id, now, session_max_cards, daily_new_limit):
    """The queue as built before the review/new split: one shared window of
    session_max_cards * 5, with the new budget applied while walking it."""
    new_budget = max(0, daily_new_limit - await count_new_cards_today(session, user_id, now, 4))
    candidates = await due_cards(session, user_id, now, limit=session_max_cards * 5)
    queue: list[str] = []
    new_used = 0
    for card in candidates:
        if card.reps == 0:
            if new_used >= new_budget:
                continue
            new_used += 1
        queue.append(card.id)
        if len(queue) >= session_max_cards:
            break
    return queue


@pytest.mark.parametrize("session_max_cards", [2, 3, 5, 20])
@pytest.mark.parametrize("daily_new_limit", [0, 1, 3, 10])
async def test_without_starvation_the_order_matches_the_old_single_window(
    session_factory, session_max_cards, daily_new_limit
):
    # 10 cards fit the old window for every session size here (2 * 5 = 10),
    # so nothing was ever starved - and then the split must change nothing.
    now = datetime(2026, 8, 24, 10, 0, tzinfo=UTC)
    tied = now - timedelta(hours=3)
    async with session_factory() as session:
        session.add(User(id=1))
        for n, age in (("note-a", 2), ("note-b", 1)):
            note = make_note(n, 1)
            note.created_at = now - timedelta(days=age)
            session.add(note)
        await session.flush()
        rows = [
            ("b-parti", "note-b", tied, "partitiivi", 0),
            ("b-genet", "note-b", tied, "genetiivi", 2),
            ("a-illat", "note-a", tied, "illatiivi", 0),
            ("a-genet", "note-a", tied, "genetiivi", 0),
            ("a-parti", "note-a", tied, "partitiivi", 1),
            ("b-iness", "note-b", now - timedelta(days=2), "inessiivi", 0),
            ("a-iness", "note-a", now - timedelta(hours=1), "inessiivi", 3),
        ]
        for card_id, note_id, due, form, reps in rows:
            session.add(make_inflection_card(card_id, note_id, 1, due, form, reps=reps))
        session.add(make_card("a-word", "note-a", 1, due=tied, reps=1))
        session.add(make_card("b-word", "note-b", 1, due=tied, reps=0))
        session.add(make_card("c-word", "note-b", 1, due=now - timedelta(minutes=5), reps=0))
        await session.commit()

        queue = await build_session_queue(
            session,
            1,
            now,
            session_max_cards=session_max_cards,
            daily_new_limit=daily_new_limit,
            boundary_hour=4,
        )
        expected = await _old_single_window_queue(
            session, 1, now, session_max_cards, daily_new_limit
        )

    assert queue == expected


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


async def test_debt_ignores_cards_that_have_never_been_reviewed(session_factory):
    # A note now opens an inflection card per form, all due immediately. If
    # those counted as debt, the backlog prompt would fire on day one over
    # cards the user has simply not reached yet.
    now = datetime(2026, 8, 24, 10, 0, tzinfo=UTC)
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(make_note("note-1", 1))
        await session.flush()
        session.add(make_card("seen", "note-1", 1, due=now - timedelta(days=3), reps=4))
        for i in range(12):
            session.add(make_card(f"new-{i}", "note-1", 1, due=now, reps=0))
        await session.commit()

        everything_due = await overdue_count(session, 1, now)
        debt = await overdue_count(session, 1, now, reviewed_only=True)

    assert everything_due == 13  # the deck screen still counts them all
    assert debt == 1


async def test_defer_overdue_tail_leaves_never_reviewed_cards_alone(session_factory):
    now = datetime(2026, 8, 24, 10, 0, tzinfo=UTC)
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(make_note("note-1", 1))
        await session.flush()
        for i in range(3):
            session.add(
                make_card(f"seen-{i}", "note-1", 1, due=now - timedelta(days=3 - i), reps=2)
            )
        session.add(make_card("fresh", "note-1", 1, due=now, reps=0))
        await session.commit()

        postponed = await defer_overdue_tail(session, 1, now, keep_n=1, postpone_days=7)
        await session.commit()

        fresh = await session.get(Card, "fresh")

    assert postponed == 2
    assert fresh.due == now


# --- deck scoping -------------------------------------------------------------

DECK_NOW = datetime(2026, 8, 24, 10, 0, tzinfo=UTC)


async def _seed_two_decks(session_factory) -> tuple[str, str]:
    """Two decks of one user with deliberately different numbers, so any
    count that leaks across decks comes out wrong.

    deck A: 2 overdue reviews, 1 due new card, 1 future card, 1 unopened form.
    deck B: 4 overdue reviews, 2 unopened forms.
    """
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(Deck(id="deck-a", user_id=1, name="A"))
        session.add(Deck(id="deck-b", user_id=1, name="B"))
        session.add(make_note("note-a", 1, deck_id="deck-a"))
        session.add(make_note("note-b", 1, deck_id="deck-b"))
        await session.flush()
        for i in range(2):
            session.add(make_card(f"a-seen-{i}", "note-a", 1, DECK_NOW - timedelta(days=9 - i), 1))
        session.add(make_card("a-new", "note-a", 1, DECK_NOW - timedelta(hours=1)))
        session.add(make_card("a-future", "note-a", 1, DECK_NOW + timedelta(days=3), 1))
        session.add(_unopened_form("a-form-0", "note-a"))
        for i in range(4):
            session.add(make_card(f"b-seen-{i}", "note-b", 1, DECK_NOW - timedelta(days=5 - i), 1))
        for i in range(2):
            session.add(_unopened_form(f"b-form-{i}", "note-b"))
        await session.commit()
    return "deck-a", "deck-b"


def _unopened_form(card_id: str, note_id: str) -> Card:
    return Card(
        id=card_id,
        note_id=note_id,
        user_id=1,
        type=CardType.inflection,
        form="genetiivi",
        due=DECK_NOW - timedelta(days=1),
        status=CardStatus.not_introduced,
    )


async def test_overdue_count_counts_only_the_given_deck(session_factory):
    deck_a, deck_b = await _seed_two_decks(session_factory)
    async with session_factory() as session:
        due_a = await overdue_count(session, 1, DECK_NOW, deck_id=deck_a)
        debt_a = await overdue_count(session, 1, DECK_NOW, deck_id=deck_a, reviewed_only=True)
        debt_b = await overdue_count(session, 1, DECK_NOW, deck_id=deck_b, reviewed_only=True)
        debt_all = await overdue_count(session, 1, DECK_NOW, reviewed_only=True)

    assert due_a == 3
    assert debt_a == 2
    assert debt_b == 4
    assert debt_all == 6


async def test_card_counters_count_only_the_given_deck(session_factory):
    deck_a, deck_b = await _seed_two_decks(session_factory)
    async with session_factory() as session:
        counters_a = await card_counters(session, 1, DECK_NOW, deck_id=deck_a)
        counters_b = await card_counters(session, 1, DECK_NOW, deck_id=deck_b)

    assert counters_a == CardCounters(total=5, introduced=4, due=3, overdue=2, not_introduced=1)
    assert counters_b == CardCounters(total=6, introduced=4, due=4, overdue=4, not_introduced=2)


async def test_defer_overdue_tail_postpones_only_the_given_deck(session_factory):
    deck_a, deck_b = await _seed_two_decks(session_factory)
    async with session_factory() as session:
        before_a = {
            c.id: c.due for c in await session.scalars(select(Card).where(Card.note_id == "note-a"))
        }
        # Deck A's reviews are due *earlier* than all of deck B's: a defer that
        # ignored the deck would keep them and postpone B's instead.
        postponed = await defer_overdue_tail(
            session, 1, DECK_NOW, keep_n=1, postpone_days=7, deck_id=deck_b
        )
        await session.commit()

        after_a = {
            c.id: c.due for c in await session.scalars(select(Card).where(Card.note_id == "note-a"))
        }
        b_seen = {
            c.id: c.due for c in await session.scalars(select(Card).where(Card.id.like("b-seen-%")))
        }
        debt_a = await overdue_count(session, 1, DECK_NOW, deck_id=deck_a, reviewed_only=True)

    assert postponed == 3
    assert after_a == before_a
    assert debt_a == 2
    assert b_seen["b-seen-0"] == DECK_NOW - timedelta(days=5)  # the oldest one is kept
    assert all(b_seen[f"b-seen-{i}"] == DECK_NOW + timedelta(days=7) for i in (1, 2, 3))


# --- ordering contract ------------------------------------------------------


def make_inflection_card(
    card_id: str, note_id: str, user_id: int, due: datetime, form: str, reps: int = 1
) -> Card:
    return Card(
        id=card_id,
        note_id=note_id,
        user_id=user_id,
        type=CardType.inflection,
        form=form,
        due=due,
        reps=reps,
    )


async def test_cards_sharing_a_due_second_come_out_in_syllabus_order_not_insert_order(
    session_factory,
):
    # Every form the curriculum opens on one study day gets `due = now`, and
    # UTCDateTime stores whole epoch seconds - so a batch of forms shares one
    # due value exactly. `ORDER BY due` alone leaves those rows unordered,
    # and what SQLite actually returned was insert order, i.e. the key order
    # of the note's principal_forms JSON. Everything below is arranged to
    # contradict the syllabus: the note inserted first is the *younger* one,
    # and inside a note both the insert order and the (uuid4-shaped) card ids
    # run backwards against FORM_TASKS and against "word before its forms".
    now = datetime(2026, 8, 24, 10, 0, tzinfo=UTC)
    due = now - timedelta(hours=1)
    async with session_factory() as session:
        session.add(User(id=1))

        younger = make_note("note-younger", 1)
        younger.created_at = now
        session.add(younger)
        older = make_note("note-older", 1)
        older.created_at = now - timedelta(days=1)
        session.add(older)
        await session.flush()

        # Inserted younger-note-first, and inside each note the forms run
        # backwards through FORM_TASKS (illatiivi sits after partitiivi,
        # which sits after genetiivi). Ids sort the wrong way too.
        session.add(make_inflection_card("a-y-illat", "note-younger", 1, due, "illatiivi"))
        session.add(make_inflection_card("b-y-genet", "note-younger", 1, due, "genetiivi"))
        session.add(make_inflection_card("c-o-illat", "note-older", 1, due, "illatiivi"))
        session.add(make_inflection_card("d-o-parti", "note-older", 1, due, "partitiivi"))
        session.add(make_inflection_card("e-o-genet", "note-older", 1, due, "genetiivi"))
        # The word's own card, inserted last and with the largest id, still
        # belongs in front of that same note's forms.
        session.add(make_card("z-o-recog", "note-older", 1, due=due, reps=1))
        await session.commit()

        queue = await build_session_queue(
            session, 1, now, session_max_cards=50, daily_new_limit=50, boundary_hour=4
        )

    assert queue == [
        "z-o-recog",  # older note first, and its word before its forms
        "e-o-genet",
        "d-o-parti",
        "c-o-illat",
        "b-y-genet",  # then the younger note, again in FORM_TASKS order
        "a-y-illat",
    ]


async def test_the_fetch_window_cuts_a_tied_batch_at_a_fixed_point(session_factory):
    # The tie-break has to live in SQL, not in a Python sort afterwards:
    # `LIMIT` is applied by the database, so with an ambiguous ORDER BY it is
    # undefined *which* of the tied rows are fetched at all - a later sort
    # could only reorder whichever ones happened to arrive. Here 6 cards
    # share one due second and the window holds 5, so exactly one must be
    # left out, and it must be the last one in syllabus order.
    now = datetime(2026, 8, 24, 10, 0, tzinfo=UTC)
    due = now - timedelta(hours=1)
    forms = ["genetiivi", "partitiivi", "illatiivi", "inessiivi", "elatiivi", "adessiivi"]
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(make_note("note-1", 1))
        await session.flush()
        # Inserted last-form-first, so insert order is the reverse of the
        # order the learner should meet them in.
        for i, form in enumerate(reversed(forms)):
            session.add(make_inflection_card(f"card-{i}", "note-1", 1, due, form))
        await session.commit()

        # due_cards is where LIMIT meets the tie; the queue's own review
        # fetch (window = session_max_cards) is cut the same way.
        queue = await build_session_queue(
            session, 1, now, session_max_cards=1, daily_new_limit=50, boundary_hour=4
        )
        fetched = await due_cards(session, 1, now, limit=5)

    assert [card.form for card in fetched] == forms[:5]  # adessiivi is the one dropped
    assert queue == [fetched[0].id]
