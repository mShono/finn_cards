from datetime import UTC, datetime, timedelta

import fsrs
import pytest

from kielikaveri.db.models import CardState
from kielikaveri.srs import scheduler as srs_scheduler
from kielikaveri.srs.scheduler import Rating, SrsState, review

NOW = datetime(2026, 8, 24, 10, 0, tzinfo=UTC)


def new_card() -> SrsState:
    return SrsState(
        state=CardState.learning, due=NOW, stability=None, difficulty=None, reps=0, lapses=0
    )


def test_first_review_initializes_stability_and_bumps_reps():
    result = review(new_card(), Rating.Good, NOW)

    assert result.reps == 1
    assert result.lapses == 0
    assert result.stability is not None and result.stability > 0
    assert result.due > NOW


def test_reps_increments_on_every_review_regardless_of_rating():
    state = new_card()
    for rating in (Rating.Again, Rating.Hard, Rating.Good, Rating.Easy):
        state = review(state, rating, state.due)
    assert state.reps == 4


def test_two_good_reviews_graduate_a_card_to_review_state():
    state = review(new_card(), Rating.Good, NOW)
    state = review(state, Rating.Good, state.due)
    assert state.state == CardState.review


def test_forgetting_a_learned_card_counts_as_a_lapse():
    state = review(new_card(), Rating.Good, NOW)
    state = review(state, Rating.Good, state.due)
    assert state.state == CardState.review
    lapses_before = state.lapses

    lapsed = review(state, Rating.Again, state.due)

    assert lapsed.state != CardState.review
    assert lapsed.lapses == lapses_before + 1


def test_reviewing_a_review_state_card_with_good_does_not_add_a_lapse():
    state = review(new_card(), Rating.Good, NOW)
    state = review(state, Rating.Good, state.due)
    assert state.state == CardState.review

    kept = review(state, Rating.Good, state.due)

    assert kept.lapses == state.lapses


def test_a_lapse_before_ever_reaching_review_state_is_not_counted():
    # Rating a still-learning card Again is expected friction, not "forgetting
    # something already learned" - only a demotion out of State.review counts.
    lapsed = review(new_card(), Rating.Again, NOW)
    assert lapsed.lapses == 0


def test_step_round_trips_so_a_reload_does_not_restart_the_learning_phase():
    # Simulates the bot restarting between two reviews of the same card:
    # SrsState is rebuilt from plain values (as loaded from the DB), not
    # kept as the same Python object.
    state = review(new_card(), Rating.Good, NOW)
    reloaded = SrsState(
        state=state.state,
        due=state.due,
        stability=state.stability,
        difficulty=state.difficulty,
        reps=state.reps,
        lapses=state.lapses,
        step=state.step,
    )

    continued = review(reloaded, Rating.Good, state.due)

    assert continued.state == CardState.review
    assert continued.reps == 2


# --- the wrapper against py-fsrs itself ---------------------------------------
# The columns above (reps, lapses, state) all moved correctly even while the
# wrapper handed FSRS `last_review=None`, which inflated stability by orders
# of magnitude from the second answer on (two "Good" in one sitting bought
# 143 days instead of 2). Only a comparison against the library's own
# scheduling catches that, so these tests run the real py-fsrs - never a mock.

ON_TIME = timedelta(0)
LATE = timedelta(days=3)

# (rating, how late the answer is relative to when the card came due)
SEQUENCES = {
    "two_good_in_one_sitting": [(Rating.Good, ON_TIME), (Rating.Good, ON_TIME)],
    "three_good": [(Rating.Good, ON_TIME)] * 3,
    "good_until_forgotten_then_relearned": [
        (Rating.Good, ON_TIME),
        (Rating.Good, ON_TIME),
        (Rating.Good, ON_TIME),
        (Rating.Again, ON_TIME),
        (Rating.Good, ON_TIME),
    ],
    "failing_the_learning_steps_twice": [
        (Rating.Again, ON_TIME),
        (Rating.Again, ON_TIME),
        (Rating.Good, ON_TIME),
    ],
    "easy_graduates_straight_to_review": [(Rating.Easy, ON_TIME), (Rating.Good, ON_TIME)],
    "answered_days_late_every_time": [
        (Rating.Good, ON_TIME),
        (Rating.Good, LATE),
        (Rating.Good, LATE),
        (Rating.Again, LATE),
        (Rating.Good, LATE),
    ],
}


@pytest.fixture
def reference_scheduler(monkeypatch) -> fsrs.Scheduler:
    """A py-fsrs scheduler configured exactly like the wrapper's own, with
    fuzzing off on both sides - FSRS randomizes Review-state intervals by
    design, and two independently fuzzed runs can't be compared.
    """
    monkeypatch.setattr(srs_scheduler._scheduler, "enable_fuzzing", False)
    return fsrs.Scheduler(enable_fuzzing=False)


@pytest.mark.parametrize("sequence", SEQUENCES.values(), ids=SEQUENCES.keys())
def test_wrapper_schedules_a_card_exactly_like_py_fsrs(reference_scheduler, sequence):
    """Both sides answer the same card at the same moments; every column has
    to agree at every step, not just at the end - a divergence in stability
    shows up one answer before it reaches `due`.
    """
    reference = fsrs.Card(due=NOW)
    ours = new_card()
    last_review = None

    for step, (rating, lateness) in enumerate(sequence, start=1):
        # Answered when the card actually came up, plus whatever lateness the
        # sequence asks for. Both sides are driven by the reference's `due`
        # so a wrong interval on our side can't also move our own clock.
        now = NOW if step == 1 else reference.due + lateness

        reference, _ = reference_scheduler.review_card(reference, rating, now)
        ours = review(ours, rating, now, last_review)
        last_review = now

        where = f"after step {step} ({rating.name})"
        # CardState's values are py-fsrs's State names, lowercased.
        assert ours.state.value == reference.state.name.lower(), where
        assert ours.step == reference.step, where
        assert ours.stability == pytest.approx(reference.stability), where
        assert ours.difficulty == pytest.approx(reference.difficulty), where
        assert abs(ours.due - reference.due) <= timedelta(seconds=1), where


def test_a_cards_first_review_has_no_previous_answer_to_schedule_from():
    """`last_review=None` is right exactly once - FSRS reads it as "never
    answered" and picks the initial stability itself. Passing it on every
    later review is the bug these comparisons guard.
    """
    first = review(new_card(), Rating.Good, NOW, last_review=None)
    second = review(first, Rating.Good, first.due, last_review=NOW)
    forgotten_instead = review(first, Rating.Good, first.due, last_review=None)

    assert second.stability < forgotten_instead.stability
