from datetime import UTC, datetime

from kielikaveri.db.models import CardState
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
