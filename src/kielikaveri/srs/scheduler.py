"""Thin wrapper over py-fsrs - no DB, no network.

Converts between our Card ORM's flat SRS columns and fsrs.Card/fsrs.Scheduler,
and back. Nothing here talks to the database or Telegram; callers own that.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import fsrs

from kielikaveri.db.models import Card, CardState

Rating = fsrs.Rating

RATING_LABELS: dict[Rating, str] = {
    Rating.Again: "Забыл",
    Rating.Hard: "Трудно",
    Rating.Good: "Хорошо",
    Rating.Easy: "Легко",
}

_STATE_TO_FSRS = {
    CardState.learning: fsrs.State.Learning,
    CardState.review: fsrs.State.Review,
    CardState.relearning: fsrs.State.Relearning,
}
_FSRS_TO_STATE = {value: key for key, value in _STATE_TO_FSRS.items()}

_scheduler = fsrs.Scheduler()


@dataclass
class SrsState:
    """Mirrors Card's SRS columns - kept separate from the ORM so this module
    is testable without a database."""

    state: CardState
    due: datetime
    stability: float | None
    difficulty: float | None
    reps: int
    lapses: int
    step: int | None = None


def review(
    current: SrsState, rating: Rating, now: datetime, last_review: datetime | None = None
) -> SrsState:
    """Apply one review to `current`, returning the resulting SRS state.

    `reps`/`lapses` aren't fields py-fsrs tracks on its own Card - we count
    them ourselves: reps increments on every review, lapses increments only
    when a card that was in State.review (i.e. already learned) gets
    demoted out of it - the FSRS definition of "forgetting" a card.

    `last_review` is when this card was answered the time before - the
    caller's `reviews` row for it, None if there is none yet. FSRS needs it
    to know how much was forgotten since: without it, retrievability reads
    as 0 ("fully forgotten, yet recalled"), which inflates stability by
    orders of magnitude from the second answer on - two "Good" in one
    sitting bought 143 days instead of 2. We don't store it on the card
    itself: `reviews` already holds every answer, so a column here would be
    a second copy of the same fact.
    """
    was_review = current.state == CardState.review

    fsrs_card = fsrs.Card(
        state=_STATE_TO_FSRS[current.state],
        step=current.step,
        stability=current.stability,
        difficulty=current.difficulty,
        due=current.due,
        last_review=last_review,
    )
    updated, _log = _scheduler.review_card(fsrs_card, rating, now)

    new_state = _FSRS_TO_STATE[updated.state]
    lapses = current.lapses + (1 if was_review and new_state != CardState.review else 0)

    return SrsState(
        state=new_state,
        due=updated.due,
        stability=updated.stability,
        difficulty=updated.difficulty,
        reps=current.reps + 1,
        lapses=lapses,
        step=updated.step,
    )


def apply_review(
    card: Card, rating: Rating, now: datetime, last_review: datetime | None = None
) -> None:
    """Apply one review to `card`'s SRS columns, in place.

    The Card <-> SrsState mapping lives here, not at the call sites, because
    there were two hand-written copies of it (the bot's rating handler and the
    tests' `answer` helper) and every column has to appear in both halves:
    dropping `step` on the write-back alone silently restarts a card's
    learning phase on the next review.

    Still no DB access - this only assigns attributes on an ORM instance.
    Reading `last_review` out of `reviews`, writing the new row there and
    committing all stay with the caller, which is the half that holds a
    session (see bot/learn.py's rating handler).
    """
    updated = review(
        SrsState(
            state=card.state,
            due=card.due,
            stability=card.stability,
            difficulty=card.difficulty,
            reps=card.reps,
            lapses=card.lapses,
            step=card.step,
        ),
        rating,
        now,
        last_review,
    )
    card.state = updated.state
    card.due = updated.due
    card.stability = updated.stability
    card.difficulty = updated.difficulty
    card.reps = updated.reps
    card.lapses = updated.lapses
    card.step = updated.step
