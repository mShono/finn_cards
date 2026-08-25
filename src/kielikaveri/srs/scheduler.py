"""Thin wrapper over py-fsrs - no DB, no network.

Converts between our Card ORM's flat SRS columns and fsrs.Card/fsrs.Scheduler,
and back. Nothing here talks to the database or Telegram; callers own that.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import fsrs

from kielikaveri.db.models import CardState

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


def review(current: SrsState, rating: Rating, now: datetime) -> SrsState:
    """Apply one review to `current`, returning the resulting SRS state.

    `reps`/`lapses` aren't fields py-fsrs tracks on its own Card - we count
    them ourselves: reps increments on every review, lapses increments only
    when a card that was in State.review (i.e. already learned) gets
    demoted out of it - the FSRS definition of "forgetting" a card.
    """
    was_review = current.state == CardState.review

    fsrs_card = fsrs.Card(
        state=_STATE_TO_FSRS[current.state],
        step=current.step,
        stability=current.stability,
        difficulty=current.difficulty,
        due=current.due,
        last_review=None,
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
