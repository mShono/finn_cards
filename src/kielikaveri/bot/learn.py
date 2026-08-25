"""The /learn command: FSRS-scheduled review sessions.

Flow: show front -> "Показать ответ" reveals back + rating buttons -> rating
applies the review via srs.scheduler and advances to the next card, until the
session hits its card/time limit or the queue runs dry. No LLM, no network -
this must keep working when OpenAI is unreachable (see plan 3.10).
"""

from __future__ import annotations

import random
from datetime import UTC, datetime

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kielikaveri.config import Settings
from kielikaveri.db.models import Card, CardType, Note, Review
from kielikaveri.srs.graduation import ensure_card_types, sync_user_card_types
from kielikaveri.srs.queue import build_session_queue, defer_overdue_tail, overdue_count
from kielikaveri.srs.scheduler import RATING_LABELS, Rating, SrsState
from kielikaveri.srs.scheduler import review as apply_review

router = Router(name="learn")


class LearnStates(StatesGroup):
    debt_choice = State()
    reviewing = State()


def render_card(card: Card, note: Note) -> tuple[str, str]:
    """Return (front, back) text for one card, by its type."""
    if card.type == CardType.recognition:
        return f"🇫🇮 {note.lemma}", f"{note.translation_ru}\n\n{note.example_fi}\n{note.example_ru}"
    if card.type == CardType.production:
        return f"🇷🇺 {note.translation_ru}", f"{note.lemma}\n\n{note.example_fi}"
    if card.type == CardType.inflection:
        forms: dict = note.meta.get("principal_forms") or {}
        if forms:
            form_name, form_value = random.choice(list(forms.items()))
            return f"{note.lemma} → {form_name}?", form_value
        return note.lemma, note.lemma
    return note.lemma, note.translation_ru


def _reveal_keyboard(card_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Показать ответ", callback_data=f"learn:reveal:{card_id}")]
        ]
    )


def _rating_keyboard(card_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=label, callback_data=f"learn:rate:{card_id}:{rating.value}"
                )
                for rating, label in RATING_LABELS.items()
            ]
        ]
    )


def _debt_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Заниматься как обычно", callback_data="learn:debt:batch"
                ),
                InlineKeyboardButton(text="Отложить остальное", callback_data="learn:debt:defer"),
            ]
        ]
    )


async def _start_session(
    answer_to: Message,
    state: FSMContext,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    user_id: int,
    now: datetime,
) -> None:
    async with session_factory() as session:
        queue = await build_session_queue(
            session,
            user_id,
            now,
            session_max_cards=settings.session_max_cards,
            daily_new_limit=settings.daily_new_limit,
            boundary_hour=settings.day_boundary_hour,
        )

    if not queue:
        await state.clear()
        await answer_to.answer("Нечего повторять - все карточки выучены на сегодня.")
        return

    await state.set_state(LearnStates.reviewing)
    await state.update_data(
        queue=queue,
        session_started_at=now.isoformat(),
        reviewed_count=0,
        session_max_cards=settings.session_max_cards,
        session_max_minutes=settings.session_max_minutes,
    )
    await _show_next_card(answer_to, state, session_factory)


async def _show_next_card(
    answer_to: Message, state: FSMContext, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    data = await state.get_data()
    queue: list[str] = data["queue"]
    started_at = datetime.fromisoformat(data["session_started_at"])
    reviewed_count: int = data["reviewed_count"]
    elapsed_minutes = (datetime.now(UTC) - started_at).total_seconds() / 60

    if (
        not queue
        or reviewed_count >= data["session_max_cards"]
        or elapsed_minutes >= data["session_max_minutes"]
    ):
        remaining = len(queue)
        await state.clear()
        text = f"Сессия окончена: {reviewed_count} карточек пройдено"
        text += (
            f", осталось {remaining} - /learn чтобы продолжить."
            if remaining
            else " - всё на сегодня!"
        )
        await answer_to.answer(text)
        return

    card_id = queue[0]
    async with session_factory() as session:
        card = await session.get(Card, card_id)
        note = await session.get(Note, card.note_id)
        front, _back = render_card(card, note)

    await answer_to.answer(front, reply_markup=_reveal_keyboard(card_id))


@router.message(Command("learn"))
async def learn_start(
    message: Message,
    state: FSMContext,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
) -> None:
    user_id = message.from_user.id
    now = datetime.now(UTC)

    async with session_factory() as session:
        await sync_user_card_types(session, user_id, now)
        await session.commit()
        overdue = await overdue_count(session, user_id, now)

    if overdue > settings.debt_threshold:
        await state.set_state(LearnStates.debt_choice)
        await state.update_data(debt_now=now.isoformat())
        await message.answer(
            f"Просрочено {overdue} карточек - это много за одну сессию.\n"
            f"Разгребать как обычно (по {settings.session_max_cards} за раз) "
            f"или отложить остальное на {settings.debt_postpone_days} дн.?",
            reply_markup=_debt_keyboard(),
        )
        return

    await _start_session(message, state, session_factory, settings, user_id, now)


@router.callback_query(F.data.startswith("learn:debt:"), LearnStates.debt_choice)
async def learn_debt_choice(
    callback: CallbackQuery,
    state: FSMContext,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
) -> None:
    action = callback.data.split(":")[2]
    user_id = callback.from_user.id
    data = await state.get_data()
    now = datetime.fromisoformat(data["debt_now"])

    if action == "defer":
        async with session_factory() as session:
            postponed = await defer_overdue_tail(
                session,
                user_id,
                now,
                keep_n=settings.session_max_cards,
                postpone_days=settings.debt_postpone_days,
            )
            await session.commit()
        await callback.message.answer(
            f"Отложено {postponed} карточек на {settings.debt_postpone_days} дн."
        )

    await callback.answer()
    await _start_session(callback.message, state, session_factory, settings, user_id, now)


@router.callback_query(F.data.startswith("learn:reveal:"), LearnStates.reviewing)
async def learn_reveal(
    callback: CallbackQuery,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    card_id = callback.data.split(":", 2)[2]
    async with session_factory() as session:
        card = await session.get(Card, card_id)
        note = await session.get(Note, card.note_id)
        _front, back = render_card(card, note)

    await callback.message.answer(back, reply_markup=_rating_keyboard(card_id))
    await callback.answer()


@router.callback_query(F.data.startswith("learn:rate:"), LearnStates.reviewing)
async def learn_rate(
    callback: CallbackQuery,
    state: FSMContext,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    _, _, card_id, rating_value = callback.data.split(":")

    data = await state.get_data()
    queue: list[str] = data.get("queue", [])
    if not queue or queue[0] != card_id:
        # A stale button - e.g. a duplicate tap on a message whose card was
        # already rated and is no longer at the head of the queue. Telegram
        # doesn't disable a button once it's used, so the old keyboard stays
        # live; applying it again would silently record a phantom review the
        # user never actually made and corrupt that card's FSRS history.
        await callback.answer("Эта карточка уже учтена.", show_alert=True)
        return

    # Claim the head *before* the first real await (the DB writes below) -
    # two callback_query updates for one genuine double-tap can otherwise
    # both pass the check above and race: each opens its own session, reads
    # the same not-yet-updated card, and both commit a review for it (proven
    # by test_concurrent_double_tap_..., which without this line records two
    # Review rows and crashes the second call with a KeyError from a queue
    # already cleared out from under it). Popping now makes the second call
    # see an empty/mismatched head and take the stale-button exit above.
    await state.update_data(queue=queue[1:], reviewed_count=data.get("reviewed_count", 0) + 1)

    rating = Rating(int(rating_value))
    now = datetime.now(UTC)

    async with session_factory() as session:
        card = await session.get(Card, card_id)
        current = SrsState(
            state=card.state,
            due=card.due,
            stability=card.stability,
            difficulty=card.difficulty,
            reps=card.reps,
            lapses=card.lapses,
            step=card.step,
        )
        updated = apply_review(current, rating, now)
        card.state = updated.state
        card.due = updated.due
        card.stability = updated.stability
        card.difficulty = updated.difficulty
        card.reps = updated.reps
        card.lapses = updated.lapses
        card.step = updated.step

        session.add(
            Review(card_id=card.id, user_id=card.user_id, rating=rating.value, reviewed_at=now)
        )

        note = await session.get(Note, card.note_id)
        await ensure_card_types(session, note, now)
        await session.commit()

    await callback.answer(f"Записано: {RATING_LABELS[rating]}")
    await _show_next_card(callback.message, state, session_factory)


@router.callback_query(F.data.startswith("learn:"))
async def learn_stray_callback(callback: CallbackQuery) -> None:
    # Reaches here only when reveal/rate/debt fired outside their expected
    # state - e.g. a button from a session already ended by the time-limit.
    await callback.answer(
        "Эта сессия уже неактуальна - начните заново через /learn", show_alert=True
    )
