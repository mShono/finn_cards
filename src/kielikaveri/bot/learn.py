"""The /learn command: FSRS-scheduled review sessions.

Flow: pick a deck (when there is more than one), pick which side to be shown
-> show front -> "Показать ответ" reveals back + rating buttons -> rating
applies the review via srs.scheduler and advances to the next card, until the
session hits its time limit or the queue - already capped at session_max_cards
by build_session_queue - runs dry. No LLM, no network -
this must keep working when OpenAI is unreachable (see plan 3.10).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from enum import StrEnum

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from openai import APIError, APIStatusError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kielikaveri.config import Settings
from kielikaveri.db.decks import list_decks
from kielikaveri.db.models import Card, CardType, Deck, Note, Review
from kielikaveri.grammar import FORM_TASKS
from kielikaveri.srs.curriculum import introduce_due_forms
from kielikaveri.srs.graduation import ensure_card_types, sync_user_card_types
from kielikaveri.srs.queue import build_session_queue, defer_overdue_tail, overdue_count
from kielikaveri.srs.scheduler import RATING_LABELS, Rating, apply_review
from kielikaveri.tts import make_tts_client, synthesize_speech

logger = logging.getLogger(__name__)

router = Router(name="learn")

DECK_ALL_TOKEN = "all"
DELETED_CARD_TEXT = "Это слово удалено."
SIDE_PROMPT = "Что показывать на карточке?"


class LearnSide(StrEnum):
    """Which side of the cards a session shows.

    A side is a filter, not a flip: recognition and production are separate
    cards with FSRS schedules of their own, so showing a recognition card
    Russian-first would record a production answer into recognition's
    history. Inflection cards ask a form of a Finnish lemma, so they go with
    Finnish.
    """

    fi = "fi"
    ru = "ru"
    mix = "mix"

    @property
    def card_types(self) -> tuple[CardType, ...] | None:
        """The card types this side shows; None - every type, unfiltered."""
        if self is LearnSide.fi:
            return (CardType.recognition, CardType.inflection)
        if self is LearnSide.ru:
            return (CardType.production,)
        return None

    @property
    def shows_forms(self) -> bool:
        return self.card_types is None or CardType.inflection in self.card_types


class LearnStates(StatesGroup):
    deck_choice = State()
    side_choice = State()
    # Between a side or debt tap and the session (or debt prompt) it leads
    # to - the mark those handlers claim so a second tap starts nothing.
    starting = State()
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
        # The card names its own form (one card per form), so the question
        # is stable across reviews and its FSRS interval means something.
        # A form with no FORM_TASKS entry is never asked - the front must
        # not carry a bare key like "nut_partisiippi".
        task = FORM_TASKS.get(card.form)
        if task is not None and card.form in forms:
            return f"{note.lemma} → {task.cue}", f"{forms[card.form]}\n\n✅ {task.label}"
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
            ],
            [InlineKeyboardButton(text="🔊 Послушать", callback_data=f"learn:listen:{card_id}")],
        ]
    )


def _deck_choice_keyboard(decks: list[Deck]) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(text=deck.name, callback_data=f"learn:deck:{deck.id}")]
        for deck in decks
    ]
    rows.append([InlineKeyboardButton(text="Все", callback_data=f"learn:deck:{DECK_ALL_TOKEN}")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _side_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🇫🇮 Финский", callback_data=f"learn:side:{LearnSide.fi}"),
                InlineKeyboardButton(text="🇷🇺 Русский", callback_data=f"learn:side:{LearnSide.ru}"),
            ],
            [
                InlineKeyboardButton(
                    text="🔀 Вперемешку", callback_data=f"learn:side:{LearnSide.mix}"
                )
            ],
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
    deck_id: str | None,
    side: LearnSide,
) -> None:
    async with session_factory() as session:
        queue = await build_session_queue(
            session,
            user_id,
            now,
            session_max_cards=settings.session_max_cards,
            daily_new_limit=settings.daily_new_limit,
            boundary_hour=settings.day_boundary_hour,
            deck_id=deck_id,
            card_types=side.card_types,
        )
        no_production = (
            side is LearnSide.ru
            and not queue
            and not await _has_production(session, user_id, deck_id)
        )

    if not queue:
        logger.info("event=learn.session_empty deck_id=%s side=%s", deck_id, side)
        await state.clear()
        await answer_to.answer(_empty_session_text(side, no_production))
        return

    logger.info("event=learn.session_start deck_id=%s side=%s queue=%d", deck_id, side, len(queue))
    await state.set_state(LearnStates.reviewing)
    await state.update_data(
        queue=queue,
        session_started_at=now.isoformat(),
        reviewed_count=0,
        session_max_minutes=settings.session_max_minutes,
    )
    await _show_next_card(answer_to, state, session_factory)


async def _has_production(session: AsyncSession, user_id: int, deck_id: str | None) -> bool:
    stmt = select(Card.id).where(Card.user_id == user_id, Card.type == CardType.production)
    if deck_id is not None:
        stmt = stmt.join(Note, Card.note_id == Note.id).where(Note.deck_id == deck_id)
    return await session.scalar(stmt.limit(1)) is not None


def _empty_session_text(side: LearnSide, no_production: bool) -> str:
    if side is LearnSide.mix:
        return "Нечего повторять - все карточки выучены на сегодня."
    # Only this side is done - the other may still have cards waiting.
    text = "На этой стороне сейчас нечего повторять."
    if no_production:
        # Production cards open only once the Finnish side is known well
        # (graduation.py) - until the first one does, Russian is empty.
        text += "\nРусская сторона открывается для слова, когда его финская уже хорошо запомнилась."
    return text


async def _drop_deleted_head(session: AsyncSession, queue: list[str]) -> list[str]:
    """The queue without deleted cards at its head. A word deleted
    mid-session (/delete, 🗑 in a deck) leaves its card ids in the FSM queue -
    delete_confirm has no state filter and knows nothing about /learn."""
    while queue and await session.get(Card, queue[0]) is None:
        logger.info("event=learn.skip_deleted card_id=%s", queue[0])
        queue = queue[1:]
    return queue


async def _skip_deleted_card(
    callback: CallbackQuery,
    state: FSMContext,
    session_factory: async_sessionmaker[AsyncSession],
    card_id: str,
) -> None:
    await callback.answer(DELETED_CARD_TEXT)
    # Only a tap on the current card moves the session on - re-showing the
    # head from an old message's button would duplicate the card on screen.
    queue: list[str] = (await state.get_data()).get("queue", [])
    if queue and queue[0] == card_id:
        await _show_next_card(callback.message, state, session_factory)


async def _show_next_card(
    answer_to: Message, state: FSMContext, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    data = await state.get_data()
    async with session_factory() as session:
        queue = await _drop_deleted_head(session, data["queue"])
    if queue != data["queue"]:
        await state.update_data(queue=queue)
    started_at = datetime.fromisoformat(data["session_started_at"])
    reviewed_count: int = data["reviewed_count"]
    elapsed_minutes = (datetime.now(UTC) - started_at).total_seconds() / 60

    # No card-count check here on purpose: build_session_queue already caps the
    # queue at session_max_cards and learn_rate pops exactly one card per
    # review, so reviewed_count can never reach the cap while cards remain -
    # the queue is the single place that limit is applied.
    if not queue or elapsed_minutes >= data["session_max_minutes"]:
        remaining = len(queue)
        reason = "queue_empty" if not queue else "max_minutes"
        logger.info(
            "event=learn.session_end reason=%s reviewed=%d remaining=%d",
            reason,
            reviewed_count,
            remaining,
        )
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


async def _claim_state(state: FSMContext, expected: State, claimed: State) -> bool:
    """Move from `expected` to `claimed`; False if another update already left `expected`."""
    if await state.get_state() != expected.state:
        return False
    await state.set_state(claimed)
    return True


@asynccontextmanager
async def _clear_on_error(state: FSMContext) -> AsyncIterator[None]:
    """Drop the claimed state if starting the session fails.

    Left in `starting`, every learn button would be answered silently and
    any text would get the mid-session hint, with no session to go with
    it. Cleared, a stale button says to start over via /learn.
    """
    try:
        yield
    except Exception:
        await state.clear()
        raise


async def _ask_side(answer_to: Message, state: FSMContext, deck_id: str | None) -> None:
    await state.set_state(LearnStates.side_choice)
    await state.update_data(deck_id=deck_id)
    await answer_to.answer(SIDE_PROMPT, reply_markup=_side_keyboard())


async def _proceed_past_side_choice(
    answer_to: Message,
    state: FSMContext,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    user_id: int,
    now: datetime,
    deck_id: str | None,
    side: LearnSide,
) -> None:
    async with session_factory() as session:
        await sync_user_card_types(session, user_id, now)
        # Cards first, then the curriculum decides which of the new forms
        # the learner actually meets today - see srs/curriculum.py. Not for
        # a session that shows no forms: opening them there would spend the
        # day's form budget on cards nobody sees.
        if side.shows_forms:
            await introduce_due_forms(
                session,
                user_id,
                now,
                daily_new_forms=settings.daily_new_forms,
                boundary_hour=settings.day_boundary_hour,
                deck_id=deck_id,
            )
        await session.commit()
        overdue = await overdue_count(
            session,
            user_id,
            now,
            deck_id=deck_id,
            reviewed_only=True,
            card_types=side.card_types,
        )

    if overdue > settings.debt_threshold:
        logger.info(
            "event=learn.debt_prompt deck_id=%s side=%s overdue=%d threshold=%d",
            deck_id,
            side,
            overdue,
            settings.debt_threshold,
        )
        await state.set_state(LearnStates.debt_choice)
        await state.update_data(debt_now=now.isoformat(), deck_id=deck_id, side=side)
        await answer_to.answer(
            f"Просрочено {overdue} карточек - это много за одну сессию.\n"
            f"Разгребать как обычно (по {settings.session_max_cards} за раз) "
            f"или отложить остальное на {settings.debt_postpone_days} дн.?",
            reply_markup=_debt_keyboard(),
        )
        return

    await _start_session(answer_to, state, session_factory, settings, user_id, now, deck_id, side)


@router.message(Command("learn"))
@router.message(F.text == "📚 Учить")
async def learn_start(
    message: Message,
    state: FSMContext,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
) -> None:
    async with session_factory() as session:
        decks = await list_decks(session, message.from_user.id)

    logger.debug("event=learn.start decks=%d", len(decks))
    if len(decks) <= 1:
        # No deck to choose between - straight to the side question.
        await _ask_side(message, state, deck_id=None)
        return

    await state.set_state(LearnStates.deck_choice)
    await message.answer("Какую колоду учить?", reply_markup=_deck_choice_keyboard(decks))


@router.callback_query(F.data.startswith("learn:deck:"), LearnStates.deck_choice)
async def learn_deck_choice(callback: CallbackQuery, state: FSMContext) -> None:
    raw_deck_id = callback.data.split(":", 2)[2]
    deck_id = None if raw_deck_id == DECK_ALL_TOKEN else raw_deck_id

    logger.debug("event=learn.deck_choice deck_id=%s", deck_id)
    # Claimed before the first await that yields, so a double tap asks the
    # side only once - see learn_side_choice.
    if not await _claim_state(state, LearnStates.deck_choice, LearnStates.side_choice):
        await callback.answer()
        return
    await state.update_data(deck_id=deck_id)
    await callback.answer()
    await callback.message.answer(SIDE_PROMPT, reply_markup=_side_keyboard())


@router.callback_query(F.data.startswith("learn:side:"), LearnStates.side_choice)
async def learn_side_choice(
    callback: CallbackQuery,
    state: FSMContext,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
) -> None:
    raw_side = callback.data.split(":", 2)[2]
    if raw_side not in LearnSide:
        await callback.answer()
        return
    side = LearnSide(raw_side)
    # Claim the start before anything that yields - the same reason as
    # learn_rate's early pop. Updates run as concurrent tasks, and the card
    # sync and curriculum below take a while: a second tap would start a
    # second session, both syncs racing to create the same cards. The state
    # filter alone doesn't stop it - aiogram reads the state before running
    # the filters, which yield, so both taps pass it. Hence the re-check
    # here; with MemoryStorage neither call yields, so check-and-claim is
    # atomic.
    if not await _claim_state(state, LearnStates.side_choice, LearnStates.starting):
        await learn_tap_while_starting(callback)
        return
    deck_id = (await state.get_data()).get("deck_id")

    logger.debug("event=learn.side_choice deck_id=%s side=%s", deck_id, side)
    async with _clear_on_error(state):
        await callback.answer()
        await _proceed_past_side_choice(
            callback.message,
            state,
            session_factory,
            settings,
            callback.from_user.id,
            datetime.now(UTC),
            deck_id,
            side,
        )


@router.callback_query(F.data.startswith("learn:debt:"), LearnStates.debt_choice)
async def learn_debt_choice(
    callback: CallbackQuery,
    state: FSMContext,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
) -> None:
    # Two taps - say "Отложить" and "как обычно" - would otherwise defer
    # the tail and start two sessions; see learn_side_choice.
    if not await _claim_state(state, LearnStates.debt_choice, LearnStates.starting):
        await learn_tap_while_starting(callback)
        return
    action = callback.data.split(":")[2]
    user_id = callback.from_user.id
    data = await state.get_data()
    now = datetime.fromisoformat(data["debt_now"])
    deck_id = data.get("deck_id")
    side = LearnSide(data["side"])

    async with _clear_on_error(state):
        if action == "defer":
            async with session_factory() as session:
                postponed = await defer_overdue_tail(
                    session,
                    user_id,
                    now,
                    keep_n=settings.session_max_cards,
                    postpone_days=settings.debt_postpone_days,
                    deck_id=deck_id,
                    card_types=side.card_types,
                )
                await session.commit()
            logger.info("event=learn.debt_choice action=defer postponed=%d", postponed)
            await callback.message.answer(
                f"Отложено {postponed} карточек на {settings.debt_postpone_days} дн."
            )
        else:
            logger.debug("event=learn.debt_choice action=%s", action)

        await callback.answer()
        await _start_session(
            callback.message, state, session_factory, settings, user_id, now, deck_id, side
        )


@router.callback_query(F.data.startswith("learn:reveal:"), LearnStates.reviewing)
async def learn_reveal(
    callback: CallbackQuery,
    session_factory: async_sessionmaker[AsyncSession],
    state: FSMContext,
) -> None:
    card_id = callback.data.split(":", 2)[2]
    logger.debug("event=learn.reveal card_id=%s", card_id)
    async with session_factory() as session:
        card = await session.get(Card, card_id)
        if card is not None:
            note = await session.get(Note, card.note_id)
            _front, back = render_card(card, note)

    if card is None:
        await _skip_deleted_card(callback, state, session_factory, card_id)
        return
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
        logger.debug("event=learn.rate_stale card_id=%s", card_id)
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
        if card is None:
            # Deleted mid-session. The head is already popped above - undo
            # only the count claimed with it, nothing was reviewed.
            await state.update_data(reviewed_count=data.get("reviewed_count", 0))
            await callback.answer(DELETED_CARD_TEXT)
            await _show_next_card(callback.message, state, session_factory)
            return
        # When this card was last answered - FSRS schedules off the gap since
        # then. Read *before* the new row below is added to the session:
        # adding it first would let autoflush include this very answer in the
        # max(), making last_review == now, and a zero gap reads as a perfect
        # recall. None here means "never answered", which is what FSRS wants
        # for a card's first review.
        last_review = await session.scalar(
            select(func.max(Review.reviewed_at)).where(Review.card_id == card.id)
        )
        # Moves every SRS column on the card; the `reviews` row below is ours.
        apply_review(card, rating, now, last_review)

        session.add(
            Review(card_id=card.id, user_id=card.user_id, rating=rating.value, reviewed_at=now)
        )

        note = await session.get(Note, card.note_id)
        await ensure_card_types(session, note, now)
        await session.commit()

    logger.info("event=db.save entity=review card_id=%s rating=%d", card_id, rating.value)
    await callback.answer(f"Записано: {RATING_LABELS[rating]}")
    await _show_next_card(callback.message, state, session_factory)


@router.callback_query(F.data.startswith("learn:listen:"), LearnStates.reviewing)
async def learn_listen(
    callback: CallbackQuery,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    state: FSMContext,
) -> None:
    if not settings.openai_api_key:
        await callback.answer("Озвучка недоступна - не настроен OpenAI.", show_alert=True)
        return

    card_id = callback.data.split(":", 2)[2]
    async with session_factory() as session:
        card = await session.get(Card, card_id)
        note = await session.get(Note, card.note_id) if card is not None else None

    if card is None:
        await _skip_deleted_card(callback, state, session_factory, card_id)
        return
    logger.debug("event=learn.listen card_id=%s", card_id)
    # Answer the tap before calling OpenAI: Telegram only accepts an answer
    # for a short while, and a slow TTS must not leave the button spinning.
    # Anything that goes wrong after this is reported as a plain message.
    await callback.answer()
    try:
        async with make_tts_client(
            settings.openai_api_key, settings.openai_tts_timeout_seconds
        ) as client:
            audio = await synthesize_speech(
                client, settings.openai_tts_model, note.example_fi, speed=settings.openai_tts_speed
            )
    except APIError as error:
        logger.exception("event=learn.listen_error card_id=%s", card_id)
        await callback.message.reply(_tts_error_text(error))
        return
    # A reply, not a plain message: the user may rate the card while TTS is
    # still running, and the audio must not look like the next card's.
    await callback.message.reply_audio(BufferedInputFile(audio, filename="example.mp3"))


def _tts_error_text(error: APIError) -> str:
    # A 4xx other than 429 is a bad key or model name - retrying never helps.
    if (
        isinstance(error, APIStatusError)
        and error.status_code != 429
        and 400 <= error.status_code < 500
    ):
        return "Озвучка настроена неверно - подробности в логах."
    return "Не получилось озвучить - попробуй ещё раз чуть позже."


@router.callback_query(F.data.startswith("learn:"), LearnStates.starting)
async def learn_tap_while_starting(callback: CallbackQuery) -> None:
    # A second tap on the side or debt buttons while the first one's
    # session is still being built - that session is coming, nothing is
    # stale.
    logger.debug("event=learn.tap_while_starting")
    await callback.answer()


@router.callback_query(F.data.startswith("learn:"))
async def learn_stray_callback(callback: CallbackQuery) -> None:
    # Reaches here only when deck/side/reveal/rate/debt fired outside their expected
    # state - e.g. a button from a session already ended by the time-limit.
    logger.debug("event=learn.stray_callback")
    await callback.answer(
        "Эта сессия уже неактуальна - начните заново через /learn", show_alert=True
    )
