"""The /decks command: create decks, pick which one new notes go to, and
drill into one to see and edit its cards.

Decks are manual and user-named on purpose (plan: chose this over an
automatic new-vs-mature split - the learner decides what's grouped with
what). This module manages the deck list, the "active" one /add saves into,
and the per-deck card list; editing a card itself lives in bot/edit.py -
this module only links to it via the noteedit: callback. /learn's own deck
picker lives in bot/learn.py.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from math import ceil

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kielikaveri.db.decks import active_deck, create_deck, list_decks, set_active_deck
from kielikaveri.db.models import Deck, Note
from kielikaveri.srs.queue import card_counters

logger = logging.getLogger(__name__)

router = Router(name="decks")

NEW_DECK_PROMPT = "Как назвать новую колоду?"

# One row per card, so a deck screen is paged - keeps the keyboard under
# Telegram's button-count limit and the message under its length limit.
NOTES_PER_PAGE = 40
MAX_BUTTON_TRANSLATION = 40


class DeckStates(StatesGroup):
    naming = State()


def _parse_open(data: str) -> tuple[str, int]:
    # "decks:open:<deck_id>" or "decks:open:<deck_id>:<page>"
    parts = data.split(":")
    page = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0
    return parts[2], page


def _pager_row(deck_id: str, page: int, pages: int) -> list[list[InlineKeyboardButton]]:
    if pages < 2:
        return []
    row = []
    if page > 0:
        row.append(
            InlineKeyboardButton(text="⬅️ назад", callback_data=f"decks:open:{deck_id}:{page - 1}")
        )
    if page < pages - 1:
        row.append(
            InlineKeyboardButton(text="вперёд ➡️", callback_data=f"decks:open:{deck_id}:{page + 1}")
        )
    return [row]


def _note_button_text(index: int, note: Note) -> str:
    # Lemma plus translation, so an edit starts from what you read
    translation = " ".join((note.translation_ru or "").split())
    if len(translation) > MAX_BUTTON_TRANSLATION:
        translation = translation[: MAX_BUTTON_TRANSLATION - 1].rstrip() + "…"
    label = f"✍️ {index}. {note.lemma}"
    return f"{label} - {translation}" if translation else label


def _decks_keyboard(decks: list[Deck], current_id: str) -> InlineKeyboardMarkup:
    rows = []
    for deck in decks:
        row = [InlineKeyboardButton(text="📂 " + deck.name, callback_data=f"decks:open:{deck.id}")]
        if deck.id != current_id:
            row.append(
                InlineKeyboardButton(text="✅ выбрать", callback_data=f"decks:activate:{deck.id}")
            )
        rows.append(row)
    rows.append([InlineKeyboardButton(text="➕ Новая колода", callback_data="decks:new")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def _decks_list_text_and_keyboard(
    session_factory: async_sessionmaker[AsyncSession], user_id: int, now: datetime
) -> tuple[str, InlineKeyboardMarkup]:
    async with session_factory() as session:
        # active_deck() first - it may create a default deck, and list_decks()
        # must see it (a fresh user with zero decks would otherwise get an
        # empty "Колод пока нет." list while the trailing line below still
        # names a deck that isn't shown anywhere).
        current = await active_deck(session, user_id)
        await session.commit()
        decks = await list_decks(session, user_id)

        lines = []
        for deck in decks:
            notes_count = await session.scalar(
                select(func.count()).select_from(Note).where(Note.deck_id == deck.id)
            )
            counters = await card_counters(session, user_id, now, deck_id=deck.id)
            marker = "📌 " if deck.id == current.id else "• "
            lines.append(f"{marker}{deck.name} - слов: {notes_count}, к повторению: {counters.due}")

    text = "Твои колоды:\n" + "\n".join(lines) if lines else "Колод пока нет."
    text += f"\n\nСейчас новое сохраняется в «{current.name}» - нажми 📂, чтобы открыть колоду."
    return text, _decks_keyboard(decks, current.id)


@router.message(Command("decks"))
@router.message(F.text == "🗂 Колоды")
async def decks_list(message: Message, session_factory: async_sessionmaker[AsyncSession]) -> None:
    logger.debug("event=decks.list")
    text, keyboard = await _decks_list_text_and_keyboard(
        session_factory, message.from_user.id, datetime.now(UTC)
    )
    await message.answer(text, reply_markup=keyboard)


@router.callback_query(F.data == "decks:list")
async def decks_back(
    callback: CallbackQuery, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    text, keyboard = await _decks_list_text_and_keyboard(
        session_factory, callback.from_user.id, datetime.now(UTC)
    )
    await callback.message.answer(text, reply_markup=keyboard)
    await callback.answer()


@router.callback_query(F.data.startswith("decks:open:"))
async def decks_open(
    callback: CallbackQuery, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    deck_id, page = _parse_open(callback.data)
    user_id = callback.from_user.id
    now = datetime.now(UTC)
    logger.debug("event=decks.open deck_id=%s page=%s", deck_id, page)

    async with session_factory() as session:
        deck = await session.get(Deck, deck_id)
        if deck is None or deck.user_id != user_id:
            logger.debug("event=decks.not_found deck_id=%s", deck_id)
            await callback.answer("Не нашла колоду.", show_alert=True)
            return

        total = (
            await session.scalar(
                select(func.count()).select_from(Note).where(Note.deck_id == deck_id)
            )
            or 0
        )
        pages = max(1, ceil(total / NOTES_PER_PAGE))
        page = min(page, pages - 1)
        notes = list(
            (
                await session.scalars(
                    # Newest first: a freshly added word is the one you most
                    # often come back to fix.
                    select(Note)
                    .where(Note.deck_id == deck_id)
                    .order_by(Note.created_at.desc())
                    .offset(page * NOTES_PER_PAGE)
                    .limit(NOTES_PER_PAGE)
                )
            ).all()
        )
        counters = await card_counters(session, user_id, now, deck_id=deck_id)

    lines = [
        f"📂 «{deck.name}» - слов: {total}, к повторению: {counters.due}",
        f"формы: изучается {counters.introduced}, ещё не открыто {counters.not_introduced}",
    ]
    if not notes:
        lines.append("\nКарточек пока нет.")
    elif pages > 1:
        lines.append(f"страница {page + 1} из {pages}")

    # The words themselves live on the buttons, not in the message text
    rows = [
        [
            InlineKeyboardButton(
                text=_note_button_text(page * NOTES_PER_PAGE + i, note),
                callback_data=f"noteedit:{note.id}",
            )
        ]
        for i, note in enumerate(notes, start=1)
    ]
    rows.extend(_pager_row(deck_id, page, pages))
    rows.append([InlineKeyboardButton(text="⬅️ Колоды", callback_data="decks:list")])

    await callback.message.answer(
        "\n".join(lines), reply_markup=InlineKeyboardMarkup(inline_keyboard=rows)
    )
    await callback.answer()


@router.callback_query(F.data.startswith("decks:activate:"))
async def decks_activate(
    callback: CallbackQuery, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    deck_id = callback.data.split(":", 2)[2]
    user_id = callback.from_user.id
    async with session_factory() as session:
        await set_active_deck(session, user_id, deck_id)
        await session.commit()
    logger.info("event=decks.activate deck_id=%s", deck_id)
    await callback.answer("Готово - новое пойдёт сюда.")


@router.callback_query(F.data == "decks:new")
async def decks_new_prompt(callback: CallbackQuery, state: FSMContext) -> None:
    logger.debug("event=decks.new_prompt")
    await state.set_state(DeckStates.naming)
    await callback.message.answer(NEW_DECK_PROMPT)
    await callback.answer()


@router.message(DeckStates.naming)
async def decks_new_save(
    message: Message, state: FSMContext, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    name = (message.text or "").strip()
    if not name:
        await message.answer(NEW_DECK_PROMPT)
        return

    user_id = message.from_user.id
    async with session_factory() as session:
        deck = await create_deck(session, user_id, name)
        await set_active_deck(session, user_id, deck.id)
        await session.commit()

    logger.info("event=decks.create deck_id=%s name=%r", deck.id, deck.name)
    await state.clear()
    await message.answer(f"Колода «{deck.name}» создана и стала активной - новое пойдёт туда.")
