"""The /decks command: create decks and pick which one new notes go to.

Decks are manual and user-named on purpose (plan: chose this over an
automatic new-vs-mature split - the learner decides what's grouped with
what). This module only manages the deck list and the "active" one /add
saves into; /learn's own deck picker lives in bot/learn.py.
"""

from __future__ import annotations

from datetime import UTC, datetime

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kielikaveri.db.decks import active_deck, create_deck, list_decks, set_active_deck
from kielikaveri.db.models import Deck, Note
from kielikaveri.srs.queue import overdue_count

router = Router(name="decks")

NEW_DECK_PROMPT = "Как назвать новую колоду?"


class DeckStates(StatesGroup):
    naming = State()


def _decks_keyboard(decks: list[Deck], current_id: str) -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton(
                text=("✅ " if deck.id == current_id else "") + deck.name,
                callback_data=f"decks:activate:{deck.id}",
            )
        ]
        for deck in decks
        if deck.id != current_id
    ]
    rows.append([InlineKeyboardButton(text="➕ Новая колода", callback_data="decks:new")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


@router.message(Command("decks"))
@router.message(F.text == "🗂 Колоды")
async def decks_list(message: Message, session_factory: async_sessionmaker[AsyncSession]) -> None:
    user_id = message.from_user.id
    now = datetime.now(UTC)

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
            due = await overdue_count(session, user_id, now, deck_id=deck.id)
            marker = "📌 " if deck.id == current.id else "• "
            lines.append(f"{marker}{deck.name} - слов: {notes_count}, к повторению: {due}")

    text = "Твои колоды:\n" + "\n".join(lines) if lines else "Колод пока нет."
    text += f"\n\nСейчас новое сохраняется в «{current.name}» - нажми другую, чтобы переключить."
    await message.answer(text, reply_markup=_decks_keyboard(decks, current.id))


@router.callback_query(F.data.startswith("decks:activate:"))
async def decks_activate(
    callback: CallbackQuery, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    deck_id = callback.data.split(":", 2)[2]
    user_id = callback.from_user.id
    async with session_factory() as session:
        await set_active_deck(session, user_id, deck_id)
        await session.commit()
    await callback.answer("Готово - новое пойдёт сюда.")


@router.callback_query(F.data == "decks:new")
async def decks_new_prompt(callback: CallbackQuery, state: FSMContext) -> None:
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

    await state.clear()
    await message.answer(f"Колода «{deck.name}» создана и стала активной - новое пойдёт туда.")
