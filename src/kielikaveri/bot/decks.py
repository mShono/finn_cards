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

# A deck screen with more cards than this only shows the first page - keeps
# the message under Telegram's length limit and the keyboard under its
# button-count limit (one row per card).
MAX_NOTES_SHOWN = 40


class DeckStates(StatesGroup):
    naming = State()


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
            due = await overdue_count(session, user_id, now, deck_id=deck.id)
            marker = "📌 " if deck.id == current.id else "• "
            lines.append(f"{marker}{deck.name} - слов: {notes_count}, к повторению: {due}")

    text = "Твои колоды:\n" + "\n".join(lines) if lines else "Колод пока нет."
    text += f"\n\nСейчас новое сохраняется в «{current.name}» - нажми 📂, чтобы открыть колоду."
    return text, _decks_keyboard(decks, current.id)


@router.message(Command("decks"))
@router.message(F.text == "🗂 Колоды")
async def decks_list(message: Message, session_factory: async_sessionmaker[AsyncSession]) -> None:
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
    deck_id = callback.data.split(":", 2)[2]
    user_id = callback.from_user.id
    now = datetime.now(UTC)

    async with session_factory() as session:
        deck = await session.get(Deck, deck_id)
        if deck is None or deck.user_id != user_id:
            await callback.answer("Не нашла колоду.", show_alert=True)
            return

        notes = list(
            (
                await session.scalars(
                    select(Note)
                    .where(Note.deck_id == deck_id)
                    .order_by(Note.created_at)
                    .limit(MAX_NOTES_SHOWN)
                )
            ).all()
        )
        total = await session.scalar(
            select(func.count()).select_from(Note).where(Note.deck_id == deck_id)
        )
        due = await overdue_count(session, user_id, now, deck_id=deck_id)

    lines = [f"📂 «{deck.name}» - слов: {total}, к повторению: {due}"]
    if not notes:
        lines.append("\nКарточек пока нет.")
    else:
        for i, note in enumerate(notes, start=1):
            pos_suffix = f" ({note.pos})" if note.pos else ""
            lines.append(f"{i}. {note.lemma}{pos_suffix} - {note.translation_ru}")
        if total > len(notes):
            lines.append(f"\n… показаны первые {len(notes)} из {total}.")

    rows = [
        [InlineKeyboardButton(text=f"✍️ {i}. {note.lemma}", callback_data=f"noteedit:{note.id}")]
        for i, note in enumerate(notes, start=1)
    ]
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
