from __future__ import annotations

from datetime import UTC, datetime

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import KeyboardButton, Message, ReplyKeyboardMarkup
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kielikaveri.bot.text import split_message
from kielikaveri.db.models import Card, Note

router = Router(name="core")

# Persistent menu (plan: no more hunting for /start or remembering command
# names) - sent once on /start and stays until Telegram clears it. Button
# labels double as message text the other routers match on directly (see
# bot/learn.py, bot/decks.py, bot/add.py) - keep them in sync if these change.
MAIN_KEYBOARD = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="📚 Учить"), KeyboardButton(text="💬 Добавить")],
        [KeyboardButton(text="🗂 Колоды"), KeyboardButton(text="📊 Статистика")],
    ],
    resize_keyboard=True,
)


@router.message(Command("start"))
async def start(message: Message) -> None:
    await message.answer(
        "Привет! Kielikaveri на связи - бот для практики финского.\n"
        "Кнопки внизу - учить, добавлять слова, колоды, статистика. "
        "Добавлять можно и просто текстом: напиши мне слово, текст на "
        "финском или свой перевод - отвечу в чате.",
        reply_markup=MAIN_KEYBOARD,
    )


@router.message(Command("help"))
async def help_(message: Message) -> None:
    await message.answer(
        "/start - поздороваться\n"
        "/help - эта справка\n"
        "/stats - сколько заметок и карточек к повторению\n"
        "/learn - повторить карточки, которым пора\n"
        "/decks - список колод, создать новую, переключить активную\n"
        "/add <текст> - то же самое, что просто написать текст в чат"
    )


@router.message(Command("stats"))
@router.message(F.text == "📊 Статистика")
async def stats(message: Message, session_factory: async_sessionmaker[AsyncSession]) -> None:
    user_id = message.from_user.id
    async with session_factory() as session:
        notes_count = await session.scalar(
            select(func.count()).select_from(Note).where(Note.user_id == user_id)
        )
        due_cards_count = await session.scalar(
            select(func.count())
            .select_from(Card)
            .where(Card.user_id == user_id, Card.due <= datetime.now(UTC))
        )

    text = f"Заметок: {notes_count}\nКарточек к повторению: {due_cards_count}"
    for chunk in split_message(text):
        await message.answer(chunk)
