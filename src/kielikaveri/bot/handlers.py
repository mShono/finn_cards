from __future__ import annotations

from datetime import UTC, datetime

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kielikaveri.bot.text import split_message
from kielikaveri.db.models import Card, Note

router = Router(name="core")


@router.message(Command("start"))
async def start(message: Message) -> None:
    await message.answer(
        "Привет! Kielikaveri на связи - бот для практики финского.\n"
        "Команды: /help, /stats."
    )


@router.message(Command("help"))
async def help_(message: Message) -> None:
    await message.answer(
        "/start - поздороваться\n"
        "/help - эта справка\n"
        "/stats - сколько заметок и карточек к повторению"
    )


@router.message(Command("stats"))
async def stats(
    message: Message, session_factory: async_sessionmaker[AsyncSession]
) -> None:
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
