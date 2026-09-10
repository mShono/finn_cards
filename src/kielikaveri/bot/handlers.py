from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import KeyboardButton, Message, ReplyKeyboardMarkup

router = Router(name="core")

# Persistent menu (plan: no more hunting for /start or remembering command
# names) - sent once on /start and stays until Telegram clears it. Button
# labels double as message text the other routers match on directly (see
# bot/learn.py, bot/decks.py, bot/add.py) - keep them in sync if these change.
MAIN_KEYBOARD = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="📚 Учить"), KeyboardButton(text="💬 Добавить")],
        [KeyboardButton(text="🗂 Колоды")],
    ],
    resize_keyboard=True,
)


@router.message(Command("start"))
async def start(message: Message) -> None:
    await message.answer(
        "Привет! Kielikaveri на связи - бот для практики финского.\n"
        "Кнопки внизу - учить, добавлять слова, колоды. "
        "Добавлять можно и просто текстом: напиши мне слово, текст на "
        "финском или свой перевод - отвечу в чате.",
        reply_markup=MAIN_KEYBOARD,
    )


@router.message(Command("help"))
async def help_(message: Message) -> None:
    await message.answer(
        "/start - поздороваться\n"
        "/help - эта справка\n"
        "/learn - повторить карточки, которым пора\n"
        "/decks - список колод, создать новую, переключить активную\n"
        "/add <текст> - то же самое, что просто написать текст в чат\n"
        "/delete <слово> - удалить слово из колоды"
    )
