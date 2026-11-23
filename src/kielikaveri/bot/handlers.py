from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
    WebAppInfo,
)

from kielikaveri.config import Settings

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
async def start(message: Message, state: FSMContext) -> None:
    # A fresh start - whatever the bot was waiting for is dropped
    await state.clear()
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
        "/read - открыть читалку книг\n"
        "/decks - список колод, создать новую, посмотреть и поправить карточки\n"
        "/add <текст> - то же самое, что просто написать текст в чат\n"
        "/delete <слово> - удалить слово из колоды\n\n"
        "Пришли книгу файлом .txt - переведу её на финский по главам."
    )


READER_NOT_CONFIGURED_TEXT = "Читалка не настроена."


@router.message(Command("read"))
async def read(message: Message, settings: Settings) -> None:
    if not settings.reader_url:
        await message.answer(READER_NOT_CONFIGURED_TEXT)
        return
    # An inline button, not a reply-keyboard one: a Mini App opened from a
    # keyboard button gets no signed initData (sendData only), and
    # reader/auth.py needs it. Inline and menu buttons both pass it.
    await message.answer(
        "Открываю читалку:",
        reply_markup=InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="📖 Открыть читалку", web_app=WebAppInfo(url=settings.reader_url)
                    )
                ]
            ]
        ),
    )
