from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    CallbackQuery,
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
# learn_entry below, bot/decks.py, bot/add.py, bot/main.py's PICKER_BREAKERS)
# - keep them in sync if these change.
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
        "/read - открыть читалку книг\n"
        "/decks - список колод, создать новую, посмотреть и поправить карточки\n"
        "/add <текст> - то же самое, что просто написать текст в чат\n"
        "/delete <слово> - удалить слово из колоды\n\n"
        "Пришли книгу файлом .txt - переведу её на финский по главам."
    )


READER_NOT_CONFIGURED_TEXT = "Читалка не настроена."
LEARN_ENTRY_TEXT = "Карточки теперь учатся в читалке:"
LEARN_MOVED_TEXT = "Учёба теперь в читалке: 📚 Учить"


def _web_app_markup(text: str, url: str) -> InlineKeyboardMarkup:
    # An inline button, not a reply-keyboard one: a Mini App opened from a
    # keyboard button gets no signed initData (sendData only), and
    # reader/auth.py needs it. Inline and menu buttons both pass it.
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text=text, web_app=WebAppInfo(url=url))]]
    )


def with_query_param(url: str, name: str, value: str) -> str:
    """`url` with `name=value` in its query - added to whatever query it
    already has (replacing an old `name`), the fragment kept."""
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k != name]
    query.append((name, value))
    return urlunsplit(parts._replace(query=urlencode(query)))


@router.message(Command("read"))
async def read(message: Message, settings: Settings) -> None:
    if not settings.reader_url:
        await message.answer(READER_NOT_CONFIGURED_TEXT)
        return
    await message.answer(
        "Открываю читалку:", reply_markup=_web_app_markup("📖 Открыть читалку", settings.reader_url)
    )


# Learning lives in the Mini App (plan 2026-11-20-kirjakaveri-reader, П5):
# the menu button and a typed /learn only hand out the button that opens it.
# The bot's own review sessions are gone (У4).
@router.message(Command("learn"))
@router.message(F.text == "📚 Учить")
async def learn_entry(message: Message, state: FSMContext, settings: Settings) -> None:
    # Whatever the bot waited for is dropped, as the bot's own /learn used
    # to - the deck picker's words included (PICKER_BREAKERS warns about
    # those first).
    await state.clear()
    if not settings.reader_url:
        await message.answer(READER_NOT_CONFIGURED_TEXT)
        return
    url = with_query_param(settings.reader_url, "screen", "learn")
    await message.answer(LEARN_ENTRY_TEXT, reply_markup=_web_app_markup("📚 Учить", url))


# Buttons of old bot sessions still sit in the chat history - answered, so
# they don't spin, with no new message.
@router.callback_query(F.data.startswith("learn:"))
async def old_learn_button(callback: CallbackQuery) -> None:
    await callback.answer(LEARN_MOVED_TEXT)
