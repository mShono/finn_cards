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
# learn_entry and decks_entry below, bot/add.py, bot/main.py's PICKER_BREAKERS)
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
        "/decks - колоды в читалке: слова, правка, перенос, удаление, поиск\n"
        "/add <текст> - то же самое, что просто написать текст в чат\n\n"
        "Пришли книгу файлом .txt - переведу её на финский по главам."
    )


READER_NOT_CONFIGURED_TEXT = "Читалка не настроена."
LEARN_ENTRY_TEXT = "Карточки теперь учатся в читалке:"
LEARN_MOVED_TEXT = "Учёба теперь в читалке: 📚 Учить"
DECKS_ENTRY_TEXT = "Колоды теперь в читалке:"
DELETE_ENTRY_TEXT = "Слова теперь удаляются в читалке: найди слово поиском и нажми 🗑."
DECKS_MOVED_TEXT = "Колоды теперь в читалке: 🗂 Колоды"


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


# Decks live in the Mini App too (plan 2026-11-20-kirjakaveri-reader, П6/К3):
# the menu button and /decks only hand out the button that opens them.
@router.message(Command("decks"))
@router.message(F.text == "🗂 Колоды")
async def decks_entry(message: Message, settings: Settings) -> None:
    # State left as is, unlike learn_entry: an /add deck pick survives a look
    # at the decks (bot/main.py's PICKER_BREAKERS) - its buttons still work
    # after. A pending input (a new deck's name) is already dropped by
    # InputEscapeMiddleware, with UnsavedWordsMiddleware's warning first.
    if not settings.reader_url:
        await message.answer(READER_NOT_CONFIGURED_TEXT)
        return
    url = with_query_param(settings.reader_url, "screen", "decks")
    await message.answer(DECKS_ENTRY_TEXT, reply_markup=_web_app_markup("🗂 Колоды", url))


# Out of the commands menu, but a typed /delete (with a word or without)
# still gets an answer: the reader's search, where 🗑 is. Here, in the core
# router, so it wins over bot/add.py's delete_command.
@router.message(Command("delete"))
async def delete_entry(message: Message, settings: Settings) -> None:
    if not settings.reader_url:
        await message.answer(READER_NOT_CONFIGURED_TEXT)
        return
    url = with_query_param(settings.reader_url, "screen", "search")
    await message.answer(DELETE_ENTRY_TEXT, reply_markup=_web_app_markup("🔎 Найти слово", url))


# The old deck screens' buttons (bot/decks.py, bot/edit.py, /delete's list)
# in the chat history - answered like old_learn_button. Not the /add picker's
# adddeck:/addnewdeck:, which bot/add.py still handles.
@router.callback_query(F.data.startswith("decks:"))
@router.callback_query(F.data.startswith("noteedit"))
@router.callback_query(F.data.startswith("delnote:"))
async def old_decks_button(callback: CallbackQuery) -> None:
    await callback.answer(DECKS_MOVED_TEXT)
