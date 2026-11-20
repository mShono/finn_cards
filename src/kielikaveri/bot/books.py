"""A book sent as a .txt document (plan 2026-11-20-kirjakaveri-reader, "П1").

No upload button: the bot notices a .txt outside the free-text input states,
stores it split into chapters and asks whether to translate it. "Да" queues
chapter 1 for the background translator (translate/worker.py); later
chapters are queued by the reader as the previous one is opened. A failed
chapter comes back with a retry button.

The book's text never goes to the log - ids, counts and the hash only.
"""

from __future__ import annotations

import logging
from pathlib import Path

from aiogram import Bot, F, Router
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kielikaveri.config import Settings
from kielikaveri.db.books import (
    chapter_by_number,
    delete_book,
    find_book,
    queue_chapter,
    store_book,
    text_hash,
)
from kielikaveri.db.models import Book, BookChapter, ChapterStatus
from kielikaveri.translate.chapters import UndecodableError, decode_text, split_book
from kielikaveri.translate.worker import ChapterEvent, Translator

logger = logging.getLogger(__name__)

router = Router(name="books")

NOT_TXT_TEXT = "Книги принимаю только в .txt. Другие файлы пока не читаю."
BUSY_TEXT = "Сейчас жду от тебя ответ на прошлый вопрос. Ответь или нажми /start, потом пришли книгу ещё раз."
UNDECODABLE_TEXT = "Не смогла прочитать файл как текст. Пришли книгу в .txt (UTF-8 или cp1251)."
TOO_SHORT_TEXT = "В файле почти нет текста - это точно книга?"
# Shorter than this is a note, not a book - and not worth a glossary call.
MIN_BOOK_CHARS = 1000


def _confirm_keyboard(book_id: str) -> InlineKeyboardMarkup:
    # "book:yes:" + a uuid4 = 45 bytes, under Telegram's 64-byte callback_data cap.
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ Да, перевести", callback_data=f"book:yes:{book_id}"),
                InlineKeyboardButton(text="❌ Нет", callback_data=f"book:no:{book_id}"),
            ]
        ]
    )


def _retry_keyboard(chapter_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🔁 Повторить", callback_data=f"book:retry:{chapter_id}")]
        ]
    )


def _title(file_name: str | None) -> str:
    stem = Path(file_name or "").stem.replace("_", " ").strip()
    return stem or "Книга"


def _is_txt(message: Message) -> bool:
    document = message.document
    name = (document.file_name or "").lower()
    return name.endswith(".txt") or document.mime_type == "text/plain"


def failure_reason(error: str | None) -> str:
    """worker._error_text puts the exception class first - what the user can do differs."""
    kind = (error or "").split(":", 1)[0]
    if kind == "RefusedError":
        return "модель отказалась переводить этот фрагмент"
    if kind == "MisalignedError":
        return "в переводе потерялись абзацы"
    if kind == "CircuitOpenError":
        return "слишком много обращений к OpenAI подряд, сработал предохранитель"
    return "ошибка при переводе"


async def send_chapter_event(bot: Bot, event: ChapterEvent) -> None:
    """Translator's notify callback - the one place it reaches Telegram."""
    if event.status is ChapterStatus.done:
        await bot.send_message(
            event.user_id, f"📖 «{event.book_title}»: глава {event.number} переведена."
        )
        return
    await bot.send_message(
        event.user_id,
        f"Не получилось перевести главу {event.number} книги «{event.book_title}»: "
        f"{failure_reason(event.error)}.",
        reply_markup=_retry_keyboard(event.chapter_id),
    )


# Registered before add.py's catch-all; a document has no text, so the two
# never compete - the ordering only matters for readability of main.py.
@router.message(F.document)
async def book_upload(
    message: Message,
    bot: Bot,
    raw_state: str | None,
    input_states: set[str],
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
) -> None:
    if not _is_txt(message):
        await message.answer(NOT_TXT_TEXT)
        return
    # A pending deck name or card value must not be silently abandoned by a
    # file - the user decided books are only noticed outside input modes.
    if raw_state in input_states:
        await message.answer(BUSY_TEXT)
        return
    document = message.document
    if document.file_size and document.file_size > settings.max_book_bytes:
        await message.answer(
            f"Файл слишком большой: больше {settings.max_book_bytes // 1_000_000} МБ."
        )
        return

    downloaded = await bot.download(document)
    try:
        text = decode_text(downloaded.read())
    except UndecodableError:
        await message.answer(UNDECODABLE_TEXT)
        return
    book = split_book(text)
    chars = sum(c.chars for c in book.chapters)
    if chars < MIN_BOOK_CHARS:
        await message.answer(TOO_SHORT_TEXT)
        return

    user_id = message.from_user.id
    source_hash = text_hash(text)
    async with session_factory() as session:
        existing = await find_book(session, user_id, source_hash)
        if existing is not None:
            first = await chapter_by_number(session, existing.id, 1)
            logger.info("event=book.duplicate book_id=%s", existing.id)
            if first is not None and first.status is ChapterStatus.pending:
                await message.answer(
                    f"Эта книга уже загружена как «{existing.title}». Перевести?",
                    reply_markup=_confirm_keyboard(existing.id),
                )
            else:
                await message.answer(f"Эта книга уже загружена как «{existing.title}».")
            return
        row = await store_book(session, user_id, _title(document.file_name), text, book)
        await session.commit()

    logger.info(
        "event=book.stored book_id=%s chapters=%d chars=%d lang=%s hash=%s",
        row.id,
        len(book.chapters),
        chars,
        row.source_lang,
        source_hash[:12],
    )
    await message.answer(
        f"Книга «{row.title}»: глав - {len(book.chapters)}, "
        f"около {max(1, round(chars / 1000))} тыс. знаков.\n"
        "Перевести на финский? Сначала переведу главу 1 - это несколько минут, "
        "следующие главы - по мере чтения.",
        reply_markup=_confirm_keyboard(row.id),
    )


async def _owned_book(session: AsyncSession, book_id: str, user_id: int) -> Book | None:
    book = await session.get(Book, book_id)
    return book if book is not None and book.user_id == user_id else None


@router.callback_query(F.data.startswith("book:yes:"))
async def book_confirm(
    callback: CallbackQuery,
    session_factory: async_sessionmaker[AsyncSession],
    translator: Translator,
) -> None:
    book_id = callback.data.removeprefix("book:yes:")
    async with session_factory() as session:
        book = await _owned_book(session, book_id, callback.from_user.id)
        first = await chapter_by_number(session, book_id, 1) if book else None
        if first is None:
            await callback.answer("Этой книги уже нет.", show_alert=True)
            return
        # Conditional: a double tap queues chapter 1 once.
        queued = await queue_chapter(session, first.id)
        await session.commit()
    await callback.answer()
    if not queued:
        await callback.message.answer("Эта книга уже переводится.")
        return
    logger.info("event=book.confirmed book_id=%s", book_id)
    translator.wake()
    await callback.message.answer(
        f"Перевожу главу 1 книги «{book.title}». Напишу, когда будет готово."
    )


@router.callback_query(F.data.startswith("book:no:"))
async def book_decline(
    callback: CallbackQuery, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    book_id = callback.data.removeprefix("book:no:")
    async with session_factory() as session:
        book = await _owned_book(session, book_id, callback.from_user.id)
        if book is None:
            await callback.answer("Этой книги уже нет.", show_alert=True)
            return
        started = await session.scalar(
            select(func.count())
            .select_from(BookChapter)
            .where(BookChapter.book_id == book_id, BookChapter.status != ChapterStatus.pending)
        )
        if started:
            # Already confirmed (an old message's button) - its chapters
            # may be translated and paid for; not deleted by a stray tap.
            await callback.answer("Эта книга уже переводится.", show_alert=True)
            return
        await delete_book(session, book_id)
        await session.commit()
    logger.info("event=book.declined book_id=%s", book_id)
    await callback.answer()
    await callback.message.answer("Хорошо, не перевожу. Книгу удалила.")


@router.callback_query(F.data.startswith("book:retry:"))
async def chapter_retry(
    callback: CallbackQuery,
    session_factory: async_sessionmaker[AsyncSession],
    translator: Translator,
) -> None:
    chapter_id = callback.data.removeprefix("book:retry:")
    async with session_factory() as session:
        chapter = await session.get(BookChapter, chapter_id)
        book = (
            await _owned_book(session, chapter.book_id, callback.from_user.id) if chapter else None
        )
        if book is None:
            await callback.answer("Этой книги уже нет.", show_alert=True)
            return
        # Only a failed chapter: queue_chapter also takes pending ones.
        queued = chapter.status is ChapterStatus.failed and await queue_chapter(session, chapter_id)
        await session.commit()
    await callback.answer()
    if not queued:
        await callback.message.answer("Эта глава уже переводится или готова.")
        return
    logger.info("event=book.retry book_id=%s chapter=%d", book.id, chapter.number)
    translator.wake()
    await callback.message.answer(f"Перевожу главу {chapter.number} ещё раз.")
