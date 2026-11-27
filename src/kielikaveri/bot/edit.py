"""Editing an existing note's word or translation (drill-down from /decks).

Only two fields are editable: lemma and translation_ru. Changing the lemma
re-resolves principal_forms through the FST the same way /add does
(kielikaveri.ingest.resolve_note_forms) - a stale form set tied to the old
lemma would silently teach a wrong Finnish form, which cards/instructions.md
treats as a hard rule ("Формы - только через FST"), not a nice-to-have.

Field codes in callback_data are "lm"/"tr", not the full field name - Telegram
caps callback_data at 64 bytes and a note id is already a 36-char uuid.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import openai
from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kielikaveri.config import Settings
from kielikaveri.db.models import Note
from kielikaveri.decks_core import (
    DuplicateNoteError,
    find_lemma_clash,
    needs_forms,
    save_lemma,
    set_translation,
    settle_lemma,
)
from kielikaveri.ingest import resolve_note_forms
from kielikaveri.llm.breaker import CallBreaker, CircuitOpenError
from kielikaveri.llm.client import make_client

logger = logging.getLogger(__name__)

router = Router(name="edit")

FIELD_CODES = {"lm": "lemma", "tr": "translation_ru"}
FIELD_LABELS = {"lemma": "слово", "translation_ru": "перевод"}
CANCEL_WORDS = {"отмена", "cancel", "/cancel"}


class EditStates(StatesGroup):
    # Data: note_id, field ("lemma" | "translation_ru").
    awaiting_value = State()


def _field_keyboard(note_id: str) -> InlineKeyboardMarkup:
    # 🗑 reuses delnote: (bot/add.py's delete_confirm) rather than a new
    # handler here - same note-scoped delete-and-cleanup-cards-and-reviews
    # logic /delete already uses, no reason for a second copy of it.
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✍️ Слово", callback_data=f"noteeditfield:{note_id}:lm"),
                InlineKeyboardButton(text="✍️ Перевод", callback_data=f"noteeditfield:{note_id}:tr"),
            ],
            [InlineKeyboardButton(text="🗑 Удалить", callback_data=f"delnote:{note_id}")],
            [InlineKeyboardButton(text="Отмена", callback_data="noteeditcancel")],
        ]
    )


@router.callback_query(F.data.startswith("noteedit:"))
async def note_edit_menu(
    callback: CallbackQuery, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    note_id = callback.data.split(":", 1)[1]
    async with session_factory() as session:
        note = await session.get(Note, note_id)
    if note is None or note.user_id != callback.from_user.id:
        logger.debug("event=edit.not_found note_id=%s", note_id)
        await callback.answer("Не нашла эту карточку.", show_alert=True)
        return

    logger.debug("event=edit.menu note_id=%s lemma=%s", note_id, note.lemma)
    pos_suffix = f" ({note.pos})" if note.pos else ""
    await callback.message.answer(
        f"«{note.lemma}{pos_suffix} - {note.translation_ru}» - что править?",
        reply_markup=_field_keyboard(note.id),
    )
    await callback.answer()


@router.callback_query(F.data == "noteeditcancel", EditStates.awaiting_value)
async def note_edit_cancel(callback: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    await callback.answer("Отменено.")


@router.callback_query(F.data == "noteeditcancel")
async def note_edit_cancel_stray(callback: CallbackQuery) -> None:
    # No edit in progress - the state belongs to someone else (an /add deck
    # pick, a clarifying question) and an old menu's Отмена must not wipe it.
    await callback.answer("Отменено.")


@router.callback_query(F.data.startswith("noteeditfield:"))
async def note_edit_field_choice(
    callback: CallbackQuery,
    state: FSMContext,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    _, note_id, field_code = callback.data.split(":", 2)
    field = FIELD_CODES[field_code]

    async with session_factory() as session:
        note = await session.get(Note, note_id)
    if note is None or note.user_id != callback.from_user.id:
        await callback.answer("Не нашла эту карточку.", show_alert=True)
        return

    current = note.lemma if field == "lemma" else note.translation_ru
    logger.debug("event=edit.field_choice note_id=%s field=%s", note_id, field)
    await state.set_state(EditStates.awaiting_value)
    await state.update_data(note_id=note_id, field=field)
    await callback.message.answer(
        f"Текущее {FIELD_LABELS[field]}: «{current}». Пришли новое значение (или «отмена»)."
    )
    await callback.answer()


@router.message(EditStates.awaiting_value)
async def note_edit_apply(
    message: Message,
    state: FSMContext,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    breaker: CallBreaker,
) -> None:
    value = (message.text or "").strip()
    if not value:
        await message.answer("Пустое значение не сохраню - пришли текст ещё раз.")
        return
    if value.lower() in CANCEL_WORDS:
        logger.debug("event=edit.cancelled")
        await state.clear()
        await message.answer("Отменено.")
        return

    data = await state.get_data()
    note_id, field = data["note_id"], data["field"]
    await state.clear()

    async with session_factory() as session:
        note = await session.get(Note, note_id)
    if note is None or note.user_id != message.from_user.id:
        logger.debug("event=edit.not_found note_id=%s", note_id)
        await message.answer("Не нашла эту карточку - её уже удалили.")
        return

    if field == "translation_ru":
        async with session_factory() as session:
            edited = await set_translation(session, message.from_user.id, note_id, value)
        if edited is None:
            logger.debug("event=edit.not_found note_id=%s", note_id)
            await message.answer("Не нашла эту карточку - её уже удалили.")
            return
        note, old = edited
        logger.info(
            "event=edit.save note_id=%s field=translation_ru lemma=%s old=%r new=%r",
            note_id,
            note.lemma,
            old,
            value,
        )
        await message.answer(f"Перевод «{note.lemma}»: «{old}» → «{value}».")
        return

    await _apply_lemma_edit(message, session_factory, settings, breaker, note, value)


def _clash_reply(lemma: str) -> str:
    return f"«{lemma}» уже есть в этой колоде отдельной карточкой - сначала удали одну из них."


async def _apply_lemma_edit(
    message: Message,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    breaker: CallBreaker,
    note: Note,
    raw_value: str,
) -> None:
    new_lemma = settle_lemma(note, raw_value)

    async with session_factory() as session:
        clash = await find_lemma_clash(session, note, new_lemma)
    if clash is not None:
        logger.debug(
            "event=edit.lemma_clash note_id=%s new_lemma=%s clash_note_id=%s",
            note.id,
            new_lemma,
            clash.id,
        )
        await message.answer(_clash_reply(new_lemma))
        return

    old_lemma = note.lemma
    resolved = None
    if needs_forms(note):
        client = make_client(settings.openai_api_key, settings.openai_timeout_seconds)
        try:
            resolved, _usage = await resolve_note_forms(
                client,
                breaker,
                settings.openai_text_model,
                new_lemma,
                note.pos,
                datetime.now(UTC),
                context=note.example_fi,
            )
        except CircuitOpenError:
            logger.warning("event=edit.lemma_forms_skipped reason=breaker_open lemma=%s", new_lemma)
            await message.answer(
                "Слово сохраню, но формы не пересчитала - предохранитель сработал. "
                "Повтори позже, если нужны формы для склонения."
            )
        except openai.APIError:
            logger.exception("event=llm.error op=resolve_note_forms lemma=%s", new_lemma)
            await message.answer("Слово сохраню, но формы не пересчитала - OpenAI недоступен.")

    # Read off before the session below: a rollback there expires every object
    # it holds, and `note.id` would then lazy-load mid-exception handler -
    # which, on an async session, is a MissingGreenlet crash, not a query.
    note_id = note.id

    async with session_factory() as session:
        try:
            saved = await save_lemma(session, note_id, new_lemma, resolved, datetime.now(UTC))
        except DuplicateNoteError:
            # The clash lookup above said this lemma was free, but a
            # concurrent /add or /edit took it in the meantime. Same answer as
            # the lookup's, just later - and the note keeps its old lemma.
            logger.info("event=edit.lemma_clash_race note_id=%s new_lemma=%s", note_id, new_lemma)
            await message.answer(_clash_reply(new_lemma))
            return
    if saved is None:
        logger.debug("event=edit.not_found note_id=%s", note_id)
        await message.answer("Не нашла эту карточку - её уже удалили.")
        return

    logger.info(
        "event=edit.save note_id=%s field=lemma old=%r new=%r forms_source=%s "
        "cards_deleted=%d cards_created=%d",
        note_id,
        old_lemma,
        new_lemma,
        resolved.forms_source if resolved is not None else "unchanged",
        saved.cards_deleted,
        saved.cards_created,
    )
    await message.answer(f"Слово: «{old_lemma}» → «{new_lemma}».")
