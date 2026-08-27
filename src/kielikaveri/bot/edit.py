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
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kielikaveri.config import Settings
from kielikaveri.db.models import Note, NoteKind
from kielikaveri.ingest import canonical_key, resolve_note_forms
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
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✍️ Слово", callback_data=f"noteeditfield:{note_id}:lm"),
                InlineKeyboardButton(text="✍️ Перевод", callback_data=f"noteeditfield:{note_id}:tr"),
            ],
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
        await callback.answer("Не нашла эту карточку.", show_alert=True)
        return

    pos_suffix = f" ({note.pos})" if note.pos else ""
    await callback.message.answer(
        f"«{note.lemma}{pos_suffix} - {note.translation_ru}» - что править?",
        reply_markup=_field_keyboard(note.id),
    )
    await callback.answer()


@router.callback_query(F.data == "noteeditcancel")
async def note_edit_cancel(callback: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
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
        await state.clear()
        await message.answer("Отменено.")
        return

    data = await state.get_data()
    note_id, field = data["note_id"], data["field"]
    await state.clear()

    async with session_factory() as session:
        note = await session.get(Note, note_id)
    if note is None or note.user_id != message.from_user.id:
        await message.answer("Не нашла эту карточку - её уже удалили.")
        return

    if field == "translation_ru":
        old = note.translation_ru
        async with session_factory() as session:
            note = await session.get(Note, note_id)
            note.translation_ru = value
            await session.commit()
        await message.answer(f"Перевод «{note.lemma}»: «{old}» → «{value}».")
        return

    await _apply_lemma_edit(message, session_factory, settings, breaker, note, value)


async def _apply_lemma_edit(
    message: Message,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    breaker: CallBreaker,
    note: Note,
    raw_value: str,
) -> None:
    # canonical_key() lemmatizes the same way /add's dedup does (kind=pattern
    # has pos=None and is returned unchanged - a pattern's "lemma" is the
    # construction string itself, not something the FST can resolve).
    new_lemma, _pos = canonical_key(raw_value, note.pos)

    async with session_factory() as session:
        pos_filter = Note.pos.is_(None) if note.pos is None else Note.pos == note.pos
        clash = await session.scalar(
            select(Note).where(
                Note.user_id == message.from_user.id,
                Note.lemma == new_lemma,
                pos_filter,
                Note.id != note.id,
            )
        )
    if clash is not None:
        await message.answer(
            f"«{new_lemma}» уже есть в базе отдельной карточкой - сначала удали одну из них."
        )
        return

    old_lemma = note.lemma
    resolved = None
    if note.kind == NoteKind.word and note.pos is not None:
        client = make_client(settings.openai_api_key, settings.openai_timeout_seconds)
        try:
            resolved, _usage = await resolve_note_forms(
                client, breaker, settings.openai_text_model, new_lemma, note.pos, datetime.now(UTC)
            )
        except CircuitOpenError:
            await message.answer(
                "Слово сохраню, но формы не пересчитала - предохранитель сработал. "
                "Повтори позже, если нужны формы для склонения."
            )
        except openai.APIError:
            logger.exception("edit.resolve_note_forms failed lemma=%s", new_lemma)
            await message.answer("Слово сохраню, но формы не пересчитала - OpenAI недоступен.")

    async with session_factory() as session:
        note = await session.get(Note, note.id)
        note.lemma = new_lemma
        if resolved is not None:
            new_meta = dict(note.meta)
            new_meta["principal_forms"] = resolved.principal_forms
            new_meta["forms_source"] = resolved.forms_source
            new_meta["forms_verified"] = resolved.forms_verified
            note.meta = new_meta
        await session.commit()

    await message.answer(f"Слово: «{old_lemma}» → «{new_lemma}».")
