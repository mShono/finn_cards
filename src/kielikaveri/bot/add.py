"""The /add command: turn pasted Finnish text into note candidates (plan 3.11, phase 3).

Flow: text -> LLM proposes candidates -> duplicates against the user's
existing notes are dropped automatically -> the rest are shown one at a time
for a manual keep/skip decision (plan 3: "экран подтверждения - беру не всё,
что предложила модель"). A kept candidate only then gets its word forms
resolved (FST first, LLM only to break ties - see ingest.resolve_note_forms)
and is written to the database.

Must fail honestly, never crash the bot, when OpenAI is unreachable or the
breaker trips - /learn has no LLM dependency and must keep working regardless
(plan 3.10).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import jsonschema
import openai
from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kielikaveri.config import Settings
from kielikaveri.db.models import Note, Source, SourceType
from kielikaveri.import_cards import load_validator
from kielikaveri.ingest import (
    build_full_note,
    canonical_key,
    existing_note_keys,
    generate_candidates,
    get_cached_candidates,
    hash_text,
    resolve_note_forms,
    store_cached_candidates,
)
from kielikaveri.llm.breaker import CallBreaker, CircuitOpenError
from kielikaveri.llm.client import make_client

logger = logging.getLogger(__name__)

router = Router(name="add")

# How much of the source text to keep as a quote (plan 3.9: "ссылка и цитата,
# не весь текст целиком" - the /add input can be an entire pasted article).
SOURCE_QUOTE_CHARS = 200


class AddStates(StatesGroup):
    reviewing = State()


def _render_candidate(candidate: dict) -> str:
    pos = candidate.get("pos")
    head = f"🇫🇮 {candidate['lemma']}" + (f" ({pos})" if pos else "")
    return f"{head}\n{candidate['translation_ru']}\n\n{candidate['example_fi']}\n{candidate['example_ru']}"


def _confirm_keyboard(cursor: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ Добавить", callback_data=f"add:keep:{cursor}"),
                InlineKeyboardButton(text="⏭ Пропустить", callback_data=f"add:skip:{cursor}"),
            ]
        ]
    )


@router.message(Command("add"))
async def add_start(
    message: Message,
    command: CommandObject,
    state: FSMContext,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    breaker: CallBreaker,
) -> None:
    text = (command.args or "").strip()
    if not text:
        await message.answer("Пришли текст после /add, например:\n/add Haen töitä kaupungista.")
        return
    if not settings.openai_api_key:
        await message.answer("Добавление недоступно - не настроен OpenAI.")
        return

    now = datetime.now(UTC)
    text_hash = hash_text(text)

    async with session_factory() as session:
        candidates = await get_cached_candidates(session, text_hash)

    if candidates is not None:
        logger.info("ingest cache hit hash=%s", text_hash)
    else:
        client = make_client(settings.openai_api_key, settings.openai_timeout_seconds)
        try:
            candidates, _usage = await generate_candidates(
                client, breaker, settings.openai_text_model, text, now
            )
        except CircuitOpenError:
            await message.answer(
                "Слишком много обращений к OpenAI подряд - похоже на баг, я остановилась. "
                "Попробуй позже."
            )
            return
        except openai.APIError:
            logger.exception("ingest.generate_candidates failed")
            await message.answer(
                "OpenAI сейчас недоступен - попробуй позже. /learn при этом работает как обычно."
            )
            return

        async with session_factory() as session:
            await store_cached_candidates(
                session, text_hash, settings.openai_text_model, candidates
            )
            await session.commit()

    if not candidates:
        await message.answer("Не нашла в этом тексте лексики выше текущего уровня.")
        return

    user_id = message.from_user.id
    async with session_factory() as session:
        existing = await existing_note_keys(session, user_id)

    seen: set[tuple[str, str | None]] = set()
    to_review: list[dict] = []
    duplicates = 0
    for candidate in candidates:
        key = canonical_key(candidate["lemma"], candidate.get("pos"))
        if key in existing or key in seen:
            duplicates += 1
            continue
        seen.add(key)
        to_review.append(candidate)

    if not to_review:
        await message.answer(f"Все {duplicates} кандидатов уже есть в базе - добавлять нечего.")
        return

    await state.set_state(AddStates.reviewing)
    await state.update_data(
        candidates=to_review,
        source_text=text,
        source_id=None,
        cursor=0,
        kept_count=0,
        skipped_count=0,
        duplicates_count=duplicates,
    )
    await _show_next(message, state)


async def _show_next(answer_to: Message, state: FSMContext) -> None:
    data = await state.get_data()
    candidates = data["candidates"]
    cursor = data["cursor"]
    if cursor >= len(candidates):
        await state.clear()
        await answer_to.answer(
            f"Готово: добавлено {data['kept_count']}, пропущено {data['skipped_count']} вручную, "
            f"{data['duplicates_count']} дубликатов пропущено автоматически."
        )
        return
    await answer_to.answer(
        _render_candidate(candidates[cursor]), reply_markup=_confirm_keyboard(cursor)
    )


@router.callback_query(F.data.startswith("add:skip:"), AddStates.reviewing)
async def add_skip(callback: CallbackQuery, state: FSMContext) -> None:
    cursor = int(callback.data.split(":")[2])
    data = await state.get_data()
    if cursor != data["cursor"]:
        await callback.answer("Эта карточка уже обработана.", show_alert=True)
        return

    await state.update_data(cursor=cursor + 1, skipped_count=data["skipped_count"] + 1)
    await callback.answer()
    await _show_next(callback.message, state)


@router.callback_query(F.data.startswith("add:keep:"), AddStates.reviewing)
async def add_keep(
    callback: CallbackQuery,
    state: FSMContext,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    breaker: CallBreaker,
) -> None:
    cursor = int(callback.data.split(":")[2])
    data = await state.get_data()
    if cursor != data["cursor"]:
        await callback.answer("Эта карточка уже обработана.", show_alert=True)
        return

    # Claim the slot before the first await, same double-tap race guard as
    # learn.py's learn_rate - see its comment for why this must come first.
    await state.update_data(cursor=cursor + 1)

    candidate = data["candidates"][cursor]
    user_id = callback.from_user.id
    now = datetime.now(UTC)

    resolved = None
    if candidate["kind"] == "word":
        client = make_client(settings.openai_api_key, settings.openai_timeout_seconds)
        try:
            resolved, _usage = await resolve_note_forms(
                client,
                breaker,
                settings.openai_text_model,
                candidate["lemma"],
                candidate["pos"],
                now,
            )
        except CircuitOpenError:
            await callback.answer(
                "Предохранитель сработал - карточка не сохранена.", show_alert=True
            )
            await _show_next(callback.message, state)
            return
        except openai.APIError:
            logger.exception("ingest.resolve_note_forms failed lemma=%s", candidate["lemma"])
            await callback.answer(
                "OpenAI недоступен - карточка не сохранена, попробуй позже.", show_alert=True
            )
            await _show_next(callback.message, state)
            return

    full_note = build_full_note(candidate, resolved)

    try:
        load_validator().validate(full_note)
    except jsonschema.ValidationError:
        logger.exception(
            "ingest candidate failed schema validation lemma=%s", candidate.get("lemma")
        )
        await callback.answer(
            "Не удалось сохранить - невалидные данные от модели.", show_alert=True
        )
        await _show_next(callback.message, state)
        return

    async with session_factory() as session:
        source_id = data["source_id"]
        if source_id is None:
            source = Source(
                type=SourceType.other,
                ref=f"Telegram /add, {now.date().isoformat()}",
                context_fi=data["source_text"][:SOURCE_QUOTE_CHARS],
            )
            session.add(source)
            await session.flush()
            source_id = source.id
            await state.update_data(source_id=source_id)

        session.add(
            Note(
                id=full_note["id"],
                user_id=user_id,
                lemma=full_note["lemma"],
                pos=full_note.get("pos"),
                translation_ru=full_note["translation_ru"],
                example_fi=full_note["example_fi"],
                example_ru=full_note["example_ru"],
                kind=full_note["kind"],
                source_id=source_id,
                meta=full_note["meta"],
            )
        )
        await session.commit()

    await state.update_data(kept_count=data["kept_count"] + 1)
    await callback.answer("Добавлено.")
    await _show_next(callback.message, state)


@router.callback_query(F.data.startswith("add:"))
async def add_stray_callback(callback: CallbackQuery) -> None:
    # Same reasoning as learn.py's learn_stray_callback - a button surviving
    # past its session (e.g. after the flow already finished).
    await callback.answer(
        "Эта сессия добавления уже неактуальна - начните заново через /add", show_alert=True
    )
