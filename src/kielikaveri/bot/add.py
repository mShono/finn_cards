"""Conversational card creation (plan 3.11, "Текст", v2).

Any plain message - pasted Finnish text, a translation attempt, a plain
question - goes to one LLM call (ingest.check_and_suggest) that replies in
chat *and* proposes note candidates in the same response. Candidates are
shown all at once, each with its own "add" button - no forced one-by-one
swipe, no filtering down to a fixed list the learner then has to click
through hoping the word they wanted is in it.

Must fail honestly, never crash the bot, when OpenAI is unreachable or the
breaker trips - /learn has no LLM dependency and must keep working regardless
(plan 3.10).
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

import jsonschema
import openai
from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kielikaveri.bot.text import split_message
from kielikaveri.config import Settings
from kielikaveri.db.decks import active_deck
from kielikaveri.db.models import Note, Source, SourceType
from kielikaveri.import_cards import load_validator
from kielikaveri.ingest import (
    build_full_note,
    canonical_key,
    check_and_suggest,
    existing_note_keys,
    get_cached_chat,
    hash_text,
    resolve_note_forms,
    store_cached_chat,
)
from kielikaveri.llm.breaker import CallBreaker, CircuitOpenError
from kielikaveri.llm.client import make_client

logger = logging.getLogger(__name__)

router = Router(name="add")

# How much of the source text to keep as a quote (plan 3.9: "ссылка и цитата,
# не весь текст целиком" - a chat message can be an entire pasted article).
SOURCE_QUOTE_CHARS = 200

ADD_BUTTON_TEXT = "💬 Добавить"
ADD_PROMPT = "Просто напиши мне текст на финском или свой перевод - отвечу в чате."


def _render_candidate(candidate: dict) -> str:
    pos = candidate.get("pos")
    head = f"🇫🇮 {candidate['lemma']}" + (f" ({pos})" if pos else "")
    return f"{head}\n{candidate['translation_ru']}\n\n{candidate['example_fi']}\n{candidate['example_ru']}"


def _keep_keyboard(batch_id: str, idx: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="✅ Добавить", callback_data=f"chat:add:{batch_id}:{idx}")]
        ]
    )


@router.message(F.text == ADD_BUTTON_TEXT)
async def add_button_prompt(message: Message) -> None:
    await message.answer(ADD_PROMPT)


@router.message(Command("add"))
async def add_command(
    message: Message,
    command: CommandObject,
    state: FSMContext,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    breaker: CallBreaker,
) -> None:
    if not (command.args or "").strip():
        await message.answer(ADD_PROMPT)
        return
    await _handle_chat_turn(message, state, session_factory, settings, breaker, command.args)


@router.message(F.text & ~F.text.startswith("/"))
async def chat_message(
    message: Message,
    state: FSMContext,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    breaker: CallBreaker,
) -> None:
    await _handle_chat_turn(message, state, session_factory, settings, breaker, message.text)


async def _handle_chat_turn(
    message: Message,
    state: FSMContext,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    breaker: CallBreaker,
    raw_text: str | None,
) -> None:
    text = (raw_text or "").strip()
    if not text:
        return
    if not settings.openai_api_key:
        await message.answer("Ответить не могу - не настроен OpenAI.")
        return

    now = datetime.now(UTC)
    text_hash = hash_text(text)

    async with session_factory() as session:
        cached = await get_cached_chat(session, text_hash)

    if cached is not None:
        logger.info("chat cache hit hash=%s", text_hash)
        reply_ru, candidates = cached
    else:
        client = make_client(settings.openai_api_key, settings.openai_timeout_seconds)
        try:
            reply_ru, candidates, _usage = await check_and_suggest(
                client, breaker, settings.openai_text_model, text, now
            )
        except CircuitOpenError:
            await message.answer(
                "Слишком много обращений к OpenAI подряд - похоже на баг, я остановилась. "
                "Попробуй позже."
            )
            return
        except openai.APIError:
            logger.exception("ingest.check_and_suggest failed")
            await message.answer(
                "OpenAI сейчас недоступен - попробуй позже. /learn при этом работает как обычно."
            )
            return

        async with session_factory() as session:
            await store_cached_chat(
                session, text_hash, settings.openai_text_model, reply_ru, candidates
            )
            try:
                await session.commit()
            except IntegrityError:
                # A concurrent turn for the same text (double-tap, a resent
                # message) already cached it first - text_hash is the PK.
                # Our own freshly generated result is still valid to show,
                # there's just nothing left to store.
                await session.rollback()

    # reply_ru is a required schema field but strict mode can't enforce
    # non-empty - fall back rather than silently sending nothing back, which
    # would look like the bot ignored the message entirely.
    reply_chunks = split_message(reply_ru.strip()) or (["Нашла кое-что:"] if candidates else None)
    if reply_chunks is None:
        await message.answer("Не нашла, что ответить - попробуй переформулировать.")
        return
    for chunk in reply_chunks:
        await message.answer(chunk)

    if not candidates:
        return

    user_id = message.from_user.id
    async with session_factory() as session:
        existing = await existing_note_keys(session, user_id)
        deck = await active_deck(session, user_id)
        await session.commit()  # active_deck() may have just created a default deck

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
        if duplicates:
            await message.answer(f"Все {duplicates} кандидатов уже есть в базе.")
        return

    # Created here, once, rather than lazily on the first "add" tap: unlike
    # the old sequential swipe, candidates in a batch are all actionable at
    # once, so two taps on different candidates can race - both would see
    # source_id=None and create their own Source otherwise.
    async with session_factory() as session:
        source = Source(
            type=SourceType.other,
            ref=f"Telegram чат, {now.date().isoformat()}",
            context_fi=text[:SOURCE_QUOTE_CHARS],
        )
        session.add(source)
        await session.commit()
        source_id = source.id

    batch_id = str(uuid.uuid4())
    await state.update_data(
        batch_id=batch_id,
        candidates=to_review,
        added=[],
        source_id=source_id,
        deck_id=deck.id,
    )

    await message.answer(f"Добавляю в колоду «{deck.name}» (сменить: /decks):")
    for idx, candidate in enumerate(to_review):
        await message.answer(
            _render_candidate(candidate), reply_markup=_keep_keyboard(batch_id, idx)
        )


@router.callback_query(F.data.startswith("chat:add:"))
async def chat_add_keep(
    callback: CallbackQuery,
    state: FSMContext,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    breaker: CallBreaker,
) -> None:
    _, _, batch_id, idx_raw = callback.data.split(":", 3)
    idx = int(idx_raw)
    data = await state.get_data()

    candidates: list[dict] = data.get("candidates", [])
    added: list[int] = data.get("added", [])
    if data.get("batch_id") != batch_id or idx >= len(candidates):
        await callback.answer(
            "Эта подборка уже неактуальна - пришли текст ещё раз.", show_alert=True
        )
        return
    if idx in added:
        await callback.answer("Уже добавлено.", show_alert=True)
        return

    # Claim the index before the first await - same double-tap race guard as
    # learn.py's learn_rate: two callback_query updates for one genuine
    # double-tap would otherwise both pass the checks above and both save.
    await state.update_data(added=[*added, idx])

    candidate = candidates[idx]
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
            return
        except openai.APIError:
            logger.exception("ingest.resolve_note_forms failed lemma=%s", candidate["lemma"])
            await callback.answer(
                "OpenAI недоступен - карточка не сохранена, попробуй позже.", show_alert=True
            )
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
        return

    async with session_factory() as session:
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
                source_id=data["source_id"],
                deck_id=data["deck_id"],
                meta=full_note["meta"],
            )
        )
        await session.commit()

    await callback.answer("Добавлено.")
