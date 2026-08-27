"""Conversational card creation (plan: /add redesign).

Any plain message - pasted Finnish text, a translation attempt, a plain
question - goes to one LLM call (ingest.check_and_suggest) that replies in
chat and decides for itself whether it has enough to act on. A bare Finnish
text with no instruction gets a clarifying question instead of an unsolicited
translation or a dumped candidate list (`needs_clarification`, see
ingest._chat_instructions) - the student says what to do with it (translate
it herself for checking, or name specific words/phrases to translate or add),
and that answer is resolved in a follow-up call carrying the original text as
context (AddStates.awaiting_instruction).

Candidates only ever come from something the student explicitly asked for -
never "extra vocabulary from the text, just in case". Saving them asks which
deck first when there is more than one (AddStates.choosing_deck), then
reports what actually landed where, rather than a silent per-candidate
"add" button.

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
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kielikaveri.bot.text import split_message
from kielikaveri.config import Settings
from kielikaveri.db.decks import active_deck, list_decks, set_active_deck
from kielikaveri.db.models import Card, Deck, Note, Review, Source, SourceType
from kielikaveri.import_cards import load_validator
from kielikaveri.ingest import (
    build_chat_input,
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


class AddStates(StatesGroup):
    # Data: pending_text - the Finnish text the clarifying question was about.
    awaiting_instruction = State()
    # Data: batch_id, candidates, source_id - waiting for a deck pick before saving.
    choosing_deck = State()


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


@router.message(Command("delete"))
async def delete_command(
    message: Message,
    command: CommandObject,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    word = (command.args or "").strip()
    if not word:
        await message.answer("Какое слово удалить? Например: /delete naapuri")
        return

    user_id = message.from_user.id
    async with session_factory() as session:
        # Matched in Python, not SQL: SQLite's built-in lower() only folds
        # ASCII, so func.lower('Äiti') stays 'Äiti' and would never match a
        # user typing "äiti" - str.lower() handles Finnish's ä/ö/å correctly.
        all_notes = (await session.scalars(select(Note).where(Note.user_id == user_id))).all()
        word_lower = word.lower()
        notes = [note for note in all_notes if note.lemma.lower() == word_lower]
        if not notes:
            await message.answer("Не нашла такое слово.")
            return

        rows = []
        for note in notes:
            deck = await session.get(Deck, note.deck_id) if note.deck_id else None
            deck_name = deck.name if deck else "без колоды"
            pos_suffix = f" ({note.pos})" if note.pos else ""
            rows.append(
                [
                    InlineKeyboardButton(
                        text=f"🗑 {note.lemma}{pos_suffix} - «{deck_name}»",
                        callback_data=f"delnote:{note.id}",
                    )
                ]
            )
    rows.append([InlineKeyboardButton(text="Отмена", callback_data="delnote:cancel")])
    await message.answer("Что удалить?", reply_markup=InlineKeyboardMarkup(inline_keyboard=rows))


@router.callback_query(F.data == "delnote:cancel")
async def delete_cancel(callback: CallbackQuery) -> None:
    await callback.answer("Отменено.")


@router.callback_query(F.data.startswith("delnote:"))
async def delete_confirm(
    callback: CallbackQuery, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    note_id = callback.data.split(":", 1)[1]
    async with session_factory() as session:
        note = await session.get(Note, note_id)
        if note is None or note.user_id != callback.from_user.id:
            await callback.answer("Уже удалено.", show_alert=True)
            return

        lemma = note.lemma
        deck_id = note.deck_id
        deck = await session.get(Deck, deck_id) if deck_id else None
        deck_name = deck.name if deck else "без колоды"

        # Two callback_query updates for one genuine double-tap both pass the
        # None-check above (neither has committed yet) - same race class as
        # learn_rate/ingest_cache. There's no FSM state to claim ahead of the
        # first await here (the note to delete isn't known until session.get()
        # itself returns), so the DELETE's rowcount is the guard instead: only
        # the invocation that actually removed a row may report success or
        # touch its cards/reviews.
        result = await session.execute(delete(Note).where(Note.id == note.id))
        if result.rowcount == 0:
            await session.rollback()
            await callback.answer("Уже удалено.", show_alert=True)
            return

        # No ON DELETE CASCADE anywhere (db/models.py) - a Card left pointing
        # at a deleted note_id would make learn.py's render_card crash on the
        # next session.get(Note, ...) returning None.
        card_ids = (await session.scalars(select(Card.id).where(Card.note_id == note.id))).all()
        if card_ids:
            await session.execute(delete(Review).where(Review.card_id.in_(card_ids)))
            await session.execute(delete(Card).where(Card.id.in_(card_ids)))
        await session.commit()

        count = 0
        if deck_id is not None:
            count = await session.scalar(
                select(func.count()).select_from(Note).where(Note.deck_id == deck_id)
            )

    await callback.answer()
    await callback.message.answer(
        f"Удалила «{lemma}» из колоды «{deck_name}». Осталось {count} слов."
    )


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

    current_state = await state.get_state()
    context_text: str | None = None
    if current_state == AddStates.awaiting_instruction.state:
        data = await state.get_data()
        context_text = data.get("pending_text")
        await state.clear()

    now = datetime.now(UTC)
    text_hash = hash_text(
        build_chat_input(text, context_text),
        settings.openai_text_model,
        is_follow_up=context_text is not None,
    )

    async with session_factory() as session:
        cached = await get_cached_chat(session, text_hash)

    if cached is not None:
        logger.info("chat cache hit hash=%s", text_hash)
        reply_ru, needs_clarification, candidates = cached
    else:
        client = make_client(settings.openai_api_key, settings.openai_timeout_seconds)
        try:
            reply_ru, needs_clarification, candidates, _usage = await check_and_suggest(
                client, breaker, settings.openai_text_model, text, now, context_text=context_text
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
                session,
                text_hash,
                settings.openai_text_model,
                reply_ru,
                needs_clarification,
                candidates,
            )
            try:
                await session.commit()
            except IntegrityError:
                # A concurrent turn for the same input (double-tap, a resent
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

    if needs_clarification:
        # context_text is the text this very question is about, even on a
        # (should-not-happen) second clarification round after a follow-up.
        await state.set_state(AddStates.awaiting_instruction)
        await state.update_data(pending_text=context_text or text)
        return

    if not candidates:
        return

    user_id = message.from_user.id
    try:
        async with session_factory() as session:
            existing = await existing_note_keys(session, user_id)

        seen: set[tuple[str, str | None]] = set()
        to_add: list[dict] = []
        duplicates = 0
        for candidate in candidates:
            key = canonical_key(candidate["lemma"], candidate.get("pos"))
            if key in existing or key in seen:
                duplicates += 1
                continue
            seen.add(key)
            # canonical_key() lemmatizes to catch dupes even when the LLM
            # handed back an inflected form as "lemma" (cards/instructions.md)
            # - without this, the dedup check sees the corrected lemma but
            # the saved card still gets the raw inflected string.
            if key[0] != candidate["lemma"]:
                candidate = {**candidate, "lemma": key[0]}
            to_add.append(candidate)

        # Found live 27.08.2026: with only a token-count log line, "the model
        # said candidates=4 but nothing got saved" took three round-trips to
        # even start narrowing down. This one line pins the exact stage - did
        # dedup eat everything, or did nothing even reach dedup.
        logger.info(
            "chat_message dedup candidates=%d to_add=%d duplicates=%d",
            len(candidates),
            len(to_add),
            duplicates,
        )

        if not to_add:
            if duplicates:
                await message.answer(f"Все {duplicates} кандидатов уже есть в базе.")
            return

        quote_text = context_text or text
        async with session_factory() as session:
            source = Source(
                type=SourceType.other,
                ref=f"Telegram чат, {now.date().isoformat()}",
                context_fi=quote_text[:SOURCE_QUOTE_CHARS],
            )
            session.add(source)
            # active_deck() first - guarantees at least one deck (creating the
            # default "Общая" for a brand new user) before the picker below is
            # built, so it's never shown empty.
            await active_deck(session, user_id)
            await session.commit()
            source_id = source.id
            decks = await list_decks(session, user_id)

        batch_id = str(uuid.uuid4())
        await state.set_state(AddStates.choosing_deck)
        await state.update_data(batch_id=batch_id, candidates=to_add, source_id=source_id)
        await message.answer(
            "В какую колоду добавить?", reply_markup=_deck_choice_keyboard(decks, batch_id)
        )
    except Exception:
        # Everything above this point used to be able to die silently - the
        # user would see "Добавляю." and then nothing, ever, with no error
        # anywhere they could see (found live 27.08.2026). Better to admit
        # failure than to leave them guessing whether it worked.
        logger.exception("chat_message failed while saving candidates")
        await message.answer(
            "Не получилось сохранить - что-то пошло не так на моей стороне. "
            "Попробуй прислать список ещё раз."
        )


def _deck_choice_keyboard(decks: list[Deck], batch_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=deck.name, callback_data=f"adddeck:{batch_id}:{deck.id}")]
            for deck in decks
        ]
    )


@router.callback_query(F.data.startswith("adddeck:"), AddStates.choosing_deck)
async def add_deck_choice(
    callback: CallbackQuery,
    state: FSMContext,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    breaker: CallBreaker,
) -> None:
    _, batch_id, deck_id = callback.data.split(":", 2)
    data = await state.get_data()
    if data.get("batch_id") != batch_id:
        await callback.answer(
            "Эта подборка уже неактуальна - пришли текст ещё раз.", show_alert=True
        )
        return

    candidates: list[dict] = data.get("candidates", [])
    source_id = data.get("source_id")
    user_id = callback.from_user.id
    await state.clear()
    await callback.answer()

    await _save_candidates_and_report(
        callback.message,
        session_factory,
        settings,
        breaker,
        user_id,
        deck_id,
        source_id,
        candidates,
    )


@router.callback_query(F.data.startswith("adddeck:"))
async def add_deck_choice_stray(callback: CallbackQuery) -> None:
    # Reaches here only when the picker is tapped outside choosing_deck - a
    # batch from a session that already resolved or expired.
    await callback.answer("Эта подборка уже неактуальна - пришли текст ещё раз.", show_alert=True)


async def _save_candidates_and_report(
    answer_to: Message,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    breaker: CallBreaker,
    user_id: int,
    deck_id: str,
    source_id: str,
    candidates: list[dict],
) -> None:
    now = datetime.now(UTC)

    async with session_factory() as session:
        deck = await session.get(Deck, deck_id)
        deck_name = deck.name if deck else "?"

    saved: list[tuple[str, str]] = []
    failed: list[tuple[str, str]] = []
    for candidate in candidates:
        resolved = None
        if candidate["kind"] == "word":
            client = make_client(settings.openai_api_key, settings.openai_timeout_seconds)
            try:
                resolved, _usage = await resolve_note_forms(
                    client,
                    breaker,
                    settings.openai_text_model,
                    candidate["lemma"],
                    candidate.get("pos"),
                    now,
                )
            except CircuitOpenError:
                failed.append((candidate["lemma"], "предохранитель сработал"))
                continue
            except openai.APIError:
                logger.exception("ingest.resolve_note_forms failed lemma=%s", candidate["lemma"])
                failed.append((candidate["lemma"], "OpenAI недоступен"))
                continue

        full_note = build_full_note(candidate, resolved)
        try:
            load_validator().validate(full_note)
        except jsonschema.ValidationError:
            logger.exception(
                "ingest candidate failed schema validation lemma=%s", candidate.get("lemma")
            )
            failed.append((candidate["lemma"], "невалидные данные от модели"))
            continue

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
                    source_id=source_id,
                    deck_id=deck_id,
                    meta=full_note["meta"],
                )
            )
            await session.commit()
        saved.append((full_note["lemma"], full_note["translation_ru"]))

    lines = [f"🇫🇮 {lemma} → {translation}" for lemma, translation in saved]
    if saved:
        async with session_factory() as session:
            await set_active_deck(session, user_id, deck_id)
            count = await session.scalar(
                select(func.count()).select_from(Note).where(Note.deck_id == deck_id)
            )
            await session.commit()
        lines.append(f"\nКолода «{deck_name}»: теперь {count} слов.")
    for lemma, reason in failed:
        lines.append(f"Не удалось сохранить «{lemma}» - {reason}.")

    if lines:
        await answer_to.answer("\n".join(lines))
