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
breaker trips - learning (in the reader) has no LLM dependency and must keep
working regardless (plan 3.10).
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import UTC, datetime

import openai
from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kielikaveri.bot.decks import NEW_DECK_PROMPT
from kielikaveri.bot.edit import CANCEL_WORDS
from kielikaveri.bot.text import split_message
from kielikaveri.cards_save import (
    DedupState,
    SaveStatus,
    SettledCandidate,
    settle_lemma,
    store_note,
)
from kielikaveri.config import Settings
from kielikaveri.db.decks import create_deck, get_or_create_default_deck, list_decks
from kielikaveri.db.models import (
    Card,
    Deck,
    Note,
    Review,
)
from kielikaveri.ingest import (
    build_chat_input,
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

ADD_BUTTON_TEXT = "💬 Добавить"
ADD_PROMPT = "Просто напиши мне текст на финском или свой перевод - отвечу в чате."
UNKNOWN_COMMAND_TEXT = "Не знаю такой команды. Список команд - /help"
RESEND_HINT = "Чтобы сохранить, пришли текст ещё раз."
WORDS_NOT_SAVED_TEXT = f"Слова из прошлого текста не сохранила. {RESEND_HINT}"
NOT_IN_DICTIONARY_HINT = (
    "⚠️ - не нашла в словаре, проверь написание: /decks → колода → слово → «✍️ Слово»."
)


class AddStates(StatesGroup):
    # Data: pending_text - the Finnish text the clarifying question was about.
    awaiting_instruction = State()
    # Data: batch_id, candidates - waiting for a deck pick before saving.
    choosing_deck = State()
    # Data: same as choosing_deck, carried over - waiting for the new deck's name.
    naming_new_deck = State()


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
        # ingest_cache. There's no FSM state to claim ahead of the first await
        # here (the note to delete isn't known until session.get() itself
        # returns), so the note DELETE's rowcount is the guard instead:
        # only the invocation that actually removed the row may report success.
        #
        # Children first: there is no ON DELETE CASCADE (db/models.py) and
        # PRAGMA foreign_keys is on (db/engine.py), so a note with cards, or a
        # card with reviews, cannot be deleted before them. SQLite serializes
        # the writes, so the loser of a double tap runs these after the
        # winner's commit, finds nothing left and rolls back.
        card_ids = select(Card.id).where(Card.note_id == note.id)
        await session.execute(delete(Review).where(Review.card_id.in_(card_ids)))
        await session.execute(delete(Card).where(Card.note_id == note.id))
        result = await session.execute(delete(Note).where(Note.id == note.id))
        if result.rowcount == 0:
            await session.rollback()
            await callback.answer("Уже удалено.", show_alert=True)
            return
        await session.commit()

        count = 0
        if deck_id is not None:
            count = await session.scalar(
                select(func.count())
                .select_from(Note)
                .where(Note.deck_id == deck_id, Note.user_id == callback.from_user.id)
            )

    await callback.answer()
    await callback.message.answer(
        f"Удалила «{lemma}» из колоды «{deck_name}». Осталось {count} слов."
    )


# Must stay above chat_message: aiogram stops propagation at the first
# handler whose filters pass, in registration order, and chat_message has no
# state filter - registered after it, the new deck's name went to the LLM
# as plain chat and no deck was ever created (bug since a4e78e2).
@router.message(AddStates.naming_new_deck)
async def add_new_deck_save(
    message: Message,
    state: FSMContext,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    breaker: CallBreaker,
) -> None:
    name = (message.text or "").strip()
    if not name:
        await message.answer(NEW_DECK_PROMPT)
        return
    # Same cancel words as a card edit - they used to become a deck's name
    if name.lower() in CANCEL_WORDS:
        logger.debug("event=add.new_deck_cancelled")
        await state.clear()
        await message.answer(f"Отменено, слова не сохранила. {RESEND_HINT}")
        return

    data = await state.get_data()
    candidates: list[dict] = data.get("candidates", [])
    user_id = message.from_user.id

    async with session_factory() as session:
        deck = await create_deck(session, user_id, name)
        await session.commit()
        deck_id = deck.id

    await state.clear()
    await message.answer(f"Колода «{deck.name}» создана.")
    await _save_candidates_and_report(
        message, session_factory, settings, breaker, user_id, deck_id, candidates
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


# Last message handler of the last router: every known command is matched
# above, so a "/..." that got this far is a typo or a removed command.
# State is left as is - an unknown command must not end, say, a deck pick.
@router.message(F.text.startswith("/"))
async def unknown_command(message: Message) -> None:
    logger.debug("event=route.unknown_command")
    await message.answer(UNKNOWN_COMMAND_TEXT)


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

    # Raw text itself is already in middleware's update_received.detail
    # (DEBUG) - this line only adds the business-level fact middleware can't
    # know: whether this turn answers a pending clarifying question.
    logger.debug("event=add.chat_turn follow_up=%s", context_text is not None)

    now = datetime.now(UTC)
    text_hash = hash_text(
        build_chat_input(text, context_text),
        settings.openai_text_model,
        is_follow_up=context_text is not None,
    )

    async with session_factory() as session:
        cached = await get_cached_chat(session, text_hash)

    if cached is not None:
        logger.info("event=chat.cache_hit hash=%s", text_hash)
        reply_ru, needs_clarification, candidates = cached
    else:
        logger.debug("event=chat.cache_miss hash=%s route=llm", text_hash)
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
                "OpenAI сейчас недоступен - попробуй позже. "
                "Учить карточки (📚 Учить) при этом можно как обычно."
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

    logger.debug(
        "event=chat.decision needs_clarification=%s candidates=%d",
        needs_clarification,
        len(candidates),
    )

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
        logger.debug("event=chat.awaiting_clarification")
        # context_text is the text this very question is about, even on a
        # (should-not-happen) second clarification round after a follow-up.
        await state.set_state(AddStates.awaiting_instruction)
        await state.update_data(pending_text=context_text or text)
        return

    if not candidates:
        logger.debug("event=chat.no_candidates")
        return

    user_id = message.from_user.id
    try:
        # Dedup happens after the deck is picked (_save_candidates_and_report),
        # not here - whether a word is "already there" depends on which deck
        # you're adding to (plan: per-deck dedup, requested 03.09.2026, was
        # global-per-user before). Nothing to filter on yet at this point.
        async with session_factory() as session:
            # Default deck first - guarantees at least one deck (creating the
            # "Общая" for a brand new user) before the picker below is built,
            # so it's never shown empty.
            await get_or_create_default_deck(session, user_id)
            await session.commit()
            decks = await list_decks(session, user_id)

        # Short, not a full uuid4 (36 chars) - callback_data below packs this
        # plus a full deck id and Telegram caps callback_data at 64 bytes.
        # "adddeck:" (8) + 12 hex chars + ":" (1) + a uuid4 deck id (36) = 57,
        # comfortably under the limit. Found live 27.08.2026: the old 36-char
        # batch_id pushed callback_data to 81 bytes, so Telegram rejected the
        # whole picker message with BUTTON_DATA_INVALID - silently, since
        # nothing here caught it yet at the time.
        batch_id = uuid.uuid4().hex[:12]
        logger.debug(
            "event=add.deck_prompt batch_id=%s candidates=%d decks=%d",
            batch_id,
            len(candidates),
            len(decks),
        )
        await state.set_state(AddStates.choosing_deck)
        await state.update_data(batch_id=batch_id, candidates=candidates)
        await message.answer(
            "В какую колоду добавить?", reply_markup=_deck_choice_keyboard(decks, batch_id)
        )
    except Exception:
        # Everything above this point used to be able to die silently - the
        # user would see "Добавляю." and then nothing, ever, with no error
        # anywhere they could see (found live 27.08.2026). Better to admit
        # failure than to leave them guessing whether it worked.
        logger.exception("event=add.error stage=save_candidates")
        await message.answer(
            "Не получилось сохранить - что-то пошло не так на моей стороне. "
            "Попробуй прислать список ещё раз."
        )


def _deck_choice_keyboard(decks: list[Deck], batch_id: str) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(text=deck.name, callback_data=f"adddeck:{batch_id}:{deck.id}")]
        for deck in decks
    ]
    rows.append(
        [InlineKeyboardButton(text="➕ Новая колода", callback_data=f"addnewdeck:{batch_id}")]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


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
    user_id = callback.from_user.id
    # Cleared before anything that yields: with MemoryStorage get_data ->
    # check -> clear is atomic, so a double tap's second update finds no
    # batch and takes the stale exit above instead of a second full run.
    await state.clear()

    # batch_id above only proves the picker is this chat's current one - the
    # deck id is whatever the callback_data says, and a forged one would land
    # this user's words in someone else's deck (or a nonexistent one). Same
    # ownership check as decks_open/noteedit/delnote. The batch is gone by
    # now; only forged callback_data gets here, so no restoring it.
    async with session_factory() as session:
        deck = await session.get(Deck, deck_id)
    if deck is None or deck.user_id != user_id:
        logger.debug("event=add.deck_not_found deck_id=%s", deck_id)
        await callback.answer("Не нашла колоду.", show_alert=True)
        return

    await callback.answer()

    # Saving + card generation takes seconds - acknowledge the tap first so
    # the picker doesn't look frozen while it runs.
    await callback.message.answer("Добавляю...")

    await _save_candidates_and_report(
        callback.message,
        session_factory,
        settings,
        breaker,
        user_id,
        deck_id,
        candidates,
    )


@router.callback_query(F.data.startswith("adddeck:"))
async def add_deck_choice_stray(callback: CallbackQuery) -> None:
    # Reaches here only when the picker is tapped outside choosing_deck - a
    # batch from a session that already resolved or expired.
    await callback.answer("Эта подборка уже неактуальна - пришли текст ещё раз.", show_alert=True)


@router.callback_query(F.data.startswith("addnewdeck:"), AddStates.choosing_deck)
async def add_new_deck_prompt(callback: CallbackQuery, state: FSMContext) -> None:
    _, batch_id = callback.data.split(":", 1)
    data = await state.get_data()
    if data.get("batch_id") != batch_id:
        await callback.answer(
            "Эта подборка уже неактуальна - пришли текст ещё раз.", show_alert=True
        )
        return

    # batch_id/candidates stay in state data - set_state() alone
    # doesn't touch them, only the name reply below needs to read them back.
    await state.set_state(AddStates.naming_new_deck)
    await callback.message.answer(NEW_DECK_PROMPT)
    await callback.answer()


@router.callback_query(F.data.startswith("addnewdeck:"))
async def add_new_deck_prompt_stray(callback: CallbackQuery) -> None:
    await callback.answer("Эта подборка уже неактуальна - пришли текст ещё раз.", show_alert=True)


async def _save_candidates_and_report(
    answer_to: Message,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    breaker: CallBreaker,
    user_id: int,
    deck_id: str,
    candidates: list[dict],
) -> None:
    now = datetime.now(UTC)
    start = time.monotonic()

    async with session_factory() as session:
        deck = await session.get(Deck, deck_id)
        deck_name = deck.name if deck else "?"
        existing = await existing_note_keys(session, user_id, deck_id=deck_id)

    client = make_client(settings.openai_api_key, settings.openai_timeout_seconds)
    dedup = DedupState(existing)
    to_add: list[SettledCandidate] = []
    duplicate_lemmas: list[str] = []
    failed: list[tuple[str, str]] = []
    for candidate in candidates:
        settled = await settle_lemma(
            client, breaker, settings.openai_text_model, candidate, dedup, now
        )
        if isinstance(settled, SettledCandidate):
            to_add.append(settled)
        elif settled.status is SaveStatus.duplicate:
            duplicate_lemmas.append(settled.lemma)
        else:
            failed.append((settled.lemma, settled.reason or ""))

    logger.info(
        "event=add.dedup deck=%s candidates=%d to_add=%d duplicates=%d",
        deck_id,
        len(candidates),
        len(to_add),
        len(duplicate_lemmas),
    )

    saved: list[tuple[str, str, bool]] = []
    for settled in to_add:
        outcome = await store_note(
            session_factory,
            client,
            breaker,
            settings.openai_text_model,
            user_id,
            deck_id,
            settled,
            dedup,
            now,
            # This module's name, not ingest's: the one /add's tests replace.
            resolve_forms=resolve_note_forms,
        )
        if outcome.status is SaveStatus.saved:
            saved.append((outcome.lemma, outcome.translation_ru or "", outcome.in_dictionary))
        elif outcome.status is SaveStatus.duplicate:
            duplicate_lemmas.append(outcome.lemma)
        else:
            failed.append((outcome.lemma, outcome.reason or ""))

    duration_ms = int((time.monotonic() - start) * 1000)
    logger.info(
        "event=add.save deck=%s saved=%d duplicates=%d failed=%d duration_ms=%d",
        deck_id,
        len(saved),
        len(duplicate_lemmas),
        len(failed),
        duration_ms,
    )

    lines = [
        f"🇫🇮 {lemma} → {translation}" + ("" if in_dictionary else " ⚠️")
        for lemma, translation, in_dictionary in saved
    ]
    if not all(in_dictionary for _lemma, _translation, in_dictionary in saved):
        lines.append(NOT_IN_DICTIONARY_HINT)
    if saved:
        async with session_factory() as session:
            count = await session.scalar(
                select(func.count())
                .select_from(Note)
                .where(Note.deck_id == deck_id, Note.user_id == user_id)
            )
        lines.append(f"\nКолода «{deck_name}»: теперь {count} слов.")
    if duplicate_lemmas:
        lines.append(f"Уже есть в «{deck_name}», не дублирую: {', '.join(duplicate_lemmas)}.")
    for lemma, reason in failed:
        lines.append(f"Не удалось сохранить «{lemma}» - {reason}.")

    if lines:
        await answer_to.answer("\n".join(lines))
