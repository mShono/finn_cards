"""Saving one card candidate as a note, without any UI (plan
2026-11-20-kirjakaveri-reader, "П4").

Shared by /add (bot/add.py, a batch of candidates and a chat report) and the
reader's ➕ (reader/cards.py, one word and a JSON answer). The steps are
/add's, unchanged:

1. settle_lemma: the FST's lemma (ingest.resolve_note_lemma), then the dedup
   check against the deck's notes and the batch so far;
2. store_note: the forms (ingest.resolve_note_forms, kind="word" only), the
   part of speech the FST corrected - with the dedup check re-run on it -,
   build_full_note, the schema check, the INSERT. Losing a race to another
   save of the same word (uq_notes_user_deck_lemma_pos) is a duplicate, not
   an error.

/add runs step 1 for the whole batch before step 2 (its "add.dedup" line
counts the batch in between); the reader runs both for its one word
(save_candidate).

Cards are not created here: learning in the reader creates them
(sync_user_card_types, learn_core.prepare_session), as for /add's notes.

The log lines quote the lemma, as /add's always have. On the reader's path
they are cut to their event name (reader.lookup.RedactBookText covers this
module's logger).
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime

import jsonschema
import openai
from openai import AsyncOpenAI
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kielikaveri.db.models import Note, is_note_duplicate_error
from kielikaveri.import_cards import load_validator
from kielikaveri.ingest import (
    SURFACE_FIELD,
    ResolvedForms,
    TokenUsage,
    build_full_note,
    resolve_note_forms,
    resolve_note_lemma,
)
from kielikaveri.llm.breaker import CallBreaker, CircuitOpenError

logger = logging.getLogger(__name__)

# The reasons /add prints after "Не удалось сохранить «…» - ".
REASON_BREAKER = "предохранитель сработал"
REASON_OPENAI = "OpenAI недоступен"
REASON_INVALID = "невалидные данные от модели"

Key = tuple[str, str | None]
ResolveForms = Callable[..., Awaitable[tuple[ResolvedForms, TokenUsage | None]]]


class SaveStatus(enum.Enum):
    saved = "saved"
    duplicate = "duplicate"
    failed = "failed"


@dataclass
class SaveOutcome:
    status: SaveStatus
    lemma: str
    translation_ru: str | None = None
    # ingest.ResolvedLemma.in_dictionary of a saved note.
    in_dictionary: bool = True
    # For a failure: the reason /add prints, and the exception behind it
    # (CircuitOpenError, openai.APIError, jsonschema.ValidationError), so a
    # caller can tell a tripped breaker from OpenAI from bad data.
    reason: str | None = None
    error: Exception | None = None


@dataclass
class DedupState:
    """The dedup universe of one save: the deck's existing (lemma, pos)
    keys (ingest.existing_note_keys) and the keys this batch already took."""

    existing: set[Key]
    seen: set[Key] = field(default_factory=set)

    def taken(self, key: Key) -> bool:
        return key in self.existing or key in self.seen


@dataclass
class SettledCandidate:
    """A candidate past step 1: its lemma is the FST's, its key is new."""

    candidate: dict
    in_dictionary: bool


@dataclass
class Usage:
    """Tokens of the OpenAI calls a save made (resolve_note_lemma/forms)."""

    input: int = 0
    output: int = 0

    def add(self, usage: TokenUsage | None) -> None:
        if usage is not None:
            self.input += usage.input_tokens
            self.output += usage.output_tokens


async def settle_lemma(
    client: AsyncOpenAI,
    breaker: CallBreaker,
    model: str,
    candidate: dict,
    dedup: DedupState,
    now: datetime,
    *,
    context: str | None = None,
    usage: Usage | None = None,
) -> SettledCandidate | SaveOutcome:
    """Step 1. `context` is the sentence the lemma is settled in (the
    candidate's example_fi when not given). A SaveOutcome means the candidate
    stops here: a duplicate or a failure."""
    # Settle the lemma before the dedup check - the FST's, not the LLM's
    # raw string, which may be an inflected form ("töitä") or made up
    # entirely ("riensiä"). Without this the check would compare one
    # lemma and the saved card would still get the other.
    try:
        resolved_lemma, lemma_usage = await resolve_note_lemma(
            client,
            breaker,
            model,
            candidate["lemma"],
            candidate.get("pos"),
            candidate.get(SURFACE_FIELD),
            context if context is not None else candidate.get("example_fi"),
            now,
        )
    except CircuitOpenError as error:
        return SaveOutcome(
            SaveStatus.failed, candidate["lemma"], reason=REASON_BREAKER, error=error
        )
    except openai.APIError as error:
        logger.exception("event=llm.error op=resolve_note_lemma lemma=%s", candidate["lemma"])
        return SaveOutcome(SaveStatus.failed, candidate["lemma"], reason=REASON_OPENAI, error=error)
    if usage is not None:
        usage.add(lemma_usage)
    key = (resolved_lemma.lemma, candidate.get("pos"))
    if dedup.taken(key):
        return SaveOutcome(SaveStatus.duplicate, key[0])
    dedup.seen.add(key)
    if key[0] != candidate["lemma"]:
        candidate = {**candidate, "lemma": key[0]}
    return SettledCandidate(candidate, resolved_lemma.in_dictionary)


async def store_note(
    session_factory: async_sessionmaker[AsyncSession],
    client: AsyncOpenAI,
    breaker: CallBreaker,
    model: str,
    user_id: int,
    deck_id: str,
    settled: SettledCandidate,
    dedup: DedupState,
    now: datetime,
    *,
    context: str | None = None,
    usage: Usage | None = None,
    resolve_forms: ResolveForms = resolve_note_forms,
) -> SaveOutcome:
    """Step 2: forms, schema, INSERT. `resolve_forms` is
    ingest.resolve_note_forms; /add passes its own module's name for it, the
    one its tests replace."""
    candidate = settled.candidate
    logger.debug("event=add.candidate lemma=%s kind=%s", candidate["lemma"], candidate["kind"])
    resolved = None
    if candidate["kind"] == "word":
        try:
            resolved, forms_usage = await resolve_forms(
                client,
                breaker,
                model,
                candidate["lemma"],
                candidate.get("pos"),
                now,
                context=context if context is not None else candidate.get("example_fi"),
            )
        except CircuitOpenError as error:
            return SaveOutcome(
                SaveStatus.failed, candidate["lemma"], reason=REASON_BREAKER, error=error
            )
        except openai.APIError as error:
            logger.exception("event=llm.error op=resolve_note_forms lemma=%s", candidate["lemma"])
            return SaveOutcome(
                SaveStatus.failed, candidate["lemma"], reason=REASON_OPENAI, error=error
            )
        if usage is not None:
            usage.add(forms_usage)

    # The FST may have corrected the part of speech the LLM guessed (see
    # ingest.resolve_note_pos: "tuli" is a noun, never the verb "tulla").
    # Save the corrected one - and re-run the dedup check with it, because
    # the key computed in settle_lemma still carried the LLM's answer.
    if resolved is not None and resolved.pos is not None and resolved.pos != candidate.get("pos"):
        candidate = {**candidate, "pos": resolved.pos}
        key = (candidate["lemma"], resolved.pos)
        if dedup.taken(key):
            logger.info(
                "event=add.duplicate_after_pos_fix deck=%s lemma=%s pos=%s",
                deck_id,
                key[0],
                key[1],
            )
            return SaveOutcome(SaveStatus.duplicate, key[0])
        dedup.seen.add(key)

    full_note = build_full_note(candidate, resolved)
    try:
        load_validator().validate(full_note)
    except jsonschema.ValidationError as error:
        logger.exception("event=ingest.schema_invalid lemma=%s", candidate.get("lemma"))
        return SaveOutcome(
            SaveStatus.failed, candidate["lemma"], reason=REASON_INVALID, error=error
        )

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
                deck_id=deck_id,
                meta=full_note["meta"],
            )
        )
        try:
            await session.commit()
        except IntegrityError as error:
            # Lost the race: another save of the same word in the same deck
            # committed between our dedup lookup and this insert. The other
            # note is just as good, so this is the same outcome as the lookup
            # having caught it - a duplicate, never a crash.
            if not is_note_duplicate_error(error):
                raise
            await session.rollback()
            logger.info(
                "event=add.duplicate_race deck=%s lemma=%s",
                deck_id,
                full_note["lemma"],
            )
            return SaveOutcome(SaveStatus.duplicate, full_note["lemma"])
    return SaveOutcome(
        SaveStatus.saved,
        full_note["lemma"],
        translation_ru=full_note["translation_ru"],
        in_dictionary=settled.in_dictionary,
    )


async def save_candidate(
    session_factory: async_sessionmaker[AsyncSession],
    client: AsyncOpenAI,
    breaker: CallBreaker,
    model: str,
    user_id: int,
    deck_id: str,
    candidate: dict,
    dedup: DedupState,
    now: datetime,
    *,
    context: str | None = None,
    usage: Usage | None = None,
) -> SaveOutcome:
    """Both steps for one candidate."""
    settled = await settle_lemma(
        client, breaker, model, candidate, dedup, now, context=context, usage=usage
    )
    if isinstance(settled, SaveOutcome):
        return settled
    return await store_note(
        session_factory,
        client,
        breaker,
        model,
        user_id,
        deck_id,
        settled,
        dedup,
        now,
        context=context,
        usage=usage,
    )
