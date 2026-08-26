"""Phase 3 (v2, conversational): turn a chat message into a reply plus note
candidates (plan 3.11, "Текст", revised - a chat instead of a fixed
one-by-one confirmation list).

Two separate LLM calls, both structured (strict: true):

1. check_and_suggest() - reads one chat message (pasted text, or a
   translation attempt) and returns a conversational Russian reply plus
   notes shaped like cards/schema.json's note def, minus the fields our own
   code fills in (id, principal_forms, forms_source, forms_verified,
   origin). The LLM never touches word forms - see resolve_note_forms().
2. resolve_note_forms() - plan 3.4: the FST (finn_cards.morphology) is the
   only thing allowed to produce a word form. When it resolves a form to
   several equally-weighted candidates ("ambiguous"), a second, separate LLM
   call picks the literary one - constrained by an enum built from exactly
   those FST candidates, so the schema itself makes it impossible for the
   model to return anything the FST didn't already generate.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from finn_cards.morphology import FormsResult, forms_for_pos, generate_forms, lemmatize
from finn_cards.strict_schema import convert_to_strict
from kielikaveri.db.models import IngestCache, Note
from kielikaveri.import_cards import SCHEMA_PATH
from kielikaveri.llm.breaker import CallBreaker

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTRUCTIONS_PATH = REPO_ROOT / "cards" / "instructions.md"

# Filled by our own code, never asked of the LLM - see cards/instructions.md's
# "Strict-схема для LLM" section. `origin` is added here on top of that list:
# phase 3 ingest is always origin="text" (phase 6's origin="error" is a
# different entry point), so there's nothing for the LLM to decide.
EXCLUDED_FIELDS = frozenset({"id", "principal_forms", "forms_source", "forms_verified", "origin"})

CHAT_SCHEMA_NAME = "kielikaveri_chat_reply"
FORM_CHOICE_SCHEMA_NAME = "kielikaveri_form_choice"


@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass
class ResolvedForms:
    principal_forms: dict[str, str]
    forms_source: str
    forms_verified: bool


def _load_note_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def _chat_schema(note_schema: dict) -> dict:
    note_strict = convert_to_strict(note_schema, "note", exclude=EXCLUDED_FIELDS)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["reply_ru", "candidates"],
        "properties": {
            "reply_ru": {"type": "string"},
            "candidates": {"type": "array", "items": note_strict},
        },
    }


def _chat_instructions() -> str:
    return (
        INSTRUCTIONS_PATH.read_text() + "\n\n---\n\n"
        "Ты - разговорный ассистент по финскому в Telegram-боте. Пользователь "
        "пишет одним сообщением одно из трёх: (1) финский текст, из которого "
        "нужно набрать карточки; (2) свой перевод текста (на финский или с "
        "финского) - сверь его с оригиналом или со смыслом и разбери "
        "неточности; (3) обычный вопрос про язык.\n\n"
        "В `reply_ru` - разговорный ответ на русском: для перевода - "
        "перечисли неточности (что не так и почему) и дай исправленный "
        "вариант; для текста - коротко скажи, что нашла; для вопроса - "
        "ответь на него. В `candidates` - карточки по правилам выше на ЛЮБУЮ "
        "лексику и конструкции в сообщении, которые могут быть незнакомы "
        "ученику или использованы неуверенно - не отсеивай агрессивно, лучше "
        "предложить лишнее, чем пропустить нужное. Пустой список, если "
        "предлагать нечего (например, чистый вопрос без нового текста)."
    )


def hash_text(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _usage_from(response) -> TokenUsage:
    return TokenUsage(
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        total_tokens=response.usage.total_tokens,
    )


async def get_cached_chat(session: AsyncSession, text_hash: str) -> tuple[str, list[dict]] | None:
    cached = await session.get(IngestCache, text_hash)
    # reply_ru is None for rows written before this field existed (see
    # models.py) - treat those as a miss rather than replaying an empty reply.
    if cached is None or cached.reply_ru is None:
        return None
    return cached.reply_ru, cached.candidates


async def store_cached_chat(
    session: AsyncSession, text_hash: str, model: str, reply_ru: str, candidates: list[dict]
) -> None:
    session.add(
        IngestCache(text_hash=text_hash, model=model, reply_ru=reply_ru, candidates=candidates)
    )


async def check_and_suggest(
    client: AsyncOpenAI,
    breaker: CallBreaker,
    model: str,
    text: str,
    now: datetime,
) -> tuple[str, list[dict], TokenUsage]:
    """One chat message in, a reply plus note candidates out. Raises CircuitOpenError via breaker."""
    breaker.check(now)
    response = await client.responses.create(
        model=model,
        instructions=_chat_instructions(),
        input=text,
        text={
            "format": {
                "type": "json_schema",
                "name": CHAT_SCHEMA_NAME,
                "schema": _chat_schema(_load_note_schema()),
                "strict": True,
            }
        },
    )
    payload = json.loads(response.output_text)
    usage = _usage_from(response)
    logger.info(
        "ingest.check_and_suggest model=%s input_tokens=%d output_tokens=%d total_tokens=%d "
        "candidates=%d",
        model,
        usage.input_tokens,
        usage.output_tokens,
        usage.total_tokens,
        len(payload["candidates"]),
    )
    return payload["reply_ru"], payload["candidates"], usage


async def resolve_ambiguous_forms(
    client: AsyncOpenAI,
    breaker: CallBreaker,
    model: str,
    lemma: str,
    pos: str,
    ambiguous: dict[str, list[str]],
    now: datetime,
) -> tuple[dict[str, str], TokenUsage]:
    """Have the LLM pick the literary form out of FST-generated candidates - never invent one."""
    breaker.check(now)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": list(ambiguous.keys()),
        "properties": {
            name: {"type": "string", "enum": candidates} for name, candidates in ambiguous.items()
        },
    }
    response = await client.responses.create(
        model=model,
        instructions=(
            f"Лемма '{lemma}' ({pos}). Для каждой формы ниже несколько кандидатов "
            "от морфологического анализатора для одной и той же формы. Выбери "
            "литературный вариант, не архаичный и не разговорный. Схема ответа "
            "разрешает только значения из присланного списка."
        ),
        input=json.dumps(ambiguous, ensure_ascii=False),
        text={
            "format": {
                "type": "json_schema",
                "name": FORM_CHOICE_SCHEMA_NAME,
                "schema": schema,
                "strict": True,
            }
        },
    )
    chosen = json.loads(response.output_text)
    usage = _usage_from(response)
    logger.info(
        "ingest.resolve_ambiguous_forms model=%s lemma=%s input_tokens=%d output_tokens=%d "
        "total_tokens=%d",
        model,
        lemma,
        usage.input_tokens,
        usage.output_tokens,
        usage.total_tokens,
    )
    return chosen, usage


async def resolve_note_forms(
    client: AsyncOpenAI,
    breaker: CallBreaker,
    model: str,
    lemma: str,
    pos: str | None,
    now: datetime,
) -> tuple[ResolvedForms, TokenUsage | None]:
    """FST first (finn_cards.morphology); LLM only breaks ties among real FST forms.

    generate_forms()/forms_for_pos() only have a principal-forms table for
    verbi/substantiivi/adjektiivi and raise ValueError on anything else -
    including pos=None. The LLM's strict schema can still hand us either:
    note.pos allows all 11 cards/schema.json parts of speech, and the
    kind=word -> pos required rule lives in schema.json's `allOf`, which
    strict_schema._make_strict() drops (no strict-mode equivalent), so pos
    is nullable there regardless of kind. Treat both as "FST has nothing"
    rather than letting the ValueError crash the /add confirmation handler.
    """
    try:
        result: FormsResult = generate_forms(lemma, pos)
    except ValueError:
        return ResolvedForms({}, "llm", False), None
    covered = result.principal_forms.keys() | result.ambiguous.keys()
    missing = set(forms_for_pos(pos)) - covered

    forms = dict(result.principal_forms)
    usage: TokenUsage | None = None
    if result.ambiguous:
        chosen, usage = await resolve_ambiguous_forms(
            client, breaker, model, lemma, pos, result.ambiguous, now
        )
        forms.update(chosen)

    if missing:
        forms_source, forms_verified = "llm", False
    elif result.ambiguous:
        forms_source, forms_verified = "fst+llm", True
    else:
        forms_source, forms_verified = "fst", True

    return ResolvedForms(forms, forms_source, forms_verified), usage


def canonical_key(lemma: str, pos: str | None) -> tuple[str, str | None]:
    """Dedup key (plan phase 3): lemmatize the LLM's candidate, don't trust it as a lemma.

    `cards/instructions.md`: the LLM sometimes returns an inflected form as
    "lemma" (e.g. töitä instead of työ) - lemmatize() resolves that. `pos` is
    part of the key so homonyms with different parts of speech stay distinct
    (kuusi "spruce" vs kuusi "six" - both nouns, but see detect_pos() for the
    general case). kind="pattern" has no real lemma to resolve - pos is None
    there, so the raw construction string is the key as-is.
    """
    if pos is None:
        return lemma, None
    lemmas = lemmatize(lemma)
    if lemma in lemmas:
        return lemma, pos
    return (lemmas[0], pos) if lemmas else (lemma, pos)


async def existing_note_keys(session: AsyncSession, user_id: int) -> set[tuple[str, str | None]]:
    rows = (
        await session.execute(select(Note.lemma, Note.pos).where(Note.user_id == user_id))
    ).all()
    return {(lemma, pos) for lemma, pos in rows}


def _drop_nulls(value):
    """Recursively drop dict keys whose value is None.

    Strict-mode structured outputs can only express "optional" as a
    ["type", "null"] union (see strict_schema.py) - every optional field the
    LLM skipped (pos on a pattern candidate, cognates, cefr, source, ...)
    comes back explicitly `null` rather than omitted. cards/schema.json
    itself doesn't allow null on most of those fields, only omission, so the
    two need reconciling before the result can validate against it.
    """
    if isinstance(value, dict):
        return {k: _drop_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_nulls(v) for v in value]
    return value


def build_full_note(
    candidate: dict,
    resolved: ResolvedForms | None,
) -> dict:
    """Fill in the fields excluded from the LLM's schema (see EXCLUDED_FIELDS)."""
    candidate = _drop_nulls(candidate)
    meta = candidate.get("meta", {})
    meta["origin"] = "text"
    if resolved is not None:
        meta["principal_forms"] = resolved.principal_forms
        meta["forms_source"] = resolved.forms_source
        meta["forms_verified"] = resolved.forms_verified
    else:
        meta["principal_forms"] = {}
        meta["forms_source"] = "llm"
        meta["forms_verified"] = False

    note = {
        "id": str(uuid.uuid4()),
        "lemma": candidate["lemma"],
        "translation_ru": candidate["translation_ru"],
        "example_fi": candidate["example_fi"],
        "example_ru": candidate["example_ru"],
        "kind": candidate["kind"],
        "meta": meta,
    }
    if "pos" in candidate:
        note["pos"] = candidate["pos"]
    return note
