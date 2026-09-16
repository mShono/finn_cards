"""Phase 3 (v2, conversational): turn a chat message into a reply plus note
candidates (plan 3.11, "Текст", revised - a chat instead of a fixed
one-by-one confirmation list).

Up to three separate LLM calls, all structured (strict: true) - the third
only fires for a genuinely ambiguous lemma:

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
3. choose_pos() - the same pattern one level up, for the part of speech
   itself. The FST says which parts of speech a lemma can have
   (morphology.pos_set_for_lemma); when the LLM's answer isn't one of them
   and there is more than one to choose from ("hakea" is both a verb and a
   noun), only the sentence can decide, so the LLM picks again from an enum
   of exactly the FST's set. See resolve_note_pos().
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from finn_cards.morphology import (
    FormsResult,
    forms_for_pos,
    generate_forms,
    lemmatize,
    pos_set_for_lemma,
)
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
POS_CHOICE_SCHEMA_NAME = "kielikaveri_pos_choice"


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
    # The part of speech the forms were actually generated for - the LLM's
    # own answer when the FST confirmed it, the FST's when it corrected it
    # (see resolve_note_pos). Callers persist this, not the LLM's `pos`.
    pos: str | None = None


def _load_note_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def _chat_schema(note_schema: dict) -> dict:
    note_strict = convert_to_strict(note_schema, "note", exclude=EXCLUDED_FIELDS)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["reply_ru", "needs_clarification", "candidates"],
        "properties": {
            "reply_ru": {"type": "string"},
            "needs_clarification": {"type": "boolean"},
            "candidates": {"type": "array", "items": note_strict},
        },
    }


def _chat_instructions(*, is_follow_up: bool) -> str:
    """`is_follow_up=True`: this call carries a student's answer to a clarifying
    question this same function asked on a previous turn (see build_chat_input()) -
    the model must act now, not ask a second time.
    """
    text = (
        INSTRUCTIONS_PATH.read_text() + "\n\n---\n\n"
        "Ты - разговорный ассистент по финскому в Telegram-боте. Один ход - "
        "одно сообщение ученика. Определи, что перед тобой, и веди себя по "
        "одному из трёх сценариев:\n\n"
        "1. **Сообщение уже содержит инструкцию** - свой перевод текста (на "
        "финский или с финского) на проверку, явная просьба перевести "
        "конкретное слово/фразу ('как будет...', 'переведи...'), или явная "
        "просьба добавить конкретные слова в карточки. **Упоминание колоды "
        "в этой просьбе ('добавь в колоду X', 'в колоду talkoot: ...') - "
        "не твоя часть работы, это не делает просьбу неясной.** Игнорируй "
        "название колоды при классификации и при ответе - колоду ученик "
        "выбирает сам через кнопки уже после этого ответа, ты её не "
        "выбираешь, не создаёшь и не называешь никогда. Действуй сразу: для "
        "перевода на проверку - перечисли неточности (что не так и почему) и "
        "дай исправленный вариант в `reply_ru`; для просьбы перевести - "
        "переведи именно названное; для просьбы добавить - в `reply_ru` "
        "напиши только что-то вроде «Добавляю» - никогда не пиши, что слова "
        "уже сохранены. `needs_clarification: false`. В `candidates` - "
        "карточки **только** по тем словам/фразам, которые ученик сам назвал "
        "или перевёл неточно - никогда не добавляй лишнюю лексику из текста "
        "'на всякий случай', даже если она выше уровня ученика.\n\n"
        "2. **Голый финский текст или фраза без инструкции** - непонятно, "
        "что с ним делать. Ничего не переводи и не разбирай. "
        "`needs_clarification: true`, `candidates: []`, а в `reply_ru` - "
        "короткий вопрос: пришлёт ли ученик свой перевод этого текста на "
        "проверку, или сам назовёт слова/фразы, которые перевести или "
        "добавить в карточки.\n\n"
        "3. **Обычный вопрос про язык, не про новый текст** - ответь на него "
        "в `reply_ru`. `needs_clarification: false`, `candidates: []`.\n\n"
        "**`reply_ru` обязан звучать так, как соответствует твоему же "
        "`needs_clarification`** - если `true`, `reply_ru` обязан быть "
        "вопросом (сценарий 2), а не утверждением о действии вроде "
        "«Добавляю»/«Хорошо»/«Сохранила»; если `false` - `reply_ru` не "
        "должен звучать как вопрос о том, что делать с уже понятным "
        "запросом. Проверяй это сама перед ответом - расхождение между "
        "`needs_clarification` и текстом `reply_ru` вводит ученика в "
        "заблуждение о том, что реально произошло.\n\n"
        "Важно для `lemma` и `example_fi` в любом кандидате: только финские "
        "слова и предложения. Никогда не подставляй английский или русский "
        "перевод вместо финской леммы (проверяй сам себя - лемма должна быть "
        "словом финского языка, а не его переводом)."
    )
    if is_follow_up:
        text += (
            "\n\n---\n\nЭто продолжение диалога: на предыдущем ходу ты уже "
            "получила сценарий 2 и спросила, что делать с текстом - входные "
            "данные ниже содержат исходный текст и ответ ученика на твой "
            "вопрос. Действуй по этому ответу как по сценарию 1 - "
            "`needs_clarification` обязан быть false, второй раз "
            "переспрашивать нельзя."
        )
    return text


def build_chat_input(text: str, context_text: str | None) -> str:
    """The literal model input for one turn - also the cache key basis (see
    hash_text() callers in bot/add.py), so this is the single place that
    combines a follow-up reply with the original text it answers.
    """
    if context_text is None:
        return text
    return (
        f"Исходный текст ученика:\n{context_text}\n\n"
        f"Ответ ученика на мой вопрос, что с ним сделать:\n{text}"
    )


def _chat_schema_fingerprint() -> str:
    """sha256 of the strict schema check_and_suggest() actually sends.

    Canonical JSON (sorted keys, no whitespace), so reformatting
    cards/schema.json alone doesn't invalidate the cache - only a change the
    model can see does, including one made through EXCLUDED_FIELDS.
    """
    canonical = json.dumps(
        _chat_schema(_load_note_schema()),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def hash_text(text: str, model: str, *, is_follow_up: bool = False) -> str:
    """Cache key for one chat turn.

    Folds in `model` and the exact instructions text, not just the student's
    input - a cache keyed on input alone never invalidates when the prompt or
    model changes (found live 27.08.2026: a prompt fix landed, but resending
    the same text kept replaying the old, already-fixed answer from cache,
    because check_and_suggest() was never called again for that text). This
    also closes the model-mismatch gap noted in the plan for the same reason -
    `store_cached_chat` already recorded `model`, but nothing compared it back.

    The response schema is folded in for the same reason: a cached candidate
    shaped by an older schema would otherwise be replayed and then rejected
    by the note validator on every resend (found 16.09.2026, when
    meta.cognates/meta.cefr were dropped from cards/schema.json).
    """
    composite = (
        f"{model}\n{_chat_schema_fingerprint()}\n"
        f"{_chat_instructions(is_follow_up=is_follow_up)}\n{text.strip()}"
    )
    return hashlib.sha256(composite.encode("utf-8")).hexdigest()


def _usage_from(response) -> TokenUsage:
    return TokenUsage(
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        total_tokens=response.usage.total_tokens,
    )


def _log_llm_request(op: str, model: str, **extra: object) -> None:
    """Every LLM call logs through here, never logger.debug() directly - the
    point is that op/model can't silently go missing at a future call site
    the way two hand-written logger calls eventually would (found while
    auditing: check_and_suggest and resolve_ambiguous_forms already agreed
    by coincidence, not by anything enforcing it)."""
    suffix = "".join(f" {key}={value}" for key, value in extra.items())
    logger.debug("event=llm.request op=%s model=%s%s", op, model, suffix)


def _log_llm_response(
    op: str, model: str, duration_ms: int, usage: TokenUsage, **extra: object
) -> None:
    """Counterpart to _log_llm_request() - op/model/duration_ms/token counts
    are always present and always in this order; extra is only for fields
    specific to one op (e.g. lemma, needs_clarification)."""
    suffix = "".join(f" {key}={value}" for key, value in extra.items())
    logger.info(
        "event=llm.response op=%s model=%s duration_ms=%d input_tokens=%d output_tokens=%d "
        "total_tokens=%d%s",
        op,
        model,
        duration_ms,
        usage.input_tokens,
        usage.output_tokens,
        usage.total_tokens,
        suffix,
    )


async def get_cached_chat(
    session: AsyncSession, text_hash: str
) -> tuple[str, bool, list[dict]] | None:
    cached = await session.get(IngestCache, text_hash)
    # reply_ru is None for rows written before this field existed (see
    # models.py) - treat those as a miss rather than replaying an empty reply.
    if cached is None or cached.reply_ru is None:
        return None
    # needs_clarification is None for rows written before that column existed -
    # those rows always came from a call that acted immediately, never asked
    # back, so False reproduces their actual behavior.
    return cached.reply_ru, bool(cached.needs_clarification), cached.candidates


async def store_cached_chat(
    session: AsyncSession,
    text_hash: str,
    model: str,
    reply_ru: str,
    needs_clarification: bool,
    candidates: list[dict],
) -> None:
    session.add(
        IngestCache(
            text_hash=text_hash,
            model=model,
            reply_ru=reply_ru,
            needs_clarification=needs_clarification,
            candidates=candidates,
        )
    )


async def check_and_suggest(
    client: AsyncOpenAI,
    breaker: CallBreaker,
    model: str,
    text: str,
    now: datetime,
    context_text: str | None = None,
) -> tuple[str, bool, list[dict], TokenUsage]:
    """One chat message in, a reply plus note candidates out. Raises CircuitOpenError via breaker.

    `context_text` is set only for a follow-up turn answering this function's
    own previous clarifying question - see build_chat_input() and
    _chat_instructions(is_follow_up=...).
    """
    breaker.check(now)
    _log_llm_request("check_and_suggest", model)
    start = time.monotonic()
    response = await client.responses.create(
        model=model,
        instructions=_chat_instructions(is_follow_up=context_text is not None),
        input=build_chat_input(text, context_text),
        text={
            "format": {
                "type": "json_schema",
                "name": CHAT_SCHEMA_NAME,
                "schema": _chat_schema(_load_note_schema()),
                "strict": True,
            }
        },
    )
    duration_ms = int((time.monotonic() - start) * 1000)
    payload = json.loads(response.output_text)
    usage = _usage_from(response)
    _log_llm_response(
        "check_and_suggest",
        model,
        duration_ms,
        usage,
        needs_clarification=payload["needs_clarification"],
        candidates=len(payload["candidates"]),
    )
    return payload["reply_ru"], payload["needs_clarification"], payload["candidates"], usage


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
    _log_llm_request("resolve_ambiguous_forms", model, lemma=lemma)
    start = time.monotonic()
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
    duration_ms = int((time.monotonic() - start) * 1000)
    chosen = json.loads(response.output_text)
    usage = _usage_from(response)
    _log_llm_response("resolve_ambiguous_forms", model, duration_ms, usage, lemma=lemma)
    return chosen, usage


async def choose_pos(
    client: AsyncOpenAI,
    breaker: CallBreaker,
    model: str,
    lemma: str,
    allowed: list[str],
    context: str | None,
    now: datetime,
) -> tuple[str, TokenUsage]:
    """Have the LLM pick one part of speech out of the FST's own set - never invent one.

    Same shape as resolve_ambiguous_forms(): the enum is built from exactly
    what the FST allows, so the schema itself makes an answer outside that
    set impossible. The difference is what is being decided - which reading
    of the lemma the sentence actually uses, which is a question about
    meaning that the FST has never seen the material for.
    """
    breaker.check(now)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["pos"],
        "properties": {"pos": {"type": "string", "enum": allowed}},
    }
    _log_llm_request("choose_pos", model, lemma=lemma)
    start = time.monotonic()
    response = await client.responses.create(
        model=model,
        instructions=(
            f"Лемма '{lemma}' по данным морфологического анализатора может быть "
            "несколькими частями речи - это разные слова, у которых совпало "
            "написание. Выбери ту часть речи, в которой слово употреблено в "
            "присланном предложении. Если предложения нет, выбери самое "
            "обычное для этой леммы значение. Схема ответа разрешает только "
            "значения из присланного списка."
        ),
        input=json.dumps(
            {"lemma": lemma, "allowed_pos": allowed, "example_fi": context or ""},
            ensure_ascii=False,
        ),
        text={
            "format": {
                "type": "json_schema",
                "name": POS_CHOICE_SCHEMA_NAME,
                "schema": schema,
                "strict": True,
            }
        },
    )
    duration_ms = int((time.monotonic() - start) * 1000)
    usage = _usage_from(response)
    _log_llm_response("choose_pos", model, duration_ms, usage, lemma=lemma)
    return json.loads(response.output_text)["pos"], usage


async def resolve_note_pos(
    client: AsyncOpenAI,
    breaker: CallBreaker,
    model: str,
    lemma: str,
    pos: str | None,
    context: str | None,
    now: datetime,
) -> tuple[str | None, TokenUsage | None]:
    """Reconcile the LLM's part of speech with the one(s) the FST allows for `lemma`.

    The invariant: the FST bounds the set of possible parts of speech, the
    LLM picks inside it. The FST is never allowed to *choose* - its readings
    all carry weight 0.0, so preferring one over another would be reading a
    ranking into an arbitrary order (plain `pos_set_for_lemma(lemma)` returns
    a set for exactly this reason).

    Four cases:

    * the FST knows nothing about `lemma` - keep the LLM's answer, nothing
      can confirm or refute it (unknown words must behave as before);
    * the LLM's answer is in the set - keep it;
    * the answer is outside the set and the set holds exactly one part of
      speech - take it. This is not guessing: there is nothing to choose
      between. "tuli" is the live case - the LLM sees "hän tuli kotiin" and
      answers "verbi", which belongs to the lemma "tulla", while the lemma
      "tuli" is only ever a noun;
    * the answer is outside the set and the set holds several - "hakea" is
      both a real verb and a real noun, and only the sentence says which -
      so ask the LLM again, constrained to the FST's set (choose_pos()).
    """
    if pos is None:
        return None, None

    allowed = pos_set_for_lemma(lemma)
    if not allowed:
        logger.debug("event=resolve_note_pos.unknown_lemma lemma=%s pos=%s", lemma, pos)
        return pos, None
    if pos in allowed:
        return pos, None

    if len(allowed) == 1:
        corrected = next(iter(allowed))
        logger.info(
            "event=resolve_note_pos.corrected lemma=%s llm_pos=%s pos=%s",
            lemma,
            pos,
            corrected,
        )
        return corrected, None

    logger.debug(
        "event=resolve_note_pos.ambiguous lemma=%s llm_pos=%s allowed=%s",
        lemma,
        pos,
        ",".join(sorted(allowed)),
    )
    chosen, usage = await choose_pos(client, breaker, model, lemma, sorted(allowed), context, now)
    if chosen not in allowed:
        # The strict enum should make this unreachable; if it ever happens,
        # fall back to the LLM's original answer rather than to an arbitrary
        # member of the set - generate_forms() then simply finds no forms and
        # the note degrades to forms_verified=False, which is honest.
        logger.warning(
            "event=resolve_note_pos.invalid_choice lemma=%s pos=%s chosen=%s",
            lemma,
            pos,
            chosen,
        )
        return pos, usage
    logger.info(
        "event=resolve_note_pos.chosen lemma=%s llm_pos=%s pos=%s allowed=%s",
        lemma,
        pos,
        chosen,
        ",".join(sorted(allowed)),
    )
    return chosen, usage


def _add_usage(first: TokenUsage | None, second: TokenUsage | None) -> TokenUsage | None:
    """One note can now cost two tie-break calls (pos, then forms) - report both."""
    if first is None:
        return second
    if second is None:
        return first
    return TokenUsage(
        input_tokens=first.input_tokens + second.input_tokens,
        output_tokens=first.output_tokens + second.output_tokens,
        total_tokens=first.total_tokens + second.total_tokens,
    )


async def resolve_note_forms(
    client: AsyncOpenAI,
    breaker: CallBreaker,
    model: str,
    lemma: str,
    pos: str | None,
    now: datetime,
    context: str | None = None,
) -> tuple[ResolvedForms, TokenUsage | None]:
    """FST first (finn_cards.morphology); LLM only breaks ties among real FST forms.

    `pos` arrives from the LLM and is checked against the FST before it is
    used - resolve_note_pos() does that, and this is the single production
    path into generate_forms(), so no caller can bypass the check. The part
    of speech the forms were really generated for comes back on
    ResolvedForms.pos; persist that one. `context` is the note's Finnish
    example sentence, needed only for a lemma the FST allows in several
    parts of speech.

    generate_forms()/forms_for_pos() only have a principal-forms table for
    verbi/substantiivi/adjektiivi and raise ValueError on anything else -
    including pos=None. The LLM's strict schema can still hand us either:
    note.pos allows all 11 cards/schema.json parts of speech, and the
    kind=word -> pos required rule lives in schema.json's `allOf`, which
    strict_schema._make_strict() drops (no strict-mode equivalent), so pos
    is nullable there regardless of kind. Treat both as "FST has nothing"
    rather than letting the ValueError crash the /add confirmation handler.
    """
    pos, usage = await resolve_note_pos(client, breaker, model, lemma, pos, context, now)
    try:
        result: FormsResult = generate_forms(lemma, pos)
    except ValueError:
        # Always forms_verified=False - a real degradation (unverified LLM
        # guess stands in for the FST), not a routine branch.
        logger.warning(
            "event=resolve_note_forms.fallback reason=no_fst_table lemma=%s pos=%s "
            "forms_source=llm",
            lemma,
            pos,
        )
        return ResolvedForms({}, "llm", False, pos), usage
    covered = result.principal_forms.keys() | result.ambiguous.keys()
    missing = set(forms_for_pos(pos)) - covered

    forms = dict(result.principal_forms)
    if result.ambiguous:
        # Routine, not a degradation: the FST did resolve every form, it just
        # returned several equally-weighted candidates for some of them - the
        # follow-up LLM call still only picks among real FST output.
        logger.debug(
            "event=resolve_note_forms.ambiguous lemma=%s pos=%s forms=%s",
            lemma,
            pos,
            list(result.ambiguous),
        )
        chosen, forms_usage = await resolve_ambiguous_forms(
            client, breaker, model, lemma, pos, result.ambiguous, now
        )
        forms.update(chosen)
        usage = _add_usage(usage, forms_usage)

    if missing:
        forms_source, forms_verified = "llm", False
        logger.warning(
            "event=resolve_note_forms.fallback reason=fst_missing lemma=%s pos=%s missing=%s",
            lemma,
            pos,
            sorted(missing),
        )
    elif result.ambiguous:
        forms_source, forms_verified = "fst+llm", True
    else:
        forms_source, forms_verified = "fst", True

    logger.debug(
        "event=resolve_note_forms.decision lemma=%s pos=%s forms_source=%s forms_verified=%s",
        lemma,
        pos,
        forms_source,
        forms_verified,
    )
    return ResolvedForms(forms, forms_source, forms_verified, pos), usage


def canonical_key(lemma: str, pos: str | None) -> tuple[str, str | None]:
    """Dedup key (plan phase 3): lemmatize the LLM's candidate, don't trust it as a lemma.

    `cards/instructions.md`: the LLM sometimes returns an inflected form as
    "lemma" (e.g. töitä instead of työ) - lemmatize() resolves that. `pos` is
    part of the key so homonyms with different parts of speech stay distinct
    (kuusi the noun "spruce" vs kuusi the numeral "six" - pos_set_for_lemma()
    is what says which of the two a note means). kind="pattern" has no real
    lemma to resolve - pos is None there, so the raw construction string is
    the key as-is.

    Only the lemma is settled here. The part of speech is passed through
    untouched and reconciled with the FST later, in resolve_note_pos().
    """
    if pos is None:
        return lemma, None
    lemmas = lemmatize(lemma)
    if lemma in lemmas:
        return lemma, pos
    return (lemmas[0], pos) if lemmas else (lemma, pos)


async def existing_note_keys(
    session: AsyncSession, user_id: int, deck_id: str | None = None
) -> set[tuple[str, str | None]]:
    """Dedup universe for one /add batch.

    Scoped to `deck_id` when given (plan: per-deck dedup, requested
    03.09.2026 - a word already in one deck should still be addable to
    another; the same word shouldn't live twice in the *same* deck).
    `deck_id=None` keeps the old global-per-user behaviour for callers that
    don't yet know which deck (there are none left in bot/add.py, but the
    parameter is optional rather than required so this isn't a breaking
    change for any other future caller).
    """
    stmt = select(Note.lemma, Note.pos).where(Note.user_id == user_id)
    if deck_id is not None:
        stmt = stmt.where(Note.deck_id == deck_id)
    rows = (await session.execute(stmt)).all()
    return {(lemma, pos) for lemma, pos in rows}


def _drop_nulls(value):
    """Recursively drop dict keys whose value is None.

    Strict-mode structured outputs can only express "optional" as a
    ["type", "null"] union (see strict_schema.py) - every optional field the
    LLM skipped (pos on a pattern candidate, rektio, source, ...)
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
