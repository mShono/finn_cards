"""A tapped word, explained (plan 2026-11-20-kirjakaveri-reader, "П3").

The client names a word by its place only - chapter, paragraph, index in
fst_check.words(paragraph) - and the word, its sentence and the original's
paragraph are taken from the database (resolve_target). One strict-schema
OpenAI call gives the lemma, part of speech, a Russian breakdown of the form
and the word's translation in context. The sentence's translation is not
asked for: the Russian original is stored paragraph by paragraph next to the
Finnish (chapters.source_paragraphs), so the model only names which of the
original's numbered sentences the Finnish one renders, and those are shown -
the author's words, not a translation back. Then ingest.resolve_note_lemma
brings the lemma to the FST's, as /add does.

Colloquial words (puhekieli: mä, oon) get the literary lemma (minä, olla)
and the form as the text spells it in `puhekieli` (plan, "Открытые вопросы").

Results are cached in lookup_cache (lookup_key). Two identical taps in a row
make one OpenAI call: the second waits on the first's lock and finds the
cache filled.

Logs carry the chapter id, paragraph and word numbers, cache hits, tokens
and time - never the word, the sentence or the model's answer. That covers
the shared code called from here too: ingest logs the lemma and the surface
it resolves, so its records are cut to their event name while a lookup runs
(RedactBookText).
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

from openai import AsyncOpenAI
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kielikaveri.db.models import BookChapter, LookupCache
from kielikaveri.import_cards import SCHEMA_PATH
from kielikaveri.ingest import resolve_note_lemma
from kielikaveri.llm.breaker import CallBreaker
from kielikaveri.reader.sentences import sentence_of_word, split_sentences, word_span

logger = logging.getLogger(__name__)

# Marks the tapped word inside its sentence: the same form can stand twice
# in one sentence, and the model must know which one is meant.
MARK_OPEN = "⟦"
MARK_CLOSE = "⟧"

# Loggers of the shared code a lookup calls whose lines quote the word.
_REDACTED_LOGGERS = ("kielikaveri.ingest", "finn_cards.morphology")
_book_text_in_flight: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "reader_book_text_in_flight", default=False
)


class RedactBookText(logging.Filter):
    """While a lookup runs in this task (and only in it - /add in the bot
    is another task and keeps its full lines), cuts a record to its event
    name: "event=resolve_note_lemma.corrected (details hidden)"."""

    def filter(self, record: logging.LogRecord) -> bool:
        if _book_text_in_flight.get():
            words = str(record.msg).split(maxsplit=1)
            record.msg = f"{words[0] if words else ''} (details hidden)"
            record.args = None
            record.exc_info = None
            record.exc_text = None
        return True


def install_log_redaction() -> None:
    """Idempotent: make_app runs it, and tests build many apps."""
    for name in _REDACTED_LOGGERS:
        target = logging.getLogger(name)
        if not any(isinstance(f, RedactBookText) for f in target.filters):
            target.addFilter(RedactBookText())


class NoSuchWordError(Exception):
    """The paragraph or the word index is not in the chapter (422)."""


class NoOriginalError(NoSuchWordError):
    """The paragraph has no original to take the sentence from (422)."""


class LookupFailedError(Exception):
    """The model answered nothing usable: a refusal or broken JSON (502)."""


@dataclass(frozen=True)
class Target:
    chapter_id: str
    paragraph: int
    token: int
    surface: str
    # The Finnish sentence with the word between MARK_OPEN and MARK_CLOSE.
    marked_sentence: str
    source_paragraph: str
    source_sentences: list[str]
    fst_flagged: bool


def resolve_target(chapter: BookChapter, paragraph: int, token: int) -> Target:
    """The tapped word and its contexts, from the stored chapter only."""
    fi_paragraphs = chapter.fi_paragraphs or []
    if not 0 <= paragraph < len(fi_paragraphs):
        raise NoSuchWordError("no such paragraph")
    text = fi_paragraphs[paragraph]
    span = word_span(text, token)
    if span is None:
        raise NoSuchWordError("no such word")
    sentence = sentence_of_word(text, token)
    assert sentence is not None  # the word exists, so its sentence does
    start, end = sentence
    marked = (
        text[start : span[0]]
        + MARK_OPEN
        + text[span[0] : span[1]]
        + MARK_CLOSE
        + text[span[1] : end]
    )
    surface = text[span[0] : span[1]]

    # Every book so far is a translation, so every paragraph has its
    # original. A book uploaded in Finnish (no original) is a later stage:
    # it will need the model to translate the sentence instead.
    sources = chapter.source_paragraphs or []
    source = sources[paragraph] if paragraph < len(sources) else ""
    source_sentences = [source[s:e] for s, e in split_sentences(source)]
    if not source_sentences:
        raise NoOriginalError("no original paragraph")

    flagged = surface in (chapter.fst_flags or {}).get(str(paragraph), [])
    return Target(
        chapter.id,
        paragraph,
        token,
        surface,
        marked,
        source,
        source_sentences,
        flagged,
    )


# --- the model ---------------------------------------------------------------------

_POS_VALUES = json.loads(SCHEMA_PATH.read_text())["$defs"]["pos"]["enum"]

LOOKUP_SCHEMA_NAME = "reader_lookup"
LOOKUP_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "lemma",
        "pos",
        "form_ru",
        "translation_ru",
        "puhekieli",
        "source_sentences",
    ],
    "properties": {
        "lemma": {"type": "string"},
        "pos": {"type": "string", "enum": _POS_VALUES},
        "form_ru": {"type": "string"},
        "translation_ru": {"type": "string"},
        "puhekieli": {"type": ["string", "null"]},
        "source_sentences": {"type": "array", "items": {"type": "integer"}},
    },
}

LOOKUP_INSTRUCTIONS = f"""\
Ты помогаешь русскоязычной ученице (уровень B1-B2) читать финский перевод \
русской книги. Тебе присылают финское предложение, в котором одно слово \
отмечено скобками {MARK_OPEN} {MARK_CLOSE}, само это слово и пронумерованные \
предложения русского оригинала того же абзаца. Ответь по схеме:
- lemma: словарная форма отмеченного слова, литературная. Если слово \
разговорное (puhekieli: mä, sä, oon, tuu, mennää), lemma - литературная \
лемма (mä -> minä, oon -> olla).
- puhekieli: если слово разговорное, его форма ровно как в тексте; иначе null.
- pos: часть речи леммы.
- form_ru: разбор формы по-русски, коротко: падеж и число или время, лицо и \
наклонение, и смысл формы в кавычках. Пример для talossa: «инессив ед. ч., \
„в доме“». Для начальной формы так и напиши: «начальная форма».
- translation_ru: перевод слова в этом предложении. Если в русском оригинале \
есть слово, которое оно переводит, возьми его (в начальной форме).
- source_sentences: номера предложений оригинала, которые переводит финское \
предложение. Обычно одно, иногда два соседних.
"""


def _fingerprint() -> str:
    """The schema version in the cache key: whatever the model is shown."""
    canonical = json.dumps(
        {"instructions": LOOKUP_INSTRUCTIONS, "schema": LOOKUP_SCHEMA},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


_FINGERPRINT = _fingerprint()


def lookup_key(model: str, target: Target) -> str:
    composite = "\n".join(
        [
            model,
            _FINGERPRINT,
            target.surface,
            target.marked_sentence,
            hashlib.sha256(target.source_paragraph.encode("utf-8")).hexdigest(),
        ]
    )
    return hashlib.sha256(composite.encode("utf-8")).hexdigest()


@dataclass
class _Tokens:
    input: int = 0
    output: int = 0


async def _ask_model(
    client: AsyncOpenAI, breaker: CallBreaker, model: str, target: Target, tokens: _Tokens
) -> dict:
    breaker.check(datetime.now(UTC))
    payload = {
        "sentence_fi": target.marked_sentence,
        "word": target.surface,
        "original_sentences": [
            {"n": number, "ru": sentence} for number, sentence in enumerate(target.source_sentences)
        ],
    }
    response = await client.responses.create(
        model=model,
        instructions=LOOKUP_INSTRUCTIONS,
        input=json.dumps(payload, ensure_ascii=False),
        text={
            "format": {
                "type": "json_schema",
                "name": LOOKUP_SCHEMA_NAME,
                "schema": LOOKUP_SCHEMA,
                "strict": True,
            }
        },
    )
    tokens.input += response.usage.input_tokens
    tokens.output += response.usage.output_tokens
    output = (response.output_text or "").strip()
    if not output:
        raise LookupFailedError("empty response or refusal")
    try:
        answer = json.loads(output)
    except json.JSONDecodeError:
        raise LookupFailedError("response is not JSON") from None
    if not isinstance(answer, dict) or set(answer) != set(LOOKUP_SCHEMA["required"]):
        raise LookupFailedError("response does not match the schema")
    return answer


def _sentence_ru(target: Target, numbers: list) -> tuple[str, bool]:
    """The original's sentences the model named, in order; the whole
    original paragraph if it named none or one that isn't there. The bool
    says whether that fallback was taken."""
    count = len(target.source_sentences)
    chosen = sorted({n for n in numbers if isinstance(n, int) and not isinstance(n, bool)})
    if not chosen or any(not 0 <= n < count for n in chosen):
        return target.source_paragraph.strip(), True
    return " ".join(target.source_sentences[n] for n in chosen), False


async def _explain(
    client: AsyncOpenAI, breaker: CallBreaker, model: str, target: Target, tokens: _Tokens
) -> dict:
    answer = await _ask_model(client, breaker, model, target, tokens)
    puhekieli = answer["puhekieli"] or None
    sentence_ru, fallback = _sentence_ru(target, answer["source_sentences"])
    if fallback:
        logger.warning(
            "event=reader.lookup_source_fallback chapter_id=%s paragraph=%d token=%d "
            "named=%d sentences=%d",
            target.chapter_id,
            target.paragraph,
            target.token,
            len(answer["source_sentences"]),
            len(target.source_sentences),
        )
    plain_sentence = target.marked_sentence.replace(MARK_OPEN, "").replace(MARK_CLOSE, "")
    flag = _book_text_in_flight.set(True)
    try:
        # A colloquial spelling says nothing to the FST (or something
        # wrong): the literary lemma alone is resolved then.
        resolved, usage = await resolve_note_lemma(
            client,
            breaker,
            model,
            answer["lemma"],
            answer["pos"],
            None if puhekieli else target.surface,
            plain_sentence,
            datetime.now(UTC),
        )
    finally:
        _book_text_in_flight.reset(flag)
    if usage is not None:
        tokens.input += usage.input_tokens
        tokens.output += usage.output_tokens
    return {
        "lemma": resolved.lemma,
        "pos": answer["pos"],
        "in_dictionary": resolved.in_dictionary,
        "form_ru": answer["form_ru"],
        "translation_ru": answer["translation_ru"],
        "sentence_ru": sentence_ru,
        "puhekieli": puhekieli,
    }


class KeyedLocks:
    """One asyncio.Lock per key while anyone holds or waits for it - the
    double tap's guard. In this process's memory only: the bot is one process."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._users: dict[str, int] = {}

    @asynccontextmanager
    async def hold(self, key: str) -> AsyncIterator[None]:
        lock = self._locks.setdefault(key, asyncio.Lock())
        self._users[key] = self._users.get(key, 0) + 1
        try:
            async with lock:
                yield
        finally:
            self._users[key] -= 1
            if not self._users[key]:
                del self._users[key]
                del self._locks[key]

    def __len__(self) -> int:
        return len(self._locks)


async def lookup(
    session_factory: async_sessionmaker[AsyncSession],
    client: AsyncOpenAI,
    breaker: CallBreaker,
    model: str,
    locks: KeyedLocks,
    target: Target,
) -> dict:
    """The /api/lookup answer for `target`. Raises CircuitOpenError (and
    BudgetExceededError), LookupFailedError and OpenAI's errors."""
    key = lookup_key(model, target)
    started = time.monotonic()
    tokens = _Tokens()
    async with locks.hold(key):
        async with session_factory() as session:
            row = await session.get(LookupCache, key)
            cached = row.result if row is not None else None
        if cached is None:
            result = await _explain(client, breaker, model, target, tokens)
            await _store(session_factory, key, target, model, result)
        else:
            result = cached
    logger.info(
        "event=reader.lookup chapter_id=%s paragraph=%d token=%d cached=%s "
        "input_tokens=%d output_tokens=%d in_dictionary=%s fst_flagged=%s puhekieli=%s "
        "duration_ms=%d",
        target.chapter_id,
        target.paragraph,
        target.token,
        cached is not None,
        tokens.input,
        tokens.output,
        result["in_dictionary"],
        target.fst_flagged,
        result["puhekieli"] is not None,
        (time.monotonic() - started) * 1000,
    )
    return {
        "surface": target.surface,
        "lemma": result["lemma"],
        "pos": result["pos"],
        "in_dictionary": result["in_dictionary"],
        "fst_flagged": target.fst_flagged,
        "form_ru": result["form_ru"],
        "translation_ru": result["translation_ru"],
        "sentence_ru": result["sentence_ru"],
        "puhekieli": result["puhekieli"],
        "cached": cached is not None,
    }


async def _store(
    session_factory: async_sessionmaker[AsyncSession],
    key: str,
    target: Target,
    model: str,
    result: dict,
) -> None:
    async with session_factory() as session:
        try:
            await session.execute(
                insert(LookupCache)
                .values(key=key, chapter_id=target.chapter_id, model=model, result=result)
                .on_conflict_do_nothing()
            )
            await session.commit()
        except IntegrityError:
            # The book was deleted while the model was answering: the
            # answer is still given, just not kept.
            logger.info("event=reader.lookup_not_cached chapter_id=%s", target.chapter_id)
