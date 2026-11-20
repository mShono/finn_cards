"""Translation pipeline: book prep -> block translation -> FST repair -> review -> edit.

Each step is a separate call shape so the P0 spike can switch steps on and
off and compare what a single pass gives against translation + review
(plan, "П0"). Every step returns paragraphs keyed by the source paragraph's
id - the 1:1 alignment is checked in code after every call, because a
silently dropped paragraph is the typical failure of LLM book translation
(TransAgents, plan "Как это делают другие").

Book text never goes to the log: lines carry ids, counts, durations and
token usage only (plan, "Логи").
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from openai import AsyncOpenAI

from kielikaveri.llm.breaker import CallBreaker
from kielikaveri.translate.chapters import Book, Chapter
from kielikaveri.translate.fst_check import likely_names, unknown_words

logger = logging.getLogger(__name__)

BLOCK_CHARS = 6000
# Paragraphs of already translated Finnish passed along for continuity only.
CONTEXT_PARAGRAPHS = 2
SAMPLE_CHARS_PER_CHAPTER = 1500

REVIEW_CATEGORIES = [
    "rektio",  # wrong case government
    "muoto",  # existing but wrong form
    "kalkki",  # calque from the source language
    "sanajarjestys",
    "rekisteri",  # register doesn't match the source
    "merkitys",  # meaning distorted
    "puute",  # something left out or added
    "muu",
]


class TranslationError(Exception):
    pass


class RefusedError(TranslationError):
    """The model returned no text - typically a refusal (explicit scenes,
    violence). Plan: a refusal on this book is a blocker for the model."""


class MisalignedError(TranslationError):
    pass


@dataclass
class StepUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0

    def add(self, response, seconds: float) -> None:
        self.calls += 1
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens
        self.seconds += seconds


@dataclass
class Usage:
    steps: dict[str, StepUsage] = field(default_factory=dict)

    def step(self, name: str) -> StepUsage:
        return self.steps.setdefault(name, StepUsage())


@dataclass
class BookGuide:
    """Glossary and style guide - prepared once per book, sent with every block."""

    characters: list[dict]
    terms: list[dict]
    style_notes: str

    @property
    def names_fi(self) -> list[str]:
        names = [c["name_fi"] for c in self.characters] + [
            t["fi"] for t in self.terms if t["fi"][:1].isupper()
        ]
        return [word for name in names for word in name.split()]

    def as_prompt(self) -> str:
        return json.dumps(
            {"characters": self.characters, "terms": self.terms, "style_notes": self.style_notes},
            ensure_ascii=False,
        )


@dataclass
class Issue:
    paragraph_id: int
    category: str
    quote_fi: str
    explanation_ru: str


@dataclass
class ChapterResult:
    chapter: Chapter
    draft: dict[int, str] = field(default_factory=dict)  # after translate
    after_fst: dict[int, str] = field(default_factory=dict)
    final: dict[int, str] = field(default_factory=dict)
    fst_flags: dict[int, list[str]] = field(default_factory=dict)  # on the draft
    fst_sent: dict[int, list[str]] = field(default_factory=dict)  # sent to repair
    fst_flags_final: dict[int, list[str]] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)


# Shared by every step: a reviewer that doesn't know them "fixes" what the
# translator did on purpose (found in the first P0 run, 20.11.2026: review
# turned a correct "(ukrainaksi)" back into Cyrillic).
HOUSE_RULES = """\
- Смысл, ритм и интонацию передавай полностью. Ничего не пропускай, не
  добавляй и не смягчай. Откровенные сцены, насилие и мат - часть
  художественного текста; переводи их с тем же уровнем откровенности и
  грубости, что в оригинале.
- Регистр следует оригиналу. Литературный текст - литературный финский
  (kirjakieli). Разговорная речь, жаргон, мат - разговорный финский
  (puhekieli, slangi) того же уровня. Ломаная речь иностранца - такой же
  ломаный финский. Как говорит каждый персонаж - в глоссарии.
- Реплики на украинском переводи на финский и ставь после реплики
  пометку «(ukrainaksi)». Если в тексте обсуждается сам язык (звуки,
  слова как примеры), слова-примеры оставь как в оригинале.
- Английские вставки в оригинале оставляй по-английски. Прочие
  иноязычные слова транслитерируй по-фински.
- Стихи переводи стихами, сохраняя размер и рифму, если это возможно.
- Имена и названия - строго по глоссарию; новые русские имена
  транслитерируй по финскому стандарту (SFS 4900: Ася - Asja,
  Далецкая - Daletskaja).
- Прямая речь - по финским правилам. Реплика начинается с тире «–»,
  слова автора отделяются только запятой, без второго тире:
  «– Tulkaa ulos, Sozantšuk huusi.» Если реплика продолжается после слов
  автора, перед продолжением снова тире: «– Tulkaa, hän sanoi. – Heti.»
  Русского оформления «..., – сказал он» в финском нет.
"""

TRANSLATOR_RULES = f"""\
Ты профессиональный литературный переводчик на финский язык. Переводишь
роман для издания: живой, естественный современный финский, каким пишут
финские авторы, а не подстрочник.

Правила:
{HOUSE_RULES}- Каждый абзац оригинала - ровно один абзац перевода с тем же id.
  Поле previous_fi - уже готовый перевод перед этим фрагментом, только
  для связности; его не переводи и не возвращай.
"""

_RULES_FOR_EDITORS = f"""
Перевод этой книги делается по таким правилам - не исправляй то, что им
соответствует, и исправляй то, что их нарушает:
{HOUSE_RULES}"""

FST_REPAIR_RULES = (
    """\
Ты финский литературный редактор. В каждом присланном абзаце перевода
морфологический анализатор не узнал слова из списка unknown_words.
Скорее всего, это несуществующие формы или неудачные сращения, но может
быть и редкое, но правильное слово. Исправь абзац так, чтобы в нём были
только существующие финские слова, не меняя смысла, стиля и регистра.
Если слово правильное (разговорная форма, редкое слово, слово-пример из
рассуждений о языке), оставь его. Верни каждый присланный абзац с его id.
"""
    + _RULES_FOR_EDITORS
)

REVIEW_RULES = (
    """\
Ты редактор литературных переводов с русского на финский. Сравни оригинал
и перевод по абзацам и найди ошибки: неправильное управление (rektio),
существующую, но неверную форму, кальки, неестественный порядок слов,
неверный регистр, искажения смысла, пропуски и добавления. Мелкие
вкусовые правки не нужны - только то, что финский редактор исправил бы
обязательно. Для каждого абзаца с ошибками верни исправленный абзац
целиком. Абзацы без ошибок не возвращай. Пояснения пиши по-русски.
"""
    + _RULES_FOR_EDITORS
)

EDIT_RULES = (
    """\
Ты финский литературный редактор. Оригинала у тебя нет - только перевод.
Сделай текст естественным для финского читателя: убери кальки и
неуклюжие обороты, не меняя смысла, событий, имён и регистра речи
персонажей. Каждый абзац - ровно один абзац с тем же id.
"""
    + _RULES_FOR_EDITORS
)

GUIDE_RULES = """\
Перед литературным переводом романа на финский составь для переводчика
глоссарий и стайлгайд по фрагментам книги:
- персонажи: русское имя (как в тексте), финская транслитерация по
  SFS 4900, как и каким языком говорит персонаж и каким финским
  регистром это передавать (kirjakieli, puhekieli, slangi, ломаный
  финский и т.п.);
- термины, названия, реалии с финским вариантом;
- общие заметки о стиле автора и о том, как их сохранить по-фински.
"""


def _paragraphs_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "paragraphs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}, "fi": {"type": "string"}},
                    "required": ["id", "fi"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["paragraphs"],
        "additionalProperties": False,
    }


def _review_schema() -> dict:
    issue = {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "category": {"type": "string", "enum": REVIEW_CATEGORIES},
            "quote_fi": {"type": "string"},
            "explanation_ru": {"type": "string"},
        },
        "required": ["id", "category", "quote_fi", "explanation_ru"],
        "additionalProperties": False,
    }
    fixed = {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "fi": {"type": "string"}},
        "required": ["id", "fi"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "issues": {"type": "array", "items": issue},
            "fixed": {"type": "array", "items": fixed},
        },
        "required": ["issues", "fixed"],
        "additionalProperties": False,
    }


def _guide_schema() -> dict:
    character = {
        "type": "object",
        "properties": {
            "name_ru": {"type": "string"},
            "name_fi": {"type": "string"},
            "speech": {"type": "string"},
            "register_fi": {"type": "string"},
        },
        "required": ["name_ru", "name_fi", "speech", "register_fi"],
        "additionalProperties": False,
    }
    term = {
        "type": "object",
        "properties": {
            "ru": {"type": "string"},
            "fi": {"type": "string"},
            "note": {"type": "string"},
        },
        "required": ["ru", "fi", "note"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "characters": {"type": "array", "items": character},
            "terms": {"type": "array", "items": term},
            "style_notes": {"type": "string"},
        },
        "required": ["characters", "terms", "style_notes"],
        "additionalProperties": False,
    }


def _refusal_text(response) -> str:
    for item in getattr(response, "output", None) or []:
        for part in getattr(item, "content", None) or []:
            if getattr(part, "type", None) == "refusal":
                return getattr(part, "refusal", "") or ""
    return ""


async def _call(
    client: AsyncOpenAI,
    breaker: CallBreaker,
    model: str,
    usage: StepUsage,
    op: str,
    instructions: str,
    payload: dict,
    schema_name: str,
    schema: dict,
) -> dict:
    breaker.check(datetime.now(UTC))
    logger.debug("event=llm.request op=%s model=%s", op, model)
    start = time.monotonic()
    response = await client.responses.create(
        model=model,
        instructions=instructions,
        input=json.dumps(payload, ensure_ascii=False),
        text={
            "format": {"type": "json_schema", "name": schema_name, "schema": schema, "strict": True}
        },
    )
    seconds = time.monotonic() - start
    usage.add(response, seconds)
    logger.info(
        "event=llm.response op=%s model=%s duration_ms=%d input_tokens=%d output_tokens=%d",
        op,
        model,
        int(seconds * 1000),
        response.usage.input_tokens,
        response.usage.output_tokens,
    )
    if not (response.output_text or "").strip():
        logger.warning("event=translate.refused op=%s", op)
        raise RefusedError(_refusal_text(response) or "empty response")
    return json.loads(response.output_text)


def blocks(paragraphs: list[str], max_chars: int = BLOCK_CHARS) -> list[list[int]]:
    """Paragraph ids grouped into blocks of about max_chars; a paragraph is never split."""
    result: list[list[int]] = []
    current: list[int] = []
    size = 0
    for pid, paragraph in enumerate(paragraphs):
        if current and size + len(paragraph) > max_chars:
            result.append(current)
            current, size = [], 0
        current.append(pid)
        size += len(paragraph)
    if current:
        result.append(current)
    return result


def _aligned(payload: dict, expected: list[int]) -> dict[int, str]:
    got = {p["id"]: p["fi"] for p in payload["paragraphs"]}
    if sorted(got) != sorted(expected) or len(payload["paragraphs"]) != len(expected):
        missing = sorted(set(expected) - set(got))
        extra = sorted(set(got) - set(expected))
        raise MisalignedError(f"paragraph ids differ: missing={missing} extra={extra}")
    if any(not text.strip() for text in got.values()):
        raise MisalignedError("empty paragraph in translation")
    return got


async def prepare_guide(
    client: AsyncOpenAI,
    breaker: CallBreaker,
    model: str,
    usage: Usage,
    book: Book,
    full_chapters: list[Chapter],
) -> BookGuide:
    """Glossary from the opening of every chapter plus the chapters about
    to be translated in full - the whole book in one prompt is mostly
    paying for text the guide doesn't need."""
    full_numbers = {c.number for c in full_chapters}
    fragments = []
    for chapter in book.chapters:
        text = "\n".join(chapter.paragraphs)
        if chapter.number not in full_numbers:
            text = text[:SAMPLE_CHARS_PER_CHAPTER]
        fragments.append({"chapter": chapter.number, "part": chapter.part, "text": text})
    payload = await _call(
        client,
        breaker,
        model,
        usage.step("guide"),
        "translate.guide",
        GUIDE_RULES,
        {"front_matter": book.front_matter, "fragments": fragments},
        "book_guide",
        _guide_schema(),
    )
    return BookGuide(payload["characters"], payload["terms"], payload["style_notes"])


async def translate_chapter(
    client: AsyncOpenAI,
    breaker: CallBreaker,
    model: str,
    usage: Usage,
    guide: BookGuide,
    chapter: Chapter,
    *,
    fst: bool = True,
    review: bool = True,
    edit: bool = False,
    block_chars: int = BLOCK_CHARS,
) -> ChapterResult:
    result = ChapterResult(chapter=chapter)
    source = chapter.paragraphs
    chapter_blocks = blocks(source, block_chars)

    for ids in chapter_blocks:
        previous = [result.draft[i] for i in range(max(0, ids[0] - CONTEXT_PARAGRAPHS), ids[0])]
        payload = {
            "glossary": json.loads(guide.as_prompt()),
            "previous_fi": previous,
            "paragraphs": [{"id": i, "text": source[i]} for i in ids],
        }
        try:
            answer = await _call(
                client,
                breaker,
                model,
                usage.step("translate"),
                "translate.block",
                TRANSLATOR_RULES,
                payload,
                "translation",
                _paragraphs_schema(),
            )
            result.draft.update(_aligned(answer, ids))
        except MisalignedError:
            # One retry: a dropped paragraph is usually not repeated.
            answer = await _call(
                client,
                breaker,
                model,
                usage.step("translate"),
                "translate.block_retry",
                TRANSLATOR_RULES,
                payload,
                "translation",
                _paragraphs_schema(),
            )
            result.draft.update(_aligned(answer, ids))
        logger.info(
            "event=translate.block chapter=%d first_id=%d paragraphs=%d",
            chapter.number,
            ids[0],
            len(ids),
        )

    names = guide.names_fi
    current = dict(result.draft)
    for pid, text in current.items():
        flagged = unknown_words(text, names)
        if flagged:
            result.fst_flags[pid] = flagged

    if fst:
        for ids in chapter_blocks:
            # Capitalized mid-sentence = a name the glossary missed (Pavlivka);
            # reported, but a repair call can't make the FST know it.
            to_repair = {}
            for pid in ids:
                flagged = result.fst_flags.get(pid, [])
                names_here = set(likely_names(current[pid], flagged))
                repairable = [word for word in flagged if word not in names_here]
                if repairable:
                    to_repair[pid] = repairable
            if not to_repair:
                continue
            result.fst_sent.update(to_repair)
            answer = await _call(
                client,
                breaker,
                model,
                usage.step("fst_repair"),
                "translate.fst_repair",
                FST_REPAIR_RULES,
                {
                    "paragraphs": [
                        {"id": pid, "ru": source[pid], "fi": current[pid], "unknown_words": words}
                        for pid, words in to_repair.items()
                    ]
                },
                "repair",
                _paragraphs_schema(),
            )
            for fixed in answer["paragraphs"]:
                if fixed["id"] in to_repair and fixed["fi"].strip():
                    current[fixed["id"]] = fixed["fi"]
    result.after_fst = dict(current)

    if review:
        for ids in chapter_blocks:
            answer = await _call(
                client,
                breaker,
                model,
                usage.step("review"),
                "translate.review",
                REVIEW_RULES,
                {
                    "glossary": json.loads(guide.as_prompt()),
                    "paragraphs": [{"id": i, "ru": source[i], "fi": current[i]} for i in ids],
                },
                "review",
                _review_schema(),
            )
            for issue in answer["issues"]:
                if issue["id"] in ids:
                    result.issues.append(
                        Issue(
                            issue["id"],
                            issue["category"],
                            issue["quote_fi"],
                            issue["explanation_ru"],
                        )
                    )
            for fixed in answer["fixed"]:
                if fixed["id"] in ids and fixed["fi"].strip():
                    current[fixed["id"]] = fixed["fi"]

    if edit:
        for ids in chapter_blocks:
            answer = await _call(
                client,
                breaker,
                model,
                usage.step("edit"),
                "translate.edit",
                EDIT_RULES,
                {"paragraphs": [{"id": i, "fi": current[i]} for i in ids]},
                "edit",
                _paragraphs_schema(),
            )
            current.update(_aligned(answer, ids))

    result.final = current
    for pid, text in current.items():
        flagged = unknown_words(text, names)
        if flagged:
            result.fst_flags_final[pid] = flagged
    return result
