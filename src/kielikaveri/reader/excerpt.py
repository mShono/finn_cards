"""The card's example: a piece of the book, shorter than its sentence (plan
2026-11-20-kirjakaveri-reader, "П4").

Made at ➕, never at a tap: one small strict-schema call is sent the Finnish
sentence, the tapped word and the Russian original's sentence(s) and answers
a short Finnish fragment with the word and the piece of the original it
renders. The model may only point, not write: both fragments must be
verbatim pieces of their sentences and the Finnish one must hold the tapped
word itself (the same occurrence, as a whole word). Anything else - or an
OpenAI error, an empty or broken answer - and the rule below is used.

The rule: the Finnish sentence is cut at , ; : – —, the part the word stands
in is taken (by the word's offset, not by searching for its spelling), and
while it has fewer than MIN_WORDS words a neighbouring part is added - the
next one, or the previous one at the sentence's end. The Russian example is
then the whole original sentence.

Either way the edges are cleaned: whitespace, dashes, quotes and the closing
punctuation are cut, the first letter is capitalized, and the example ends
with "." - or "?"/"!" when its sentence ends with one.

A tripped breaker is not worked around: it raises, and ➕ answers 503 as
every other reader call does.

Nothing here logs the text: only the outcome and token counts.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime

import openai
from openai import AsyncOpenAI

from kielikaveri.cards_save import Usage
from kielikaveri.llm.breaker import CallBreaker
from kielikaveri.reader.warmup import Warmup
from kielikaveri.translate.fst_check import words

logger = logging.getLogger(__name__)

MIN_WORDS = 3
_SEPARATORS = ",;:–—"
_LEADING_JUNK = " \t\n\r ,;:–—-«„“”\"'(["
_TRAILING_JUNK = ' \t\n\r .,;:–—-!?…”»"’)]'
_CLOSERS = ' \t\n\r ”»"’)]'

EXCERPT_SCHEMA_NAME = "reader_excerpt"
EXCERPT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["fragment_fi", "fragment_ru"],
    "properties": {
        "fragment_fi": {"type": "string"},
        "fragment_ru": {"type": "string"},
    },
}
EXCERPT_INSTRUCTIONS = """\
You make a flashcard example from a book. You get a Finnish sentence, one \
word of it, and the Russian original the sentence was translated from. \
Answer:
- fragment_fi: a short piece of the Finnish sentence, copied exactly, \
character for character, that holds the word and reads as a complete \
thought on its own (usually the clause with the word, 3-8 words). Do not \
change, add or reorder anything.
- fragment_ru: the piece of the Russian original that this Finnish piece \
renders, also copied exactly.
"""


@dataclass(frozen=True)
class Excerpt:
    example_fi: str
    example_ru: str
    # False: the rule made it (the model's answer was not used).
    from_model: bool


def _end_mark(sentence: str) -> str:
    tail = sentence.rstrip(_CLOSERS)
    if tail.endswith("?"):
        return "?"
    if tail.endswith("!"):
        return "!"
    return "."


def tidy(fragment: str, sentence: str) -> str:
    """Edges cleaned, first letter capitalized, the end mark of `sentence`."""
    core = fragment.lstrip(_LEADING_JUNK).rstrip(_TRAILING_JUNK)
    if not core:
        return ""
    return core[0].upper() + core[1:] + _end_mark(sentence)


def _covers_word(fragment: str, sentence: str, span: tuple[int, int], surface: str) -> bool:
    """`fragment` is a verbatim piece of `sentence` that holds the tapped
    word - at its own offset (`span`) and as a whole word."""
    if not fragment or surface not in words(fragment):
        return False
    start = sentence.find(fragment)
    while start != -1:
        if start <= span[0] and span[1] <= start + len(fragment):
            # A whole word at that place, not a piece of a longer one.
            inner = fragment[span[0] - start : span[1] - start]
            before = fragment[: span[0] - start]
            after = fragment[span[1] - start :]
            if inner == surface and not re.search(r"\w$", before) and not re.search(r"^\w", after):
                return True
        start = sentence.find(fragment, start + 1)
    return False


def fallback(sentence: str, span: tuple[int, int], sentence_ru: str) -> Excerpt:
    """The rule: the part of `sentence` between separators that holds the
    word, widened to MIN_WORDS words; the whole original."""
    parts: list[tuple[int, int]] = []
    start = 0
    for i, char in enumerate(sentence):
        if char in _SEPARATORS:
            parts.append((start, i))
            start = i + 1
    parts.append((start, len(sentence)))
    # Parts with no word (the dash opening a line of dialogue) don't count.
    parts = [p for p in parts if words(sentence[p[0] : p[1]])]
    index = next(
        (i for i, (s, e) in enumerate(parts) if s <= span[0] < e),
        None,
    )
    if index is None:  # unreachable: the word lies in a part with a word
        return Excerpt(tidy(sentence, sentence), tidy(sentence_ru, sentence_ru), False)
    first = last = index

    def count() -> int:
        return len(words(sentence[parts[first][0] : parts[last][1]]))

    while count() < MIN_WORDS and (first > 0 or last < len(parts) - 1):
        if last < len(parts) - 1:
            last += 1
        else:
            first -= 1
    piece = sentence[parts[first][0] : parts[last][1]]
    return Excerpt(tidy(piece, sentence), tidy(sentence_ru, sentence_ru), False)


async def make_excerpt(
    client: AsyncOpenAI,
    breaker: CallBreaker,
    model: str,
    sentence: str,
    span: tuple[int, int],
    sentence_ru: str,
    usage: Usage,
    warmup: Warmup | None = None,
) -> Excerpt:
    """The example for the word at `span` (offsets in `sentence`, the plain
    Finnish sentence). Raises CircuitOpenError; everything else falls back."""
    surface = sentence[span[0] : span[1]]
    breaker.check(datetime.now(UTC))
    if warmup is not None:
        warmup.touch()
    try:
        response = await client.responses.create(
            model=model,
            instructions=EXCERPT_INSTRUCTIONS,
            input=json.dumps(
                {"sentence_fi": sentence, "word": surface, "original_ru": sentence_ru},
                ensure_ascii=False,
            ),
            text={
                "format": {
                    "type": "json_schema",
                    "name": EXCERPT_SCHEMA_NAME,
                    "schema": EXCERPT_SCHEMA,
                    "strict": True,
                }
            },
        )
    except openai.OpenAIError as exc:
        logger.warning("event=reader.excerpt_fallback reason=error error=%s", type(exc).__name__)
        return fallback(sentence, span, sentence_ru)
    finally:
        if warmup is not None:
            warmup.touch()
    usage_info = getattr(response, "usage", None)
    if usage_info is not None:
        usage.input += usage_info.input_tokens
        usage.output += usage_info.output_tokens
    try:
        answer = json.loads((response.output_text or "").strip())
    except json.JSONDecodeError:
        answer = None
    if not isinstance(answer, dict) or not all(
        isinstance(answer.get(name), str) for name in EXCERPT_SCHEMA["required"]
    ):
        logger.warning("event=reader.excerpt_fallback reason=format")
        return fallback(sentence, span, sentence_ru)
    fragment_fi = answer["fragment_fi"].strip()
    fragment_ru = answer["fragment_ru"].strip()
    if not _covers_word(fragment_fi, sentence, span, surface):
        logger.info("event=reader.excerpt_fallback reason=fi_not_verbatim")
        return fallback(sentence, span, sentence_ru)
    if not words_ru(fragment_ru) or fragment_ru not in sentence_ru:
        logger.info("event=reader.excerpt_fallback reason=ru_not_verbatim")
        return fallback(sentence, span, sentence_ru)
    example_fi = tidy(fragment_fi, sentence)
    example_ru = tidy(fragment_ru, sentence_ru)
    if not example_fi or not example_ru:
        logger.info("event=reader.excerpt_fallback reason=empty")
        return fallback(sentence, span, sentence_ru)
    return Excerpt(example_fi, example_ru, True)


def words_ru(text: str) -> list[str]:
    """Any letters at all (Cyrillic included) - words() knows only Latin."""
    return re.findall(r"\w+", text)
