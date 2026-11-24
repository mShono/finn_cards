"""Paragraphs cut into sentences (plan 2026-11-20-kirjakaveri-reader, "П3").

A tap's context is the sentence, not the paragraph: a paragraph can run for
half a page. Both sides are cut - the Finnish one (the sentence the tapped
word stands in) and the Russian original (numbered for the model to point
at, see reader/lookup.py) - by the same rules, since the book's punctuation
is the same kind on both:

- a sentence ends at . ! ? or an ellipsis (… or ...), with any closing
  quotes or brackets right after it;
- only if what follows the whitespace starts a sentence: a capital letter, a
  digit, or a dialogue dash (– or —) or an opening quote followed by a
  capital. "– Minä… minä en tiedä." and "«Да?» — спросил он." stay one
  sentence: after the break the text goes on in lowercase;
- never after an abbreviation that does not end sentences (esim., n., ts.,
  т.е., им., г., ...) or a single capital initial (J. K. Rowling, А. С.
  Пушкин). "т.д." and "т.п." usually do end one, so a capital after them
  splits.

Plain functions over strings and offsets: nothing here logs.
"""

from __future__ import annotations

import re

from kielikaveri.translate.fst_check import split_words

_CLOSERS = '”»"’)]'
_OPENERS = "«„“”\"'(["
_DASHES = "–—-"
# A sentence-ending punctuation run with the closers after it, then the
# whitespace before the next sentence.
_BREAK_RE = re.compile(rf"(?:\.\.\.|[.!?…])+[{re.escape(_CLOSERS)}]*(?=\s)")
# Lowercased; matched against the word right before the final dot.
_NO_BREAK_ABBREVIATIONS = frozenset(
    {
        # Finnish
        "esim",
        "n",
        "ns",
        "ts",
        "mm",
        "ym",
        "ks",
        "vrt",
        "ko",
        "klo",
        "tri",
        "prof",
        "s",
        "v",
        "nk",
        "huom",
        # Russian
        "им",
        "г",
        "гг",
        "ул",
        "д",
        "см",
        "стр",
        "проф",
        "акад",
        "св",
        "напр",
        "тов",
        "др",
    }
)
# Two-letter dotted ones: "т.е." / "т. е." (то есть), "т.к.", "т.н." -
# the last word is "е", "к", "н", so they are told by the "т." before it.
_DOTTED_NO_BREAK_RE = re.compile(r"(?:^|[\s(«\"„])т\.\s?[екн]$", re.IGNORECASE)
# "т.д." and "т.п." end a sentence as often as not - their "д" must not be
# taken for "д." (дом).
_DOTTED_MAY_BREAK_RE = re.compile(r"(?:^|[\s(«\"„])т\.\s?[дп]$", re.IGNORECASE)
_LAST_WORD_RE = re.compile(r"(\w+)$")


def _starts_sentence(text: str, pos: int) -> bool:
    """Whether text from `pos` (past the whitespace) opens a new sentence."""
    i = pos
    if i < len(text) and text[i] in _DASHES:
        i += 1
        while i < len(text) and text[i].isspace():
            i += 1
    while i < len(text) and text[i] in _OPENERS:
        i += 1
    if i >= len(text):
        return False
    return text[i].isupper() or (i == pos and text[i].isdigit())


def _ends_with_abbreviation(before: str) -> bool:
    """`before` is the text up to (not including) a single final dot."""
    if _DOTTED_NO_BREAK_RE.search(before):
        return True
    if _DOTTED_MAY_BREAK_RE.search(before):
        return False
    match = _LAST_WORD_RE.search(before)
    if match is None:
        return False
    word = match.group(1)
    if len(word) == 1 and word.isupper():
        return True  # an initial
    return word.lower() in _NO_BREAK_ABBREVIATIONS


def split_sentences(text: str) -> list[tuple[int, int]]:
    """(start, end) offsets of each sentence of `text`, whitespace between
    them left out. Empty or blank text gives []."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _BREAK_RE.finditer(text):
        end = match.end()
        punctuation = match.group(0).rstrip(_CLOSERS)
        if punctuation == "." and _ends_with_abbreviation(text[: match.start()]):
            continue
        after = end
        while after < len(text) and text[after].isspace():
            after += 1
        if not _starts_sentence(text, after):
            continue
        spans.append((start, end))
        start = after
    spans.append((start, len(text)))
    return [_strip(text, s, e) for s, e in spans if text[s:e].strip()]


def _strip(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def sentences(text: str) -> list[str]:
    return [text[s:e] for s, e in split_sentences(text)]


def word_span(text: str, index: int) -> tuple[int, int] | None:
    """Offsets of words(text)[index] - computed from split_words(), so it is
    exactly the word the reader's data-i names. None if there is no such word."""
    pieces = split_words(text)
    word_count = len(pieces) // 2
    if not 0 <= index < word_count:
        return None
    offset = sum(len(piece) for piece in pieces[: 2 * index + 1])
    return offset, offset + len(pieces[2 * index + 1])


def sentence_of_word(text: str, index: int) -> tuple[int, int] | None:
    """(start, end) of the sentence words(text)[index] stands in, or None if
    there is no such word."""
    span = word_span(text, index)
    if span is None:
        return None
    for start, end in split_sentences(text):
        if start <= span[0] < end:
            return start, end
    return None  # unreachable: every word lies in some sentence
