"""FST check of translated Finnish: which word forms does omorfi not know?

This is the translation-side counterpart of how /add uses the FST, but
after the fact: free text can't be restricted to FST candidates the way a
single card's forms are, so every word of the output is analyzed instead.
A word with no analysis at all is a strong hint of an invented form or a
broken compound - and nearly free to find (plan, "Что ловит наша
FST-проверка"). It says nothing about forms that exist but are wrong.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

from finn_cards.morphology import lemmatize

_WORD_RE = re.compile(r"[A-Za-zÅÄÖåäöŠšŽžÉé]+(?:[-:'’][A-Za-zÅÄÖåäöŠšŽžÉé]+)*")
# Names inflect ("Asjan", "Kostjalle") and the FST knows none of them - a
# capitalized word starting with a glossary name this long is not flagged.
_MIN_NAME_STEM = 3


_SPLIT_RE = re.compile(f"({_WORD_RE.pattern})")


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def split_words(text: str) -> list[str]:
    """`text` cut into [gap, word, gap, word, ..., gap]: the words sit at odd
    indices and are exactly words(text) - what the reader wraps in tappable
    spans, so a tap names the same word the FST check saw."""
    return _SPLIT_RE.split(text)


def _known(word: str, analyze: Callable[[str], list[str]]) -> bool:
    if analyze(word) or analyze(word.lower()):
        return True
    # "EU:n", "Moskova-joella": the FST may know the parts but not the whole.
    parts = [p for p in re.split(r"[-:'’]", word) if p]
    return len(parts) > 1 and all(analyze(p) or analyze(p.lower()) for p in parts)


def _is_name(word: str, names: Iterable[str]) -> bool:
    if not word[:1].isupper():
        return False
    lowered = word.lower()
    for name in names:
        # Drop the last letter: "Asja" -> "Asj" still matches "Asjan", "Asjalle".
        stem = name.lower()[: max(_MIN_NAME_STEM, len(name) - 1)]
        if len(name) >= _MIN_NAME_STEM and lowered.startswith(stem):
            return True
    return False


def unknown_words(
    text: str,
    names: Iterable[str] = (),
    allow: Iterable[str] = (),
    analyze: Callable[[str], list[str]] | None = None,
) -> list[str]:
    """Distinct words of `text` the FST has no reading for, in order of
    appearance. `names` are glossary names (inflected forms pass too);
    `allow` are exact words accepted as they are (foreign insertions)."""
    analyze = analyze or lemmatize
    names = list(names)
    allowed = {a.lower() for a in allow}
    found: list[str] = []
    for word in words(text):
        if word in found or word.lower() in allowed or _is_name(word, names):
            continue
        if not _known(word, analyze):
            found.append(word)
    return found


# What may stand right before a sentence-initial capital: nothing, sentence
# punctuation, a dialogue dash or an opening quote.
_SENTENCE_START = set('.!?…–—"”“«»:(')


def likely_names(text: str, flagged: Iterable[str]) -> list[str]:
    """The flagged words that occur capitalized in mid-sentence - most
    likely names or places the glossary didn't list (Pavlivka). A capital
    after a full stop proves nothing, so those words stay repairable."""
    result = []
    for word in flagged:
        if not word[:1].isupper():
            continue
        for match in re.finditer(rf"(?<![\w-]){re.escape(word)}(?![\w-])", text):
            before = text[: match.start()].rstrip()
            if before and before[-1] not in _SENTENCE_START:
                result.append(word)
                break
    return result
