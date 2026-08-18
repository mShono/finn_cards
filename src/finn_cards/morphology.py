"""Wrapper over uralicNLP. The FST generates word forms; nothing here invents them.

generate_forms() never returns an unverified guess: if the FST gives one
candidate, that candidate is used; if it gives several, they are reported
as ambiguous rather than silently taking the first one - the first candidate
uralicNLP returns is often archaic or colloquial (e.g. "oon" before "olen").
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from uralicNLP import uralicApi

LANG = "fin"

# uralicNLP caches loaded FST models in module-level dicts with no lock
# (check-then-set on generator_cache/analyzer_cache). Two concurrent calls
# before the cache is warm both load the same model - each load peaks at
# ~260MB, which is significant on a small VPS. Serialize our own calls into
# the library so a cold cache only pays that cost once.
_FST_LOCK = threading.Lock()

POS_TAG = {
    "verbi": "+V",
    "substantiivi": "+N",
    "adjektiivi": "+A",
}

VERB_FORMS = {
    "preesens_1s": "+Act+Ind+Prs+Sg1",
    "preesens_3s": "+Act+Ind+Prs+Sg3",
    "imperfekti_3s": "+Act+Ind+Prt+Sg3",
    "nut_partisiippi": "+Act+PrfPrc+Sg+Nom",
    "passiivi": "+Pss+Ind+Prs+Pe4",
}

NOMINAL_FORMS = {
    "nominatiivi": "+Sg+Nom",
    "genetiivi": "+Sg+Gen",
    "partitiivi": "+Sg+Par",
    "illatiivi": "+Sg+Ill",
    "monikon_genetiivi": "+Pl+Gen",
    "monikon_partitiivi": "+Pl+Par",
}

# FST POS tag -> our vocabulary (cards/schema.json $defs/pos). Checked as
# exact "+"-separated components, not substrings, so "+A" never matches
# inside "+Adv".
FST_TAG_TO_POS = {
    "V": "verbi",
    "N": "substantiivi",
    "A": "adjektiivi",
    "Adv": "adverbi",
    "Pron": "pronomini",
    "Num": "numeraali",
    "Pr": "prepositio",
    "Po": "postpositio",
    "CS": "konjunktio",
    "CC": "konjunktio",
    "Interj": "interjektio",
    "Pcle": "partikkeli",
}


def forms_for_pos(pos: str) -> dict[str, str]:
    if pos == "verbi":
        return VERB_FORMS
    if pos in ("substantiivi", "adjektiivi"):
        return NOMINAL_FORMS
    raise ValueError(f"no principal-forms table for pos={pos!r}")


@dataclass
class FormsResult:
    principal_forms: dict[str, str]
    ambiguous: dict[str, list[str]]
    forms_source: str  # "fst" | "fst+llm" | "llm" - see cards/schema.json $defs/formsSource
    forms_verified: bool


def generate_forms(lemma: str, pos: str) -> FormsResult:
    """Resolve every principal form of `pos` via the FST, without guessing on ambiguity."""
    tag_prefix = POS_TAG.get(pos)
    if tag_prefix is None:
        raise ValueError(f"no FST tag for pos={pos!r}")

    resolved: dict[str, str] = {}
    ambiguous: dict[str, list[str]] = {}
    any_ambiguous = False
    any_missing = False

    for name, suffix in forms_for_pos(pos).items():
        query = f"{lemma}{tag_prefix}{suffix}"

        # dictionary_forms=True can itself return more than one candidate
        # (e.g. hampaitten/hampaiden are both standard literary plural
        # genitives with equal FST weight) - fall back to the plain query
        # only when it gave nothing, never to break a tie.
        with _FST_LOCK:
            candidates = _dedupe(
                w for w, _weight in uralicApi.generate(query, LANG, dictionary_forms=True)
            )
            if not candidates:
                candidates = _dedupe(w for w, _weight in uralicApi.generate(query, LANG))

        if len(candidates) == 1:
            resolved[name] = candidates[0]
        elif len(candidates) > 1:
            ambiguous[name] = candidates
            any_ambiguous = True
        else:
            any_missing = True

    if any_missing:
        source = "llm"
    elif any_ambiguous:
        source = "fst+llm"
    else:
        source = "fst"

    return FormsResult(
        principal_forms=resolved,
        ambiguous=ambiguous,
        forms_source=source,
        forms_verified=not any_ambiguous and not any_missing,
    )


def validate_form(word: str) -> bool:
    """True if `word` parses as some existing Finnish word form.

    Does not check whether the form fits its sentence context, and must
    not be used on puhekieli (colloquial forms fall outside the FST).
    """
    with _FST_LOCK:
        return bool(uralicApi.analyze(word, LANG))


def lemmatize(word: str) -> list[str]:
    """Distinct dictionary forms `word` can be an inflection of.

    A reading for a compound splits into components at "Cmp#" (e.g.
    "koti+N+Sg+Nom+Cmp#kaupunki+N+Sg+Ine"); the lemma is each component's
    own head joined back together ("kotikaupunki"), not just the text
    before the reading's first "+" ("koti"). Some compounds (e.g.
    "keittiöpöytä") have no whole-word entry at all - only the Cmp# split -
    so skipping the join would silently return a wrong, truncated lemma
    instead of the real one.
    """
    lemmas: list[str] = []
    with _FST_LOCK:
        readings = uralicApi.analyze(word, LANG)
    for reading, _weight in readings:
        lemma = "".join(segment.split("+", 1)[0] for segment in reading.split("Cmp#"))
        if lemma not in lemmas:
            lemmas.append(lemma)
    return lemmas


def detect_pos(word: str) -> list[str]:
    """Distinct parts of speech `word` analyzes as, in our vocabulary.

    Must run before generate_forms(): adjectives need the +A tag, nouns
    +N, and generate() on the wrong tag silently returns nothing.

    A Cmp#-split reading (e.g. "sini+N+Sg+Nom+Cmp#valkoinen+N+Sg+Nom" for
    "sinivalkoinen") tags each *component* with its own class, which need
    not match the compound's actual class - "sinivalkoinen" is only ever
    an adjective, but its first component "sini" is a noun. A lexicalized
    whole-word reading with no Cmp# (e.g. "sinivalkoinen+A+Sg+Nom") gives
    the real class, so it takes priority; Cmp# readings are only a
    fallback for words with no whole-word dictionary entry at all.
    """
    with _FST_LOCK:
        readings = uralicApi.analyze(word, LANG)

    whole_word = [reading for reading, _weight in readings if "Cmp#" not in reading]
    return _pos_from_readings(whole_word) or _pos_from_readings(
        reading for reading, _weight in readings
    )


def _pos_from_readings(readings) -> list[str]:
    found: list[str] = []
    for reading in readings:
        for part in reading.split("+")[1:]:
            pos = FST_TAG_TO_POS.get(part)
            if pos:
                if pos not in found:
                    found.append(pos)
                break
    return found


def _dedupe(items) -> list[str]:
    seen: list[str] = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen
