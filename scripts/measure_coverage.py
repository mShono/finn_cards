"""Measure whether the FST covers principal_forms without needing an LLM
to disambiguate or fill gaps.

This uses the real generate_forms() algorithm, not a naive "take
candidate[0]" approach - candidate[0] is often archaic or colloquial (see
generate_forms()'s docstring), which would silently inflate the coverage
number. A form counts as "resolved" here only if generate_forms() resolved
it unambiguously.

Usage: python scripts/measure_coverage.py
"""

from __future__ import annotations

from finn_cards.morphology import detect_pos, generate_forms, forms_for_pos

VERBS = [
    "hakea", "olla", "tulla", "juosta", "tarvita", "opiskella",
    "voida", "syödä", "pitää", "mennä", "nähdä", "antaa",
]
NOUNS = [
    "käsi", "työ", "vesi", "ihminen", "puhelin", "kaupunki", "hammas",
    "perhe", "tyttö", "kirje", "huone", "vastaus", "sisar", "kieli",
    "päivä", "auto",
]
ADJECTIVES = ["lyhyt", "pitkä", "hyvä", "iso", "kaunis", "vaikea", "helppo", "uusi"]

WORDS = [(w, "verbi") for w in VERBS]
WORDS += [(w, "substantiivi") for w in NOUNS]
WORDS += [(w, "adjektiivi") for w in ADJECTIVES]


def main() -> None:
    total_forms = 0
    resolved_forms = 0
    ambiguous_forms = 0
    missing_forms = 0
    notes_fully_verified = 0
    pos_mismatches = []
    problems = []

    for lemma, pos in WORDS:
        detected = detect_pos(lemma)
        if pos not in detected:
            pos_mismatches.append((lemma, pos, detected))

        result = generate_forms(lemma, pos)
        n_forms = len(forms_for_pos(pos))
        total_forms += n_forms
        resolved_forms += len(result.principal_forms)
        ambiguous_forms += sum(len(v) for v in result.ambiguous.values())
        missing_forms += n_forms - len(result.principal_forms) - len(result.ambiguous)

        if result.forms_verified:
            notes_fully_verified += 1
        else:
            problems.append((lemma, pos, result))

    print(f"words: {len(WORDS)}  (verbs={len(VERBS)} nouns={len(NOUNS)} adjectives={len(ADJECTIVES)})")
    print(f"notes fully verified (no ambiguity, no gaps): {notes_fully_verified}/{len(WORDS)} "
          f"({notes_fully_verified / len(WORDS):.1%})")
    print(f"forms total: {total_forms}")
    print(f"  resolved (unambiguous):   {resolved_forms} ({resolved_forms / total_forms:.1%})")
    print(f"  ambiguous (needs LLM):    {ambiguous_forms}")
    print(f"  missing (FST empty):      {missing_forms}")

    if pos_mismatches:
        print("\npos mismatches (detect_pos() disagreed with the curated list):")
        for lemma, expected, detected in pos_mismatches:
            print(f"  {lemma}: expected {expected}, detect_pos()={detected}")

    if problems:
        print("\nwords needing attention (ambiguous or missing forms):")
        for lemma, pos, result in problems:
            print(f"  {lemma} ({pos}) source={result.forms_source}")
            for name, candidates in result.ambiguous.items():
                print(f"    {name}: ambiguous {candidates}")
            missing = set(forms_for_pos(pos)) - set(result.principal_forms) - set(result.ambiguous)
            for name in missing:
                print(f"    {name}: no candidates at all")


if __name__ == "__main__":
    main()
