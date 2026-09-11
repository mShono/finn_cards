"""What an inflection card asks, when it explains it, and when it opens.

An inflection card trains *use* of a form, not recall of its name. The
front carries only a functional cue - a case question ("mihin?"), a frame
to slot the word into ("ei ole ___"), or a person+time pair for verbs
("hän, eilen") - and the grammatical term appears on the back, next to
the answer:

    front:  kauppa -> mihin?
    back:   kauppaan
            illatiivi - mihin? (sisään)

Each task also carries a curriculum `level` and a `group`. Those are data,
not policy: srs/curriculum.py reads them to decide which form a learner
meets next, so re-ordering the syllabus means editing this table, never the
scheduler.

Keys match finn_cards.morphology's VERB_FORMS/NOMINAL_FORMS exactly, minus
NEVER_ASKED. tests/test_grammar.py enforces that partition, so a form added
to the FST tables can never reach the user as a raw identifier
("rientää -> nut_partisiippi?" is what this module exists to prevent).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class FormCategory(str, enum.Enum):
    """Verb forms are not one flat list - the task differs by category.

    `compound` covers the participle too: a NUT-participle is trained as the
    whole perfect construction ("hän on mennyt"), never as a bare morpheme.
    No infinitive category exists on purpose - see NEVER_ASKED.
    """

    case = "case"
    case_plural = "case_plural"
    finite = "finite"
    conditional = "conditional"
    imperative = "imperative"
    passive = "passive"
    compound = "compound"


class CurriculumLevel(str, enum.Enum):
    """How early a form is worth meeting. Order is LEVEL_ORDER, not the
    member order - an enum has no ordering of its own."""

    core = "core"
    extended = "extended"
    later = "later"


LEVEL_ORDER: tuple[CurriculumLevel, ...] = (
    CurriculumLevel.core,
    CurriculumLevel.extended,
    CurriculumLevel.later,
)

# Related forms a learner should meet as one system rather than scattered
# months apart. Cards inside a group sit next to each other in FORM_TASKS,
# so the curriculum's ordering introduces them back to back.
GROUP_INNER_LOCATIVE = "sisäpaikallissijat"  # mihin? missä? mistä?
GROUP_OUTER_LOCATIVE = "ulkopaikallissijat"  # mille? millä? miltä?
GROUP_PLURAL = "monikko"
GROUP_PRESENT = "preesens"
GROUP_STATE_CHANGE = "olotila"  # essiivi/translatiivi: being vs becoming


@dataclass(frozen=True)
class FormTask:
    cue: str  # front, shown after "lemma -> "
    label: str  # back, the grammatical name - only ever seen after answering
    category: FormCategory
    level: CurriculumLevel
    group: str | None = None


# A noun's lemma *is* its nominative singular, and the front already shows
# the lemma - quizzing it would ask the user to repeat the question. The
# same holds for a verb's A-infinitive, which is why no infinitive is in the
# FST tables at all. Anything listed here is generated and stored, just
# never turned into a question - distinct from a card that exists but has
# not been introduced yet (db.models.CardStatus).
NEVER_ASKED = frozenset({"nominatiivi"})

_NOMINAL_TASKS: dict[str, FormTask] = {
    "genetiivi": FormTask(
        cue="kenen? minkä?",
        label="genetiivi - kenen? minkä? (-n)",
        category=FormCategory.case,
        level=CurriculumLevel.core,
    ),
    # No single question fits the partitive (object, negation, quantity,
    # measure), and "ketä? mitä?" alone would teach a wrong rule. A frame
    # that any noun can be slotted into is honest instead: "ei ole" takes
    # the partitive singular for count and mass nouns alike.
    "partitiivi": FormTask(
        cue="ei ole ___",
        label="partitiivi - ei ole mitä? juon mitä? paljon mitä? (-a/-ä, -ta/-tä)",
        category=FormCategory.case,
        level=CurriculumLevel.core,
    ),
    # The inner locative series is one system, but the illative is core B1
    # on its own (it is the case B1 material drills first), so it opens a
    # level earlier than its two siblings and they follow immediately.
    "illatiivi": FormTask(
        cue="mihin?",
        label="illatiivi - mihin? sisään (-Vn, -seen, -hVn)",
        category=FormCategory.case,
        level=CurriculumLevel.core,
        group=GROUP_INNER_LOCATIVE,
    ),
    "inessiivi": FormTask(
        cue="missä?",
        label="inessiivi - missä? sisällä (-ssa/-ssä)",
        category=FormCategory.case,
        level=CurriculumLevel.extended,
        group=GROUP_INNER_LOCATIVE,
    ),
    "elatiivi": FormTask(
        cue="mistä?",
        label="elatiivi - mistä? sisältä (-sta/-stä)",
        category=FormCategory.case,
        level=CurriculumLevel.extended,
        group=GROUP_INNER_LOCATIVE,
    ),
    "adessiivi": FormTask(
        cue="millä? kenellä?",
        label="adessiivi - millä? kenellä? päällä (-lla/-llä)",
        category=FormCategory.case,
        level=CurriculumLevel.extended,
        group=GROUP_OUTER_LOCATIVE,
    ),
    "ablatiivi": FormTask(
        cue="miltä? keneltä?",
        label="ablatiivi - miltä? keneltä? päältä (-lta/-ltä)",
        category=FormCategory.case,
        level=CurriculumLevel.extended,
        group=GROUP_OUTER_LOCATIVE,
    ),
    "allatiivi": FormTask(
        cue="mille? kenelle?",
        label="allatiivi - mille? kenelle? päälle (-lle)",
        category=FormCategory.case,
        level=CurriculumLevel.extended,
        group=GROUP_OUTER_LOCATIVE,
    ),
    # The textbook question for the essive is "minä?", which a learner reads
    # as the pronoun "I" - so the cue asks by function (role, state) and the
    # term is left for the back.
    "essiivi": FormTask(
        cue="millaisena? (rooli, tila)",
        label="essiivi - millaisena? minä? esim. opettajana (-na/-nä)",
        category=FormCategory.case,
        level=CurriculumLevel.later,
        group=GROUP_STATE_CHANGE,
    ),
    # Same trap: "miksi?" also means "why?".
    "translatiivi": FormTask(
        cue="miksi muuttuu? (tulla joksikin)",
        label="translatiivi - miksi? joksikin, esim. opettajaksi (-ksi)",
        category=FormCategory.case,
        level=CurriculumLevel.later,
        group=GROUP_STATE_CHANGE,
    ),
    "monikon_genetiivi": FormTask(
        cue="kenen? minkä? (monikko)",
        label="monikon genetiivi - kenen? minkä? monikossa (-jen, -ien, -ten)",
        category=FormCategory.case_plural,
        level=CurriculumLevel.core,
        group=GROUP_PLURAL,
    ),
    "monikon_partitiivi": FormTask(
        cue="paljon ___",
        label="monikon partitiivi - paljon mitä? (-ja/-jä, -ia/-iä)",
        category=FormCategory.case_plural,
        level=CurriculumLevel.core,
        group=GROUP_PLURAL,
    ),
}

_VERB_TASKS: dict[str, FormTask] = {
    "preesens_1s": FormTask(
        cue="minä, nyt → ?",
        label="preesens, 1. persoona yksikkö (minä)",
        category=FormCategory.finite,
        level=CurriculumLevel.core,
        group=GROUP_PRESENT,
    ),
    "preesens_3s": FormTask(
        cue="hän, nyt → ?",
        label="preesens, 3. persoona yksikkö (hän)",
        category=FormCategory.finite,
        level=CurriculumLevel.core,
        group=GROUP_PRESENT,
    ),
    "imperfekti_3s": FormTask(
        cue="hän, eilen → ?",
        label="imperfekti, 3. persoona yksikkö (hän)",
        category=FormCategory.finite,
        level=CurriculumLevel.core,
    ),
    "konditionaali_1s": FormTask(
        cue="minä, jos... → ?",
        label="konditionaali, 1. persoona yksikkö (minä ...isin)",
        category=FormCategory.conditional,
        level=CurriculumLevel.later,
    ),
    "imperatiivi_2s": FormTask(
        cue="sinä, käsky → ?",
        label="imperatiivi, 2. persoona yksikkö (käsky sinulle)",
        category=FormCategory.imperative,
        level=CurriculumLevel.later,
    ),
    # Trained as the perfect construction, not as a participle in isolation:
    # the gap sits after the auxiliary, so the answer is only correct if it
    # works inside "hän on ___".
    "nut_partisiippi": FormTask(
        cue="hän on jo ___",
        label="perfekti: olla + NUT-partisiippi",
        category=FormCategory.compound,
        level=CurriculumLevel.extended,
    ),
    "passiivi": FormTask(
        cue="täällä ___ (kuka tahansa)",
        label="passiivi, preesens - tekijää ei sanota (-taan/-tään, -daan/-dään)",
        category=FormCategory.passive,
        level=CurriculumLevel.extended,
    ),
}

FORM_TASKS: dict[str, FormTask] = {**_NOMINAL_TASKS, **_VERB_TASKS}

# Position in this table is the tie-break the curriculum introduces forms
# in, which is what keeps a group's members consecutive.
FORM_ORDER: dict[str, int] = {name: i for i, name in enumerate(FORM_TASKS)}
