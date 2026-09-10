from finn_cards.morphology import NOMINAL_FORMS, VERB_FORMS
from kielikaveri.grammar import FORM_TASKS, NOT_QUIZZED, FormCategory


def test_every_fst_form_is_either_quizzed_or_explicitly_excluded():
    # The guard against the original bug: a form added to the FST tables
    # with no task defined used to reach the user as its raw key
    # ("rientää → nut_partisiippi?"). Now it must be one or the other.
    all_forms = set(VERB_FORMS) | set(NOMINAL_FORMS)

    assert set(FORM_TASKS) | NOT_QUIZZED == all_forms
    assert set(FORM_TASKS) & NOT_QUIZZED == set()


def test_no_task_leaks_a_technical_term_into_the_cue():
    # The whole point: the grammatical name belongs on the back. If a cue
    # ever contains the form's own key, the card is asking for the name
    # again instead of for the use.
    for name, task in FORM_TASKS.items():
        assert name not in task.cue
        stem = name.removeprefix("monikon_").split("_")[0]
        assert stem not in task.cue.lower(), f"{name}: cue names the category"


def test_every_task_names_the_category_on_the_back():
    for name, task in FORM_TASKS.items():
        assert task.label, f"{name}: empty label"
        assert task.cue, f"{name}: empty cue"


def test_nominative_is_not_quizzed_because_it_equals_the_lemma():
    assert "nominatiivi" in NOT_QUIZZED
    assert "nominatiivi" not in FORM_TASKS


def test_verb_forms_are_not_one_flat_category():
    # plan 3.3 / the card must distinguish finite forms from conditional,
    # imperative, passive and the compound tenses - not lump them together.
    verb_categories = {FORM_TASKS[name].category for name in VERB_FORMS if name in FORM_TASKS}

    assert verb_categories == {
        FormCategory.finite,
        FormCategory.conditional,
        FormCategory.imperative,
        FormCategory.passive,
        FormCategory.compound,
    }


def test_participle_is_trained_as_the_perfect_construction():
    # Not as a bare morpheme tag: the cue puts the gap after the auxiliary,
    # so only a form that works inside "hän on ___" counts.
    task = FORM_TASKS["nut_partisiippi"]

    assert task.category is FormCategory.compound
    assert "on" in task.cue.split()
    assert "___" in task.cue
