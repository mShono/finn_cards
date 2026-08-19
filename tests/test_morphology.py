import threading

import pytest

from finn_cards.morphology import (
    detect_pos,
    generate_forms,
    lemmatize,
    validate_form,
)


def test_validate_form_accepts_real_word():
    assert validate_form("kättä") is True


def test_validate_form_rejects_made_up_word():
    assert validate_form("xyzquu") is False


def test_validate_form_rejects_made_up_verb_form():
    assert validate_form("hakelisi") is False


def test_lemmatize_returns_dictionary_form():
    assert "käsi" in lemmatize("kättä")
    assert "työ" in lemmatize("töitä")


def test_lemmatize_unknown_word_returns_empty():
    assert lemmatize("xyzquu") == []


def test_lemmatize_joins_compound_across_cmp_boundary():
    # bug: reading.split("+", 1)[0] on a Cmp# reading ("auto+N+Sg+Nom+Cmp#talli+N+Sg+Ine")
    # returned just "auto", not the word's real lemma "autotalli".
    assert lemmatize("autotallissa") == ["autotalli"]


def test_lemmatize_compound_with_no_whole_word_entry():
    # "keittiöpöytä" has no lexicalized whole-word FST entry at all, only
    # the Cmp# split - the old code returned "keittiö" (wrong) here, not
    # merely a lesser-evil duplicate like in the autotalli case above.
    assert lemmatize("keittiöpöydässä") == ["keittiöpöytä"]


def test_detect_pos_verb():
    assert "verbi" in detect_pos("hakea")


def test_detect_pos_adjective_not_confused_with_adverb():
    # "+A" is a substring of "+Adv" - detect_pos must match whole tag
    # components, not substrings, or every adjective would look like an
    # adverb too.
    assert detect_pos("lyhyt") == ["adjektiivi"]


def test_detect_pos_compound_not_polluted_by_modifier_class():
    # bug: "sinivalkoinen" (adjective) also has a Cmp#-split reading
    # "sini+N+Sg+Nom+Cmp#valkoinen+N+Sg+Nom" where the *modifier*
    # component is tagged +N. Taking the first tag from every reading
    # returned ["substantiivi", "adjektiivi"] - the lexicalized
    # non-compound reading ("sinivalkoinen+A+Sg+Nom") must win.
    assert detect_pos("sinivalkoinen") == ["adjektiivi"]


def test_generate_forms_unambiguous_verb():
    result = generate_forms("hakea", "verbi")
    assert result.principal_forms["preesens_1s"] == "haen"
    assert result.forms_source == "fst"
    assert result.forms_verified is True
    assert result.ambiguous == {}


def test_generate_forms_dictionary_forms_avoids_archaic_plural():
    # naive candidate[0] gives "perhehiä", not "perheitä" - an archaic plural
    result = generate_forms("perhe", "substantiivi")
    assert result.principal_forms["monikon_partitiivi"] == "perheitä"


def test_generate_forms_reports_ambiguity_instead_of_guessing():
    # candidate[0] for olla+Sg1 is the colloquial "oon", not "olen"
    result = generate_forms("olla", "verbi")
    assert "preesens_1s" not in result.principal_forms
    assert set(result.ambiguous["preesens_1s"]) == {"oon", "olen"}
    assert result.forms_verified is False
    assert result.forms_source == "fst+llm"


def test_generate_forms_reports_ambiguity_in_dictionary_forms_too():
    # dictionary_forms=True is not immune to ambiguity: hampaitten/hampaiden
    # are both standard literary plural genitives with equal FST weight.
    # Taking dict_forms[0] blindly (the original bug) marked this as a
    # verified "fst" form instead of reporting it.
    result = generate_forms("hammas", "substantiivi")
    assert "monikon_genetiivi" not in result.principal_forms
    assert set(result.ambiguous["monikon_genetiivi"]) == {"hampaitten", "hampaiden"}
    assert result.forms_verified is False
    assert result.forms_source == "fst+llm"


def test_generate_forms_unknown_pos_raises():
    with pytest.raises(ValueError):
        generate_forms("hakea", "adverbi")


def test_concurrent_calls_do_not_crash():
    # regression guard for the uralicApi cache race (uralicApi's
    # generator_cache/analyzer_cache are plain dicts, no lock): hammer
    # every wrapper function from multiple threads at once, cold cache
    # included. This can't prove the lock removes the double-load window
    # in the dependency, only that our own serialization doesn't deadlock
    # or corrupt results under concurrent access.
    errors: list[Exception] = []

    def worker(lemma: str, pos: str) -> None:
        try:
            assert detect_pos(lemma)
            assert lemmatize(lemma)
            assert validate_form(lemma)
            generate_forms(lemma, pos)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    words = [("hakea", "verbi"), ("perhe", "substantiivi"), ("hammas", "substantiivi")]
    threads = [
        threading.Thread(target=worker, args=(lemma, pos)) for lemma, pos in words for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
