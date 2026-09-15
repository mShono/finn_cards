import logging
import threading

import pytest
from conftest import log_fields

from finn_cards.morphology import (
    detect_pos,
    generate_forms,
    lemmatize,
    pos_set_for_lemma,
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


# --- pos_set_for_lemma --------------------------------------------------------
#
# The point of every test here: detect_pos() answers "what can this string
# be", pos_set_for_lemma() answers "what can this lemma be". The two differ
# exactly when a string also happens to be some other lemma's inflected form,
# which is where generate_forms() used to be handed a part of speech that
# produced nothing.


def test_pos_set_for_lemma_ignores_another_lemmas_reading():
    # "tuli" analyzes as tulla+V+Act+Ind+Prt+Sg3 ("he came") *and* as
    # tuli+N+Sg+Nom ("fire"). The verb reading belongs to the lemma "tulla",
    # so it must not end up in the lemma "tuli"'s own set.
    assert "verbi" in detect_pos("tuli")
    assert pos_set_for_lemma("tuli") == {"substantiivi"}


def test_pos_set_for_lemma_drops_voida_readings_from_voi():
    # Five of "voi"'s eight readings are forms of the verb "voida"; the rest
    # are the lemma "voi" itself (butter / particle / interjection).
    assert "verbi" in detect_pos("voi")
    assert pos_set_for_lemma("voi") == {"substantiivi", "partikkeli", "interjektio"}


def test_pos_set_for_lemma_drops_a_different_lemmas_possessive_reading():
    # "kuusi" analyzes as kuu+N+...+PxSg2 ("your moon") beside its own two
    # readings. Only the latter say anything about the lemma "kuusi", and
    # both of them are real - a set, not a pick.
    assert pos_set_for_lemma("kuusi") == {"substantiivi", "numeraali"}


def test_pos_set_for_lemma_keeps_both_when_one_lemma_really_has_two():
    # "hakea" is both the verb "to fetch" and a noun. Nothing in the FST
    # ranks them (both readings carry weight 0.0), so both stay and the
    # choice belongs to whoever has the sentence.
    assert pos_set_for_lemma("hakea") == {"verbi", "substantiivi"}


def test_pos_set_for_lemma_unambiguous_word():
    assert pos_set_for_lemma("kissa") == {"substantiivi"}
    assert pos_set_for_lemma("lyhyt") == {"adjektiivi"}


def test_pos_set_for_lemma_unknown_word_is_empty():
    assert pos_set_for_lemma("xyzquu") == set()


def test_pos_set_for_lemma_not_polluted_by_compound_modifier_class():
    # Same trap detect_pos() guards against: "sinivalkoinen" has a Cmp#
    # reading whose first component "sini" is a noun, but the lexicalized
    # whole-word reading says adjective and wins.
    assert pos_set_for_lemma("sinivalkoinen") == {"adjektiivi"}


def test_pos_set_for_lemma_compound_with_no_whole_word_entry_uses_the_head():
    # "keittiöpöytä" has only the Cmp# split - the compound's class is its
    # head's ("pöytä"), not its modifier's.
    assert pos_set_for_lemma("keittiöpöytä") == {"substantiivi"}


def test_generate_forms_unambiguous_verb():
    result = generate_forms("hakea", "verbi")
    assert result.principal_forms["preesens_1s"] == "haen"
    assert result.forms_source == "fst"
    assert result.forms_verified is True
    assert result.ambiguous == {}


def test_generate_forms_logs_fst_resolve_with_duration(caplog):
    with caplog.at_level(logging.DEBUG, logger="finn_cards.morphology"):
        generate_forms("hakea", "verbi")

    events = [log_fields(r.message) for r in caplog.records]
    resolved = next(f for f in events if f.get("event") == "fst.resolve")
    assert resolved["lemma"] == "hakea"
    assert resolved["forms_source"] == "fst"
    assert int(resolved["duration_ms"]) >= 0


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
