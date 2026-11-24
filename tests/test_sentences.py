"""reader/sentences.py: paragraphs cut into sentences, Finnish and Russian
(plan 2026-11-20-kirjakaveri-reader, "П3"). The texts are made up here,
not taken from a book."""

import pytest

from kielikaveri.reader.sentences import sentence_of_word, sentences, split_sentences, word_span
from kielikaveri.translate.fst_check import words


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Finnish dialogue: a dash opens each line.
        (
            "– Mä oon täällä, hän sanoi. – Missä sä oot? – Tuu tänne! Hän nauroi.",
            ["– Mä oon täällä, hän sanoi.", "– Missä sä oot?", "– Tuu tänne!", "Hän nauroi."],
        ),
        # A speech tag after the line stays with it.
        (
            "– Tule tänne! hän huusi. – En tule.",
            ["– Tule tänne! hän huusi.", "– En tule."],
        ),
        # Finnish quotes: ” on both sides.
        (
            "Hän sanoi: ”Tule.” Sitten hän lähti. ”Minne?” hän kysyi.",
            ["Hän sanoi: ”Tule.”", "Sitten hän lähti.", "”Minne?” hän kysyi."],
        ),
        # Abbreviations and initials don't end a sentence.
        (
            (
                "Hän osti esim. Omenoita ja n. 5 kiloa perunoita. J. K. Rowling kirjoitti sen. "
                "Klo 8 ts. Aamulla."
            ),
            [
                "Hän osti esim. Omenoita ja n. 5 kiloa perunoita.",
                "J. K. Rowling kirjoitti sen.",
                "Klo 8 ts. Aamulla.",
            ],
        ),
        # An ellipsis ends one only before a capital.
        (
            "– Minä… minä en tiedä. Sitten... Ei mitään. Hän odotti… Hiljaisuus.",
            ["– Minä… minä en tiedä.", "Sitten...", "Ei mitään.", "Hän odotti…", "Hiljaisuus."],
        ),
        # Russian dialogue: em dash, speech tags in lowercase.
        (
            "— Пойдём, — сказал он. — Нет! «Да?» — спросил он. Она молчала.",
            ["— Пойдём, — сказал он.", "— Нет!", "«Да?» — спросил он.", "Она молчала."],
        ),
        # Russian abbreviations: т.е., г., ул., им., initials; т.д. may end one.
        (
            (
                "Это т.е. Москва, а не Тверь и т.д. Потом пришёл А. С. Пушкин. "
                "Он жил в г. Москве на ул. Ленина в доме им. Горького."
            ),
            [
                "Это т.е. Москва, а не Тверь и т.д.",
                "Потом пришёл А. С. Пушкин.",
                "Он жил в г. Москве на ул. Ленина в доме им. Горького.",
            ],
        ),
        ("Это т. е. Москва. Вот.", ["Это т. е. Москва.", "Вот."]),
        # A sentence ending in a one-letter word still ends.
        ("Это был я. Потом ушёл.", ["Это был я.", "Потом ушёл."]),
        # A digit after a break starts a sentence.
        ("Hän tuli. 1990 oli hyvä vuosi.", ["Hän tuli.", "1990 oli hyvä vuosi."]),
        ("Yksi lause ilman pistettä", ["Yksi lause ilman pistettä"]),
        ("", []),
        ("   ", []),
    ],
)
def test_sentences(text, expected):
    assert sentences(text) == expected


def test_spans_cover_the_text_without_the_whitespace_between():
    text = "  Hän tuli.   Hän meni!  "
    spans = split_sentences(text)
    assert [text[s:e] for s, e in spans] == ["Hän tuli.", "Hän meni!"]
    assert spans[0][0] == 2


def test_word_span_is_the_index_in_words():
    text = "– Mä oon EU:n Moskova-joella, eikö? Hän sanoi."
    for index, word in enumerate(words(text)):
        start, end = word_span(text, index)
        assert text[start:end] == word
    assert word_span(text, len(words(text))) is None
    assert word_span(text, -1) is None


def test_sentence_of_word():
    text = "– Mä oon talossa, hän sanoi. – Missä? Hän nauroi."
    found = {
        word: text[slice(*sentence_of_word(text, index))] for index, word in enumerate(words(text))
    }
    assert found["talossa"] == "– Mä oon talossa, hän sanoi."
    assert found["Missä"] == "– Missä?"
    assert found["nauroi"] == "Hän nauroi."


def test_sentence_of_word_out_of_range():
    assert sentence_of_word("Hän tuli.", 5) is None
    assert sentence_of_word("", 0) is None


def test_same_word_twice_in_a_paragraph_finds_its_own_sentence():
    text = "Hän tuli. Hän meni."
    assert text[slice(*sentence_of_word(text, 0))] == "Hän tuli."
    assert text[slice(*sentence_of_word(text, 2))] == "Hän meni."
