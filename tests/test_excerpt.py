"""reader/excerpt.py: the card's example is a verbatim piece of the book,
shorter than its sentence (plan 2026-11-20-kirjakaveri-reader, "П4"). The
sentences are made up here, in the spirit of the plan's example; OpenAI is
a fake."""

import json
from datetime import timedelta
from types import SimpleNamespace

import openai
import pytest

from kielikaveri.cards_save import Usage
from kielikaveri.llm.breaker import CallBreaker, CircuitOpenError
from kielikaveri.reader.excerpt import EXCERPT_SCHEMA_NAME, fallback, make_excerpt, tidy

FI = "Vain isä käveli puutarhassa, unohtaneena, mitä oli etsimässä ja minne oli menossa"
RU = "Только отец бродил по саду, забыв, что искал и куда шёл"


def span(sentence: str, word: str, nth: int = 0) -> tuple[int, int]:
    start = -1
    for _ in range(nth + 1):
        start = sentence.index(word, start + 1)
    return start, start + len(word)


class FakeOpenAI:
    def __init__(self, answer=None, text=None, error=None):
        self.answer = answer
        self.text = text
        self.error = error
        self.requests = []
        self.responses = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error is not None:
            raise self.error
        assert kwargs["text"]["format"]["name"] == EXCERPT_SCHEMA_NAME
        assert kwargs["text"]["format"]["strict"] is True
        output = self.text if self.text is not None else json.dumps(self.answer, ensure_ascii=False)
        return SimpleNamespace(
            output_text=output,
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        )


def breaker(max_calls: int = 60) -> CallBreaker:
    return CallBreaker(max_calls, timedelta(minutes=10))


async def excerpt(client, sentence=FI, word="käveli", ru=RU, nth=0, usage=None, brk=None):
    return await make_excerpt(
        client,
        brk or breaker(),
        "gpt-test",
        sentence,
        span(sentence, word, nth),
        ru,
        usage if usage is not None else Usage(),
    )


# --- the model's fragments -----------------------------------------------------------


async def test_verbatim_fragments_are_used():
    client = FakeOpenAI(
        {"fragment_fi": "Vain isä käveli puutarhassa", "fragment_ru": "Только отец бродил по саду"}
    )
    usage = Usage()

    result = await excerpt(client, usage=usage)

    assert result.example_fi == "Vain isä käveli puutarhassa."
    assert result.example_ru == "Только отец бродил по саду."
    assert result.from_model is True
    assert (usage.input, usage.output) == (100, 20)
    payload = json.loads(client.requests[0]["input"])
    assert payload == {"sentence_fi": FI, "word": "käveli", "original_ru": RU}


async def test_edges_are_cleaned_and_capitalized():
    client = FakeOpenAI(
        {"fragment_fi": " isä käveli puutarhassa, ", "fragment_ru": "отец бродил по саду,"}
    )

    result = await excerpt(client)

    assert result.example_fi == "Isä käveli puutarhassa."
    assert result.example_ru == "Отец бродил по саду."


@pytest.mark.parametrize(
    "answer",
    [
        # Made up: not a piece of the sentence.
        {"fragment_fi": "Isä käveli puistossa", "fragment_ru": "Только отец бродил по саду"},
        # Reordered.
        {"fragment_fi": "käveli isä puutarhassa", "fragment_ru": "Только отец бродил по саду"},
        # The Russian one made up.
        {"fragment_fi": "Vain isä käveli puutarhassa", "fragment_ru": "Отец гулял по саду"},
        # Verbatim, but without the word.
        {"fragment_fi": "mitä oli etsimässä", "fragment_ru": "что искал"},
        # Half of the word only.
        {"fragment_fi": "Vain isä käve", "fragment_ru": "Только отец бродил"},
        {"fragment_fi": "", "fragment_ru": ""},
    ],
)
async def test_anything_but_verbatim_with_the_word_falls_back(answer):
    result = await excerpt(FakeOpenAI(answer))

    assert result.from_model is False
    assert result.example_fi == "Vain isä käveli puutarhassa."
    assert result.example_ru == RU + "."


async def test_the_same_word_elsewhere_in_the_sentence_is_not_enough():
    sentence = "Hän oli kotona, kun äiti oli töissä ja isä oli matkalla"
    # The tap is on the second "oli"; the model picks the first one's clause.
    client = FakeOpenAI({"fragment_fi": "Hän oli kotona", "fragment_ru": "Он был дома"})

    result = await excerpt(client, sentence, "oli", "Он был дома, когда мама работала", nth=1)

    assert result.from_model is False
    assert result.example_fi == "Kun äiti oli töissä ja isä oli matkalla."


@pytest.mark.parametrize("text", ["", "{not json", '{"fragment_fi": 5, "fragment_ru": "x"}', "[]"])
async def test_a_broken_answer_falls_back(text):
    result = await excerpt(FakeOpenAI(text=text))

    assert result.from_model is False
    assert result.example_fi == "Vain isä käveli puutarhassa."


async def test_an_openai_error_falls_back():
    client = FakeOpenAI(error=openai.APIConnectionError(request=SimpleNamespace()))

    result = await excerpt(client)

    assert result.from_model is False
    assert result.example_fi == "Vain isä käveli puutarhassa."


async def test_a_tripped_breaker_is_raised_not_worked_around():
    client = FakeOpenAI({"fragment_fi": "x", "fragment_ru": "y"})

    with pytest.raises(CircuitOpenError):
        await excerpt(client, brk=breaker(max_calls=0))
    assert client.requests == []


# --- the rule -------------------------------------------------------------------------


def rule(sentence: str, word: str, ru: str = "Русское предложение.", nth: int = 0):
    return fallback(sentence, span(sentence, word, nth), ru)


def test_the_rule_takes_the_part_with_the_word():
    result = rule(FI, "puutarhassa", RU)

    assert result.example_fi == "Vain isä käveli puutarhassa."
    assert result.example_ru == RU + "."


def test_the_part_is_found_by_position_not_by_spelling():
    sentence = "Hän oli väsynyt, mutta hän oli iloinen; kaikki oli hyvin"
    assert rule(sentence, "oli", nth=1).example_fi == "Mutta hän oli iloinen."
    assert rule(sentence, "oli", nth=2).example_fi == "Kaikki oli hyvin."


def test_a_short_part_gets_the_next_one():
    assert rule(FI, "unohtaneena").example_fi == (
        "Unohtaneena, mitä oli etsimässä ja minne oli menossa."
    )


def test_a_short_last_part_gets_the_previous_one():
    sentence = "– Mä oon talossa, hän sanoi."
    assert rule(sentence, "sanoi").example_fi == "Mä oon talossa, hän sanoi."


@pytest.mark.parametrize(
    ("sentence", "word", "expected"),
    [
        ("Miksi sinä tulit kotiin, kun sataa?", "tulit", "Miksi sinä tulit kotiin?"),
        ("Tule heti tänne, poika!", "Tule", "Tule heti tänne!"),
        ("Hän meni ulos: sade oli loppunut.", "meni", "Hän meni ulos."),
    ],
)
def test_the_end_mark_follows_the_sentence(sentence, word, expected):
    assert rule(sentence, word).example_fi == expected


def test_dashes_and_semicolons_split_too():
    sentence = "Talo oli vanha — katto vuoti ja ikkunat olivat rikki"
    assert rule(sentence, "katto").example_fi == "Katto vuoti ja ikkunat olivat rikki."


def test_tidy_keeps_the_question_mark_of_the_original():
    assert tidy("что ты делаешь", "Что ты делаешь?»") == "Что ты делаешь?"
    assert tidy("  – ну, ", "Ну!") == "Ну!"
