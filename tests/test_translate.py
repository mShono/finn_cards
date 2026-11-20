import json
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kielikaveri.llm.breaker import CallBreaker
from kielikaveri.translate import fst_check, pipeline
from kielikaveri.translate.chapters import Chapter, read_text, split_book
from kielikaveri.translate.fst_check import likely_names, unknown_words
from kielikaveri.translate.pipeline import (
    BookGuide,
    MisalignedError,
    RefusedError,
    Usage,
    blocks,
    translate_chapter,
)

BOOK = """Автор

Название

ДВЕ СЕСТРЫ

1

Первый абзац.

Второй абзац.

2

Третий абзац.

ВОЙНА ЗА МИР

1

Четвёртый абзац.
"""


def test_split_book_numbers_chapters_across_parts():
    book = split_book(BOOK)

    assert book.front_matter == ["Автор", "Название"]
    assert [(c.number, c.part, c.title) for c in book.chapters] == [
        (1, "ДВЕ СЕСТРЫ", "1"),
        (2, "ДВЕ СЕСТРЫ", "2"),
        # The printed number restarts in a new part; the running one must not.
        (3, "ВОЙНА ЗА МИР", "1"),
    ]
    assert book.chapters[0].paragraphs == ["Первый абзац.", "Второй абзац."]


def test_split_book_without_headings_falls_back_to_chunks():
    text = "\n".join(["а" * 3000, "б" * 3000, "в" * 3000])

    book = split_book(text)

    assert [len(c.paragraphs) for c in book.chapters] == [1, 1, 1]
    assert [c.number for c in book.chapters] == [1, 2, 3]


def test_dialogue_and_short_lines_are_not_headings():
    book = split_book("1\n\n— Да.\n\nНет.\n\n2\n\nОК, пусть.")

    assert [c.paragraphs for c in book.chapters] == [["— Да.", "Нет."], ["ОК, пусть."]]


def test_read_text_handles_crlf_and_cp1251(tmp_path):
    utf = tmp_path / "utf.txt"
    utf.write_bytes("1\r\n\r\nПривет\r\n".encode())
    legacy = tmp_path / "legacy.txt"
    legacy.write_bytes("Привет\n".encode("cp1251"))

    assert read_text(utf) == "1\n\nПривет\n"
    assert read_text(legacy) == "Привет\n"


KNOWN = {"hän", "meni", "kotiin", "talo", "ja"}


def fake_analyze(word):
    return [word] if word in KNOWN else []


def test_unknown_words_flags_only_forms_the_fst_lacks():
    assert unknown_words("Hän meni kotiinsaa ja talo.", analyze=fake_analyze) == ["kotiinsaa"]


def test_unknown_words_lets_inflected_glossary_names_through():
    text = "Asjan talo ja Kostjalle meni."

    assert unknown_words(text, names=["Asja", "Kostja"], analyze=fake_analyze) == []
    # Lowercase is not a name - a typo that happens to share a prefix still counts.
    assert unknown_words("asjan", names=["Asja"], analyze=fake_analyze) == ["asjan"]


def test_unknown_words_accepts_hyphenated_parts_the_fst_knows():
    assert unknown_words("talo-ja", analyze=fake_analyze) == []


def test_blocks_never_split_a_paragraph():
    assert blocks(["a" * 4000, "b" * 3000, "c" * 100], max_chars=6000) == [[0], [1, 2]]


def response(payload=None, text=None, input_tokens=10, output_tokens=5):
    return SimpleNamespace(
        output_text=json.dumps(payload, ensure_ascii=False) if text is None else text,
        output=[],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def translation(*pairs):
    return response({"paragraphs": [{"id": i, "fi": fi} for i, fi in pairs]})


GUIDE = BookGuide(characters=[], terms=[], style_notes="")
CHAPTER = Chapter(number=1, part=None, title="1", paragraphs=["Он ушёл.", "Дом стоял."])


@pytest.fixture(autouse=True)
def known_words(monkeypatch):
    monkeypatch.setattr(fst_check, "lemmatize", fake_analyze)


def client_with(*responses):
    return SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(side_effect=responses)))


def breaker():
    return CallBreaker(100, timedelta(minutes=10))


async def test_translate_chapter_runs_fst_repair_and_review():
    client = client_with(
        translation((0, "Hän meni kotiinsaa."), (1, "Talo ja.")),
        translation((0, "Hän meni kotiin.")),
        response(
            {
                "issues": [
                    {"id": 1, "category": "kalkki", "quote_fi": "ja", "explanation_ru": "лишнее"}
                ],
                "fixed": [{"id": 1, "fi": "Talo."}],
            }
        ),
    )
    usage = Usage()

    result = await translate_chapter(client, breaker(), "m", usage, GUIDE, CHAPTER)

    assert result.draft == {0: "Hän meni kotiinsaa.", 1: "Talo ja."}
    assert result.fst_flags == {0: ["kotiinsaa"]}
    assert result.final == {0: "Hän meni kotiin.", 1: "Talo."}
    assert result.fst_flags_final == {}
    assert [i.category for i in result.issues] == ["kalkki"]
    assert {name: step.calls for name, step in usage.steps.items()} == {
        "translate": 1,
        "fst_repair": 1,
        "review": 1,
    }


async def test_dropped_paragraph_is_retried_once_then_fails():
    client = client_with(translation((0, "Hän meni.")), translation((0, "Hän meni.")))

    with pytest.raises(MisalignedError):
        await translate_chapter(client, breaker(), "m", Usage(), GUIDE, CHAPTER, review=False)
    assert client.responses.create.await_count == 2


async def test_retry_recovers_a_dropped_paragraph():
    client = client_with(translation((0, "Hän meni.")), translation((0, "Hän meni."), (1, "Talo.")))

    result = await translate_chapter(client, breaker(), "m", Usage(), GUIDE, CHAPTER, review=False)

    assert result.final == {0: "Hän meni.", 1: "Talo."}


async def test_empty_response_is_reported_as_refusal():
    client = client_with(response(text=""))

    with pytest.raises(RefusedError):
        await translate_chapter(client, breaker(), "m", Usage(), GUIDE, CHAPTER)


async def test_review_fix_for_a_paragraph_outside_the_block_is_ignored():
    client = client_with(
        translation((0, "Hän meni."), (1, "Talo.")),
        response({"issues": [], "fixed": [{"id": 7, "fi": "Väärä."}]}),
    )

    result = await translate_chapter(client, breaker(), "m", Usage(), GUIDE, CHAPTER)

    assert result.final == {0: "Hän meni.", 1: "Talo."}


async def test_book_text_stays_out_of_logs_even_at_debug(caplog):
    # Plan "Логи": the book lives in the DB, logs carry ids and counts only.
    caplog.set_level("DEBUG", logger=pipeline.__name__)
    client = client_with(
        translation((0, "Hän meni kotiinsaa."), (1, "Talo.")),
        translation((0, "Hän meni kotiin.")),
        response({"issues": [], "fixed": []}),
    )

    await translate_chapter(client, breaker(), "m", Usage(), GUIDE, CHAPTER)

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert caplog.records
    for text in ["Он ушёл", "Дом стоял", "Hän meni", "Talo", "kotiinsaa"]:
        assert text not in logged


def test_likely_names_only_counts_mid_sentence_capitals():
    text = "Kolonna saapui Pavlivkaan. Kotiinsaa he menivät. – Šulik, minne?"

    # After a full stop or a dialogue dash a capital proves nothing.
    assert likely_names(text, ["Pavlivkaan", "Kotiinsaa", "Šulik"]) == ["Pavlivkaan"]


async def test_names_are_reported_but_not_sent_to_repair():
    chapter = Chapter(number=1, part=None, title="1", paragraphs=["Он ушёл в Павловку."])
    client = client_with(
        translation((0, "Hän meni Pavlivkaan.")),
        response({"issues": [], "fixed": []}),
    )

    result = await translate_chapter(client, breaker(), "m", Usage(), GUIDE, chapter)

    assert result.fst_flags == {0: ["Pavlivkaan"]}
    assert result.fst_sent == {}
    assert client.responses.create.await_count == 2  # translate + review, no repair


async def test_repair_is_one_call_per_block():
    chapter = Chapter(number=1, part=None, title="1", paragraphs=["А.", "Б."])
    client = client_with(
        translation((0, "kotiinsaa"), (1, "talosaa")),
        translation((0, "kotiin"), (1, "talo")),
        response({"issues": [], "fixed": []}),
    )

    result = await translate_chapter(client, breaker(), "m", Usage(), GUIDE, chapter)

    assert result.final == {0: "kotiin", 1: "talo"}
    assert client.responses.create.await_count == 3
