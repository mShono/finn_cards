"""Size limits on a book (review 23.11.2026, И3): chapter count, chapter
length and the glossary prompt. All texts here are synthetic."""

import json
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

from sqlalchemy import func, select
from test_books import routed, session_factory  # noqa: F401 - pytest fixtures

from kielikaveri.db.models import Book, BookChapter
from kielikaveri.llm.breaker import CallBreaker
from kielikaveri.translate.chapters import (
    FALLBACK_CHUNK_CHARS,
    MAX_CHAPTER_CHARS,
    MAX_CHAPTERS,
    Chapter,
    split_book,
)
from kielikaveri.translate.chapters import Book as SplitBook
from kielikaveri.translate.pipeline import (
    GUIDE_FRONT_MATTER_CHARS,
    GUIDE_MAX_CHARS,
    SAMPLE_CHARS_PER_CHAPTER,
    Usage,
    prepare_guide,
)

SENTENCE = "Он ушёл из дома рано утром, и никто не видел, куда он пошёл. "


def _paragraph(index: int, size: int = 1000) -> str:
    # Distinct paragraphs, so a lost or duplicated one shows in the comparison.
    head = f"Абзац {index}. "
    return head + (SENTENCE * (size // len(SENTENCE) + 1))[: size - len(head)]


async def _counts(factory) -> tuple[int, int]:
    async with factory() as session:
        books = await session.scalar(select(func.count()).select_from(Book))
        chapters = await session.scalar(select(func.count()).select_from(BookChapter))
        return books, chapters


# --- chapter count: refusal in the bot ------------------------------------------


async def test_page_numbers_are_refused_and_nothing_is_stored(routed, session_factory):  # noqa: F811
    # OCR output: every page number on its own line looks like a chapter heading.
    lines = ["Название"]
    for page in range(1, MAX_CHAPTERS + 101):
        lines += [str(page), _paragraph(page, 300)]
    routed.content["bytes"] = "\n\n".join(lines).encode()

    [reply] = await routed.upload()

    assert f"глав - {MAX_CHAPTERS + 100}" in reply.text
    assert "номера страниц" in reply.text
    assert reply.reply_markup is None
    assert await _counts(session_factory) == (0, 0)
    routed.translator.wake.assert_not_called()


async def test_huge_text_without_headings_is_refused(routed, session_factory):  # noqa: F811
    paragraphs = [_paragraph(i) for i in range((MAX_CHAPTERS + 5) * FALLBACK_CHUNK_CHARS // 1000)]
    routed.content["bytes"] = "\n\n".join(paragraphs).encode()

    [reply] = await routed.upload()

    assert "заголовков глав" in reply.text
    assert "номера страниц" not in reply.text
    assert await _counts(session_factory) == (0, 0)


async def test_book_at_the_chapter_limit_is_stored(routed, session_factory):  # noqa: F811
    lines = ["Название"]
    for number in range(1, MAX_CHAPTERS + 1):
        lines += [str(number), _paragraph(number, 100)]
    routed.content["bytes"] = "\n\n".join(lines).encode()

    [reply] = await routed.upload()

    assert f"глав - {MAX_CHAPTERS}" in reply.text
    assert await _counts(session_factory) == (1, MAX_CHAPTERS)


# --- chapter length: splitting ----------------------------------------------------


def test_huge_chapter_is_split_with_running_numbers_and_no_lost_paragraphs():
    huge = [_paragraph(i) for i in range(70)]  # ~70 000 chars -> 3 pieces
    text = "\n\n".join(
        ["Название", "Часть первая", "1", _paragraph(100), "2", *huge, "Часть вторая", "3"]
        + [_paragraph(101)]
    )

    book = split_book(text)

    assert [c.number for c in book.chapters] == [1, 2, 3, 4, 5]
    assert [c.title for c in book.chapters] == ["1", "2 (1/3)", "2 (2/3)", "2 (3/3)", "3"]
    assert [c.part for c in book.chapters] == ["Часть первая"] * 4 + ["Часть вторая"]
    pieces = book.chapters[1:4]
    assert [p for c in pieces for p in c.paragraphs] == huge
    assert all(c.chars <= MAX_CHAPTER_CHARS for c in book.chapters)
    # Even pieces, not two full ones and a stub.
    assert min(c.chars for c in pieces) >= 20_000


def test_chapter_at_the_limit_is_kept_and_one_char_over_is_halved():
    at_limit = ["а" * 1000] * (MAX_CHAPTER_CHARS // 1000)
    over = [*at_limit, "б"]
    text = "\n\n".join(["1", *at_limit, "2", *over])

    book = split_book(text)

    assert [c.title for c in book.chapters] == ["1", "2 (1/2)", "2 (2/2)"]
    assert book.chapters[0].paragraphs == at_limit
    assert [p for c in book.chapters[1:] for p in c.paragraphs] == over
    assert [c.chars for c in book.chapters[1:]] == [16_000, 14_001]


# --- glossary prompt ----------------------------------------------------------------


def _guide_client():
    reply = SimpleNamespace(
        output_text=json.dumps({"characters": [], "terms": [], "style_notes": ""}),
        output=[],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )
    return SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(return_value=reply)))


async def _guide_input(book: SplitBook) -> dict:
    client = _guide_client()
    breaker = CallBreaker(100, timedelta(minutes=10))
    await prepare_guide(client, breaker, "m", Usage(), book, full_chapters=[])
    return json.loads(client.responses.create.await_args.kwargs["input"])


def _chapters(count: int, size: int) -> list[Chapter]:
    return [
        Chapter(number=n, part=None, title=str(n), paragraphs=[_paragraph(n, size)])
        for n in range(1, count + 1)
    ]


async def test_glossary_prompt_stays_under_the_cap_with_1000_chapters():
    front = [_paragraph(i) for i in range(50)]  # a late first heading: 50 000 chars

    sent = await _guide_input(SplitBook(front_matter=front, chapters=_chapters(1000, 3000)))

    fragments = sent["fragments"]
    assert len(fragments) == 1000
    assert sum(len(f["text"]) for f in fragments) <= GUIDE_MAX_CHARS
    assert all(f["text"] for f in fragments)  # every chapter still sampled
    assert sum(len(p) for p in sent["front_matter"]) <= GUIDE_FRONT_MATTER_CHARS


async def test_glossary_samples_of_a_normal_book_are_unchanged():
    front = ["Название", "Эпиграф"]

    sent = await _guide_input(SplitBook(front_matter=front, chapters=_chapters(49, 3000)))

    assert {len(f["text"]) for f in sent["fragments"]} == {SAMPLE_CHARS_PER_CHAPTER}
    assert sent["front_matter"] == front
