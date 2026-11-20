"""Split a plain-text book into chapters of paragraphs.

Paragraph boundaries are kept exactly: the translator returns one Finnish
paragraph per source paragraph, and that 1:1 alignment is what later lets the
reader show the original paragraph for free (plan, "Конвейер POC", point 6).

Chapters get a running `number` across the whole book, not the number printed
in the text - "Покаянные дни" restarts at 1 in every part, and a deck named
"<book> - гл.1" must not be ambiguous (decided 20.11.2026: сквозной номер).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# A bare number on its own line ("1", "12.") or an explicit heading word.
# Roman numerals stay case-sensitive, or a lone English "ill." would match.
_CHAPTER_RE = re.compile(r"^(?:(?i:глава|chapter)\s+)?(?:\d{1,3}|[IVXLC]{1,7})\.?$")
_SCENE_BREAK_RE = re.compile(r"^[*•\s]{3,}$")
# No fallback split below this: a book with fewer headings is chunked instead.
_MIN_CHAPTERS = 2
FALLBACK_CHUNK_CHARS = 5000


@dataclass
class Chapter:
    number: int  # running, 1-based across the whole book
    part: str | None  # part heading the chapter sits under, if any
    title: str  # heading as printed ("1", "Глава 3"), or "" for chunks
    paragraphs: list[str] = field(default_factory=list)

    @property
    def chars(self) -> int:
        return sum(len(p) for p in self.paragraphs)


@dataclass
class Book:
    front_matter: list[str]  # title page, epigraph, copyright - never translated as a chapter
    chapters: list[Chapter]


def read_text(path: Path) -> str:
    """UTF-8 first; Russian .txt files from older sources are often cp1251."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1251")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _is_part_heading(line: str) -> bool:
    letters = [ch for ch in line if ch.isalpha()]
    return len(line) <= 60 and len(letters) >= 3 and all(ch.isupper() for ch in letters)


def split_book(text: str) -> Book:
    paragraphs = [line.strip() for line in text.split("\n")]
    paragraphs = [p for p in paragraphs if p and not _SCENE_BREAK_RE.match(p)]

    front: list[str] = []
    chapters: list[Chapter] = []
    part: str | None = None
    for paragraph in paragraphs:
        if _CHAPTER_RE.match(paragraph):
            chapters.append(Chapter(number=len(chapters) + 1, part=part, title=paragraph))
        elif _is_part_heading(paragraph):
            part = paragraph
        elif chapters:
            chapters[-1].paragraphs.append(paragraph)
        else:
            front.append(paragraph)

    if len(chapters) < _MIN_CHAPTERS:
        return Book(front_matter=[], chapters=chunk_paragraphs(paragraphs))
    return Book(front_matter=front, chapters=chapters)


def chunk_paragraphs(paragraphs: list[str], max_chars: int = FALLBACK_CHUNK_CHARS) -> list[Chapter]:
    """For books without recognizable headings: reading-sized pieces, never
    splitting a paragraph."""
    chunks: list[Chapter] = []
    current: list[str] = []
    size = 0
    for paragraph in paragraphs:
        if current and size + len(paragraph) > max_chars:
            chunks.append(Chapter(number=len(chunks) + 1, part=None, title="", paragraphs=current))
            current, size = [], 0
        current.append(paragraph)
        size += len(paragraph)
    if current:
        chunks.append(Chapter(number=len(chunks) + 1, part=None, title="", paragraphs=current))
    return chunks
