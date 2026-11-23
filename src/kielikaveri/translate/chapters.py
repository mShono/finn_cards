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

from charset_normalizer import from_bytes

# A bare number on its own line ("1", "12.") or an explicit heading word.
# Roman numerals stay case-sensitive, or a lone English "ill." would match.
_CHAPTER_RE = re.compile(r"^(?:(?i:глава|chapter)\s+)?(?:\d{1,3}|[IVXLC]{1,7})\.?$")
# "Глава 3. Возвращение", "Chapter IV: The Return" - a heading word is
# required here, or any paragraph starting with a number would qualify.
_TITLED_CHAPTER_RE = re.compile(
    r"^(?i:глава|chapter)\s+(?:\d{1,3}|[IVXLC]{1,7})\s*[.:—–-]\s*\S.{0,80}$"
)
# "Часть первая", "Part II" - only a number or an ordinal after the word:
# "Part of the problem was..." is an ordinary English paragraph.
_PART_RE = re.compile(
    r"^(?i:часть|part)\s+(?:\d{1,2}|[IVXLC]{1,5}|(?i:первая|вторая|третья|четв[её]ртая|пятая"
    r"|шестая|седьмая|восьмая|девятая|десятая|one|two|three|four|five|six|seven|eight|nine"
    r"|ten|first|second|third|fourth|fifth))\b[.:]?[^.!?]{0,60}$"
)
_SCENE_BREAK_RE = re.compile(r"^[*•\s]{3,}$")
# No fallback split below this: a book with fewer headings is chunked instead.
_MIN_CHAPTERS = 2
FALLBACK_CHUNK_CHARS = 5000
# A chapter longer than this is cut into numbered pieces: one chapter is one
# background translation job, and the call breaker would trip halfway through
# a huge one after the calls are already paid for (review 23.11.2026, И3).
MAX_CHAPTER_CHARS = 30_000
# More "chapters" than this is a mis-split (page numbers on their own lines
# after OCR), not a book - the bot refuses it rather than storing it.
MAX_CHAPTERS = 300


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


class UndecodableError(ValueError):
    pass


# Books come in Russian or English, so only their legacy encodings compete -
# unrestricted, charset-normalizer reads a short cp1251 line as Chinese.
_LEGACY_ENCODINGS = ["cp1251", "koi8_r", "cp866", "mac_cyrillic", "iso8859_5", "cp1252"]


def decode_text(raw: bytes) -> str:
    """UTF-8 first - it never decodes non-UTF-8 Cyrillic by accident; anything
    else (cp1251, koi8-r from older sources) is left to charset-normalizer.

    UTF-16 with a BOM (Windows Notepad's "Unicode") is taken by its BOM before
    either: charset-normalizer can mistake its NUL-heavy bytes for binary."""
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            text = raw.decode("utf-16")
        except UnicodeDecodeError:
            raise UndecodableError("broken UTF-16 after a byte order mark") from None
    else:
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            best = from_bytes(raw, cp_isolation=_LEGACY_ENCODINGS).best()
            if best is None:
                raise UndecodableError("not a text file in any known encoding") from None
            text = str(best)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def read_text(path: Path) -> str:
    return decode_text(path.read_bytes())


def _is_chapter_heading(line: str) -> bool:
    return bool(_CHAPTER_RE.match(line) or _TITLED_CHAPTER_RE.match(line))


def _is_part_heading(line: str) -> bool:
    if _PART_RE.match(line):
        return True
    letters = [ch for ch in line if ch.isalpha()]
    return len(line) <= 60 and len(letters) >= 3 and all(ch.isupper() for ch in letters)


def split_book(text: str) -> Book:
    paragraphs = [line.strip() for line in text.split("\n")]
    paragraphs = [p for p in paragraphs if p and not _SCENE_BREAK_RE.match(p)]

    front: list[str] = []
    chapters: list[Chapter] = []
    part: str | None = None
    for paragraph in paragraphs:
        if _is_chapter_heading(paragraph):
            chapters.append(Chapter(number=len(chapters) + 1, part=part, title=paragraph))
        elif _is_part_heading(paragraph):
            part = paragraph
        elif chapters:
            chapters[-1].paragraphs.append(paragraph)
        else:
            front.append(paragraph)

    if len(chapters) < _MIN_CHAPTERS:
        return Book(front_matter=[], chapters=chunk_paragraphs(paragraphs))
    return Book(front_matter=front, chapters=_split_long(chapters))


def _split_long(chapters: list[Chapter], max_chars: int = MAX_CHAPTER_CHARS) -> list[Chapter]:
    """Cut chapters over `max_chars` into even pieces at paragraph boundaries.
    Numbers stay running across the book; the piece shows in the title -
    "13 (1/2)", "13 (2/2)" - and keeps the chapter's part."""
    result: list[Chapter] = []
    for chapter in chapters:
        if chapter.chars <= max_chars:
            pieces = [chapter.paragraphs]
        else:
            pieces = _even_pieces(chapter.paragraphs, chapter.chars, max_chars)
        for index, paragraphs in enumerate(pieces, start=1):
            title = (
                chapter.title if len(pieces) == 1 else f"{chapter.title} ({index}/{len(pieces)})"
            )
            result.append(
                Chapter(
                    number=len(result) + 1, part=chapter.part, title=title, paragraphs=paragraphs
                )
            )
    return result


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


def _even_pieces(paragraphs: list[str], chars: int, max_chars: int) -> list[list[str]]:
    """Greedy cut that closes a piece once it reaches an even share of the
    chapter, never going over `max_chars` - so 30 001 chars become two
    halves, not a full piece plus a one-line stub. A single paragraph over
    `max_chars` stays whole: paragraphs are never split."""
    count = -(-chars // max_chars)
    target = -(-chars // count)
    pieces: list[list[str]] = []
    current: list[str] = []
    size = 0
    for paragraph in paragraphs:
        if current and size + len(paragraph) > max_chars:
            pieces.append(current)
            current, size = [], 0
        current.append(paragraph)
        size += len(paragraph)
        if size >= target:
            pieces.append(current)
            current, size = [], 0
    if current:
        pieces.append(current)
    return pieces
