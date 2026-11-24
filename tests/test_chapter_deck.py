"""Plan И7: one reader chapter deck per chapter, even under concurrent taps.

The deck is found by decks.chapter_id (unique index), not by its name. Every
test uses a real file-backed SQLite DB so that separate sessions - like two
concurrent aiohttp handlers - really are separate connections.

Tests that run alembic stay synchronous: alembic/env.py calls asyncio.run().
"""

import asyncio
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import create_engine, select, text

from alembic import command
from kielikaveri.db import decks
from kielikaveri.db.books import delete_book
from kielikaveri.db.decks import create_deck, get_or_create_chapter_deck
from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import DECK_CHAPTER_UNIQUE_INDEX, Book, BookChapter, Deck, User

USER_ID = 1
ALEMBIC_DIR = Path(__file__).parent.parent / "alembic"


@pytest.fixture
async def make_db(tmp_path):
    engines = []

    async def _make():
        engine = make_engine(f"sqlite+aiosqlite:///{tmp_path}/chapter_deck.db")
        await create_all(engine)
        engines.append(engine)
        return make_session_factory(engine)

    yield _make
    for engine in engines:
        await engine.dispose()


async def seed(session_factory, chapters: int = 2) -> tuple[str, list[BookChapter]]:
    async with session_factory() as session:
        session.add(User(id=USER_ID))
        await session.flush()
        book = Book(
            user_id=USER_ID,
            title="Kirja",
            source_lang="ru",
            source_hash="hash",
            front_matter=[],
        )
        session.add(book)
        await session.flush()
        rows = [
            BookChapter(book_id=book.id, number=n, title=f"Глава {n}", source_paragraphs=[])
            for n in range(1, chapters + 1)
        ]
        session.add_all(rows)
        await session.commit()
    return book.id, rows


async def all_decks(session_factory) -> list[Deck]:
    async with session_factory() as session:
        return list((await session.scalars(select(Deck))).all())


async def test_concurrent_calls_create_exactly_one_deck(make_db, monkeypatch):
    session_factory = await make_db()
    _, (chapter, _) = await seed(session_factory)

    # Both callers must see "no deck" before either inserts - the race window.
    # A barrier on the first lookup of each caller guarantees the interleaving.
    barrier = asyncio.Barrier(2)
    original = decks._find_chapter_deck
    first_lookup_done: set[int] = set()
    lookups = 0

    async def lookup_at_the_barrier(session, chapter_id):
        nonlocal lookups
        lookups += 1
        result = await original(session, chapter_id)
        if id(session) not in first_lookup_done:
            first_lookup_done.add(id(session))
            assert result is None
            await barrier.wait()
        return result

    monkeypatch.setattr(decks, "_find_chapter_deck", lookup_at_the_barrier)

    async def tap() -> str:
        async with session_factory() as session:
            deck = await get_or_create_chapter_deck(session, USER_ID, chapter, "Kirja - гл.1")
            await session.commit()
            return deck.id

    first, second = await asyncio.gather(tap(), tap())

    assert first == second
    # Two misses, then the loser's re-select after its IntegrityError.
    assert lookups == 3
    rows = await all_decks(session_factory)
    assert [(d.id, d.chapter_id, d.name) for d in rows] == [(first, chapter.id, "Kirja - гл.1")]


async def test_loser_keeps_its_outer_transaction(make_db, monkeypatch):
    """The IntegrityError is contained in the savepoint: work the caller did
    earlier in the same transaction survives and commits."""
    session_factory = await make_db()
    _, (chapter, _) = await seed(session_factory)
    async with session_factory() as session:
        winner = await get_or_create_chapter_deck(session, USER_ID, chapter, "A")
        await session.commit()

    async with session_factory() as session:
        other = await create_deck(session, USER_ID, "Other")
        # Force the INSERT path as if the first lookup had missed.
        original = decks._find_chapter_deck
        calls = 0

        async def miss_once(s, chapter_id):
            nonlocal calls
            calls += 1
            return None if calls == 1 else await original(s, chapter_id)

        monkeypatch.setattr(decks, "_find_chapter_deck", miss_once)
        deck = await get_or_create_chapter_deck(session, USER_ID, chapter, "B")
        await session.commit()

    assert deck.id == winner.id and deck.name == "A"
    assert sorted(d.name for d in await all_decks(session_factory)) == ["A", "Other"]
    assert other.id in {d.id for d in await all_decks(session_factory)}


async def test_repeat_call_returns_the_same_deck(make_db):
    session_factory = await make_db()
    _, (chapter, _) = await seed(session_factory)
    async with session_factory() as session:
        first = await get_or_create_chapter_deck(session, USER_ID, chapter, "Kirja - гл.1")
        await session.commit()
    async with session_factory() as session:
        # A different label (e.g. after a book rename) does not matter.
        again = await get_or_create_chapter_deck(session, USER_ID, chapter, "Renamed - гл.1")
        await session.commit()
    assert again.id == first.id
    assert again.name == "Kirja - гл.1"
    assert len(await all_decks(session_factory)) == 1


async def test_different_chapters_get_different_decks(make_db):
    session_factory = await make_db()
    _, (one, two) = await seed(session_factory)
    async with session_factory() as session:
        a = await get_or_create_chapter_deck(session, USER_ID, one, "Kirja - гл.1")
        b = await get_or_create_chapter_deck(session, USER_ID, two, "Kirja - гл.2")
        await session.commit()
    assert a.id != b.id
    assert (a.chapter_id, b.chapter_id) == (one.id, two.id)


async def test_decks_with_the_same_name_do_not_interfere(make_db):
    session_factory = await make_db()
    _, (chapter, _) = await seed(session_factory)
    async with session_factory() as session:
        twin_one = await create_deck(session, USER_ID, "Kirja - гл.1")
        twin_two = await create_deck(session, USER_ID, "Kirja - гл.1")
        await session.commit()
    async with session_factory() as session:
        deck = await get_or_create_chapter_deck(session, USER_ID, chapter, "Kirja - гл.1")
        await session.commit()
    assert deck.id not in {twin_one.id, twin_two.id}
    assert deck.chapter_id == chapter.id
    rows = await all_decks(session_factory)
    assert len(rows) == 3
    assert sorted(d.chapter_id or "" for d in rows) == ["", "", chapter.id]


async def test_deleting_the_book_keeps_the_deck_and_nulls_its_chapter_id(make_db):
    """PRAGMA foreign_keys is on (db/engine.py), so the FK's ON DELETE SET
    NULL fires: the deck survives, no longer tied to a chapter."""
    session_factory = await make_db()
    book_id, (chapter, _) = await seed(session_factory)
    async with session_factory() as session:
        deck = await get_or_create_chapter_deck(session, USER_ID, chapter, "Kirja - гл.1")
        await session.commit()
    async with session_factory() as session:
        await delete_book(session, book_id)
        await session.commit()
    (row,) = await all_decks(session_factory)
    assert row.id == deck.id
    assert row.chapter_id is None


def _decks_schema(url: str) -> tuple[set[str], set[str], list[tuple]]:
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            columns = {c[1] for c in conn.execute(text('PRAGMA table_info("decks")'))}
            indexes = {i[1] for i in conn.execute(text('PRAGMA index_list("decks")'))}
            fks = [tuple(fk[2:7]) for fk in conn.execute(text('PRAGMA foreign_key_list("decks")'))]
            return columns, indexes, fks
    finally:
        engine.dispose()


def test_migration_adds_and_removes_chapter_id(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    config = Config()
    config.set_main_option("script_location", str(ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{tmp_path}/migrated.db")
    url = f"sqlite:///{tmp_path}/migrated.db"

    command.upgrade(config, "4d8e2f6a1c37")
    columns, indexes, fks = _decks_schema(url)
    assert "chapter_id" in columns
    assert DECK_CHAPTER_UNIQUE_INDEX in indexes
    assert ("chapters", "chapter_id", "id", "NO ACTION", "SET NULL") in fks

    command.downgrade(config, "dc204c38f9e1")
    columns, indexes, fks = _decks_schema(url)
    assert "chapter_id" not in columns
    assert DECK_CHAPTER_UNIQUE_INDEX not in indexes
    assert all(fk[0] != "chapters" for fk in fks)
