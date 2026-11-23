"""SQLite connection setup done by make_engine (plan И9, М7).

File-backed databases, as in production: the bot and the reader share one
file, and PRAGMA journal_mode=WAL only means something for a file.
"""

import pytest
from sqlalchemy import select, text

from kielikaveri.db.decks import get_or_create_chapter_deck
from kielikaveri.db.engine import BUSY_TIMEOUT_MS, create_all, make_engine, make_session_factory
from kielikaveri.db.models import Book, BookChapter, Deck, User

USER_ID = 1


@pytest.fixture
async def engine(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path}/engine.db")
    await create_all(engine)
    yield engine
    await engine.dispose()


async def _pragma(engine, name: str):
    async with engine.connect() as conn:
        return (await conn.exec_driver_sql(f"PRAGMA {name}")).scalar_one()


async def _seed_chapter(session_factory) -> BookChapter:
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
        chapter = BookChapter(book_id=book.id, number=1, title="Глава 1", source_paragraphs=[])
        session.add(chapter)
        await session.commit()
    return chapter


async def test_new_chapter_deck_is_undone_by_the_callers_rollback(engine):
    """The deck's SAVEPOINT is the first write of the session. Without an
    explicit BEGIN the sqlite3 driver would run it outside any transaction,
    so RELEASE would commit the deck for good and rollback() would miss it."""
    session_factory = make_session_factory(engine)
    chapter = await _seed_chapter(session_factory)

    async with session_factory() as session:
        await get_or_create_chapter_deck(session, USER_ID, chapter, "Kirja - гл.1")
        await session.rollback()

    async with session_factory() as session:
        assert (await session.scalars(select(Deck))).all() == []


async def test_rollback_undoes_reads_then_writes(engine):
    session_factory = make_session_factory(engine)
    async with session_factory() as session:
        await session.execute(select(User))
        session.add(User(id=USER_ID))
        await session.flush()
        await session.rollback()
    async with session_factory() as session:
        assert (await session.scalars(select(User))).all() == []


async def test_savepoint_after_an_earlier_write_shares_its_transaction(engine):
    """The driver already opened the transaction (earlier INSERT): no second
    BEGIN, and the caller's rollback undoes both writes."""
    session_factory = make_session_factory(engine)
    chapter = await _seed_chapter(session_factory)

    async with session_factory() as session:
        session.add(User(id=USER_ID + 1))
        await session.flush()
        await get_or_create_chapter_deck(session, USER_ID, chapter, "Kirja - гл.1")
        await session.rollback()

    async with session_factory() as session:
        assert (await session.scalars(select(Deck))).all() == []
        assert [u.id for u in (await session.scalars(select(User))).all()] == [USER_ID]


async def test_read_then_write_survives_a_concurrent_commit(engine):
    """A session that read before another connection committed can still
    write afterwards. SQLAlchemy's BEGIN-on-every-transaction recipe fails
    here with "database is locked" (stale WAL snapshot), which is why
    make_engine only opens a transaction early for a SAVEPOINT."""
    session_factory = make_session_factory(engine)
    async with session_factory() as reader, session_factory() as writer:
        assert (await reader.scalars(select(User))).all() == []
        writer.add(User(id=USER_ID))
        await writer.commit()
        reader.add(User(id=USER_ID + 1))
        await reader.commit()
    async with session_factory() as session:
        assert sorted(u.id for u in (await session.scalars(select(User))).all()) == [
            USER_ID,
            USER_ID + 1,
        ]


async def test_file_database_uses_wal(engine):
    assert await _pragma(engine, "journal_mode") == "wal"


async def test_busy_timeout_is_set(engine):
    assert await _pragma(engine, "busy_timeout") == BUSY_TIMEOUT_MS


async def test_in_memory_database_still_works():
    engine = make_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as conn:
            assert (await conn.exec_driver_sql("PRAGMA journal_mode")).scalar_one() == "memory"
            await conn.execute(text("CREATE TABLE t (x INTEGER)"))
            await conn.execute(text("INSERT INTO t VALUES (1)"))
            assert (await conn.execute(text("SELECT x FROM t"))).scalar_one() == 1
    finally:
        await engine.dispose()
