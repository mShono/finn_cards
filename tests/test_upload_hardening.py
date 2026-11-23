"""Book upload hardening (review 23.11.2026, "Мелочи" М4, М5, М6, М11, М12):
parsing off the event loop, the duplicate-upload race, a conditional "no",
UTF-16 with a BOM and the honest note on English books. All texts are
synthetic.

The database here runs with PRAGMA foreign_keys=ON and real users rows, so
the tests hold whether or not the project switches foreign keys on."""

import threading

import pytest
from sqlalchemy import event, func, select
from test_books import BOOK_TEXT, routed  # noqa: F401 - pytest fixture

from kielikaveri.bot import books as bot_books
from kielikaveri.db.books import chapter_by_number, delete_unstarted_book, queue_chapter
from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import Book, BookChapter, ChapterStatus, Deck, User
from kielikaveri.translate.chapters import decode_text

OWNER = 1  # bot_helpers.tg_user()
STRANGER = 2

EN_PARAGRAPH = "He left the house early in the morning, and nobody saw where he went. " * 6
EN_BOOK_TEXT = "\n\n".join(
    ["Title"] + [line for n in ("1", "2") for line in (f"Chapter {n}", EN_PARAGRAPH, EN_PARAGRAPH)]
)


@pytest.fixture
async def session_factory(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path}/hardening.db")

    @event.listens_for(engine.sync_engine, "connect")
    def _enable_fks(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    await create_all(engine)
    factory = make_session_factory(engine)
    async with factory() as session:
        session.add_all([User(id=OWNER), User(id=STRANGER)])
        await session.commit()
    yield factory
    await engine.dispose()


async def _books(factory) -> list[Book]:
    async with factory() as session:
        return list((await session.scalars(select(Book))).all())


async def _chapter_count(factory) -> int:
    async with factory() as session:
        return await session.scalar(select(func.count()).select_from(BookChapter))


# --- М4: parsing off the event loop -------------------------------------------


async def test_decode_split_and_hash_run_in_one_worker_thread(routed, monkeypatch):  # noqa: F811
    threads = {}

    def spy(name, real):
        def wrapper(*args, **kwargs):
            threads[name] = threading.current_thread()
            return real(*args, **kwargs)

        return wrapper

    for name in ("decode_text", "split_book", "text_hash", "detect_lang"):
        monkeypatch.setattr(bot_books, name, spy(name, getattr(bot_books, name)))

    [reply] = await routed.upload()

    assert "глав - 3" in reply.text
    assert set(threads) == {"decode_text", "split_book", "text_hash", "detect_lang"}
    assert len(set(threads.values())) == 1
    assert threading.main_thread() not in threads.values()


# --- М5: the same file twice at once --------------------------------------------


async def test_duplicate_upload_race_answers_already_uploaded(
    routed,  # noqa: F811
    session_factory,
    monkeypatch,
):
    await routed.upload()
    [stored] = await _books(session_factory)

    # The second upload's find_book ran before the first one committed:
    # it saw nothing, so the unique index is what stops it.
    real_find_book = bot_books.find_book
    calls = []

    async def late_find_book(session, user_id, source_hash):
        calls.append(source_hash)
        if len(calls) == 1:
            return None
        return await real_find_book(session, user_id, source_hash)

    monkeypatch.setattr(bot_books, "find_book", late_find_book)

    [reply] = await routed.upload("копия.txt")

    assert len(calls) == 2
    assert reply.text.startswith(f"Эта книга уже загружена как «{stored.title}».")
    # Chapter 1 is still pending: the same question as an ordinary repeat.
    buttons = [b.callback_data for b in reply.reply_markup.inline_keyboard[0]]
    assert buttons == [f"book:yes:{stored.id}", f"book:no:{stored.id}"]
    assert [b.id for b in await _books(session_factory)] == [stored.id]
    assert await _chapter_count(session_factory) == 3


async def test_integrity_error_that_is_not_a_duplicate_is_raised(
    routed,  # noqa: F811
    session_factory,
    monkeypatch,
):
    async def nothing(session, user_id, source_hash):
        return None

    monkeypatch.setattr(bot_books, "find_book", nothing)
    await routed.upload()

    # find_book never finds it, so the IntegrityError can't be explained
    # as a duplicate - it must not be swallowed.
    with pytest.raises(Exception, match="IntegrityError|UNIQUE"):
        await routed.upload("копия.txt")
    assert len(await _books(session_factory)) == 1


# --- М6: "no" deletes only an unstarted book ----------------------------------


async def test_no_after_chapter_one_was_queued_keeps_the_book(routed, session_factory):  # noqa: F811
    await routed.upload()
    [book] = await _books(session_factory)
    # "Yes" from another message landed first.
    async with session_factory() as session:
        first = await chapter_by_number(session, book.id, 1)
        assert await queue_chapter(session, first.id)
        await session.commit()

    replies = await routed.tap(f"book:no:{book.id}")

    assert not any("удалила" in r.text for r in replies)
    assert [b.id for b in await _books(session_factory)] == [book.id]
    assert await _chapter_count(session_factory) == 3
    async with session_factory() as session:
        first = await chapter_by_number(session, book.id, 1)
    assert first.status is ChapterStatus.queued


async def test_no_on_a_pending_book_deletes_it(routed, session_factory):  # noqa: F811
    await routed.upload()
    [book] = await _books(session_factory)

    [reply] = await routed.tap(f"book:no:{book.id}")

    assert "удалила" in reply.text
    assert await _books(session_factory) == []
    assert await _chapter_count(session_factory) == 0


async def test_delete_unstarted_book_keeps_a_book_with_any_started_chapter(
    routed,  # noqa: F811
    session_factory,
):
    await routed.upload()
    [book] = await _books(session_factory)
    async with session_factory() as session:
        last = await chapter_by_number(session, book.id, 3)
        last.status = ChapterStatus.failed
        await session.commit()

    async with session_factory() as session:
        assert not await delete_unstarted_book(session, book.id, OWNER)
        await session.commit()

    assert len(await _books(session_factory)) == 1
    assert await _chapter_count(session_factory) == 3


async def test_delete_unstarted_book_ignores_another_users_book(routed, session_factory):  # noqa: F811
    await routed.upload()
    [book] = await _books(session_factory)

    async with session_factory() as session:
        assert not await delete_unstarted_book(session, book.id, STRANGER)
        await session.commit()

    assert len(await _books(session_factory)) == 1
    assert await _chapter_count(session_factory) == 3


async def test_delete_unstarted_book_unlinks_a_chapter_deck(routed, session_factory):  # noqa: F811
    await routed.upload()
    [book] = await _books(session_factory)
    async with session_factory() as session:
        first = await chapter_by_number(session, book.id, 1)
        deck = Deck(user_id=OWNER, name="Kirja - гл.1", chapter_id=first.id)
        session.add(deck)
        await session.commit()

    async with session_factory() as session:
        assert await delete_unstarted_book(session, book.id, OWNER)
        await session.commit()

    async with session_factory() as session:
        kept = await session.get(Deck, deck.id)
    assert kept is not None and kept.chapter_id is None
    assert await _books(session_factory) == []


# --- М11: UTF-16 with a byte order mark ---------------------------------------


@pytest.mark.parametrize("encoding", ["utf-16-le", "utf-16-be"])
def test_decode_text_reads_utf16_with_bom(encoding):
    bom = b"\xff\xfe" if encoding == "utf-16-le" else b"\xfe\xff"
    raw = bom + "Глава 1\r\nОн ушёл.\r\n".encode(encoding)

    assert decode_text(raw) == "Глава 1\nОн ушёл.\n"


def test_decode_text_refuses_truncated_utf16():
    with pytest.raises(bot_books.UndecodableError):
        decode_text(b"\xff\xfe" + "Он ушёл.".encode("utf-16-le") + b"\x00")


async def test_utf16_upload_is_stored(routed, session_factory):  # noqa: F811
    routed.content["bytes"] = BOOK_TEXT.encode("utf-16")  # BOM + native order

    [reply] = await routed.upload()

    assert "глав - 3" in reply.text
    [book] = await _books(session_factory)
    assert book.source_lang == "ru"


# --- М12: English books are taken, with an honest note -------------------------


async def test_english_book_is_stored_with_a_note(routed, session_factory):  # noqa: F811
    routed.content["bytes"] = EN_BOOK_TEXT.encode()

    [reply] = await routed.upload()

    [book] = await _books(session_factory)
    assert book.source_lang == "en"
    assert bot_books.ENGLISH_NOTE in reply.text
    assert "Перевести на финский?" in reply.text
    buttons = [b.callback_data for b in reply.reply_markup.inline_keyboard[0]]
    assert buttons == [f"book:yes:{book.id}", f"book:no:{book.id}"]


async def test_english_note_repeats_on_a_duplicate(routed, session_factory):  # noqa: F811
    routed.content["bytes"] = EN_BOOK_TEXT.encode()
    await routed.upload()

    [reply] = await routed.upload("again.txt")

    assert "уже загружена" in reply.text
    assert bot_books.ENGLISH_NOTE in reply.text


async def test_russian_book_has_no_english_note(routed):  # noqa: F811
    [reply] = await routed.upload()

    assert bot_books.ENGLISH_NOTE not in reply.text
