import re

import pytest
from sqlalchemy import delete
from sqlalchemy.dialects.sqlite import insert

from kielikaveri.db.models import Book, BookChapter, User
from kielikaveri.reader import app as reader_app
from kielikaveri.reader.warmup import Warmup


class _NoWarmup(Warmup):
    def maybe_start(self, client, model):
        return None


@pytest.fixture(autouse=True)
def _no_network_warmup(monkeypatch):
    """make_app's warm-up would call the real OpenAI (models.retrieve) in the
    background whenever a test opens a chapter. Tests that check the warm-up
    put their own Warmup into the app (tests/test_reader_speed.py)."""
    monkeypatch.setattr(reader_app, "Warmup", _NoWarmup)


def log_fields(message: str) -> dict[str, str]:
    """Parse an `event=x key=val key2=val2` log message into a dict, event
    included under the "event" key - shared by every test asserting on the
    application's structured log lines instead of matching full strings.
    """
    return dict(re.findall(r"(\S+?)=(\S+)", message))


async def add_users(session_factory, *user_ids: int) -> None:
    """The `users` rows a test's decks, notes, cards and books point at.

    PRAGMA foreign_keys is on (db/engine.py), so those rows can't be saved
    without them. In the running bot WhitelistMiddleware creates the row on
    a user's first update; a test that calls a handler or a db function
    directly has to do it itself. Rows already there are left alone.
    """
    async with session_factory() as session:
        for user_id in user_ids:
            await session.execute(insert(User).values(id=user_id).on_conflict_do_nothing())
        await session.commit()


async def delete_book(session, book_id: str) -> None:
    """Removes a book outright, whatever its chapters' statuses - the tests'
    way to make a book vanish mid-translation. The bot itself only deletes
    an unstarted book (db.books.delete_unstarted_book)."""
    await session.execute(delete(BookChapter).where(BookChapter.book_id == book_id))
    await session.execute(delete(Book).where(Book.id == book_id))
