import re

from sqlalchemy.dialects.sqlite import insert

from kielikaveri.db.models import User


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
