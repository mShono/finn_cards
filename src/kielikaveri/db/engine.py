from __future__ import annotations

from sqlalchemy import event
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from kielikaveri.db.models import Base

# How long a connection waits for another one's write lock before giving up
# with "database is locked" (bot and reader write the same file).
BUSY_TIMEOUT_MS = 5000


def _is_file_database(database_url: str) -> bool:
    url = make_url(database_url)
    database = url.database or ""
    in_memory = (
        database in ("", ":memory:")
        or database.startswith("file::memory:")
        or url.query.get("mode") == "memory"
    )
    return not in_memory


def _driver_in_transaction(dbapi_connection) -> bool:
    # The aiosqlite adapter wraps an aiosqlite.Connection (`_connection`);
    # plain pysqlite hands over the sqlite3.Connection itself.
    inner = getattr(dbapi_connection, "_connection", dbapi_connection)
    return bool(inner.in_transaction)


def _install_sqlite_events(engine: AsyncEngine, *, file_database: bool) -> None:
    """Connection setup for SQLite: WAL, busy timeout, and a real
    transaction around every SAVEPOINT.

    PRAGMA foreign_keys stays off (plan М14 open): most test fixtures insert
    rows without the users/decks rows they reference.

    The sqlite3 driver opens a transaction only right before the first
    INSERT/UPDATE/DELETE/REPLACE. A SAVEPOINT emitted before any of those
    (session.begin_nested() as the first write of a session) therefore runs
    outside any transaction: SQLite makes it a transaction of its own and its
    RELEASE commits it, so the caller's later rollback() cannot undo it.
    BEGIN is emitted right before such a SAVEPOINT.

    SQLAlchemy's documented recipe (isolation_level=None plus BEGIN on every
    "begin" event) is not used: it also puts the leading SELECTs inside the
    transaction, and a deferred transaction that read first and writes later
    fails at once with "database is locked" (SQLITE_BUSY_SNAPSHOT in WAL, a
    lock deadlock in rollback-journal mode) whenever another connection
    committed in between - busy_timeout does not help there. The bot, the
    reader and the translator worker write the same file concurrently, so
    that would turn ordinary read-then-write handlers into sporadic errors.
    """
    sync_engine = engine.sync_engine

    @event.listens_for(sync_engine, "connect")
    def _on_connect(dbapi_connection, _connection_record) -> None:
        # No transaction is open on a fresh connection, so journal_mode can
        # be changed (it cannot inside one).
        cursor = dbapi_connection.cursor()
        try:
            if file_database:
                # Readers do not block the writer and vice versa. Meaningless
                # for :memory:, which keeps journal_mode=memory.
                cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        finally:
            cursor.close()

    @event.listens_for(sync_engine, "savepoint")
    def _on_savepoint(conn, _name) -> None:
        dbapi_connection = conn.connection.dbapi_connection
        if not _driver_in_transaction(dbapi_connection):
            # The driver sees this BEGIN and leaves the transaction to
            # SQLAlchemy's commit()/rollback() from here on.
            conn.exec_driver_sql("BEGIN")


def make_engine(database_url: str) -> AsyncEngine:
    # Bound values (message texts, book chapters) stay out of exception
    # messages and out of the log if SQL statement logging is ever switched
    # on - the SQL itself stays readable.
    engine = create_async_engine(database_url, hide_parameters=True)
    if engine.dialect.name == "sqlite":
        _install_sqlite_events(engine, file_database=_is_file_database(database_url))
    return engine


def make_session_factory(engine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def create_all(engine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
