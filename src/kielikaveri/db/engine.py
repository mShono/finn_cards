from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from kielikaveri.db.models import Base


def make_engine(database_url: str):
    # Bound values (message texts, book chapters) stay out of exception
    # messages and out of the log if SQL statement logging is ever switched
    # on - the SQL itself stays readable.
    return create_async_engine(database_url, hide_parameters=True)


def make_session_factory(engine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def create_all(engine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
