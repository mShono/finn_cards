"""`alembic upgrade head` must build the same schema as Base.metadata.create_all.

Every other test builds its DB with create_all, while production runs the
migrations - so a model change without a matching migration (or a migration
that drifts from the model) passes the whole suite and only shows up on deploy.

What compare_metadata does NOT see on SQLite, and what covers it instead:
- expression-based indexes (it can't reflect them) - test_migrated_indexes_match_models
  compares the raw CREATE INDEX statements from sqlite_master, all indexes included;
- CHECK constraints - autogenerate doesn't compare them, and SQLite reflection
  only reports named ones. Name every CheckConstraint
  (CheckConstraint(..., name="ck_...")) and check its migration by hand;
- VARCHAR length - SQLite ignores it, so String(50) vs String(100) is no diff.
Full table DDL is deliberately not compared: after batch migrations the column
order legitimately differs from create_all.

These tests and fixtures must stay synchronous: alembic/env.py calls
asyncio.run(), which fails inside an already running event loop (async test).
"""

import warnings
from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, text

from alembic import command
from kielikaveri.db.models import Base

ALEMBIC_DIR = Path(__file__).parent.parent / "alembic"

INDEX_SQL = text(
    "SELECT name, sql FROM sqlite_master WHERE type = 'index' AND sql IS NOT NULL ORDER BY name"
)


@pytest.fixture
def alembic_config(tmp_path, monkeypatch):
    # env.py lets DATABASE_URL override the URL - never migrate a real DB here.
    monkeypatch.delenv("DATABASE_URL", raising=False)
    # A Config without an ini file: env.py then skips fileConfig(), which would
    # otherwise reset logging for the rest of the test session.
    config = Config()
    config.set_main_option("script_location", str(ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{tmp_path}/migrated.db")
    return config


@pytest.fixture
def migrated_url(alembic_config, tmp_path):
    command.upgrade(alembic_config, "head")
    return f"sqlite:///{tmp_path}/migrated.db"


def _schema(url):
    """Tables (columns + foreign keys) and raw index DDL of a SQLite DB.

    Not the raw table DDL, and order-independent: a batch rebuild emits
    FOREIGN KEY clauses in an order that varies between runs, and a downgrade
    re-adds a dropped column at the end rather than at its old position.
    """
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            tables = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
            ).scalars()
            schema = {
                table: (
                    # (name, type, notnull, dflt_value, pk) - without cid, the position
                    sorted(
                        tuple(col[1:])
                        for col in conn.execute(text(f'PRAGMA table_info("{table}")'))
                    ),
                    # (table, from, to, on_update, on_delete)
                    sorted(
                        tuple(fk[2:7])
                        for fk in conn.execute(text(f'PRAGMA foreign_key_list("{table}")'))
                    ),
                )
                for table in tables.all()
            }
            schema["indexes"] = conn.execute(INDEX_SQL).all()
            return schema
    finally:
        engine.dispose()


def test_migrations_match_models(migrated_url):
    engine = create_engine(migrated_url)
    try:
        with engine.connect() as conn, warnings.catch_warnings():
            # SQLite can't reflect expression-based indexes, so autogenerate
            # skips every one of them (today: uq_notes_user_deck_lemma_pos).
            # test_migrated_indexes_match_models compares all index DDL raw,
            # expression indexes included, so nothing is lost here.
            warnings.filterwarnings("ignore", message=".*expression-based index")
            context = MigrationContext.configure(
                conn, opts={"compare_type": True, "compare_server_default": True}
            )
            diff = compare_metadata(context, Base.metadata)
    finally:
        engine.dispose()
    assert diff == []


def test_migrated_indexes_match_models(migrated_url, tmp_path):
    reference = create_engine(f"sqlite:///{tmp_path}/create_all.db")
    migrated = create_engine(migrated_url)
    try:
        Base.metadata.create_all(reference)
        with reference.connect() as ref_conn, migrated.connect() as mig_conn:
            assert mig_conn.execute(INDEX_SQL).all() == ref_conn.execute(INDEX_SQL).all()
    finally:
        reference.dispose()
        migrated.dispose()


def test_migrations_stairway(alembic_config, tmp_path):
    """Every revision's downgrade() must undo its upgrade() exactly.

    For each revision in order: upgrade to it, downgrade one step - the schema
    must be the previous revision's again (a leftover column can survive the
    re-upgrade unnoticed, batch add_column just rebuilds over it) - then
    upgrade again, which must match the first upgrade. Step by step rather
    than one head -> base -> head round trip: the initial schema's downgrade
    drops whole tables, which would hide an index or column a later downgrade
    left behind.
    """
    url = f"sqlite:///{tmp_path}/migrated.db"
    script = ScriptDirectory.from_config(alembic_config)
    revisions = [rev.revision for rev in script.walk_revisions("base", "heads")]
    previous = None
    for revision in reversed(revisions):
        command.upgrade(alembic_config, revision)
        upgraded = _schema(url)
        command.downgrade(alembic_config, "-1")
        # Nothing to compare at base: the fresh file had no alembic_version.
        if previous is not None:
            assert _schema(url) == previous, f"downgrade of {revision}"
        command.upgrade(alembic_config, revision)
        assert _schema(url) == upgraded, f"re-upgrade of {revision}"
        previous = upgraded
