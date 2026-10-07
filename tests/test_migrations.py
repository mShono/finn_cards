"""`alembic upgrade head` must build the same schema as Base.metadata.create_all.

Every other test builds its DB with create_all, while production runs the
migrations - so a model change without a matching migration (or a migration
that drifts from the model) passes the whole suite and only shows up on deploy.

What compare_metadata does NOT see on SQLite, and what covers it instead:
- expression-based indexes (it can't reflect them) - test_migrated_indexes_match_models
  compares the raw CREATE INDEX statement of every explicitly created index
  (not the sqlite_autoindex_* behind UNIQUE constraints: those have no DDL,
  and compare_metadata does compare unique constraints);
- CHECK constraints - autogenerate doesn't compare them, and SQLite reflection
  only reports named ones. Name every CheckConstraint
  (CheckConstraint(..., name="ck_...")) and check its migration by hand
  (test_migrations_stairway only checks that each downgrade undoes them);
- VARCHAR length - SQLite ignores it, so String(50) vs String(100) is no diff.
Full table DDL is deliberately not compared: after batch migrations the column
order legitimately differs from create_all.

These tests and fixtures must stay synchronous: alembic/env.py calls
asyncio.run(), which fails inside an already running event loop (async test).
"""

import re
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

CHECK_RE = re.compile(r"\bCHECK\s*\(", re.IGNORECASE)

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


def _checks(table_sql):
    """The CHECK (...) clauses of a CREATE TABLE statement, whitespace-normalized."""
    checks = []
    for match in CHECK_RE.finditer(table_sql):
        depth, quoted = 0, False
        for end in range(match.end() - 1, len(table_sql)):
            char = table_sql[end]
            if char == "'":
                quoted = not quoted
            elif not quoted and char == "(":
                depth += 1
            elif not quoted and char == ")":
                depth -= 1
                if depth == 0:
                    break
        checks.append(" ".join(table_sql[match.start() : end + 1].split()))
    return sorted(checks)


def _schema(url):
    """Tables (columns, foreign keys, indexes, CHECKs) and raw index DDL of a SQLite DB.

    Not the raw table DDL, and order-independent: a batch rebuild emits
    FOREIGN KEY clauses in an order that varies between runs, and a downgrade
    re-adds a dropped column at the end rather than at its old position.
    """
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            tables = conn.execute(
                text("SELECT name, sql FROM sqlite_master WHERE type = 'table' ORDER BY name")
            ).all()
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
                    # (unique, origin, partial, columns) - without the name: UNIQUE
                    # constraints get positional sqlite_autoindex_* names and no DDL
                    sorted(
                        (
                            index.unique,
                            index.origin,
                            index.partial,
                            tuple(
                                col.name
                                for col in conn.execute(text(f'PRAGMA index_info("{index.name}")'))
                            ),
                        )
                        for index in conn.execute(text(f'PRAGMA index_list("{table}")')).all()
                    ),
                    _checks(table_sql),
                )
                for table, table_sql in tables
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
    revisions = list(script.walk_revisions("base", "heads"))
    # downgrade "-1" and `previous` below assume one straight line of revisions.
    assert len(script.get_heads()) == 1 and not any(
        rev.is_branch_point or rev.is_merge_point for rev in revisions
    ), "revision history is not linear - this test needs a rework for branches/merges"
    # Base: an empty DB with only the (empty) alembic_version table.
    command.stamp(alembic_config, "base")
    previous = _schema(url)
    for revision in reversed([rev.revision for rev in revisions]):
        command.upgrade(alembic_config, revision)
        upgraded = _schema(url)
        command.downgrade(alembic_config, "-1")
        assert _schema(url) == previous, f"downgrade of {revision}"
        command.upgrade(alembic_config, revision)
        assert _schema(url) == upgraded, f"re-upgrade of {revision}"
        previous = upgraded
