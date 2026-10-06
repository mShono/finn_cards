"""`alembic upgrade head` must build the same schema as Base.metadata.create_all.

Every other test builds its DB with create_all, while production runs the
migrations - so a model change without a matching migration (or a migration
that drifts from the model) passes the whole suite and only shows up on deploy.
"""

import warnings
from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, text

from alembic import command
from kielikaveri.db.models import Base

ALEMBIC_DIR = Path(__file__).parent.parent / "alembic"

INDEX_SQL = text(
    "SELECT name, sql FROM sqlite_master WHERE type = 'index' AND sql IS NOT NULL ORDER BY name"
)


@pytest.fixture
def migrated_url(tmp_path, monkeypatch):
    # env.py lets DATABASE_URL override the URL - never migrate a real DB here.
    monkeypatch.delenv("DATABASE_URL", raising=False)
    # A Config without an ini file: env.py then skips fileConfig(), which would
    # otherwise reset logging for the rest of the test session.
    config = Config()
    config.set_main_option("script_location", str(ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{tmp_path}/migrated.db")
    command.upgrade(config, "head")
    return f"sqlite:///{tmp_path}/migrated.db"


def test_migrations_match_models(migrated_url):
    engine = create_engine(migrated_url)
    try:
        with engine.connect() as conn, warnings.catch_warnings():
            # SQLite can't reflect expression-based indexes, so autogenerate
            # skips uq_notes_user_deck_lemma_pos - test_migrated_indexes_match_models
            # covers it instead.
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
