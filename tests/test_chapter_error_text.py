"""chapters.error keeps the exception's message only for the translator's own
exceptions; for the rest the class alone (plan М17) - str() of them can quote
the book. The strings here are made up and short.

The migration tests stay synchronous: alembic/env.py calls asyncio.run().
"""

import json
from pathlib import Path
from types import SimpleNamespace

import openai
import pytest
from alembic.config import Config
from sqlalchemy import create_engine, event, text

from alembic import command
from kielikaveri.bot.books import failure_reason
from kielikaveri.llm.breaker import BudgetExceededError, CircuitOpenError
from kielikaveri.translate.pipeline import MisalignedError, RefusedError
from kielikaveri.translate.worker import _ChapterGone, _error_text

ALEMBIC_DIR = Path(__file__).parent.parent / "alembic"
BEFORE = "6b3d9e2f4a17"
SECRET = "Tuuli puhalsi yli järven"
HIDDEN = "(details hidden)"


def _json_error() -> json.JSONDecodeError:
    try:
        json.loads('{"fi": "' + SECRET)
    except json.JSONDecodeError as exc:
        return exc
    raise AssertionError("unreachable")


def _api_error() -> openai.APIError:
    return openai.APIError(f"bad input: {SECRET}", SimpleNamespace(), body=None)


@pytest.mark.parametrize(
    "exc",
    [
        MisalignedError("paragraph ids differ: missing=[3] extra=[]"),
        CircuitOpenError("more than 50 OpenAI calls in 0:10:00 - stopped"),
        BudgetExceededError("more than 500 OpenAI calls in 24 hours - stopped"),
        _ChapterGone("chapter deleted"),
    ],
)
def test_own_exceptions_keep_their_message(exc):
    assert _error_text(exc) == f"{type(exc).__name__}: {exc}"


@pytest.mark.parametrize(
    "exc",
    [
        RefusedError(f"I can't translate: {SECRET}"),
        _api_error(),
        ValueError(SECRET),
        KeyError(SECRET),
        _json_error(),
    ],
)
def test_other_exceptions_hide_their_message(exc):
    text_ = _error_text(exc)
    assert text_ == f"{type(exc).__name__}: {HIDDEN}"
    assert SECRET not in text_


def test_subclass_of_a_safe_class_is_hidden():
    # Exact types only: a subclass may carry a different message.
    class Custom(MisalignedError):
        pass

    assert _error_text(Custom(SECRET)) == f"Custom: {HIDDEN}"


def test_error_text_is_capped():
    exc = MisalignedError("paragraph ids differ: missing=" + str(list(range(500))))
    assert len(_error_text(exc)) == 500
    assert _error_text(exc).startswith("MisalignedError: paragraph ids differ")


@pytest.mark.parametrize(
    ("exc", "reason"),
    [
        (RefusedError(SECRET), "модель отказалась переводить этот фрагмент"),
        (MisalignedError("empty paragraph in translation"), "в переводе потерялись абзацы"),
        (CircuitOpenError("x"), "слишком много обращений к OpenAI подряд, сработал предохранитель"),
        (BudgetExceededError("x"), "дневной лимит переводов исчерпан, нажми 🔁 завтра"),
        (_json_error(), "ошибка при переводе"),
    ],
)
def test_failure_reason_reads_the_new_strings(exc, reason):
    assert failure_reason(_error_text(exc)) == reason


def test_json_error_prefix_kept():
    assert _error_text(_json_error()).startswith("JSONDecodeError:")


# --- the data migration ---


@pytest.fixture
def alembic_config(tmp_path, monkeypatch):
    # env.py lets DATABASE_URL override the URL - never migrate a real DB here.
    monkeypatch.delenv("DATABASE_URL", raising=False)
    config = Config()
    config.set_main_option("script_location", str(ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{tmp_path}/migrated.db")
    return config


def _engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/migrated.db")

    @event.listens_for(engine, "connect")
    def _fk_on(dbapi_connection, _record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    return engine


ROWS = {
    "null": None,
    "refused": f"RefusedError: {SECRET}",
    "json": f"JSONDecodeError: Unterminated string: {SECRET}",
    "api": f"APIError: {SECRET}",
    "empty_message": "KeyError: ",
    "misaligned": "MisalignedError: paragraph ids differ: missing=[1] extra=[]",
    "circuit": "CircuitOpenError: more than 50 OpenAI calls in 0:10:00 - stopped",
    "budget": "BudgetExceededError: more than 500 OpenAI calls in 24 hours - stopped",
    "already": f"ValueError: {HIDDEN}",
    "bare_class": "TimeoutError",
    "bare_text": SECRET,
    "text_prefix": f"{SECRET}: more",
}

EXPECTED = {
    "null": None,
    "refused": f"RefusedError: {HIDDEN}",
    "json": f"JSONDecodeError: {HIDDEN}",
    "api": f"APIError: {HIDDEN}",
    "empty_message": f"KeyError: {HIDDEN}",
    "misaligned": ROWS["misaligned"],
    "circuit": ROWS["circuit"],
    "budget": ROWS["budget"],
    "already": ROWS["already"],
    "bare_class": "TimeoutError",
    "bare_text": f"UnknownError: {HIDDEN}",
    "text_prefix": f"UnknownError: {HIDDEN}",
}


def test_migration_hides_foreign_messages(alembic_config, tmp_path):
    command.upgrade(alembic_config, BEFORE)
    engine = _engine(tmp_path)
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO users (id, created_at) VALUES (1, '2026-11-24 00:00:00')"))
        conn.execute(
            text(
                "INSERT INTO books (id, user_id, title, source_lang, source_hash, front_matter,"
                " created_at) VALUES ('b1', 1, 'Kirja', 'ru', 'h', '[]', '2026-11-24 00:00:00')"
            )
        )
        for number, (chapter_id, error) in enumerate(ROWS.items(), start=1):
            conn.execute(
                text(
                    "INSERT INTO chapters (id, book_id, number, title, source_paragraphs,"
                    " status, error) VALUES (:id, 'b1', :number, 't', '[]', 'failed', :error)"
                ),
                {"id": chapter_id, "number": number, "error": error},
            )
    engine.dispose()

    command.upgrade(alembic_config, "head")

    engine = _engine(tmp_path)
    with engine.connect() as conn:
        got = dict(conn.execute(text("SELECT id, error FROM chapters")).all())
    engine.dispose()
    assert got == EXPECTED

    # Downgrade is a no-op on the data and must not fail.
    command.downgrade(alembic_config, "-1")
    engine = _engine(tmp_path)
    with engine.connect() as conn:
        after_down = dict(conn.execute(text("SELECT id, error FROM chapters")).all())
    engine.dispose()
    assert after_down == EXPECTED
