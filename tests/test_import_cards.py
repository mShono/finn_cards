import json

import jsonschema
import pytest
from sqlalchemy import select

from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import Note, Source, SourceType
from kielikaveri.import_cards import DEFAULT_CARDS_DIR, import_notes


@pytest.fixture
async def session_factory(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    await create_all(engine)
    yield make_session_factory(engine)
    await engine.dispose()


async def test_imports_all_phase_0_examples(session_factory):
    imported = await import_notes(session_factory, user_id=1, cards_dir=DEFAULT_CARDS_DIR)

    assert len(imported) == 3

    async with session_factory() as session:
        notes = (await session.scalars(select(Note))).all()
        assert {n.lemma for n in notes} == {"hakea", "hammas", "pitää"}
        assert all(n.user_id == 1 for n in notes)


async def test_reimporting_is_idempotent(session_factory):
    await import_notes(session_factory, user_id=1, cards_dir=DEFAULT_CARDS_DIR)
    second_run = await import_notes(session_factory, user_id=1, cards_dir=DEFAULT_CARDS_DIR)

    assert second_run == []
    async with session_factory() as session:
        notes = (await session.scalars(select(Note))).all()
        assert len(notes) == 3


async def test_invalid_note_is_rejected(session_factory, tmp_path):
    bad_dir = tmp_path / "bad_cards"
    bad_dir.mkdir()
    (bad_dir / "broken.json").write_text(json.dumps({"lemma": "no id or required fields"}))

    with pytest.raises(jsonschema.ValidationError):
        await import_notes(session_factory, user_id=1, cards_dir=bad_dir)


@pytest.mark.xfail(
    reason="import_notes dedups by note.id alone, not (user_id, id) - a second "
    "user importing the same examples silently gets 0 notes. Known gap, plan's "
    "'Известные пробелы' (2026-08-21); strict so a real fix must remove this marker.",
    strict=True,
)
async def test_import_gives_each_user_their_own_notes(session_factory):
    await import_notes(session_factory, user_id=1, cards_dir=DEFAULT_CARDS_DIR)
    second_user_imported = await import_notes(
        session_factory, user_id=2, cards_dir=DEFAULT_CARDS_DIR
    )

    assert len(second_user_imported) == 3
    async with session_factory() as session:
        user_2_notes = (await session.scalars(select(Note).where(Note.user_id == 2))).all()
        assert {n.lemma for n in user_2_notes} == {"hakea", "hammas", "pitää"}


@pytest.mark.xfail(
    reason="import_notes never creates Source rows or sets Note.source_id - "
    "meta.source data is captured on import but the FK stays NULL. Known gap, "
    "plan's 'Известные пробелы' (2026-08-21); strict so a real fix must remove "
    "this marker.",
    strict=True,
)
async def test_import_populates_source_for_notes_with_meta_source(session_factory):
    await import_notes(session_factory, user_id=1, cards_dir=DEFAULT_CARDS_DIR)

    async with session_factory() as session:
        hakea = (await session.scalars(select(Note).where(Note.lemma == "hakea"))).one()
        assert hakea.source_id is not None

        source = await session.get(Source, hakea.source_id)
        assert source.type == SourceType.conversation
        assert source.ref == "разговор о поиске работы"
