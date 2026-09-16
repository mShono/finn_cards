import json

import jsonschema
import pytest
from sqlalchemy import select

from kielikaveri.db.decks import create_deck
from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import Note, User
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


async def test_import_does_not_clash_with_the_same_word_already_in_a_deck(session_factory):
    # import_notes leaves deck_id NULL, so its rows sit in their own slot of
    # the (user, deck, lemma, pos) key - an imported "hakea" and a /add-ed
    # "hakea" in a real deck are two different notes, not a constraint breach.
    async with session_factory() as session:
        session.add(User(id=1))
        deck = await create_deck(session, 1, "Общая")
        session.add(
            Note(
                id="added-hakea",
                user_id=1,
                lemma="hakea",
                pos="verbi",
                translation_ru="искать",
                example_fi="x",
                example_ru="y",
                kind="word",
                deck_id=deck.id,
                meta={},
            )
        )
        await session.commit()

    imported = await import_notes(session_factory, user_id=1, cards_dir=DEFAULT_CARDS_DIR)

    assert len(imported) == 3
    async with session_factory() as session:
        hakeas = (await session.scalars(select(Note).where(Note.lemma == "hakea"))).all()
    assert {n.deck_id for n in hakeas} == {deck.id, None}


async def test_import_refuses_a_file_duplicating_an_existing_deckless_note(
    session_factory, tmp_path
):
    await import_notes(session_factory, user_id=1, cards_dir=DEFAULT_CARDS_DIR)

    again = tmp_path / "again"
    again.mkdir()
    payload = json.loads((DEFAULT_CARDS_DIR / "hakea.json").read_text())
    payload["id"] = "a-different-id"
    (again / "hakea-copy.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="already has a deckless note"):
        await import_notes(session_factory, user_id=1, cards_dir=again)

    async with session_factory() as session:
        notes = (await session.scalars(select(Note))).all()
    assert len(notes) == 3  # the first import's rows, untouched


async def test_import_refuses_two_files_for_the_same_word_in_one_run(session_factory, tmp_path):
    both = tmp_path / "both"
    both.mkdir()
    payload = json.loads((DEFAULT_CARDS_DIR / "hakea.json").read_text())
    (both / "a.json").write_text(json.dumps(payload))
    (both / "b.json").write_text(json.dumps({**payload, "id": "a-different-id"}))

    with pytest.raises(ValueError, match="already has a deckless note"):
        await import_notes(session_factory, user_id=1, cards_dir=both)

    async with session_factory() as session:
        notes = (await session.scalars(select(Note))).all()
    assert notes == []  # one commit at the end, so a refusal imports nothing


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
