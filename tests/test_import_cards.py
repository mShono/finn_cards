import json

import jsonschema
import pytest
from sqlalchemy import select

from kielikaveri.db.decks import create_deck
from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import Note, User
from kielikaveri.import_cards import DEFAULT_CARDS_DIR, import_notes, note_id_for


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


async def test_import_gives_each_user_their_own_notes(session_factory):
    await import_notes(session_factory, user_id=1, cards_dir=DEFAULT_CARDS_DIR)
    second_user_imported = await import_notes(
        session_factory, user_id=2, cards_dir=DEFAULT_CARDS_DIR
    )

    assert len(second_user_imported) == 3
    async with session_factory() as session:
        user_2_notes = (await session.scalars(select(Note).where(Note.user_id == 2))).all()
        assert {n.lemma for n in user_2_notes} == {"hakea", "hammas", "pitää"}

    async with session_factory() as session:
        user_1_notes = (await session.scalars(select(Note).where(Note.user_id == 1))).all()
    assert len(user_1_notes) == 3
    # Separate rows, not user 1's notes reassigned or shared.
    assert not {n.id for n in user_1_notes} & {n.id for n in user_2_notes}


async def test_reimporting_is_idempotent_per_user(session_factory):
    await import_notes(session_factory, user_id=1, cards_dir=DEFAULT_CARDS_DIR)
    await import_notes(session_factory, user_id=2, cards_dir=DEFAULT_CARDS_DIR)

    assert await import_notes(session_factory, user_id=1, cards_dir=DEFAULT_CARDS_DIR) == []
    assert await import_notes(session_factory, user_id=2, cards_dir=DEFAULT_CARDS_DIR) == []
    async with session_factory() as session:
        notes = (await session.scalars(select(Note))).all()
    assert sorted(n.user_id for n in notes) == [1, 1, 1, 2, 2, 2]


async def test_reimport_skips_notes_imported_under_the_raw_file_id(session_factory):
    # Imports before per-user ids stored the file's id verbatim - a prod DB
    # still has those rows, and re-running the import must not duplicate them.
    payload = json.loads((DEFAULT_CARDS_DIR / "hakea.json").read_text())
    async with session_factory() as session:
        session.add(User(id=1))
        deck = await create_deck(session, 1, "Общая")
        session.add(
            Note(
                id=payload["id"],
                user_id=1,
                lemma="hakea",
                pos="verbi",
                translation_ru=payload["translation_ru"],
                example_fi=payload["example_fi"],
                example_ru=payload["example_ru"],
                kind="word",
                # Moved into a deck since - the deckless clash check would
                # not see it, only the id lookup does.
                deck_id=deck.id,
                meta=payload["meta"],
            )
        )
        await session.commit()

    imported = await import_notes(session_factory, user_id=1, cards_dir=DEFAULT_CARDS_DIR)

    assert len(imported) == 2
    assert note_id_for(1, payload["id"]) not in imported
    async with session_factory() as session:
        hakeas = (await session.scalars(select(Note).where(Note.lemma == "hakea"))).all()
    assert [n.id for n in hakeas] == [payload["id"]]


async def test_another_users_raw_file_id_does_not_block_import(session_factory):
    # User 1's legacy row holds the raw file id - user 2 still gets their own
    # note, and user 1's row is left as it was.
    payload = json.loads((DEFAULT_CARDS_DIR / "hakea.json").read_text())
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(
            Note(
                id=payload["id"],
                user_id=1,
                lemma="hakea",
                pos="verbi",
                translation_ru=payload["translation_ru"],
                example_fi=payload["example_fi"],
                example_ru=payload["example_ru"],
                kind="word",
                meta=payload["meta"],
            )
        )
        await session.commit()

    imported = await import_notes(session_factory, user_id=2, cards_dir=DEFAULT_CARDS_DIR)

    assert note_id_for(2, payload["id"]) in imported
    async with session_factory() as session:
        legacy = await session.get(Note, payload["id"])
        mine = await session.get(Note, note_id_for(2, payload["id"]))
    assert legacy.user_id == 1
    assert mine.user_id == 2
