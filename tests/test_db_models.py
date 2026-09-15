from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from kielikaveri.db.decks import create_deck
from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import (
    Card,
    CardState,
    CardType,
    Note,
    NoteKind,
    Review,
    User,
    is_note_duplicate_error,
)


@pytest.fixture
async def session_factory(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    await create_all(engine)
    yield make_session_factory(engine)
    await engine.dispose()


async def test_roundtrip_user_note_card_review(session_factory):
    due = datetime(2026, 9, 1, 6, 0, tzinfo=UTC)

    async with session_factory() as session:
        session.add(User(id=1))
        session.add(
            Note(
                id="note-1",
                user_id=1,
                lemma="hakea",
                pos="verbi",
                translation_ru="искать",
                example_fi="Haen töitä.",
                example_ru="Я ищу работу.",
                kind=NoteKind.word,
                meta={"forms_source": "fst", "forms_verified": True, "origin": "error"},
            )
        )
        session.add(
            Card(
                id="card-1",
                note_id="note-1",
                user_id=1,
                type=CardType.recognition,
                state=CardState.learning,
                due=due,
            )
        )
        session.add(Review(card_id="card-1", user_id=1, rating=3))
        await session.commit()

    async with session_factory() as session:
        note = await session.get(Note, "note-1")
        assert note.lemma == "hakea"
        assert note.kind == NoteKind.word
        assert note.meta["forms_source"] == "fst"

        card = await session.get(Card, "card-1")
        assert card.note_id == "note-1"
        assert card.due == due

        reviews = (await session.scalars(select(Review).where(Review.card_id == "card-1"))).all()
        assert len(reviews) == 1
        assert reviews[0].rating == 3


async def test_note_without_pos_is_allowed(session_factory):
    async with session_factory() as session:
        session.add(User(id=2))
        session.add(
            Note(
                id="note-2",
                user_id=2,
                lemma="hakea + partitiivi",
                pos=None,
                translation_ru="искать + партитив",
                example_fi="Haen töitä.",
                example_ru="Я ищу работу.",
                kind=NoteKind.pattern,
                meta={"forms_source": "fst", "forms_verified": True, "origin": "text"},
            )
        )
        await session.commit()

    async with session_factory() as session:
        note = await session.get(Note, "note-2")
        assert note.pos is None
        assert note.kind == NoteKind.pattern


# --- (user, deck, lemma, pos) uniqueness -----------------------------------
#
# The DB's half of the dedup contract /add enforces in Python
# (ingest.existing_note_keys): one note per word per deck, the same word
# allowed in another deck. A concurrency test belongs in its own task; these
# pin the constraint itself.


def make_note(note_id: str, user_id: int, lemma: str, pos: str | None, deck_id: str | None) -> Note:
    return Note(
        id=note_id,
        user_id=user_id,
        lemma=lemma,
        pos=pos,
        translation_ru="искать",
        example_fi="Haen töitä.",
        example_ru="Я ищу работу.",
        kind=NoteKind.word if pos is not None else NoteKind.pattern,
        deck_id=deck_id,
        meta={},
    )


async def _seed_deck(session_factory, user_id: int = 1, name: str = "Общая") -> str:
    async with session_factory() as session:
        session.add(User(id=user_id))
        deck = await create_deck(session, user_id, name)
        await session.commit()
        return deck.id


async def test_same_user_deck_lemma_pos_cannot_be_inserted_twice(session_factory):
    deck_id = await _seed_deck(session_factory)
    async with session_factory() as session:
        session.add(make_note("n1", 1, "hakea", "verbi", deck_id))
        await session.commit()

    async with session_factory() as session:
        session.add(make_note("n2", 1, "hakea", "verbi", deck_id))
        with pytest.raises(IntegrityError) as caught:
            await session.commit()
    assert is_note_duplicate_error(caught.value)


async def test_same_lemma_in_two_decks_is_allowed(session_factory):
    deck_a = await _seed_deck(session_factory)
    async with session_factory() as session:
        deck_b = await create_deck(session, 1, "talkoot")
        await session.commit()
        deck_b_id = deck_b.id

    async with session_factory() as session:
        session.add(make_note("n1", 1, "hakea", "verbi", deck_a))
        session.add(make_note("n2", 1, "hakea", "verbi", deck_b_id))
        await session.commit()

    async with session_factory() as session:
        notes = (await session.scalars(select(Note))).all()
    assert {n.deck_id for n in notes} == {deck_a, deck_b_id}


async def test_same_lemma_for_two_users_is_allowed(session_factory):
    deck_1 = await _seed_deck(session_factory, user_id=1)
    deck_2 = await _seed_deck(session_factory, user_id=2, name="Omat")

    async with session_factory() as session:
        session.add(make_note("n1", 1, "hakea", "verbi", deck_1))
        session.add(make_note("n2", 2, "hakea", "verbi", deck_2))
        await session.commit()

    async with session_factory() as session:
        notes = (await session.scalars(select(Note))).all()
    assert len(notes) == 2


async def test_same_lemma_with_a_different_pos_is_allowed(session_factory):
    # Homonyms across parts of speech stay distinct notes - pos is part of the
    # key exactly for this (ingest.canonical_key).
    deck_id = await _seed_deck(session_factory)
    async with session_factory() as session:
        session.add(make_note("n1", 1, "kuusi", "substantiivi", deck_id))
        session.add(make_note("n2", 1, "kuusi", "numeraali", deck_id))
        await session.commit()

    async with session_factory() as session:
        notes = (await session.scalars(select(Note))).all()
    assert len(notes) == 2


async def test_two_patterns_with_null_pos_cannot_duplicate(session_factory):
    # kind="pattern" has no pos (cards/schema.json requires it only for
    # kind="word"), and SQLite counts NULLs as distinct in a unique index - so
    # the index coalesces NULL, or patterns would be the one thing free to
    # duplicate.
    deck_id = await _seed_deck(session_factory)
    async with session_factory() as session:
        session.add(make_note("n1", 1, "hakea + partitiivi", None, deck_id))
        await session.commit()

    async with session_factory() as session:
        session.add(make_note("n2", 1, "hakea + partitiivi", None, deck_id))
        with pytest.raises(IntegrityError) as caught:
            await session.commit()
    assert is_note_duplicate_error(caught.value)


async def test_a_word_and_a_pattern_sharing_a_lemma_are_distinct(session_factory):
    deck_id = await _seed_deck(session_factory)
    async with session_factory() as session:
        session.add(make_note("n1", 1, "hakea", "verbi", deck_id))
        session.add(make_note("n2", 1, "hakea", None, deck_id))
        await session.commit()

    async with session_factory() as session:
        notes = (await session.scalars(select(Note))).all()
    assert len(notes) == 2


async def test_notes_without_a_deck_cannot_duplicate(session_factory):
    # import_cards.py leaves deck_id NULL, as do rows predating decks - the
    # same COALESCE reason as pos above.
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(make_note("n1", 1, "hakea", "verbi", None))
        await session.commit()

    async with session_factory() as session:
        session.add(make_note("n2", 1, "hakea", "verbi", None))
        with pytest.raises(IntegrityError) as caught:
            await session.commit()
    assert is_note_duplicate_error(caught.value)


async def test_a_deckless_note_does_not_block_the_same_word_in_a_deck(session_factory):
    deck_id = await _seed_deck(session_factory)
    async with session_factory() as session:
        session.add(make_note("n1", 1, "hakea", "verbi", None))
        session.add(make_note("n2", 1, "hakea", "verbi", deck_id))
        await session.commit()

    async with session_factory() as session:
        notes = (await session.scalars(select(Note))).all()
    assert len(notes) == 2


async def test_is_note_duplicate_error_rejects_an_unrelated_integrity_error(session_factory):
    # The race handler in bot/add.py must not swallow a NOT NULL miss or a
    # broken FK as "already added".
    async with session_factory() as session:
        session.add(User(id=1))
        await session.commit()

    async with session_factory() as session:
        session.add(make_note("n1", 1, None, "verbi", None))
        with pytest.raises(IntegrityError) as caught:
            await session.commit()
    assert not is_note_duplicate_error(caught.value)
