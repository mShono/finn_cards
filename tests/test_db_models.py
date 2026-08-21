from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import Card, CardState, CardType, Note, NoteKind, Review, User


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
