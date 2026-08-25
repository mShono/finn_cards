from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import Card, CardType, Note, NoteKind, User
from kielikaveri.srs.graduation import ensure_card_types, sync_user_card_types

NOW = datetime(2026, 8, 24, 10, 0, tzinfo=UTC)


@pytest.fixture
async def session_factory(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    await create_all(engine)
    yield make_session_factory(engine)
    await engine.dispose()


def make_note(note_id: str, user_id: int, **meta_overrides) -> Note:
    meta = {"forms_source": "fst", "forms_verified": False, "origin": "error"}
    meta.update(meta_overrides)
    return Note(
        id=note_id,
        user_id=user_id,
        lemma="hakea",
        pos="verbi",
        translation_ru="искать",
        example_fi="Haen töitä.",
        example_ru="Я ищу работу.",
        kind=NoteKind.word,
        meta=meta,
    )


async def _cards_for(session_factory, note_id) -> dict[CardType, Card]:
    async with session_factory() as session:
        cards = (await session.scalars(select(Card).where(Card.note_id == note_id))).all()
    return {c.type: c for c in cards}


async def test_a_note_with_no_cards_gets_a_recognition_card(session_factory):
    async with session_factory() as session:
        session.add(User(id=1))
        note = make_note("note-1", 1)
        session.add(note)
        await session.flush()

        created = await ensure_card_types(session, note, NOW)
        await session.commit()

    assert {c.type for c in created} == {CardType.recognition}
    cards = await _cards_for(session_factory, "note-1")
    assert cards[CardType.recognition].due == NOW


async def test_calling_ensure_card_types_twice_does_not_duplicate_recognition(session_factory):
    async with session_factory() as session:
        session.add(User(id=1))
        note = make_note("note-1", 1)
        session.add(note)
        await session.flush()
        await ensure_card_types(session, note, NOW)
        await ensure_card_types(session, note, NOW)
        await session.commit()

    cards = await _cards_for(session_factory, "note-1")
    async with session_factory() as session:
        count = len((await session.scalars(select(Card).where(Card.note_id == "note-1"))).all())
    assert count == 1
    assert CardType.recognition in cards


async def test_production_opens_once_recognition_stability_crosses_threshold(session_factory):
    async with session_factory() as session:
        session.add(User(id=1))
        note = make_note("note-1", 1)
        session.add(note)
        session.add(
            Card(
                id="rec-1",
                note_id="note-1",
                user_id=1,
                type=CardType.recognition,
                due=NOW,
                stability=1.0,
            )
        )
        await session.flush()

        created = await ensure_card_types(session, note, NOW)
        assert created == []  # stability below threshold - production stays closed

        (await session.get(Card, "rec-1")).stability = 3.5
        created = await ensure_card_types(session, note, NOW)
        await session.commit()

    assert {c.type for c in created} == {CardType.production}


async def test_inflection_opens_only_when_forms_are_filled_and_verified(session_factory):
    async with session_factory() as session:
        session.add(User(id=1))
        note = make_note("note-1", 1, forms_verified=False, principal_forms={"preesens_1s": "haen"})
        session.add(note)
        await session.flush()

        created = await ensure_card_types(session, note, NOW)
        assert CardType.inflection not in {c.type for c in created}

        note.meta = {**note.meta, "forms_verified": True}
        created = await ensure_card_types(session, note, NOW)
        await session.commit()

    assert CardType.inflection in {c.type for c in created}


async def test_inflection_stays_closed_without_principal_forms_even_if_verified(session_factory):
    async with session_factory() as session:
        session.add(User(id=1))
        note = make_note("note-1", 1, forms_verified=True, principal_forms={})
        session.add(note)
        await session.flush()

        created = await ensure_card_types(session, note, NOW)
        await session.commit()

    assert CardType.inflection not in {c.type for c in created}


async def test_sync_user_card_types_covers_every_note_for_that_user_only(session_factory):
    async with session_factory() as session:
        session.add(User(id=1))
        session.add(User(id=2))
        session.add(make_note("note-1", 1))
        session.add(make_note("note-2", 1))
        session.add(make_note("note-3", 2))
        await session.flush()

        created = await sync_user_card_types(session, 1, NOW)
        await session.commit()

    assert {c.note_id for c in created} == {"note-1", "note-2"}
    cards_3 = await _cards_for(session_factory, "note-3")
    assert cards_3 == {}
