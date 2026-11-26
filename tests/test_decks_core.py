"""decks_core: what the reader's deck screens add to the bot's - renaming
and deleting a deck, moving a word, the search and the word's details. No
aiogram, no HTTP (tests/test_reader_decks.py covers /api on top of it)."""

from datetime import UTC, datetime, timedelta

import pytest
from conftest import add_users
from sqlalchemy import func, select

from kielikaveri.db.books import chapter_by_number, store_book
from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import Card, CardStatus, CardType, Deck, Note, NoteKind, Review
from kielikaveri.decks_core import (
    DuplicateNoteError,
    delete_deck,
    move_note,
    note_detail,
    rename_deck,
    search_notes,
)
from kielikaveri.translate.chapters import split_book

OWNER, STRANGER = 1, 2
NOW = datetime(2026, 11, 26, 10, 0, tzinfo=UTC)
BOOK_TEXT = "Kirja\n\n1\n\nТекст.\n\n2\n\nТекст."


@pytest.fixture
async def session_factory(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    await create_all(engine)
    session_factory = make_session_factory(engine)
    await add_users(session_factory, OWNER, STRANGER)
    yield session_factory
    await engine.dispose()


async def add_deck(session_factory, deck_id, user_id=OWNER, chapter_id=None) -> str:
    async with session_factory() as session:
        session.add(Deck(id=deck_id, user_id=user_id, name=deck_id, chapter_id=chapter_id))
        await session.commit()
    return deck_id


async def add_note(
    session_factory, note_id, deck_id, user_id=OWNER, lemma=None, pos="substantiivi", **columns
) -> str:
    defaults = {
        "translation_ru": f"перевод {note_id}",
        "example_fi": "Esimerkki.",
        "example_ru": "Пример.",
        "kind": NoteKind.word,
        "meta": {},
        "created_at": NOW,
    }
    async with session_factory() as session:
        session.add(
            Note(
                id=note_id,
                user_id=user_id,
                deck_id=deck_id,
                lemma=lemma or f"sana-{note_id}",
                pos=pos,
                **{**defaults, **columns},
            )
        )
        await session.commit()
    return note_id


async def add_card_with_review(session_factory, card_id, note_id, user_id=OWNER) -> None:
    async with session_factory() as session:
        session.add(Card(id=card_id, note_id=note_id, user_id=user_id, type=CardType.recognition))
        await session.flush()
        session.add(Review(card_id=card_id, user_id=user_id, rating=3, reviewed_at=NOW))
        await session.commit()


async def count(session_factory, model, *where) -> int:
    async with session_factory() as session:
        return await session.scalar(select(func.count()).select_from(model).where(*where))


# --- rename --------------------------------------------------------------------------


async def test_rename_deck_renames_only_the_users_own(session_factory):
    await add_deck(session_factory, "mine")
    await add_deck(session_factory, "theirs", user_id=STRANGER)

    async with session_factory() as session:
        renamed = await rename_deck(session, OWNER, "mine", "Uusi")
        assert renamed is not None and renamed.name == "Uusi"
        assert await rename_deck(session, OWNER, "theirs", "Uusi") is None
        assert await rename_deck(session, OWNER, "missing", "Uusi") is None

    async with session_factory() as session:
        assert (await session.get(Deck, "theirs")).name == "theirs"
        assert (await session.get(Deck, "mine")).name == "Uusi"


# --- delete a deck -------------------------------------------------------------------


async def test_delete_deck_takes_its_notes_cards_and_reviews_and_nothing_else(session_factory):
    await add_deck(session_factory, "gone")
    await add_deck(session_factory, "kept")
    await add_deck(session_factory, "theirs", user_id=STRANGER)
    for i in range(2):
        await add_note(session_factory, f"g{i}", "gone")
        await add_card_with_review(session_factory, f"gc{i}", f"g{i}")
    await add_note(session_factory, "k", "kept")
    await add_card_with_review(session_factory, "kc", "k")
    await add_note(session_factory, "t", "theirs", user_id=STRANGER)
    await add_card_with_review(session_factory, "tc", "t", user_id=STRANGER)

    async with session_factory() as session:
        assert await delete_deck(session, OWNER, "gone") == 2

    assert await count(session_factory, Deck, Deck.id == "gone") == 0
    assert await count(session_factory, Note, Note.deck_id == "gone") == 0
    assert await count(session_factory, Card, Card.id.in_(["gc0", "gc1"])) == 0
    assert await count(session_factory, Review, Review.card_id.in_(["gc0", "gc1"])) == 0
    # The other deck of the same user and the stranger's are whole.
    assert await count(session_factory, Note) == 2
    assert await count(session_factory, Card) == 2
    assert await count(session_factory, Review) == 2
    assert await count(session_factory, Deck) == 2


async def test_delete_deck_refuses_someone_elses_and_a_second_time(session_factory):
    await add_deck(session_factory, "mine")
    await add_deck(session_factory, "theirs", user_id=STRANGER)
    await add_note(session_factory, "t", "theirs", user_id=STRANGER)

    async with session_factory() as session:
        assert await delete_deck(session, OWNER, "theirs") is None
        assert await delete_deck(session, OWNER, "mine") == 0
        assert await delete_deck(session, OWNER, "mine") is None

    assert await count(session_factory, Note, Note.id == "t") == 1


# --- move ----------------------------------------------------------------------------


async def test_move_note_keeps_its_cards_and_reviews(session_factory):
    await add_deck(session_factory, "a")
    await add_deck(session_factory, "b")
    await add_note(session_factory, "n", "a")
    await add_card_with_review(session_factory, "c", "n")

    async with session_factory() as session:
        moved = await move_note(session, OWNER, "n", "b")
    assert moved is not None and moved.deck_id == "b"

    async with session_factory() as session:
        card = await session.get(Card, "c")
        assert card.note_id == "n"
    assert await count(session_factory, Review, Review.card_id == "c") == 1


async def test_move_note_into_a_deck_with_the_same_word_is_a_duplicate(session_factory):
    await add_deck(session_factory, "a")
    await add_deck(session_factory, "b")
    await add_note(session_factory, "n", "a", lemma="talo")
    await add_note(session_factory, "there", "b", lemma="talo")
    # Another part of speech is another note: the index has pos in it.
    await add_note(session_factory, "verb", "a", lemma="talo", pos="verbi")

    async with session_factory() as session:
        with pytest.raises(DuplicateNoteError):
            await move_note(session, OWNER, "n", "b")
    async with session_factory() as session:
        assert (await session.get(Note, "n")).deck_id == "a"
        assert (await move_note(session, OWNER, "verb", "b")).deck_id == "b"


async def test_move_note_same_deck_others_and_no_deck(session_factory):
    await add_deck(session_factory, "a")
    await add_deck(session_factory, "theirs", user_id=STRANGER)
    await add_note(session_factory, "n", "a")
    await add_note(session_factory, "loose", None)
    await add_note(session_factory, "t", "theirs", user_id=STRANGER)

    async with session_factory() as session:
        assert (await move_note(session, OWNER, "n", "a")).deck_id == "a"
        assert await move_note(session, OWNER, "n", "theirs") is None
        assert await move_note(session, OWNER, "t", "a") is None
        assert await move_note(session, OWNER, "n", "missing") is None
        # A note in no deck (import_cards.py) can be put into one.
        assert (await move_note(session, OWNER, "loose", "a")).deck_id == "a"


# --- search --------------------------------------------------------------------------


async def found_ids(session_factory, query, user_id=OWNER) -> list[str]:
    async with session_factory() as session:
        return [found.note.id for found in await search_notes(session, user_id, query)]


async def test_search_ignores_case_of_ä_and_cyrillic(session_factory):
    await add_deck(session_factory, "a")
    await add_note(session_factory, "fi", "a", lemma="Äiti", translation_ru="мама")
    await add_note(session_factory, "ru", "a", lemma="talo", translation_ru="Дом")

    assert await found_ids(session_factory, "äiti") == ["fi"]
    assert await found_ids(session_factory, "ÄITI") == ["fi"]
    assert await found_ids(session_factory, "дом") == ["ru"]
    assert await found_ids(session_factory, "ДОМ") == ["ru"]
    assert await found_ids(session_factory, "МАМ") == ["fi"]


async def test_search_orders_exact_then_word_start_then_the_rest(session_factory):
    await add_deck(session_factory, "a")
    await add_note(session_factory, "inside", "a", lemma="kirjasto")
    await add_note(session_factory, "middle", "a", lemma="sanakirja")
    await add_note(session_factory, "start", "a", lemma="aaa", translation_ru="kirja- книга")
    await add_note(session_factory, "exact", "a", lemma="zzz", translation_ru="Kirja")

    # "kirjasto" starts with it too - a word start, by lemma after "aaa".
    assert await found_ids(session_factory, "kirja") == ["exact", "start", "inside", "middle"]


async def test_search_finds_puhekieli_and_only_the_users_notes(session_factory):
    await add_deck(session_factory, "a")
    await add_deck(session_factory, "theirs", user_id=STRANGER)
    await add_note(session_factory, "mä", "a", lemma="minä", meta={"puhekieli": "Mä"})
    await add_note(session_factory, "t", "theirs", user_id=STRANGER, lemma="mä")
    await add_note(session_factory, "loose", None, lemma="mäki")

    async with session_factory() as session:
        found = await search_notes(session, OWNER, "mä")
    assert [(f.note.id, f.deck.id if f.deck else None) for f in found] == [
        ("mä", "a"),
        ("loose", None),
    ]


async def test_search_empty_query_and_limit(session_factory):
    await add_deck(session_factory, "a")
    for i in range(60):
        await add_note(session_factory, f"n{i:02}", "a", lemma=f"sana{i:02}")

    assert await found_ids(session_factory, "") == []
    assert await found_ids(session_factory, "   ") == []
    assert len(await found_ids(session_factory, "sana")) == 50


# --- details -------------------------------------------------------------------------


async def test_note_detail_orders_cards_and_names_the_chapter(session_factory):
    async with session_factory() as session:
        book = await store_book(session, OWNER, "Kirja", BOOK_TEXT, split_book(BOOK_TEXT))
        await session.commit()
        chapter_id = (await chapter_by_number(session, book.id, 1)).id
    await add_deck(session_factory, "ch", chapter_id=chapter_id)
    meta = {"principal_forms": {"genetiivi": "talon", "partitiivi": "taloa"}}
    await add_note(session_factory, "n", "ch", lemma="talo", meta=meta)
    async with session_factory() as session:
        session.add_all(
            [
                Card(
                    id="i2", note_id="n", user_id=OWNER, type=CardType.inflection, form="partitiivi"
                ),
                Card(
                    id="i1", note_id="n", user_id=OWNER, type=CardType.inflection, form="genetiivi"
                ),
                Card(id="p", note_id="n", user_id=OWNER, type=CardType.production),
                Card(
                    id="r",
                    note_id="n",
                    user_id=OWNER,
                    type=CardType.recognition,
                    status=CardStatus.introduced,
                    due=NOW + timedelta(days=3),
                ),
            ]
        )
        await session.commit()

    async with session_factory() as session:
        detail = await note_detail(session, OWNER, "n")
        assert await note_detail(session, STRANGER, "n") is None

    assert [card.id for card in detail.cards] == ["r", "p", "i1", "i2"]
    assert detail.deck.id == "ch"
    assert detail.chapter.id == chapter_id
    assert detail.chapter.book_title == "Kirja"
