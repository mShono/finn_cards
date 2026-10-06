"""Import note JSON files (cards/schema.json format) into the database.

Only `notes` are imported here - `cards/examples/*.json` are notes, not
review cards. Review cards (type: recognition/production/inflection)
get created gradually per phase 2's "postpone type" logic, not on import.

Usage: uv run python -m kielikaveri.import_cards --user-id 123
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from pathlib import Path

import jsonschema
from sqlalchemy import select

from kielikaveri.config import load_settings
from kielikaveri.db.engine import make_engine, make_session_factory
from kielikaveri.db.models import Note, User

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CARDS_DIR = REPO_ROOT / "cards" / "examples"
SCHEMA_PATH = REPO_ROOT / "cards" / "schema.json"
# Fixed forever - changing it would make every re-import duplicate its notes.
_NOTE_ID_NAMESPACE = uuid.UUID("5b0e7c1e-3f4a-4d8b-9c2a-6e1f0a7d4b93")


def note_id_for(user_id: int, file_id: str) -> str:
    """The id a file's note gets in `user_id`'s collection.

    notes.id is a global primary key, but a file id is shared by everyone who
    imports that file - inserting it verbatim lets only the first importer
    have the note. A uuid5 of (file id, user) is per-user yet stable, so a
    re-run for the same user finds the row it made last time.

    Hashed as a name under a fixed namespace rather than using the file id as
    the namespace: the validator doesn't enforce `"format": "uuid"`, so a
    file id isn't guaranteed to parse as one.
    """
    return str(uuid.uuid5(_NOTE_ID_NAMESPACE, f"{user_id}:{file_id}"))


def load_validator() -> jsonschema.Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text())
    return jsonschema.Draft202012Validator(schema)


async def import_notes(session_factory, user_id: int, cards_dir: Path) -> list[str]:
    """Insert every valid, not-yet-imported note in `cards_dir`. Returns imported note ids."""
    validator = load_validator()
    imported: list[str] = []

    async with session_factory() as session:
        user = await session.get(User, user_id)
        if user is None:
            session.add(User(id=user_id))

        for note_file in sorted(cards_dir.glob("*.json")):
            payload = json.loads(note_file.read_text())
            validator.validate(payload)

            note_id = note_id_for(user_id, payload["id"])
            # The raw file id too: imports before per-user ids stored it
            # verbatim, and those rows must still count as already imported.
            # Scoped to this user - the same raw id under someone else is
            # their note, not a reason to skip this one.
            existing = await session.scalar(
                select(Note.id).where(
                    Note.user_id == user_id, Note.id.in_([note_id, payload["id"]])
                )
            )
            if existing is not None:
                continue

            # Notes are unique per (user, deck, lemma, pos) and everything
            # imported here lands deckless, so a second file for a word this
            # user already has deckless would only surface as an IntegrityError
            # on the single commit below - aborting the whole import with a
            # traceback. Say which file instead, and change nothing.
            clash = await session.scalar(
                select(Note.id).where(
                    Note.user_id == user_id,
                    Note.lemma == payload["lemma"],
                    Note.pos.is_(None)
                    if payload.get("pos") is None
                    else Note.pos == payload["pos"],
                    Note.deck_id.is_(None),
                )
            )
            if clash is not None:
                raise ValueError(
                    f"{note_file}: user {user_id} already has a deckless note "
                    f"{payload['lemma']!r} ({payload.get('pos') or 'no pos'}), id={clash}. "
                    "Nothing was imported - remove or re-point one of the two."
                )

            session.add(
                Note(
                    id=note_id,
                    user_id=user_id,
                    lemma=payload["lemma"],
                    pos=payload.get("pos"),
                    translation_ru=payload["translation_ru"],
                    example_fi=payload["example_fi"],
                    example_ru=payload["example_ru"],
                    kind=payload["kind"],
                    meta=payload["meta"],
                )
            )
            imported.append(note_id)

        await session.commit()

    return imported


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--dir", type=Path, default=DEFAULT_CARDS_DIR)
    args = parser.parse_args()

    settings = load_settings()
    engine = make_engine(settings.database_url)
    session_factory = make_session_factory(engine)

    imported = await import_notes(session_factory, args.user_id, args.dir)
    print(f"Imported {len(imported)} note(s): {', '.join(imported) or '-'}")

    await engine.dispose()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
