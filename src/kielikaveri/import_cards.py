"""Import note JSON files (cards/schema.json format) into the database.

Only `notes` are imported here - `cards/examples/*.json` are notes, not
review cards. Review cards (type: recognition/production/inflection/usage)
get created gradually per phase 2's "postpone type" logic, not on import.

Usage: uv run python -m kielikaveri.import_cards --user-id 123
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import jsonschema

from kielikaveri.config import load_settings
from kielikaveri.db.engine import make_engine, make_session_factory
from kielikaveri.db.models import Note, User

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CARDS_DIR = REPO_ROOT / "cards" / "examples"
SCHEMA_PATH = REPO_ROOT / "cards" / "schema.json"


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
            if "kind" not in payload:
                # schema.json is oneOf[note, card] - "kind" only exists on note
                raise ValueError(f"{note_file} is a card, not a note - import_cards only imports notes")

            existing = await session.get(Note, payload["id"])
            if existing is not None:
                continue

            session.add(
                Note(
                    id=payload["id"],
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
            imported.append(payload["id"])

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
