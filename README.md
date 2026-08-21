# finn-cards

> **Kielikaveri** - a Telegram bot for learning Finnish: FST-verified flashcards generated from your own mistakes, FSRS scheduling, voice practice.

FST-verified Finnish flashcard schema and morphology wrapper (phase 0),
plus the bot itself (phase 1+). "Kielikaveri" is the project's working
name - this repo, `finn-cards`, is its only home; there's no separate
`kielikaveri` repo.

## Why

An LLM will confidently produce a plausible-looking but wrong Finnish word
form, and at B1 level that's not always obvious to the ear. So forms are
never invented by an LLM here - they come from `uralicNLP` (the omorfi
FST for Finnish), which by construction can only return forms that exist.
See `cards/instructions.md` for the exact algorithm and its rationale.

## Layout

- `cards/schema.json` - the note+card JSON Schema. Single source of truth;
  nothing else redefines the card format.
- `cards/instructions.md` - how to fill in a card without inventing forms.
- `cards/examples/` - three reference notes (verb, noun with consonant
  gradation, verb with non-trivial government/rektio).
- `src/finn_cards/` - phase 0: `morphology.py` (generate_forms,
  validate_form, lemmatize, detect_pos), `strict_schema.py` (converts
  `cards/schema.json` into an OpenAI `strict: true` schema).
- `src/kielikaveri/` - phase 1+: the bot.
  - `config.py` - settings loaded from `.env` (see `.env.example`).
  - `db/models.py` - SQLAlchemy models (`users`, `notes`, `cards`,
    `reviews`, `sources`), mirroring `cards/schema.json`.
  - `db/engine.py` - async engine/session helpers.
  - `bot/main.py` - entrypoint: long polling, whitelist middleware,
    `/start` `/help` `/stats`.
  - `bot/middleware.py` - `WhitelistMiddleware`, registered on
    `dp.update.outer_middleware`.
  - `bot/text.py` - `split_message`, for replies over Telegram's 4096
    character limit.
  - `import_cards.py` - imports note JSON files (schema-validated) into
    the database.
- `alembic/` - migrations for the schema above.
- `deploy/kielikaveri.service` - systemd unit template for the bot.
- `scripts/backup_db.sh` - daily SQLite backup (off-server copy step not
  wired up yet - see the script).
- `scripts/measure_coverage.py` - measures hypothesis G3 (does the FST
  cover the forms we need without LLM help) on a curated word list.

## Setup

This project shares the venv at `~/yki/.venv` (Python 3.12), which already
has the Finnish FST models downloaded - `uralicNLP` stores them inside its
own `site-packages` folder, so a fresh venv would re-download them.

```bash
uv pip install --python ~/yki/.venv/bin/python -e ".[dev]"
cp .env.example .env  # fill in BOT_TOKEN, WHITELIST_USER_IDS at minimum
~/yki/.venv/bin/alembic upgrade head
```

## Running the bot

```bash
~/yki/.venv/bin/python -m kielikaveri.bot.main
```

## Importing phase 0 cards

```bash
~/yki/.venv/bin/python -m kielikaveri.import_cards --user-id <your telegram id>
```

## Tests

```bash
~/yki/.venv/bin/pytest
```

## Lint

```bash
~/yki/.venv/bin/ruff check .
```

## License

GPL-3.0 - the Finnish FST models this project depends on (via omorfi) are
GPL-licensed.
