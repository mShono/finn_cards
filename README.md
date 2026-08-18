# finn-cards

FST-verified Finnish flashcard schema and morphology wrapper - phase 0 of
the [Kielikaveri](https://github.com) Telegram bot project. No bot code
here yet: this repo fixes the card format and proves that word forms can
be generated from a finite-state transducer instead of an LLM guessing
them.

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
- `src/finn_cards/morphology.py` - `generate_forms`, `validate_form`,
  `lemmatize`, `detect_pos`.
- `src/finn_cards/strict_schema.py` - converts a `cards/schema.json`
  definition into an OpenAI `strict: true` schema (used by ingest in a
  later phase).
- `scripts/measure_coverage.py` - measures hypothesis G3 (does the FST
  cover the forms we need without LLM help) on a curated word list.

## Setup

This project shares the venv at `~/yki/.venv` (Python 3.12), which already
has the Finnish FST models downloaded - `uralicNLP` stores them inside its
own `site-packages` folder, so a fresh venv would re-download them.

```bash
uv pip install --python ~/yki/.venv/bin/python -e ".[dev]"
```

## Tests

```bash
~/yki/.venv/bin/pytest
```

## License

Not yet set for this standalone repo. The eventual bot repo takes
GPL-3.0, because the Finnish FST models it depends on (via omorfi) are
GPL-licensed.
