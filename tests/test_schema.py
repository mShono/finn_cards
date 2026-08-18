import json
from pathlib import Path

import jsonschema
import pytest

from finn_cards.morphology import FST_TAG_TO_POS
from finn_cards.strict_schema import convert_to_strict

CARDS_DIR = Path(__file__).parent.parent / "cards"


@pytest.fixture(scope="module")
def schema():
    return json.loads((CARDS_DIR / "schema.json").read_text())


@pytest.fixture(scope="module")
def validator(schema):
    return jsonschema.Draft202012Validator(schema)


@pytest.mark.parametrize(
    "example_file",
    sorted((CARDS_DIR / "examples").glob("*.json")),
    ids=lambda p: p.stem,
)
def test_example_notes_are_valid(validator, example_file):
    note = json.loads(example_file.read_text())
    validator.validate(note)


def test_pattern_kind_does_not_require_pos(validator):
    note = {
        "id": "00000000-0000-4000-8000-000000000000",
        "lemma": "hakea + partitiivi",
        "translation_ru": "искать + партитив",
        "example_fi": "Haen töitä.",
        "example_ru": "Я ищу работу.",
        "kind": "pattern",
        "meta": {"forms_source": "llm", "forms_verified": False, "origin": "error"},
    }
    validator.validate(note)


def test_word_kind_requires_pos(validator):
    note = {
        "id": "00000000-0000-4000-8000-000000000000",
        "lemma": "hakea",
        "translation_ru": "искать",
        "example_fi": "Haen töitä.",
        "example_ru": "Я ищу работу.",
        "kind": "word",
        "meta": {"forms_source": "fst", "forms_verified": True, "origin": "error"},
    }
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(note)


def test_convert_to_strict_excludes_fst_only_fields(schema):
    strict = convert_to_strict(
        schema,
        "note",
        exclude={"id", "principal_forms", "forms_source", "forms_verified"},
    )
    assert strict["additionalProperties"] is False
    assert "principal_forms" not in strict["properties"]["meta"]["properties"]
    assert set(strict["properties"]["meta"]["required"]) == set(
        strict["properties"]["meta"]["properties"]
    )


def test_convert_to_strict_raises_on_dynamic_map_field(schema):
    with pytest.raises(ValueError):
        convert_to_strict(schema, "note")


def test_fst_tag_to_pos_covers_exactly_the_schema_pos_enum(schema):
    # FST_TAG_TO_POS (morphology.py) and $defs/pos (schema.json) are two
    # independent lists that must name the same set of parts of speech.
    # Nothing else checks this - edit one without the other and detect_pos()
    # silently stops recognizing a pos value, with no test failure.
    schema_pos_values = set(schema["$defs"]["pos"]["enum"])
    assert set(FST_TAG_TO_POS.values()) == schema_pos_values


def test_convert_to_strict_optional_fields_become_nullable(schema):
    strict = convert_to_strict(
        schema,
        "note",
        exclude={"id", "principal_forms", "forms_source", "forms_verified"},
    )
    # gradation is genuinely optional in the source schema (not in
    # meta.required) and must become nullable, not disappear.
    gradation = strict["properties"]["meta"]["properties"]["gradation"]
    assert "null" in gradation["type"]
