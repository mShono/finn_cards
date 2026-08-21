"""Convert a definition from cards/schema.json into an OpenAI strict-mode schema.

OpenAI structured outputs require, on every object in the schema:
- "additionalProperties": false
- every property listed in "required" (optionality is expressed as a
  ["type", "null"] union instead of omission)

It also cannot resolve sibling $ref, so this bundles $defs inline, and it
cannot represent a dynamic-keyed map (a bare "additionalProperties" schema
with no fixed "properties") - callers must exclude such fields.

principal_forms/forms_source/forms_verified/id are filled by our own code
from the FST, never asked of the LLM, so any LLM-facing prompt built from
this schema is expected to exclude them via the `exclude` argument.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable


def convert_to_strict(schema: dict, def_name: str, exclude: Iterable[str] = ()) -> dict:
    """Return an OpenAI strict-mode schema for schema['$defs'][def_name]."""
    defs = schema.get("$defs", {})
    resolved = _resolve_refs({"$ref": f"#/$defs/{def_name}"}, defs)
    return _make_strict(resolved, frozenset(exclude))


def _resolve_refs(node, defs, _seen=()):
    if isinstance(node, dict):
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            if name in _seen:
                raise ValueError(f"circular $ref not supported: {node['$ref']}")
            return _resolve_refs(defs[name], defs, (*_seen, name))
        return {k: _resolve_refs(v, defs, _seen) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_refs(item, defs, _seen) for item in node]
    return node


def _make_strict(node: dict, exclude: frozenset[str]) -> dict:
    node = copy.deepcopy(node)
    node.pop("format", None)

    if node.get("type") == "object" and "properties" in node:
        properties = {
            name: prop for name, prop in node["properties"].items() if name not in exclude
        }
        required = set(node.get("required", [])) - exclude
        for name, prop in properties.items():
            prop = _make_strict(prop, exclude)
            properties[name] = prop if name in required else _make_nullable(prop)
        node["properties"] = properties
        node["required"] = list(properties.keys())
        node["additionalProperties"] = False
        node.pop("allOf", None)  # conditional requiredness has no strict-mode equivalent
        return node

    if node.get("type") == "array" and "items" in node:
        node["items"] = _make_strict(node["items"], exclude)
        return node

    if node.get("type") == "object" and "properties" not in node:
        raise ValueError(
            "dynamic-keyed object has no fixed properties and cannot be "
            "represented in a strict schema - exclude this field"
        )

    return node


def _make_nullable(prop: dict) -> dict:
    t = prop.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        if "null" not in types:
            prop["type"] = [*types, "null"]
    if "enum" in prop and None not in prop["enum"]:
        prop["enum"] = [*prop["enum"], None]
    return prop
