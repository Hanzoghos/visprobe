"""Typed decision schema: choice / boolean (aligned with Nimble's field types).

Unified record format (one json object per jsonl line):
  {"id", "image", "state"(optional text-context dict),
   "questions": {field_name: {"type": "choice"|"boolean",
                              "instructions": "natural-language question",
                              "criteria": {"A": candidate1, ...}}},  # boolean may omit criteria
   "reference": {field_name: correct_answer}}                        # boolean answers are JSON true/false

One image plus an optional piece of text; each field is classified independently; field
values map to single-token codes (A/B/C...), and probabilities come from the answer-token
logits.

- choice: pick one from a fixed candidate set; criteria is a "code -> value" mapping
  (2-26 entries; order is the candidate order).
- boolean: yes/no; candidate values are fixed to JSON false/true and codes are fixed to
  A=false, B=true.
- score (ordered candidates plus probability-weighted expected grade) is not implemented
  yet; it will later be added on top of choice by introducing an ordering relation.
"""

import json
from dataclasses import dataclass

VALID_TYPES = ("choice", "boolean")
# Legacy name compatibility: noul -> boolean, enum -> choice, score not implemented yet
_LEGACY_TYPES = {"noul": "boolean", "enum": "choice", "score": None}


@dataclass
class FieldSpec:
    name: str
    type: str
    instructions: str
    criteria: dict   # code(str) -> label(bool/str); key order is the candidate order
    choices: list    # criteria keys, in order

    @classmethod
    def from_dict(cls, name, d):
        typ = d.get("type")
        if typ in _LEGACY_TYPES:
            new = _LEGACY_TYPES[typ]
            if new is None:
                raise ValueError(
                    f"field {name!r} uses the not-yet-implemented type='score'"
                    " (ordered candidates plus expected grade; coming in a later version)")
            raise ValueError(f"field {name!r} has renamed type {typ!r}; please use {new!r} instead")
        if typ not in VALID_TYPES:
            raise ValueError(f"field {name!r} has unsupported type: {typ!r} (only {VALID_TYPES})")

        criteria = dict(d.get("criteria", {}))
        if typ == "boolean":
            if not criteria:
                criteria = {"A": False, "B": True}
            else:
                if len(criteria) != 2:
                    raise ValueError(
                        f"boolean field {name!r} must have exactly the two values false/true, got {criteria}")
                vals = list(criteria.values())
                # Only accept bool (do not use isinstance here: True/False must be strictly
                # distinguished semantically from 1/0)
                if any(type(v) is not bool for v in vals) or set(vals) != {False, True}:
                    raise ValueError(
                        f"boolean field {name!r} candidate values must be JSON false/true, got {vals}")
                # Force the fixed order A=false, B=true so gold-code lookups use one consistent convention
                criteria = {"A": False, "B": True}
        elif not 2 <= len(criteria) <= 26:
            raise ValueError(f"choice field {name!r} must have 2-26 candidates, got {len(criteria)}")

        return cls(name=name, type=typ, instructions=d.get("instructions", ""),
                   criteria=criteria, choices=list(criteria.keys()))


def parse_schema(schema):
    """schema: {field_name: field_spec} -> [FieldSpec]."""
    return [FieldSpec.from_dict(name, spec) for name, spec in schema.items()]


_PROMPT_HEADER = (
    "Classify the context using the supplied schema. Use the image and the context "
    "to answer the requested field. Return only that choice's one-letter code "
    "with no other text."
)


def build_field_prompt(field, context_text=""):
    """Render one field into Nimble-style user text (without the image).

    context_text: JSON string of state; if empty, the Context line is omitted.
    """
    schema = [{
        "name": field.name,
        "description": field.instructions,
        "choices": [{"code": code, "value": field.criteria[code]} for code in field.choices],
    }]
    lines = [_PROMPT_HEADER]
    if context_text:
        lines.append(f"Context: {context_text}")
    lines.append("Schema: " + json.dumps(schema, ensure_ascii=False))
    lines.append(f"Requested field: {field.name}")
    return "\n".join(lines)
