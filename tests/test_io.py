from __future__ import annotations

import json

from pydantic import BaseModel

from proceedings_to_eee.io import canonical_json_bytes


class _NestedModel(BaseModel):
    name: str
    optional: str | None = None


def test_canonical_json_serializes_models_nested_in_collections() -> None:
    encoded = canonical_json_bytes({"items": [_NestedModel(name="result")]})

    assert json.loads(encoded) == {"items": [{"name": "result"}]}
