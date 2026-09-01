from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from proceedings_to_eee.resolution.attribution import load_lexicon
from proceedings_to_eee.resources import (
    DEFAULT_ATTRIBUTION_LEXICON_PATH,
    DEFAULT_EEE_SCHEMA_PATH,
    EEE_SCHEMA_SHA256,
)
from proceedings_to_eee.validation.eee_schema import load_schema, validate_eee_record

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_packaged_resources_match_the_reviewable_repository_copies() -> None:
    repository_schema = PROJECT_ROOT / "schemas" / "eee-0.2.2" / "eval.schema.json"
    repository_lexicon = PROJECT_ROOT / "configs" / "attribution-cues-v0.yaml"

    assert DEFAULT_EEE_SCHEMA_PATH.read_bytes() == repository_schema.read_bytes()
    assert DEFAULT_ATTRIBUTION_LEXICON_PATH.read_bytes() == repository_lexicon.read_bytes()
    assert hashlib.sha256(DEFAULT_EEE_SCHEMA_PATH.read_bytes()).hexdigest() == EEE_SCHEMA_SHA256


def test_default_resources_work_outside_the_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    load_lexicon.cache_clear()

    schema, authority = load_schema(expected_sha256=EEE_SCHEMA_SHA256)
    lexicon = load_lexicon()

    assert authority.version == "0.2.2"
    assert lexicon.lexicon_id == "attribution-cues-v0"
    assert schema["version"] == "0.2.2"


def test_synthetic_quickstart_record_is_schema_valid() -> None:
    record_path = PROJECT_ROOT / "examples" / "quickstart" / "synthetic-eee.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    schema, _ = load_schema(expected_sha256=EEE_SCHEMA_SHA256)

    assert validate_eee_record(record, schema) == []
