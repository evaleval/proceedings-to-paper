from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from proceedings_to_eee.resolution.attribution import load_lexicon
from proceedings_to_eee.resolution.metrics import load_eee_metric_snapshot
from proceedings_to_eee.resources import (
    ATTRIBUTION_LEXICON_PATHS,
    DEFAULT_ATTRIBUTION_LEXICON_PATH,
    DEFAULT_EEE_METRIC_SNAPSHOT_PATH,
    DEFAULT_EEE_SCHEMA_PATH,
    EEE_METRIC_SNAPSHOT_SHA256,
    EEE_SCHEMA_SHA256,
    EEE_SCHEMA_SHA256_BY_VERSION,
    LEGACY_EEE_SCHEMA_PATH,
    eee_schema_path,
)
from proceedings_to_eee.validation.eee_schema import (
    load_schema,
    validate_eee_record,
    vendored_schema_authority,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("version", sorted(EEE_SCHEMA_SHA256_BY_VERSION))
def test_packaged_schemas_match_the_reviewable_repository_copies(version: str) -> None:
    repository_dir = PROJECT_ROOT / "schemas" / f"eee-{version}"
    packaged = eee_schema_path(version)

    assert packaged.read_bytes() == (repository_dir / "eval.schema.json").read_bytes()
    assert (packaged.parent / "LICENSE").read_bytes() == (repository_dir / "LICENSE").read_bytes()
    assert (
        hashlib.sha256(packaged.read_bytes()).hexdigest() == (EEE_SCHEMA_SHA256_BY_VERSION[version])
    )
    assert json.loads(packaged.read_text(encoding="utf-8"))["version"] == version


def test_packaged_lexicons_and_metric_snapshot_match_their_pins() -> None:
    for lexicon_id, packaged in ATTRIBUTION_LEXICON_PATHS.items():
        repository_lexicon = PROJECT_ROOT / "configs" / f"{lexicon_id}.yaml"
        assert packaged.read_bytes() == repository_lexicon.read_bytes()
    assert hashlib.sha256(DEFAULT_EEE_SCHEMA_PATH.read_bytes()).hexdigest() == EEE_SCHEMA_SHA256
    assert (
        hashlib.sha256(DEFAULT_EEE_METRIC_SNAPSHOT_PATH.read_bytes()).hexdigest()
        == EEE_METRIC_SNAPSHOT_SHA256
    )


@pytest.mark.parametrize("lexicon_id", sorted(ATTRIBUTION_LEXICON_PATHS))
def test_every_shipped_lexicon_loads_under_its_own_id(lexicon_id: str) -> None:
    """Older lexicons stay loadable so a finished run's verdict hash can be reproduced."""

    lexicon = load_lexicon(ATTRIBUTION_LEXICON_PATHS[lexicon_id])

    assert lexicon.lexicon_id == lexicon_id
    assert (
        lexicon.sha256
        == hashlib.sha256(ATTRIBUTION_LEXICON_PATHS[lexicon_id].read_bytes()).hexdigest()
    )


def test_default_resources_work_outside_the_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    load_lexicon.cache_clear()
    load_eee_metric_snapshot.cache_clear()

    schema, authority = load_schema(expected_sha256=EEE_SCHEMA_SHA256)
    lexicon = load_lexicon()
    snapshot = load_eee_metric_snapshot()

    assert authority.version == "0.3.0"
    assert authority.allows_null_bounds
    assert lexicon.lexicon_id == "attribution-cues-v2"
    assert ATTRIBUTION_LEXICON_PATHS["attribution-cues-v2"] == DEFAULT_ATTRIBUTION_LEXICON_PATH
    assert schema["version"] == "0.3.0"
    assert snapshot.sha256 == EEE_METRIC_SNAPSHOT_SHA256
    assert snapshot.definitions


def test_the_legacy_schema_stays_selectable_and_pinned() -> None:
    _, authority = load_schema(LEGACY_EEE_SCHEMA_PATH)

    assert authority.version == "0.2.2"
    assert authority.sha256 == EEE_SCHEMA_SHA256_BY_VERSION["0.2.2"]
    assert not authority.allows_null_bounds


def test_an_edited_copy_of_a_vendored_schema_version_is_refused(tmp_path: Path) -> None:
    schema = json.loads(DEFAULT_EEE_SCHEMA_PATH.read_text(encoding="utf-8"))
    schema["description"] = "locally edited"
    edited = tmp_path / "eval.schema.json"
    edited.write_text(json.dumps(schema), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_schema(edited)


@pytest.mark.parametrize("version", sorted(EEE_SCHEMA_SHA256_BY_VERSION))
def test_a_recorded_schema_binding_resolves_to_its_vendored_copy(version: str) -> None:
    authority = vendored_schema_authority(version, EEE_SCHEMA_SHA256_BY_VERSION[version])

    assert authority.path == eee_schema_path(version)
    assert (authority.version, authority.sha256) == (version, EEE_SCHEMA_SHA256_BY_VERSION[version])
    assert load_schema(authority.path, authority.sha256)[1] == authority


@pytest.mark.parametrize(
    ("version", "sha256"),
    [
        ("0.1.0", EEE_SCHEMA_SHA256_BY_VERSION["0.3.0"]),
        ("0.3.0", EEE_SCHEMA_SHA256_BY_VERSION["0.2.2"]),
        ("0.2.2", "0" * 64),
        (None, EEE_SCHEMA_SHA256_BY_VERSION["0.3.0"]),
    ],
)
def test_a_schema_binding_this_package_does_not_ship_is_refused(
    version: str | None, sha256: str
) -> None:
    with pytest.raises(ValueError):
        vendored_schema_authority(version, sha256)


def test_synthetic_quickstart_record_is_schema_valid() -> None:
    record_path = PROJECT_ROOT / "examples" / "quickstart" / "synthetic-eee.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    schema, _ = load_schema(expected_sha256=EEE_SCHEMA_SHA256)

    assert validate_eee_record(record, schema) == []
