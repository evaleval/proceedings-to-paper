"""Paths and identities for immutable resources shipped with the package."""

from pathlib import Path

RESOURCE_ROOT = Path(__file__).resolve().parent

#: Every vendored EEE schema, keyed by its ``version`` field. The default authority is the
#: newest one; the older one stays selectable so artifacts composed under it still
#: validate against the exact schema they were written for.
EEE_SCHEMA_SHA256_BY_VERSION: dict[str, str] = {
    "0.2.2": "088fed8029d42fb3a607aa67e1a05c39e425241b5cd90803705b37562f402f2a",
    "0.3.0": "c9c6195aec8a9dfa0b2aba4924ac1aa8a184c9d8ca97cee778cb531088e39b48",
}
EEE_SCHEMA_VERSION = "0.3.0"
EEE_SCHEMA_SHA256 = EEE_SCHEMA_SHA256_BY_VERSION[EEE_SCHEMA_VERSION]
LEGACY_EEE_SCHEMA_VERSION = "0.2.2"
LEGACY_EEE_SCHEMA_SHA256 = EEE_SCHEMA_SHA256_BY_VERSION[LEGACY_EEE_SCHEMA_VERSION]


def eee_schema_path(version: str) -> Path:
    """Return the packaged schema file for a vendored EEE version."""

    if version not in EEE_SCHEMA_SHA256_BY_VERSION:
        raise ValueError(f"EEE schema version {version!r} is not vendored")
    return RESOURCE_ROOT / "schemas" / f"eee-{version}" / "eval.schema.json"


DEFAULT_EEE_SCHEMA_PATH = eee_schema_path(EEE_SCHEMA_VERSION)
LEGACY_EEE_SCHEMA_PATH = eee_schema_path(LEGACY_EEE_SCHEMA_VERSION)

#: Every shipped cue lexicon, oldest first. Earlier files stay byte-identical so the
#: lexicon hash a finished run recorded on its verdicts can still be reproduced.
ATTRIBUTION_LEXICON_PATHS = {
    "attribution-cues-v0": RESOURCE_ROOT / "configs" / "attribution-cues-v0.yaml",
    "attribution-cues-v1": RESOURCE_ROOT / "configs" / "attribution-cues-v1.yaml",
    "attribution-cues-v2": RESOURCE_ROOT / "configs" / "attribution-cues-v2.yaml",
}
DEFAULT_ATTRIBUTION_LEXICON_PATH = ATTRIBUTION_LEXICON_PATHS["attribution-cues-v2"]

#: Curated subset of the EEE entity registry's metric list, vendored so metric identity
#: resolution stays offline and reproducible. Regenerate with
#: ``scripts/refresh_eee_metric_snapshot.py`` and update the pin together with the file.
EEE_METRIC_SNAPSHOT_DATE = "2026-10-12"
EEE_METRIC_SNAPSHOT_SHA256 = "46e349f382296e8cb1e2414eb75d16ce87037ca7c1e9e32a58787a885defe635"
DEFAULT_EEE_METRIC_SNAPSHOT_PATH = (
    RESOURCE_ROOT / "registry" / f"eee-metrics-{EEE_METRIC_SNAPSHOT_DATE}.json"
)

__all__ = [
    "ATTRIBUTION_LEXICON_PATHS",
    "DEFAULT_ATTRIBUTION_LEXICON_PATH",
    "DEFAULT_EEE_METRIC_SNAPSHOT_PATH",
    "DEFAULT_EEE_SCHEMA_PATH",
    "EEE_METRIC_SNAPSHOT_DATE",
    "EEE_METRIC_SNAPSHOT_SHA256",
    "EEE_SCHEMA_SHA256",
    "EEE_SCHEMA_SHA256_BY_VERSION",
    "EEE_SCHEMA_VERSION",
    "LEGACY_EEE_SCHEMA_PATH",
    "LEGACY_EEE_SCHEMA_SHA256",
    "LEGACY_EEE_SCHEMA_VERSION",
    "RESOURCE_ROOT",
    "eee_schema_path",
]
