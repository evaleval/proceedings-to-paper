"""Paths and identities for immutable resources shipped with the package."""

from pathlib import Path

RESOURCE_ROOT = Path(__file__).resolve().parent

EEE_SCHEMA_VERSION = "0.2.2"
EEE_SCHEMA_SHA256 = "088fed8029d42fb3a607aa67e1a05c39e425241b5cd90803705b37562f402f2a"
DEFAULT_EEE_SCHEMA_PATH = (
    RESOURCE_ROOT / "schemas" / f"eee-{EEE_SCHEMA_VERSION}" / "eval.schema.json"
)
DEFAULT_ATTRIBUTION_LEXICON_PATH = RESOURCE_ROOT / "configs" / "attribution-cues-v0.yaml"

__all__ = [
    "DEFAULT_ATTRIBUTION_LEXICON_PATH",
    "DEFAULT_EEE_SCHEMA_PATH",
    "EEE_SCHEMA_SHA256",
    "EEE_SCHEMA_VERSION",
    "RESOURCE_ROOT",
]
