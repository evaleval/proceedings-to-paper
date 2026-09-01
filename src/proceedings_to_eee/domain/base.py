"""Shared strict model base.

Kept in its own module so that domain models can reference each other without an import
cycle through ``observation``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base model that rejects accidental ontology drift."""

    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, populate_by_name=True, serialize_by_alias=True
    )
