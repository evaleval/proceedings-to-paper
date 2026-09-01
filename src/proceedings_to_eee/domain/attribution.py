"""Provenance attribution: did THIS paper produce this number, or is it reporting one.

This axis is deliberately separate from :class:`ClaimType`. ``claim_type`` is filled by
the extraction model as pure self-report and collapsed to 97.8% ``primary_result`` on the
sealed holdout, so it cannot carry attribution. Adding a member to ``ClaimType`` is also
forbidden: ``extraction/llm_schema.py`` types the wire model on it, so an enum change
would move ``provider_json_schema()`` and void the holdout seal.

The state on this axis is produced by deterministic code from structure the region index
recovers, never by a model. A model may not write it, and the independent verifier may
only ever move a candidate toward doubt, never toward acceptance.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from proceedings_to_eee.domain.base import StrictModel


class AttributionState(StrEnum):
    """Whether the paper printing a value is also the source of it."""

    PAPER_PRODUCED = "paper_produced"
    """Positive, trusted evidence that the current paper produced this number."""

    EXTERNALLY_SOURCED = "externally_sourced"
    """Positive, row-scoped evidence that another party produced this number."""

    UNRESOLVED = "unresolved"
    """Origin is ambiguous or the available structure cannot support a decision."""

    NO_SIGNAL = "no_signal"
    """A completed v0 cue inspection found nothing; never proof of paper production."""


class AttributionCue(StrictModel):
    """One cue that fired, with the exact text it fired on."""

    cue_id: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    matched_text: str = Field(min_length=1)


class AttributionVerdict(StrictModel):
    """A deterministic attribution decision, reversible to the structure behind it."""

    schema_version: str = "attribution-verdict/0.2"
    state: AttributionState
    rule_id: str = Field(min_length=1)
    lexicon_id: str | None = None
    lexicon_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    cues: list[AttributionCue] = Field(default_factory=list)
    region_id: str | None = None
    page: int | None = Field(default=None, ge=1)
    row_label: str | None = None
    table_label: str | None = None
    contrast_rows_total: int | None = Field(default=None, ge=0)
    contrast_rows_matched: int | None = Field(default=None, ge=0)

    @property
    def allows_canonical_export(self) -> bool:
        """True only for positively established current-paper production."""

        return self.state is AttributionState.PAPER_PRODUCED

    @property
    def demotes(self) -> bool:
        """True when this verdict must keep a candidate in the review layer."""

        return not self.allows_canonical_export
