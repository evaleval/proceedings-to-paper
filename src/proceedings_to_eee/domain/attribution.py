"""Provenance attribution: did THIS paper produce this number, or is it reporting one.

This axis is deliberately separate from :class:`ClaimType`. ``claim_type`` is filled by
the extraction model as self-report, so it cannot establish attribution. Keeping origin
separate also avoids changing the extraction wire schema: ``extraction/llm_schema.py``
uses ``ClaimType`` directly in its provider contract.

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


class OriginBasis(StrEnum):
    """What a record's producer-origin claim actually rests on.

    Canonical export means positively established origin. A tiered policy permits weaker
    bases without calling a record `paper_produced`; every record must carry the basis
    on which its export was granted.
    """

    HUMAN_CONFIRMED = "human_confirmed"
    """A person reviewed the origin and confirmed it."""

    POSITIVE_STRUCTURAL = "positive_structural"
    """Deterministic structure established current-paper production."""

    MODEL_REVIEWED_ORIGIN_QUOTE = "model_reviewed_origin_quote"
    """A page-scoped reviewer accepted origin against a quote verified on the page."""

    MODEL_ASSERTED_PRIMARY_NO_EXTERNAL_CUE = "model_asserted_primary_no_external_cue"
    """The extractor called it a primary result and a completed cue check found nothing."""

    MODEL_ASSERTED_PRIMARY_UNCHECKED = "model_asserted_primary_unchecked"
    """The extractor called it a primary result and the cue check never ran."""

    NONE = "none"
    """No basis. Never exports under any policy."""

    @property
    def exports_under_tiered(self) -> bool:
        return self is not OriginBasis.NONE

    @property
    def exports_under_positive_only(self) -> bool:
        return self in {OriginBasis.HUMAN_CONFIRMED, OriginBasis.POSITIVE_STRUCTURAL}


class ReviewTier(StrEnum):
    """How much review stands behind a composed record."""

    DETERMINISTIC = "deterministic"
    """Only deterministic checks: quote on page, value in quote, reference resolution."""

    MODEL_REVIEWED = "model_reviewed"
    """A page-scoped model reviewer accepted it against quotes verified on the page."""

    HUMAN_CONFIRMED = "human_confirmed"
    """A person confirmed it."""


class OriginExportPolicy(StrEnum):
    """Which origin bases a composition run is willing to export."""

    POSITIVE_ONLY = "positive_only"
    """The historical policy: only positively established origin. The default."""

    TIERED = "tiered"
    """Export any basis but `none`, with the basis carried in every record."""

    def permits(self, basis: OriginBasis) -> bool:
        if self is OriginExportPolicy.TIERED:
            return basis.exports_under_tiered
        return basis.exports_under_positive_only


#: Rule-id prefix the reviewed-export workflow stamps on a human-confirmed verdict.
REVIEWED_ORIGIN_RULE_PREFIX = "reviewed_origin:"

#: Rule id the deterministic resolver emits when a completed cue check found nothing.
NO_CUE_RULE_ID = "no_cue"

#: `ClaimType.PRIMARY_RESULT` as a plain string. Importing the enum here would make
#: domain.attribution depend on domain.status's claim axis, which this module documents
#: as deliberately separate.
PRIMARY_RESULT_CLAIM_TYPE = "primary_result"


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

    def origin_basis(self, claim_type: str | None = None) -> OriginBasis:
        """Return what this verdict lets a record claim about who produced the number.

        `claim_type` is the extractor's own label. It is self-report, so it can only ever
        name a basis as model-asserted; it can never raise one to positive.
        """

        if self.state is AttributionState.PAPER_PRODUCED:
            if self.rule_id.startswith(REVIEWED_ORIGIN_RULE_PREFIX):
                return OriginBasis.HUMAN_CONFIRMED
            return OriginBasis.POSITIVE_STRUCTURAL
        if self.state is AttributionState.EXTERNALLY_SOURCED:
            return OriginBasis.NONE
        if claim_type != PRIMARY_RESULT_CLAIM_TYPE:
            return OriginBasis.NONE
        if self.state is AttributionState.NO_SIGNAL:
            if self.rule_id == NO_CUE_RULE_ID:
                return OriginBasis.MODEL_ASSERTED_PRIMARY_NO_EXTERNAL_CUE
            return OriginBasis.NONE
        # UNRESOLVED. Cues that fired mean the check ran and could not decide, which is
        # doubt, not silence; only a check that never ran leaves the model's own label.
        if self.cues:
            return OriginBasis.NONE
        return OriginBasis.MODEL_ASSERTED_PRIMARY_UNCHECKED

    @property
    def allows_canonical_export(self) -> bool:
        """True only for positively established current-paper production."""

        return self.state is AttributionState.PAPER_PRODUCED

    @property
    def demotes(self) -> bool:
        """True when this verdict must keep a candidate in the review layer."""

        return not self.allows_canonical_export
