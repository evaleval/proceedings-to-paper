"""Orthogonal observation states that must never be collapsed."""

from enum import StrEnum


class ReportingStatus(StrEnum):
    """Whether a source reports an applicable value."""

    PRESENT = "present"
    UNKNOWN = "unknown"
    NOT_REPORTED = "not_reported"
    NOT_APPLICABLE = "not_applicable"


class TextSupportStatus(StrEnum):
    """Whether anchored source evidence supports an extracted claim."""

    UNVERIFIED = "unverified"
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"


class ReferentialStatus(StrEnum):
    """Whether a candidate is assigned to the correct entity and scope."""

    UNVERIFIED = "unverified"
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    WRONG_SCOPE = "wrong_scope"


class ClaimType(StrEnum):
    """Scientific function of a numeric or methodological statement."""

    PRIMARY_RESULT = "primary_result"
    SECONDARY_CLAIM = "secondary_claim"
    ILLUSTRATION = "illustration"
    METHOD_METADATA = "method_metadata"
    UNCERTAIN = "uncertain"


class ActorRole(StrEnum):
    """Role an actor or system plays in the reported evaluation."""

    EVALUATED_SYSTEM = "evaluated_system"
    EVALUATION_INSTRUMENT = "evaluation_instrument"
    LABEL_GENERATOR = "label_generator"
    HUMAN_REFERENCE = "human_reference"


class ExportStatus(StrEnum):
    """Deterministic EEE export decision."""

    ELIGIBLE = "eligible"
    NEEDS_REVIEW = "needs_review"
    NOT_ELIGIBLE = "not_eligible"
    EXPORTED = "exported"


class EvidenceKind(StrEnum):
    """Source structure containing the evidence."""

    TABLE = "table"
    FIGURE = "figure"
    PROSE = "prose"
    APPENDIX = "appendix"


class ValueComparator(StrEnum):
    """Relationship between the printed number and the reported quantity."""

    EXACT = "exact"
    LESS_THAN = "less_than"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"
    GREATER_THAN = "greater_than"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    APPROXIMATELY = "approximately"
