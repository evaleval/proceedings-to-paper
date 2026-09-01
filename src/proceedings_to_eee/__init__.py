"""Proceedings-to-EEE: evidence-bound paper result extraction."""

from proceedings_to_eee.domain.observation import CandidateObservation
from proceedings_to_eee.domain.status import (
    ClaimType,
    ExportStatus,
    ReferentialStatus,
    ReportingStatus,
    TextSupportStatus,
)

__all__ = [
    "CandidateObservation",
    "ClaimType",
    "ExportStatus",
    "ReferentialStatus",
    "ReportingStatus",
    "TextSupportStatus",
]
__version__ = "0.2.0"
