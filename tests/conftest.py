from __future__ import annotations

from datetime import UTC, datetime

import pytest

from proceedings_to_eee.domain.attribution import AttributionState, AttributionVerdict
from proceedings_to_eee.domain.observation import (
    CandidateObservation,
    EvidenceAnchor,
    MetricSpec,
    ObservationScope,
    ReportedValue,
    RoleAssignment,
)
from proceedings_to_eee.domain.status import (
    ActorRole,
    ClaimType,
    EvidenceKind,
    ExportStatus,
    ReferentialStatus,
    TextSupportStatus,
)
from proceedings_to_eee.sources.manifest import (
    FrozenSource,
    LicenseDisposition,
    SourceManifest,
    SourceRole,
)


@pytest.fixture
def manifest() -> SourceManifest:
    return SourceManifest(
        paper_id="synthetic-audit-study",
        title="Synthetic Audit Study",
        doi="10.1145/example",
        sources=[
            FrozenSource(
                source_id="src_paper",
                paper_id="synthetic-audit-study",
                role=SourceRole.PAPER,
                original_uri="https://example.org/paper.pdf",
                resolved_uri="https://example.org/paper.pdf",
                retrieved_at=datetime(2026, 8, 4, tzinfo=UTC),
                sha256="a" * 64,
                byte_size=123,
                media_type="application/pdf",
                cache_relpath="data/sources/aa/" + "a" * 64 + ".pdf",
                license_disposition=LicenseDisposition.DERIVED_METADATA_ONLY,
            )
        ],
    )


@pytest.fixture
def eligible_candidate() -> CandidateObservation:
    return CandidateObservation(
        paper_id="synthetic-audit-study",
        claim_type=ClaimType.PRIMARY_RESULT,
        roles=[
            RoleAssignment(
                role=ActorRole.EVALUATED_SYSTEM,
                raw_name="Atlas Moderation API",
                canonical_id="example/atlas-moderation-api",
                provider="Example Labs",
                confidence=1.0,
            ),
            RoleAssignment(
                role=ActorRole.HUMAN_REFERENCE,
                raw_name="Synthetic Speech Set labels",
                confidence=0.9,
            ),
        ],
        scope=ObservationScope(dataset_raw="Synthetic Speech Set", split="test", sample_count=6400),
        metric=MetricSpec(
            raw_name="AUC",
            canonical_id="auroc",
            kind="auroc",
            unit="percent",
            lower_is_better=False,
            min_score=0,
            max_score=100,
        ),
        value=ReportedValue(raw="74.6%", numeric=74.6, unit="percent"),
        evidence=[
            EvidenceAnchor(
                source_id="src_paper",
                page=7,
                kind=EvidenceKind.TABLE,
                label="Table 2",
                row="Atlas Moderation API · Synthetic Speech Set",
                column="AUC",
                quote="Atlas Moderation API  61.3  74.6%  58.2",
            )
        ],
        text_support=TextSupportStatus.SUPPORTED,
        referential_status=ReferentialStatus.RESOLVED,
        export_status=ExportStatus.ELIGIBLE,
        attribution=AttributionVerdict(
            state=AttributionState.PAPER_PRODUCED,
            rule_id="explicit_test_fixture",
        ),
        extraction_method="fixture",
        extraction_confidence=0.99,
    )
