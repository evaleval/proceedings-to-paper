from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

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
)
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout
from proceedings_to_eee.sources.manifest import FrozenSource, SourceManifest, SourceRole
from proceedings_to_eee.sources.processing import (
    SourceProcessingArtifact,
    SourceProcessingReason,
    SourceProcessingState,
    build_source_processing_artifact,
    validate_source_processing_artifact,
)
from proceedings_to_eee.validation.candidates import validate_non_origin_candidates


def _source(source_id: str, role: SourceRole) -> FrozenSource:
    if role is SourceRole.REPOSITORY:
        return FrozenSource(
            source_id=source_id,
            paper_id="paper",
            role=role,
            original_uri="https://example.test/repository.git",
            retrieved_at=datetime(2026, 9, 2, tzinfo=UTC),
            git_commit="a" * 40,
        )
    return FrozenSource(
        source_id=source_id,
        paper_id="paper",
        role=role,
        original_uri=f"https://example.test/{source_id}.pdf",
        retrieved_at=datetime(2026, 9, 2, tzinfo=UTC),
        sha256="b" * 64 if role is SourceRole.PAPER else "c" * 64,
        byte_size=10,
        cache_relpath=f"data/sources/{source_id}.pdf",
    )


def _bundle() -> tuple[SourceManifest, PdfLayout]:
    manifest = SourceManifest(
        paper_id="paper",
        title="Paper",
        sources=[
            _source("src_paper", SourceRole.PAPER),
            _source("src_supplement", SourceRole.SUPPLEMENT),
            _source("src_repository", SourceRole.REPOSITORY),
        ],
    )
    text = "System 0.5\n"
    layout = PdfLayout(
        source_id="src_paper",
        parser="fixture",
        parser_version="fixture/1",
        page_count=1,
        pages=[
            PageFragment(
                fragment_id="fragment-1",
                source_id="src_paper",
                page=1,
                text=text,
                text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                character_count=len(text),
                numeric_token_count=1,
                result_signal_score=1.0,
            )
        ],
    )
    return manifest, layout


def test_source_processing_exactly_partitions_primary_supplement_and_repository() -> None:
    manifest, layout = _bundle()

    artifact = build_source_processing_artifact(
        manifest=manifest,
        layout=layout,
        layout_sha256="d" * 64,
    )

    assert artifact.counts.model_dump() == {
        "sources": 3,
        "processed": 1,
        "unsupported": 2,
    }
    assert artifact.processed_source_ids == {"src_paper"}
    assert artifact.source_by_id["src_supplement"].reason is SourceProcessingReason.NO_ADAPTER
    repository = artifact.source_by_id["src_repository"]
    assert repository.state is SourceProcessingState.UNSUPPORTED
    assert repository.reason is SourceProcessingReason.COMMIT_ADVERTISED_ONLY_NO_TREE_BYTES


def test_source_processing_rejects_a_missing_manifest_source() -> None:
    manifest, layout = _bundle()
    artifact = build_source_processing_artifact(
        manifest=manifest,
        layout=layout,
        layout_sha256="d" * 64,
    )
    payload = artifact.model_dump(mode="json")
    payload["source_by_id"].pop("src_supplement")
    payload["counts"] = {"sources": 2, "processed": 1, "unsupported": 1}
    tampered = SourceProcessingArtifact.model_validate(payload)

    with pytest.raises(ValueError, match="partition every manifest source"):
        validate_source_processing_artifact(
            manifest=manifest,
            artifact=tampered,
            layout=layout,
        )


def test_candidate_citing_unprocessed_supplement_gets_typed_rejection() -> None:
    _, layout = _bundle()
    candidate = CandidateObservation(
        paper_id="paper",
        claim_type=ClaimType.PRIMARY_RESULT,
        roles=[
            RoleAssignment(
                role=ActorRole.EVALUATED_SYSTEM,
                raw_name="System",
                confidence=1.0,
            )
        ],
        scope=ObservationScope(dataset_raw="Dataset"),
        metric=MetricSpec(
            raw_name="Accuracy",
            canonical_id="accuracy",
            unit="proportion",
            lower_is_better=False,
        ),
        value=ReportedValue(raw="0.5", numeric=0.5, unit="proportion"),
        evidence=[
            EvidenceAnchor(
                source_id="src_supplement",
                page=1,
                kind=EvidenceKind.TABLE,
                quote="System 0.5",
            )
        ],
        extraction_confidence=1.0,
    )

    [validated] = validate_non_origin_candidates(
        [candidate],
        {layout.source_id: layout},
        processed_source_ids={layout.source_id},
    )

    assert validated.export_status is ExportStatus.NEEDS_REVIEW
    assert validated.export_reason == "source_not_processed"
