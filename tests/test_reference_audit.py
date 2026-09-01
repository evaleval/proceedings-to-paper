from __future__ import annotations

from proceedings_to_eee.domain.status import EvidenceKind
from proceedings_to_eee.evaluation.reference_audit import (
    audit_reference_against_layout,
    normalize_evidence_text,
)
from proceedings_to_eee.extraction.pdf_layout import PageFragment, PdfLayout
from proceedings_to_eee.reference import (
    AnnotationCoverage,
    EvidencePurpose,
    EvidenceVerificationMode,
    PaperReference,
    ReferenceEvidence,
)


def _layout(text: str) -> PdfLayout:
    return PdfLayout(
        source_id="source",
        parser="fixture",
        parser_version="fixture/1",
        page_count=1,
        pages=[
            PageFragment(
                fragment_id="fragment",
                source_id="source",
                page=1,
                text=text,
                text_sha256="a" * 64,
                character_count=len(text),
                numeric_token_count=1,
                result_signal_score=1,
            )
        ],
    )


def _reference(evidence: list[ReferenceEvidence]) -> PaperReference:
    return PaperReference(
        paper_id="paper",
        source_sha256="b" * 64,
        annotation_protocol="test",
        annotation_status="reviewed",
        coverage=AnnotationCoverage(
            inclusion_rule="fixture",
            exclusion_rule="fixture",
        ),
        evidence=evidence,
        observations=[],
    )


def test_normalization_handles_layout_hyphenation() -> None:
    assert normalize_evidence_text("Atlas Moder-\n ation API  0.74") == (
        "Atlas Moderation API 0.74"
    )


def test_audit_distinguishes_text_and_visual_review() -> None:
    reference = _reference(
        [
            ReferenceEvidence(
                evidence_id="text",
                purpose=EvidencePurpose.RESULT,
                page=1,
                kind=EvidenceKind.TABLE,
                exact_quote="Atlas Moderation API 0.74",
            ),
            ReferenceEvidence(
                evidence_id="visual",
                purpose=EvidencePurpose.RESULT,
                page=1,
                kind=EvidenceKind.TABLE,
                exact_quote="graphical value 0.43",
                verification_mode=EvidenceVerificationMode.VISUAL,
                visual_reviewed=True,
            ),
        ]
    )
    result = audit_reference_against_layout(
        reference,
        _layout("Atlas Moderation API  0.74"),
        actual_source_sha256="b" * 64,
    )
    assert result.passed
    assert result.text_verified == 1
    assert result.visual_verified == 1


def test_audit_fails_wrong_hash_and_missing_quote() -> None:
    reference = _reference(
        [
            ReferenceEvidence(
                evidence_id="missing",
                purpose=EvidencePurpose.RESULT,
                page=1,
                kind=EvidenceKind.TABLE,
                exact_quote="not present",
            )
        ]
    )
    result = audit_reference_against_layout(
        reference,
        _layout("different text"),
        actual_source_sha256="c" * 64,
    )
    assert not result.passed
    assert not result.source_hash_matches
    assert result.failed == 1
