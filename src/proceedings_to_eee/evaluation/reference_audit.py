"""Audit reference quotes against the exact frozen PDF and layout text."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from pydantic import Field

from proceedings_to_eee.domain.observation import StrictModel
from proceedings_to_eee.extraction.pdf_layout import PdfLayout, extract_pdf_layout
from proceedings_to_eee.io import sha256_file
from proceedings_to_eee.reference import (
    EvidenceVerificationMode,
    PaperReference,
    load_reference,
)


def normalize_evidence_text(value: str) -> str:
    """Normalize layout whitespace and line-end word hyphenation, but not wording."""

    normalized = unicodedata.normalize("NFKC", value)
    normalized = re.sub(r"(?<=\w)-\s+(?=\w)", "", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


class EvidenceAuditItem(StrictModel):
    evidence_id: str
    page: int
    status: str
    detail: str | None = None


class ReferenceAudit(StrictModel):
    schema_version: str = "reference-audit/0.2"
    paper_id: str
    source_hash_matches: bool
    page_count: int = Field(ge=1)
    text_verified: int = Field(ge=0)
    visual_verified: int = Field(ge=0)
    failed: int = Field(ge=0)
    passed: bool
    items: list[EvidenceAuditItem]


def audit_reference_against_layout(
    reference: PaperReference,
    layout: PdfLayout,
    *,
    actual_source_sha256: str,
) -> ReferenceAudit:
    """Check every evidence anchor without silently treating visual review as OCR proof."""

    items: list[EvidenceAuditItem] = []
    pages = {page.page: page.text for page in layout.pages}
    for evidence in reference.evidence:
        if evidence.page not in pages:
            items.append(
                EvidenceAuditItem(
                    evidence_id=evidence.evidence_id,
                    page=evidence.page,
                    status="page_out_of_range",
                    detail=f"layout has {layout.page_count} pages",
                )
            )
            continue
        if evidence.verification_mode == EvidenceVerificationMode.VISUAL:
            items.append(
                EvidenceAuditItem(
                    evidence_id=evidence.evidence_id,
                    page=evidence.page,
                    status="visual_verified",
                    detail="human visual review asserted in reference annotation",
                )
            )
            continue
        quote = normalize_evidence_text(evidence.exact_quote)
        page_text = normalize_evidence_text(pages[evidence.page])
        if quote in page_text:
            items.append(
                EvidenceAuditItem(
                    evidence_id=evidence.evidence_id,
                    page=evidence.page,
                    status="text_verified",
                )
            )
        else:
            items.append(
                EvidenceAuditItem(
                    evidence_id=evidence.evidence_id,
                    page=evidence.page,
                    status="quote_not_found",
                    detail=evidence.exact_quote,
                )
            )
    failed = sum(item.status not in {"text_verified", "visual_verified"} for item in items)
    source_hash_matches = actual_source_sha256 == reference.source_sha256
    return ReferenceAudit(
        paper_id=reference.paper_id,
        source_hash_matches=source_hash_matches,
        page_count=layout.page_count,
        text_verified=sum(item.status == "text_verified" for item in items),
        visual_verified=sum(item.status == "visual_verified" for item in items),
        failed=failed,
        passed=source_hash_matches and failed == 0,
        items=items,
    )


def audit_reference_pdf(reference_path: Path, pdf_path: Path) -> ReferenceAudit:
    reference = load_reference(reference_path)
    layout = extract_pdf_layout(pdf_path, source_id=f"reference-audit:{reference.paper_id}")
    return audit_reference_against_layout(
        reference,
        layout,
        actual_source_sha256=sha256_file(pdf_path),
    )
