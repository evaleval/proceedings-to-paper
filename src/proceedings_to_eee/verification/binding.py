"""Deterministic binding from a candidate to the exact frozen block that carries its quote.

The pipeline and any offline replay must bind candidates identically, otherwise a
replayed verification measures a different experiment than the one the pipeline would
have run. This module is the single implementation both call.
"""

from __future__ import annotations

import hashlib

from proceedings_to_eee.domain.observation import CandidateObservation, EvidenceAnchor
from proceedings_to_eee.extraction.result_blocks import ResultBlock
from proceedings_to_eee.validation.candidates import normalize_evidence_text
from proceedings_to_eee.verification.independent import FrozenEvidenceBlock


def bind_candidate_block(
    candidate: CandidateObservation,
    blocks: list[ResultBlock],
) -> tuple[ResultBlock, EvidenceAnchor] | None:
    """Return the first block/anchor pair whose block text contains the anchor quote.

    Anchors are tried in their recorded order and blocks in segmentation order, so the
    binding is a pure function of the two inputs. Matching is containment after layout
    whitespace normalization only; wording, numbers, and punctuation are never relaxed.
    """

    for anchor in candidate.evidence:
        for block in blocks:
            if block.source_id != anchor.source_id or block.page != anchor.page:
                continue
            if normalize_evidence_text(anchor.quote) in normalize_evidence_text(
                block.prompt_text()
            ):
                return block, anchor
    return None


def frozen_evidence_block(
    *,
    paper_id: str,
    block: ResultBlock,
    anchor: EvidenceAnchor,
) -> FrozenEvidenceBlock:
    """Wrap one bound block as the hash-bound evidence the verifier is allowed to see."""

    text = block.prompt_text()
    return FrozenEvidenceBlock(
        block_id=block.block_id,
        paper_id=paper_id,
        source_id=block.source_id,
        page=block.page,
        kind=anchor.kind,
        label=anchor.label,
        row=anchor.row,
        column=anchor.column,
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
