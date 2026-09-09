"""Deterministic binding from a candidate to the exact frozen block that carries its quote.

The pipeline and any offline replay must bind candidates identically, otherwise a
replayed verification measures a different experiment than the one the pipeline would
have run. This module is the single implementation both call.
"""

from __future__ import annotations

import hashlib

from proceedings_to_eee.domain.observation import CandidateObservation, EvidenceAnchor
from proceedings_to_eee.extraction.result_blocks import ResultBlock
from proceedings_to_eee.validation.candidates import bounded_claim_present
from proceedings_to_eee.verification.independent import (
    FrozenEvidenceBlock,
    FrozenEvidenceLine,
    UntrustedClaimedAnchor,
)


def bind_candidate_block(
    candidate: CandidateObservation,
    blocks: list[ResultBlock],
) -> tuple[ResultBlock, EvidenceAnchor] | None:
    """Return the strongest deterministic block/anchor evidence binding.

    A multi-anchor candidate can list caption/context evidence before its physical value
    row, and overlapping blocks can carry the same line as context in one block and body
    text in another. Prefer an anchor containing the asserted raw value, then an exact
    result-body occurrence. Recorded anchor/block order remains the final deterministic
    tie-breaker. A block is eligible only when one of its source-native lines contains
    the claimed quote bytes exactly. Candidate validation may have found the quote on
    the same page elsewhere; normalized or cross-line containment in this block is not
    evidence that this is the block the candidate came from.
    """

    matches: list[tuple[tuple[int, int, int, int, int], ResultBlock, EvidenceAnchor]] = []
    raw_value = candidate.value.raw if candidate.value is not None else None
    for anchor_index, anchor in enumerate(candidate.evidence):
        for block_index, block in enumerate(blocks):
            if block.source_id != anchor.source_id or block.page != anchor.page:
                continue
            exact_prompt_match = any(
                anchor.quote in line for line in block.prompt_text().splitlines()
            )
            if not exact_prompt_match:
                continue
            value_bearing = bool(raw_value and bounded_claim_present(anchor.quote, raw_value))
            exact_body_match = any(anchor.quote in line for line in block.body_text.splitlines())
            rank = (
                int(value_bearing and exact_body_match),
                int(value_bearing),
                int(exact_body_match),
                -anchor_index,
                -block_index,
            )
            matches.append((rank, block, anchor))
    if not matches:
        return None
    _, block, anchor = max(matches, key=lambda item: item[0])
    return block, anchor


def frozen_evidence_block(
    *,
    paper_id: str,
    block: ResultBlock,
    anchor: EvidenceAnchor,
) -> FrozenEvidenceBlock:
    """Wrap source-native lines and keep all candidate anchor metadata explicitly untrusted."""

    lines: list[FrozenEvidenceLine] = []

    def append_span(
        *,
        section: str,
        text: str,
        start_line: int | None,
        end_line: int | None,
    ) -> None:
        if not text:
            return
        if start_line is None or end_line is None:
            raise ValueError("frozen evidence text span is missing its source line range")
        source_lines = text.split("\n")
        if (
            source_lines
            and source_lines[-1] == ""
            and len(source_lines) > end_line - start_line + 1
        ):
            source_lines.pop()
        if len(source_lines) != end_line - start_line + 1:
            raise ValueError("frozen evidence text does not match its source line range")
        for offset, source_text in enumerate(source_lines):
            lines.append(
                FrozenEvidenceLine(
                    line_id=f"L{len(lines) + 1:04d}",
                    section=section,
                    source_line=start_line + offset,
                    text=source_text,
                )
            )

    append_span(
        section="leading_context",
        text=block.context_text,
        start_line=block.context_start_line,
        end_line=block.context_end_line,
    )
    append_span(
        section="result_block",
        text=block.body_text,
        start_line=block.body_start_line,
        end_line=block.body_end_line,
    )
    append_span(
        section="trailing_context",
        text=block.trailing_context_text,
        start_line=block.trailing_context_start_line,
        end_line=block.trailing_context_end_line,
    )
    text = "\n".join(line.text for line in lines)
    return FrozenEvidenceBlock(
        block_id=block.block_id,
        paper_id=paper_id,
        source_id=block.source_id,
        page=block.page,
        source_column_start=block.source_column_start,
        source_column_end=block.source_column_end,
        lines=lines,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        claimed_anchor_untrusted=UntrustedClaimedAnchor.from_anchor(anchor),
    )
