"""Propose negative-control regions from places extraction actually looks.

A negative control can distinguish a correct rejection only when it lies inside the
extractor's actual search space. A control that extraction never sees cannot test the
extractor's decision.

So proposals are drawn only from inside blocks that were actually sent to the extractor,
and only where a deterministic rule can say why the number is not a primary result of this
paper.

These are **proposals, not annotations**. Each carries the rule that produced it and a
verbatim quote taken from the frozen layout, and each is written to a separate worklist
that no scorer reads. A human confirms or rejects before anything reaches
``references/``. Nothing here may assert a fact about a paper on its own.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import Field

from proceedings_to_eee.domain.base import StrictModel
from proceedings_to_eee.extraction.pdf_layout import PdfLayout
from proceedings_to_eee.extraction.region_index import (
    RegionKind,
    build_region_index,
)
from proceedings_to_eee.extraction.result_blocks import ResultBlock
from proceedings_to_eee.reference import PaperReference
from proceedings_to_eee.resolution.attribution import AttributionLexicon, load_lexicon

CONTROL_PROPOSAL_SCHEMA_VERSION = "control-proposal-worklist/0.1"

# "Invented cohort (n = 731)", "Synthetic appendix (N=1,407)". A sample count is
# method metadata: it must never become a reported result.
_SAMPLE_COUNT = re.compile(r"\(\s*[nN]\s*=\s*[\d,]+\s*\)")


class ControlProposal(StrictModel):
    """One deterministically motivated candidate control, awaiting human confirmation."""

    paper_id: str = Field(min_length=1)
    rule_id: str = Field(min_length=1)
    expected_claim_type: str | None = None
    reason: str = Field(min_length=1)
    page: int = Field(ge=1)
    kind: str = Field(min_length=1)
    label: str | None = None
    row: str | None = None
    exact_quote: str = Field(min_length=1)
    block_id: str = Field(min_length=1)
    confirmed: bool = False


def _examined_spans(blocks: list[ResultBlock]) -> dict[int, list[tuple[int, int, str]]]:
    spans: dict[int, list[tuple[int, int, str]]] = {}
    for block in blocks:
        spans.setdefault(block.page, []).append(
            (block.body_start_line, block.body_end_line, block.block_id)
        )
    return spans


def _covering_block(
    spans: dict[int, list[tuple[int, int, str]]], page: int, line: int
) -> str | None:
    for start, end, block_id in spans.get(page, ()):
        if start <= line <= end:
            return block_id
    return None


def _annotated_quotes(reference: PaperReference | None) -> set[str]:
    if reference is None:
        return set()
    return {re.sub(r"\s+", " ", item.exact_quote).strip().casefold() for item in reference.evidence}


def _annotated_table_labels(reference: PaperReference | None) -> set[tuple[int, str]]:
    """Tables that already carry an annotated result, as (page, normalized label)."""

    if reference is None:
        return set()
    return {
        (item.page, re.sub(r"\s+", " ", item.label).strip().casefold())
        for item in reference.evidence
        if item.label and item.purpose.value in {"result", "negative_control"}
    }


def propose_controls(
    *,
    paper_id: str,
    layout: PdfLayout,
    blocks: list[ResultBlock],
    reference: PaperReference | None = None,
    lexicon: AttributionLexicon | None = None,
) -> list[ControlProposal]:
    """Return deterministic control proposals covered by an extracted block."""

    lexicon = lexicon or load_lexicon()
    indexes = build_region_index(layout)
    spans = _examined_spans(blocks)
    already = _annotated_quotes(reference)
    annotated_tables = _annotated_table_labels(reference)
    proposals: list[ControlProposal] = []

    for page in sorted(indexes):
        page_index = indexes[page]
        for region in page_index.regions:
            if region.in_references:
                continue
            if region.kind is RegionKind.TABLE:
                proposals.extend(
                    _table_proposals(
                        paper_id, region, spans, page, lexicon, already, annotated_tables
                    )
                )
            elif region.kind is RegionKind.CAPTION:
                proposals.extend(_caption_proposals(paper_id, region, spans, page, already))
    return proposals


def _table_proposals(
    paper_id: str,
    region: Any,
    spans: dict[int, list[tuple[int, int, str]]],
    page: int,
    lexicon: AttributionLexicon,
    already: set[str],
    annotated_tables: set[tuple[int, str]],
) -> list[ControlProposal]:
    found: list[ControlProposal] = []
    table_key = (
        page,
        re.sub(r"\s+", " ", region.table_label).strip().casefold() if region.table_label else "",
    )
    in_annotated_table = table_key in annotated_tables
    for row in region.rows:
        block_id = _covering_block(spans, page, row.line)
        if block_id is None:
            continue
        quote = " ".join(row.cells).strip()
        if not quote or re.sub(r"\s+", " ", quote).casefold() in already:
            continue
        label = row.effective_row_label

        cues = lexicon.decisive_matches(label) if label else []
        if cues and not row.is_header:
            found.append(
                ControlProposal(
                    paper_id=paper_id,
                    rule_id="foreign_row_cue",
                    expected_claim_type="secondary_claim",
                    reason=(
                        "The row label carries an attributing cue "
                        f"({', '.join(cue.cue_id for cue in cues)}), so the value is "
                        "reported from another source rather than produced here."
                    ),
                    page=page,
                    kind="table",
                    label=region.table_label,
                    row=label,
                    exact_quote=quote,
                    block_id=block_id,
                )
            )
            continue

        sample = _SAMPLE_COUNT.search(quote)
        if sample:
            found.append(
                ControlProposal(
                    paper_id=paper_id,
                    rule_id="sample_count_in_table",
                    expected_claim_type="method_metadata",
                    reason=(
                        f"{sample.group(0)} states how many items the column covers. A "
                        "sample count is method metadata, never a reported result."
                    ),
                    page=page,
                    kind="table",
                    label=region.table_label,
                    row=label,
                    exact_quote=quote,
                    block_id=block_id,
                )
            )
            continue

        # A sibling row of an already-annotated table sits exactly where extraction
        # looks, which makes it the cheapest place to grow the annotation. It is NOT
        # proposed as a control: a paper may legitimately report primary results for
        # many systems at once, so whether this row is the paper's own or someone
        # else's is a question only a human reading the paper can answer.
        if in_annotated_table and not row.is_header and label:
            found.append(
                ControlProposal(
                    paper_id=paper_id,
                    rule_id="sibling_row_needs_label",
                    expected_claim_type=None,
                    reason=(
                        "Sibling row of a table that already carries an annotation. "
                        "Needs a human own-or-foreign decision; no claim type is asserted."
                    ),
                    page=page,
                    kind="table",
                    label=region.table_label,
                    row=label,
                    exact_quote=quote,
                    block_id=block_id,
                )
            )
    return found


def _caption_proposals(
    paper_id: str,
    region: Any,
    spans: dict[int, list[tuple[int, int, str]]],
    page: int,
    already: set[str],
) -> list[ControlProposal]:
    if region.caption is None:
        return []
    block_id = _covering_block(spans, page, region.span.start_line)
    if block_id is None:
        return []
    sample = _SAMPLE_COUNT.search(region.caption.text)
    if not sample:
        return []
    if re.sub(r"\s+", " ", region.caption.text).casefold() in already:
        return []
    return [
        ControlProposal(
            paper_id=paper_id,
            rule_id="sample_count_in_caption",
            expected_claim_type="method_metadata",
            reason=(
                f"The caption states {sample.group(0)}. A sample count is method "
                "metadata, never a reported result."
            ),
            page=page,
            kind="table",
            label=region.caption.label,
            row=None,
            exact_quote=region.caption.text,
            block_id=block_id,
        )
    ]


def worklist(proposals: list[ControlProposal]) -> dict[str, Any]:
    """Wrap proposals in a document that is explicitly not a reference annotation."""

    by_rule: dict[str, int] = {}
    for proposal in proposals:
        by_rule[proposal.rule_id] = by_rule.get(proposal.rule_id, 0) + 1
    asserted = [item for item in proposals if item.expected_claim_type is not None]
    needs_label = [item for item in proposals if item.expected_claim_type is None]
    return {
        "schema_version": CONTROL_PROPOSAL_SCHEMA_VERSION,
        "status": "proposed-unconfirmed",
        "warning": (
            "These are deterministic proposals, not annotations. No scorer reads this "
            "file. Each entry needs human confirmation against the rendered PDF before "
            "it may be copied into references/."
        ),
        "counts": {
            "total": len(proposals),
            "with_asserted_claim_type": len(asserted),
            "needing_a_human_label": len(needs_label),
            "by_rule": dict(sorted(by_rule.items())),
        },
        "proposals": [proposal.model_dump(mode="json") for proposal in asserted],
        "rows_needing_a_human_label": [item.model_dump(mode="json") for item in needs_label],
    }
