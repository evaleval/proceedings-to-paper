"""Whether a negative control's region was ever examined by extraction.

A negative control that attracts no candidate is ambiguous on its own. It can mean the
pipeline looked at the region and correctly declined to turn it into a result, which is
the safety property the control exists to test. It can also mean the region was never
put in front of the extractor at all, which says nothing.

Collapsing those two into "no match" is what left the holdout's safety gates unmeasurable
on nine of ten papers. This module separates them deterministically: a control counts as
examined when a bounded result block that was actually sent for extraction covers the
lines its evidence quote occupies.
"""

from __future__ import annotations

from proceedings_to_eee.extraction.pdf_layout import PdfLayout
from proceedings_to_eee.extraction.region_index import build_region_index, locate_quote_in_index
from proceedings_to_eee.extraction.result_blocks import ResultBlock
from proceedings_to_eee.reference import PaperReference, ReferenceEvidence


def _block_line_ranges(blocks: list[ResultBlock]) -> dict[int, list[tuple[int, int]]]:
    """Every page line range that was sent to the extractor, by page."""

    ranges: dict[int, list[tuple[int, int]]] = {}
    for block in blocks:
        spans = [(block.body_start_line, block.body_end_line)]
        if block.context_start_line is not None and block.context_end_line is not None:
            spans.append((block.context_start_line, block.context_end_line))
        if (
            block.trailing_context_start_line is not None
            and block.trailing_context_end_line is not None
        ):
            spans.append((block.trailing_context_start_line, block.trailing_context_end_line))
        ranges.setdefault(block.page, []).extend(spans)
    return ranges


def _evidence_lines(
    evidence: ReferenceEvidence, layout: PdfLayout, indexes: dict[int, object]
) -> tuple[int, int] | None:
    """Locate an annotation quote in the frozen layout, in page line ordinals."""

    page_index = indexes.get(evidence.page)
    page = next((item for item in layout.pages if item.page == evidence.page), None)
    if page_index is None or page is None:
        return None
    location = locate_quote_in_index(page_index, page, evidence.exact_quote)
    if location is None:
        return None
    return location.span.start_line, location.span.end_line


def control_examination(
    reference: PaperReference,
    layout: PdfLayout,
    blocks: list[ResultBlock],
) -> dict[str, bool]:
    """Map each control id to whether an extracted block covered its evidence lines.

    A control whose quote cannot be located in the layout at all is reported as not
    examined. That is the conservative reading: an annotation transcribed from the
    rendered PDF may have no counterpart in the text the extractor actually saw.
    """

    indexes = build_region_index(layout)
    ranges = _block_line_ranges(blocks)
    evidence_by_id = {item.evidence_id: item for item in reference.evidence}
    examined: dict[str, bool] = {}
    for control in reference.negative_controls:
        covered = False
        for evidence_id in control.evidence_ids:
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None:
                continue
            lines = _evidence_lines(evidence, layout, indexes)
            if lines is None:
                continue
            start, end = lines
            if any(
                block_start <= end and start <= block_end
                for block_start, block_end in ranges.get(evidence.page, ())
            ):
                covered = True
                break
        examined[control.control_id] = covered
    return examined
