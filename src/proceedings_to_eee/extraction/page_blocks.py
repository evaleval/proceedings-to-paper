"""Segment selected pages into result blocks and keep bibliography blocks off the model.

A reference list is dense with years and page ranges, so the segmenter reads it as
sparse tables. The region index tracks the bibliography across page breaks; the
page-local heading rule remains as a second, independent criterion for pages where the
index could not establish section state. Both criteria describe where a block sits,
not what it says, so neither may drop a block whose own lines do not read like
bibliography entries: the index carries ``in_references`` through letter-numbered
appendices and into the second column of a page whose first column is the
bibliography, and the heading rule covers everything printed below ``References``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from proceedings_to_eee.extraction.pdf_layout import (
    PageFragment,
    PdfLayout,
    reference_heading_line,
)
from proceedings_to_eee.extraction.region_index import PageRegionIndex, build_region_index
from proceedings_to_eee.extraction.result_blocks import (
    _CAPTION_ANY,
    _DECIMAL_OR_PERCENT,
    _NUMERIC,
    _RESULT_TERMS,
    _STATISTIC,
    ResultBlock,
    ResultBlockConfig,
    _page_line_features,
    segment_page_result_blocks,
)

#: Lines above a block's first line in which a caption still belongs to that block. A
#: caption printed above a table can be clustered as the trailing context of a dense
#: preceding cluster, which inside a bibliography is the reference list itself.
_CAPTION_LOOKBACK_LINES = 3

#: Fields a bibliography entry prints and a result passage does not: a bracketed entry
#: number, a year used as a field separator, a page range closed by punctuation, a
#: volume(issue) pair, a DOI, URL or arXiv identifier, a venue or publisher, and a
#: two-digit venue year such as ``KDD '15``. Prose years and ranges inside a sentence
#: (``0-4 scale``, ``3-4 days``) are deliberately outside the pattern.
_BIBLIOGRAPHY_FIELD = re.compile(
    r"(?:^\s*\[\d{1,4}\]"
    r"|\b(?:19|20)\d{2}[a-z]?\."
    r"|\((?:[A-Za-z]+\s+)?(?:19|20)\d{2}(?:-\d{2}-\d{2})?\)"
    r"|\b\d{1,5}[-\u2013\u2014]\d{1,5}(?:[.,;]|\s*$)"
    r"|\b\d{1,4}\s?\(\d{1,4}\)\s*[:,]"
    r"|(?i:\b(?:vol|no|pp?)\.\s*\d)"
    r"|(?i:\barticle\s+\d+)"
    r"|(?i:\bdoi\b|\bhttps?:)"
    r"|\b10\.\d{4,}/"
    r"|(?i:\.(?:io|org|com|edu|net|gov)/|%[0-9a-f]{2}%[0-9a-f]{2}|\.pdf\b)"
    r"|(?i:\barxiv\b)"
    r"|(?i:\bin\s+proc(?:eedings|\.))"
    r"|\bProc\.\s+[A-Z]"
    r"|\b[A-Z][\w&]*\s+Press\b"
    r"|\b[A-Z]{2,}\s?[\u2019']\d{2}\b)"
)
_COLUMN_GAP = re.compile(r"\s{2,}")


@dataclass(frozen=True, slots=True)
class SegmentedPage:
    """Blocks retained for one page plus the count filtered out as bibliography."""

    page: int
    blocks: tuple[ResultBlock, ...]
    blocks_dropped_in_references: int


@dataclass(frozen=True, slots=True)
class PageSegmentation:
    """Document-level segmentation outcome in page order, with telemetry totals."""

    page_count: int
    pages: tuple[SegmentedPage, ...]

    @property
    def blocks(self) -> list[ResultBlock]:
        return [block for page in self.pages for block in page.blocks]

    @property
    def blocks_dropped_in_references(self) -> int:
        return sum(page.blocks_dropped_in_references for page in self.pages)

    @property
    def tables_chunked(self) -> int:
        return count_chunked_tables(self.blocks)

    def telemetry(self) -> dict[str, Any]:
        return {
            "page_count": self.page_count,
            "pages_selected": len(self.pages),
            "blocks_total": sum(len(page.blocks) for page in self.pages),
            "blocks_dropped_in_references": self.blocks_dropped_in_references,
            "tables_chunked": self.tables_chunked,
        }


def count_chunked_tables(blocks: Sequence[ResultBlock]) -> int:
    """Count tables split into several chunks: first chunks that carry data rows."""

    return sum(
        block.chunk_index == 1 and block.chunk_count > 1 and block.data_row_count > 0
        for block in blocks
    )


def _covers_block_panel(
    block: ResultBlock, column_start: int | None, column_end: int | None
) -> bool:
    if block.source_column_start is None or block.source_column_end is None:
        return True
    if column_start is None or column_end is None:
        return True
    return block.source_column_start <= column_start and column_end <= block.source_column_end


def _reference_lines(index: PageRegionIndex, block: ResultBlock) -> set[int]:
    lines: set[int] = set()
    for region in index.regions:
        if not region.in_references or not _covers_block_panel(
            block, region.span.column_start, region.span.column_end
        ):
            continue
        lines.update(range(region.span.start_line, region.span.end_line + 1))
    return lines


def _carries_result_marker(
    block: ResultBlock, page_lines: Sequence[str], config: ResultBlockConfig
) -> bool:
    """A caption or result heading in, just above, or (for a table) below the block.

    A bibliography carries none of these, so their presence keeps a block even when
    the section state says otherwise: dropping is irreversible for the run, and an
    appendix table printed after the references must survive. A trailing caption alone
    does not count, because the caption of the next table can be clustered onto the
    reference entries above it; it counts only when the body itself holds table rows.
    """

    features = _page_line_features(
        (block.context_text + block.body_text).splitlines(), config.min_signal_score
    )
    if any(feature.caption or feature.result_heading for feature in features):
        return True
    first_line = block.context_start_line or block.body_start_line
    window = page_lines[max(0, first_line - 1 - _CAPTION_LOOKBACK_LINES) : first_line - 1]
    if any(_CAPTION_ANY.search(line) for line in window):
        return True
    trailing_caption = any(
        _CAPTION_ANY.search(line) for line in block.trailing_context_text.splitlines()
    )
    return trailing_caption and any(
        "aligned_numeric_row" in feature.signal_kinds for feature in features
    )


def _reports_a_result(line: str) -> bool:
    """A column segment pairing a result term or statistic with a value, bibliography-free.

    Segments are split at the column gap so that a full-width block over a two-column
    page is judged column by column: a bibliography entry on the left cannot mask a
    score reported on the right. A title that names a metric beside its year is a
    bibliography segment and does not count.
    """

    for segment in _COLUMN_GAP.split(line.strip()):
        if not segment or _BIBLIOGRAPHY_FIELD.search(segment):
            continue
        if _RESULT_TERMS.search(segment) and _DECIMAL_OR_PERCENT.search(segment):
            return True
        if _STATISTIC.search(segment) and _NUMERIC.search(segment):
            return True
    return False


def body_reads_like_bibliography(lines: Sequence[str]) -> bool:
    """At least half of the content lines carry a bibliography field and none reports a result.

    Wrapped titles carry no field, so a strict majority would keep ordinary two-line
    entries; a reported value anywhere in the block outweighs the fields around it.
    """

    content = [line for line in lines if line.strip()]
    if not content or any(_reports_a_result(line) for line in content):
        return False
    return 2 * sum(bool(_BIBLIOGRAPHY_FIELD.search(line)) for line in content) >= len(content)


def block_in_references(
    block: ResultBlock,
    *,
    index: PageRegionIndex | None,
    heading_line: int | None,
    page_lines: Sequence[str],
    config: ResultBlockConfig | None = None,
) -> bool:
    """True when a bibliography-like body sits entirely in the bibliography.

    The first criterion is the region index, which carries ``in_references`` across page
    breaks. The second is the page-local heading: a block that starts below a
    ``References`` heading is a bibliography block even when the index could not
    classify it. Neither criterion applies unless the body itself reads like
    bibliography entries, and a block that carries a result marker is always kept.
    """

    config = config or ResultBlockConfig()
    body_lines = block.body_text.splitlines()
    content_lines = [
        block.body_start_line + offset for offset, text in enumerate(body_lines) if text.strip()
    ]
    if (
        not content_lines
        or _carries_result_marker(block, page_lines, config)
        or not body_reads_like_bibliography(body_lines)
    ):
        return False
    if index is not None:
        covered = _reference_lines(index, block)
        if all(line in covered for line in content_lines):
            return True
    return heading_line is not None and block.body_start_line > heading_line


def segment_result_pages(
    layout: PdfLayout,
    pages: Sequence[PageFragment],
    config: ResultBlockConfig | None = None,
) -> PageSegmentation:
    """Segment ``pages`` in page order and drop blocks that lie inside the bibliography.

    The region index is built from the whole layout with the same configuration as the
    segmenter so that both see identical column panels.
    """

    config = config or ResultBlockConfig()
    region_indexes = build_region_index(layout, config)
    segmented: list[SegmentedPage] = []
    for page in sorted(pages, key=lambda item: item.page):
        blocks = segment_page_result_blocks(page, config=config)
        index = region_indexes.get(page.page)
        heading_line = reference_heading_line(page.text)
        page_lines = page.text.splitlines()
        kept = tuple(
            block
            for block in blocks
            if not block_in_references(
                block,
                index=index,
                heading_line=heading_line,
                page_lines=page_lines,
                config=config,
            )
        )
        segmented.append(
            SegmentedPage(
                page=page.page,
                blocks=kept,
                blocks_dropped_in_references=len(blocks) - len(kept),
            )
        )
    return PageSegmentation(page_count=layout.page_count, pages=tuple(segmented))
